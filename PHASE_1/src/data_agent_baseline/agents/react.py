from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from data_agent_baseline.agents.aggregate_grain import (
    dual_grain_rejected_observation,
    dual_grain_satisfied,
    pre_submit_grain_notes,
)
from data_agent_baseline.agents.answer_check import (
    prune_answer_columns,
    has_full_sql_scan,
    sql_answer_rejected_observation,
)
from data_agent_baseline.agents.gate_common import is_undecided
from data_agent_baseline.agents.grain_contract import (
    submit_grain_contract_post,
    submit_grain_contract_pre,
)
from data_agent_baseline.agents.hypothesis import (
    EvidenceState,
    Hypothesis,
    build_hypotheses,
    evidence_guidance,
    format_ambiguity_probe_block,
    format_hypothesis_plan,
    update_evidence,
)
from data_agent_baseline.agents.model import ModelAdapter, ModelMessage, ModelStep
from data_agent_baseline.agents.prompt import (
    REACT_SYSTEM_PROMPT,
    build_observation_prompt,
    build_system_prompt,
    build_task_prompt,
    load_knowledge_md,
    prepare_knowledge_text,
)
from data_agent_baseline.agents.schema_link import SchemaLinkPlan, link_schema
from data_agent_baseline.agents.submit_rows import (
    submit_projection_rejection,
    submit_row_rejection,
)
from data_agent_baseline.agents.submit_validation import (
    submit_ambiguity_probe_rejection,
    submit_id_column_rejection,
    submit_sanity_rejection,
    submit_shape_rejection,
    submit_symmetric_relation_rejection,
)
from data_agent_baseline.agents.measure_identity import submit_measure_identity_rejection
from data_agent_baseline.agents.predicate_scope import submit_predicate_scope_rejection
from data_agent_baseline.agents.value_normalize import (
    NormalizePlan,
    build_normalize_plan,
    find_promotable_probe,
    format_normalize_plan,
)
from data_agent_baseline.agents.voting import vote_best_final
from data_agent_baseline.agents.runtime import AgentRunResult, AgentRuntimeState, StepRecord
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.registry import ToolExecutionResult, ToolRegistry


_RATE_LIMIT_RE = re.compile(
    r"\b429\b|rate[\s_-]?limit|insufficient[_\s]?quota|quota[\s_-]?exceeded|"
    r"usage[\s_-]?limit|too many requests|tokens?\s+(?:exhausted|exceeded)|"
    r"billing|credit|余额不足|用量超",
    flags=re.IGNORECASE,
)


def is_rate_limit_error(text: str) -> bool:
    """True when an exception / observation looks like API 429 or quota exhaustion."""
    return bool(_RATE_LIMIT_RE.search(text or ""))


@dataclass(frozen=True, slots=True)
class ReActAgentConfig:
    max_steps: int = 16


def _strip_json_fence(raw_response: str) -> str:
    text = raw_response.strip()
    fence_match = re.search(r"```json\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence_match is not None:
        return fence_match.group(1).strip()
    generic_fence_match = re.search(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
    if generic_fence_match is not None:
        return generic_fence_match.group(1).strip()
    return text


def _load_single_json_object(text: str) -> dict[str, object]:
    payload, end = json.JSONDecoder().raw_decode(text)
    remainder = text[end:].strip()
    if remainder:
        cleaned_remainder = re.sub(r"(?:\\[nrt])+", "", remainder).strip()
        if cleaned_remainder:
            raise ValueError("Model response must contain only one JSON object.")
    if not isinstance(payload, dict):
        raise ValueError("Model response must be a JSON object.")
    return payload


def parse_model_step(raw_response: str) -> ModelStep:
    normalized = _strip_json_fence(raw_response)
    payload = _load_single_json_object(normalized)

    thought = payload.get("thought", "")
    action = payload.get("action")
    action_input = payload.get("action_input", {})
    if not isinstance(thought, str):
        raise ValueError("thought must be a string.")
    if not isinstance(action, str) or not action:
        raise ValueError("action must be a non-empty string.")
    if not isinstance(action_input, dict):
        raise ValueError("action_input must be a JSON object.")

    return ModelStep(
        thought=thought,
        action=action,
        action_input=action_input,
        raw_response=raw_response,
    )


def _rejection_type(content: dict[str, Any]) -> str:
    for key in content:
        if key.endswith("_check"):
            return key[:-6] or "unknown"
    return "unknown"


class ReActAgent:
    def __init__(
        self,
        *,
        model: ModelAdapter,
        tools: ToolRegistry,
        config: ReActAgentConfig | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.config = config or ReActAgentConfig()
        self.system_prompt = system_prompt or REACT_SYSTEM_PROMPT

    def _note_repair(self, state: AgentRuntimeState, content: dict[str, Any]) -> str:
        """Increment repair_counts for a typed failure; return the error type key."""
        rtype = _rejection_type(content)
        if rtype == "unknown":
            return rtype
        state.repair_counts[rtype] = state.repair_counts.get(rtype, 0) + 1
        return rtype

    def _maybe_reject(
        self,
        state: AgentRuntimeState,
        content: dict[str, Any],
    ) -> ToolExecutionResult | None:
        """Track repeated rejection types and escalate/abort to avoid loops.

        Returns a rejection ToolExecutionResult, or None when the same error has
        fired too many times and the agent should be allowed to proceed.
        """
        rtype = self._note_repair(state, content)
        count = state.repair_counts.get(rtype, 0)
        if count >= 4:
            return None
        hint = content.get("hint", "")
        if count == 3:
            hint = (
                f"{hint} [This '{rtype}' check has fired 3 times. Do not repeat the "
                "same SQL structure. Either fix the underlying issue with a materially "
                "different query or submit the closest valid result you can produce.]"
            )
        elif count == 2:
            hint = (
                f"{hint} [This '{rtype}' check fired twice. Change the approach; "
                "re-running the same query will be rejected again.]"
            )
        content["hint"] = hint
        return ToolExecutionResult(ok=False, content=content)

    @staticmethod
    def _note_gate_telemetry(state: AgentRuntimeState, key: str) -> None:
        """Count gate events (§13.5): undecided verdicts, pre-submit notes."""
        state.gate_telemetry[key] = state.gate_telemetry.get(key, 0) + 1

    def _prepare_knowledge(self, task: PublicTask) -> str:
        full = load_knowledge_md(task) or ""
        return prepare_knowledge_text(full, task.question)

    def _prepare_schema(self, task: PublicTask) -> str:
        session = self.tools.session
        if session is None:
            return ""
        from data_agent_baseline.tools.warehouse import format_table_schema

        # force_rebuild rebuilds an in-memory warehouse (legacy on-disk leftovers and
        # spill tmp cleared) so every run/retry starts clean.
        return format_table_schema(session.for_task(task, force_rebuild=True))

    def _prepare_schema_link(
        self,
        task: PublicTask,
        *,
        knowledge_text: str,
    ) -> SchemaLinkPlan | None:
        session = self.tools.session
        if session is None or session.state is None:
            return None
        from data_agent_baseline.tools.warehouse import describe_tables

        try:
            tables = describe_tables(session.state)
        except Exception:
            return None
        return link_schema(
            question=task.question,
            knowledge_text=knowledge_text,
            tables=tables,
        )

    def _build_messages(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        *,
        knowledge_text: str,
        schema_text: str,
        schema_link_text: str,
        hypothesis_text: str,
        normalize_text: str,
        evidence: EvidenceState,
        hypotheses: list[Hypothesis],
        normalize_plan: NormalizePlan,
    ) -> list[ModelMessage]:
        system_content = build_system_prompt(
            self.tools.describe_for_prompt(),
            system_prompt=self.system_prompt,
        )
        messages = [ModelMessage(role="system", content=system_content)]
        messages.append(
            ModelMessage(
                role="user",
                content=build_task_prompt(
                    task,
                    knowledge_text=knowledge_text,
                    schema_text=schema_text,
                    schema_link_text=schema_link_text or None,
                    hypothesis_text=hypothesis_text or None,
                    normalize_text=normalize_text or None,
                ),
            )
        )
        remaining_after_last = max(self.config.max_steps - len(state.steps), 0)
        for index, step in enumerate(state.steps):
            messages.append(ModelMessage(role="assistant", content=step.raw_response))
            is_last = index == len(state.steps) - 1
            remaining_steps = remaining_after_last if is_last else None
            notes = None
            if is_last:
                notes = evidence_guidance(
                    evidence=evidence,
                    hypotheses=hypotheses,
                    question=task.question,
                    steps=state.steps,
                    remaining_steps=remaining_steps,
                    normalize_plan=normalize_plan,
                ) or None
            messages.append(
                ModelMessage(
                    role="user",
                    content=build_observation_prompt(
                        step.observation,
                        remaining_steps=remaining_steps,
                        evidence_notes=notes,
                    ),
                )
            )
        return messages

    def run(self, task: PublicTask) -> AgentRunResult:
        from data_agent_baseline.run.progress import mark as _mark

        state = AgentRuntimeState()
        knowledge_text = ""
        schema_text = ""
        schema_link_text = ""
        hypothesis_text = ""
        normalize_text = ""
        hypotheses: list[Hypothesis] = []
        evidence = EvidenceState()
        normalize_plan = NormalizePlan()
        self._knowledge_text = ""
        self._schema_link: SchemaLinkPlan | None = None
        self._normalize_plan = normalize_plan
        try:
            _mark("prepare_knowledge")
            knowledge_text = self._prepare_knowledge(task)
            self._knowledge_text = knowledge_text
            _mark("prepare_knowledge_done")
        except Exception as exc:  # noqa: BLE001
            knowledge_text = ""
            state.steps.append(
                StepRecord(
                    step_index=0,
                    thought="knowledge_prepare_failed",
                    action="__knowledge__",
                    action_input={},
                    raw_response="",
                    observation={"ok": False, "error": str(exc)},
                    ok=False,
                )
            )
        try:
            _mark("prepare_schema")
            schema_text = self._prepare_schema(task)
            _mark("prepare_schema_done", schema_chars=len(schema_text or ""))
        except Exception as exc:  # noqa: BLE001
            from data_agent_baseline.tools.doc_extract import DocumentExtractionError

            schema_text = ""
            state.steps.append(
                StepRecord(
                    step_index=0,
                    thought="schema_prepare_failed",
                    action="__schema__",
                    action_input={},
                    raw_response="",
                    observation={"ok": False, "error": str(exc)},
                    ok=False,
                )
            )
            if isinstance(exc, DocumentExtractionError):
                # Extract is isolated from ReAct: skip failed docs, keep csv/json/sqlite.
                _mark("prepare_schema_extract_failed", error=str(exc)[:300])

        # L1.0 value normalization plan (soft anchor for literal ↔ stored alignment).
        try:
            _mark("prepare_normalize")
            normalize_plan = build_normalize_plan(
                question=task.question,
                knowledge_text=knowledge_text,
            )
            self._normalize_plan = normalize_plan
            normalize_text = format_normalize_plan(normalize_plan)
            _mark("prepare_normalize_done")
        except Exception as exc:  # noqa: BLE001
            state.steps.append(
                StepRecord(
                    step_index=0,
                    thought="normalize_prepare_failed",
                    action="__normalize__",
                    action_input={},
                    raw_response="",
                    observation={"ok": False, "error": str(exc)},
                    ok=False,
                )
            )

        # L0.5 schema linking + L1.5 hypothesis plan (soft anchors for ReAct).
        try:
            _mark("prepare_schema_link")
            link = self._prepare_schema_link(task, knowledge_text=knowledge_text)
            self._schema_link = link
            if link is not None:
                schema_link_text = link.format_for_prompt()
                ambig = format_ambiguity_probe_block(link.ambiguity_groups)
                if ambig:
                    schema_link_text = f"{schema_link_text}\n\n{ambig}".strip()
                hypotheses = build_hypotheses(
                    question=task.question,
                    knowledge_text=knowledge_text,
                    link=link,
                )
                hypothesis_text = format_hypothesis_plan(hypotheses)
            _mark("prepare_schema_link_done")
        except Exception as exc:  # noqa: BLE001
            state.steps.append(
                StepRecord(
                    step_index=0,
                    thought="schema_link_prepare_failed",
                    action="__schema_link__",
                    action_input={},
                    raw_response="",
                    observation={"ok": False, "error": str(exc)},
                    ok=False,
                )
            )

        for step_index in range(1, self.config.max_steps + 1):
            raw_response = ""
            try:
                _mark("react_model_call", step=step_index)
                raw_response = self.model.complete(
                    self._build_messages(
                        task,
                        state,
                        knowledge_text=knowledge_text,
                        schema_text=schema_text,
                        schema_link_text=schema_link_text,
                        hypothesis_text=hypothesis_text,
                        normalize_text=normalize_text,
                        evidence=evidence,
                        hypotheses=hypotheses,
                        normalize_plan=normalize_plan,
                    )
                )
                _mark("react_model_done", step=step_index, response_chars=len(raw_response or ""))
                model_step = parse_model_step(raw_response)
                _mark("react_tool", step=step_index, action=model_step.action)
                tool_result = self._execute_checked_action(task, state, model_step)
                _mark("react_tool_done", step=step_index, action=model_step.action, ok=tool_result.ok)
                if (
                    not tool_result.ok
                    and model_step.action == "run_sql"
                    and isinstance(tool_result.content, dict)
                ):
                    self._note_repair(state, tool_result.content)
                observation = {
                    "ok": tool_result.ok,
                    "tool": model_step.action,
                    "content": tool_result.content,
                }
                step_record = StepRecord(
                    step_index=step_index,
                    thought=model_step.thought,
                    action=model_step.action,
                    action_input=model_step.action_input,
                    raw_response=raw_response,
                    observation=observation,
                    ok=tool_result.ok,
                )
                state.steps.append(step_record)
                update_evidence(
                    evidence,
                    step_record,
                    normalize_plan=normalize_plan,
                )
                if tool_result.is_terminal:
                    state.answer = tool_result.answer
                    break
            except Exception as exc:
                err = str(exc).strip() or type(exc).__name__
                observation = {
                    "ok": False,
                    "error": err,
                    "error_type": type(exc).__name__,
                }
                err_step = StepRecord(
                    step_index=step_index,
                    thought="",
                    action="__error__",
                    action_input={},
                    raw_response=raw_response,
                    observation=observation,
                    ok=False,
                )
                state.steps.append(err_step)
                update_evidence(
                    evidence,
                    err_step,
                    normalize_plan=normalize_plan,
                )
                if isinstance(exc, MemoryError):
                    state.failure_reason = f"Agent stopped after {type(exc).__name__}."
                    break
                if is_rate_limit_error(err):
                    state.failure_reason = (
                        "rate_limit: Model API rate limit or quota exhausted."
                    )
                    break

        if state.answer is None and state.failure_reason is None:
            if not self._try_fallback_submit(task, state):
                state.failure_reason = "Agent did not submit an answer within max_steps."

        return AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            steps=list(state.steps),
            failure_reason=state.failure_reason,
            gate_telemetry=dict(state.gate_telemetry),
        )

    def _restore_voted_final(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
    ) -> None:
        """If multiple finals exist, re-run the voted SQL so session.last_final matches.

        When no final exists, promote a nonempty coarse/normalized probe to final=true.
        """
        session = self.tools.session
        if session is None:
            return
        voted = vote_best_final(question=task.question, steps=state.steps)
        sql: str | None = None
        reason = ""
        if voted is not None:
            sql = voted.sql
            reason = f"vote restore: {voted.reason} (score={voted.score:.2f})"
            if session.last_final_sql and session.last_final_sql.strip() == sql.strip():
                return
        else:
            plan = getattr(self, "_normalize_plan", None)
            promoted = find_promotable_probe(
                question=task.question,
                steps=state.steps,
                plan=plan,
            )
            if promoted is None:
                return
            sql, promo_reason = promoted
            reason = f"probe promote: {promo_reason}"

        try:
            tool_result = self.tools.execute(
                task,
                "run_sql",
                {"sql": sql, "final": True},
            )
        except Exception:
            return
        if not tool_result.ok:
            return
        state.steps.append(
            StepRecord(
                step_index=len(state.steps) + 1,
                thought=reason,
                action="run_sql",
                action_input={"sql": sql, "final": True},
                raw_response="__vote_restore__" if voted is not None else "__probe_promote__",
                observation={
                    "ok": True,
                    "tool": "run_sql",
                    "content": tool_result.content,
                    "vote_restore": voted is not None,
                    "probe_promote": voted is None,
                    "vote_score": voted.score if voted is not None else None,
                    "vote_reason": reason,
                },
                ok=True,
            )
        )

    def _try_fallback_submit(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
    ) -> bool:
        """Submit the best final / promoted probe when the model ran out of steps.

        Votes among successful finals when several exist; otherwise may promote a
        nonempty coarse-normalized probe to final=true. Then submits through L3.
        """
        session = self.tools.session
        if session is None:
            return False

        self._restore_voted_final(task, state)
        if session.last_final is None:
            return False

        synthetic = ModelStep(
            thought="max_steps exhausted; submit voted/promoted final SQL result",
            action="answer",
            action_input={},
            raw_response="__fallback_submit__",
        )
        tool_result = self._execute_checked_action(task, state, synthetic)
        observation = {
            "ok": tool_result.ok,
            "tool": "answer",
            "content": tool_result.content,
            "fallback_submit": True,
        }
        state.steps.append(
            StepRecord(
                step_index=len(state.steps) + 1,
                thought=synthetic.thought,
                action="answer",
                action_input={},
                raw_response="__fallback_submit__",
                observation=observation,
                ok=tool_result.ok,
            )
        )
        if tool_result.is_terminal and tool_result.answer is not None:
            state.answer = tool_result.answer
            return True
        return False

    def _execute_checked_action(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        model_step: ModelStep,
    ) -> ToolExecutionResult:
        if model_step.action == "answer" and not has_full_sql_scan(state.steps):
            result = self._maybe_reject(state, sql_answer_rejected_observation())
            if result is not None:
                return result
        if model_step.action == "answer" and not dual_grain_satisfied(
            task.question, state.steps, getattr(self, "_knowledge_text", None)
        ):
            result = self._maybe_reject(
                state,
                dual_grain_rejected_observation(
                    task.question,
                    state.steps,
                    getattr(self, "_knowledge_text", None),
                ),
            )
            if result is not None:
                return result
        if model_step.action == "answer":
            session = self.tools.session
            grain_payload = submit_grain_contract_pre(
                task.question,
                state.steps,
                sql=session.last_final_sql if session is not None else None,
                knowledge_text=getattr(self, "_knowledge_text", None),
            )
            if grain_payload is not None:
                if is_undecided(grain_payload):
                    # §13.5 UNDECIDED: let the answer through, record telemetry.
                    self._note_gate_telemetry(
                        state, f"{grain_payload.get('check', 'gate')}_undecided"
                    )
                else:
                    result = self._maybe_reject(state, grain_payload)
                    if result is not None:
                        return result

        tool_result = self.tools.execute(task, model_step.action, model_step.action_input)
        if (
            tool_result.ok
            and model_step.action == "run_sql"
            and isinstance(model_step.action_input, dict)
            and model_step.action_input.get("final")
            and isinstance(tool_result.content, dict)
        ):
            # §13.5 early warning: surface gate verdicts on the final SQL before
            # the model spends a step on answer.
            gate_notes = pre_submit_grain_notes(
                task.question,
                state.steps,
                str(model_step.action_input.get("sql") or ""),
                getattr(self, "_knowledge_text", None),
            )
            if gate_notes:
                self._note_gate_telemetry(state, "grain_pre_notes")
                content = dict(tool_result.content)
                content["gate_notes"] = gate_notes
                tool_result = ToolExecutionResult(
                    ok=tool_result.ok,
                    content=content,
                    is_terminal=tool_result.is_terminal,
                    answer=tool_result.answer,
                )
        if not (tool_result.is_terminal and tool_result.answer is not None):
            return tool_result

        session = self.tools.session
        rejection = submit_row_rejection(
            task.question,
            tool_result.answer,
            sql=session.last_final_sql if session is not None else None,
            state=session.state if session is not None else None,
        )
        if rejection is not None:
            result = self._maybe_reject(state, rejection)
            if result is not None:
                return result

        membership = submit_grain_contract_post(
            task.question,
            state.steps,
            sql=session.last_final_sql if session is not None else None,
            answer=tool_result.answer,
        )
        if membership is not None:
            result = self._maybe_reject(state, membership)
            if result is not None:
                return result

        link = getattr(self, "_schema_link", None)
        ambiguity = submit_ambiguity_probe_rejection(
            ambiguity_groups=link.ambiguity_groups if link is not None else None,
            steps=state.steps,
            sql=session.last_final_sql if session is not None else None,
        )
        if ambiguity is not None:
            if is_undecided(ambiguity):
                self._note_gate_telemetry(
                    state, f"{ambiguity.get('check', 'ambiguity')}_undecided"
                )
            else:
                result = self._maybe_reject(state, ambiguity)
                if result is not None:
                    return result

        predicate_scope = submit_predicate_scope_rejection(
            task.question,
            getattr(self, "_knowledge_text", None),
            sql=session.last_final_sql if session is not None else None,
        )
        if predicate_scope is not None:
            if is_undecided(predicate_scope):
                self._note_gate_telemetry(
                    state,
                    f"{predicate_scope.get('check', 'predicate_scope')}_undecided",
                )
            else:
                result = self._maybe_reject(state, predicate_scope)
                if result is not None:
                    return result

        measure_id = submit_measure_identity_rejection(
            task.question,
            getattr(self, "_knowledge_text", None),
            sql=session.last_final_sql if session is not None else None,
            candidates=link.candidate_columns if link is not None else None,
        )
        if measure_id is not None:
            result = self._maybe_reject(state, measure_id)
            if result is not None:
                return result

        relation = submit_symmetric_relation_rejection(
            sql=session.last_final_sql if session is not None else None,
            state=session.state if session is not None else None,
        )
        if relation is not None:
            result = self._maybe_reject(state, relation)
            if result is not None:
                return result

        shape = submit_shape_rejection(
            task.question,
            tool_result.answer,
            state.steps,
        )
        if shape is not None:
            result = self._maybe_reject(state, shape)
            if result is not None:
                return result

        id_cols = submit_id_column_rejection(task.question, tool_result.answer)
        if id_cols is not None:
            result = self._maybe_reject(state, id_cols)
            if result is not None:
                return result

        sanity = submit_sanity_rejection(
            sql=session.last_final_sql if session is not None else None,
            answer=tool_result.answer,
        )
        if sanity is not None:
            result = self._maybe_reject(state, sanity)
            if result is not None:
                return result

        pruned, column_check = prune_answer_columns(
            self.model,
            question=task.question,
            answer=tool_result.answer,
        )
        projection = submit_projection_rejection(
            pruned,
            sql=session.last_final_sql if session is not None else None,
            steps=state.steps,
        )
        if projection is not None:
            projection["column_check"] = column_check
            result = self._maybe_reject(state, projection)
            if result is not None:
                return result
        content = dict(tool_result.content)
        content["column_check"] = column_check
        content["column_count"] = len(pruned.columns)
        content["row_count"] = len(pruned.rows)
        return ToolExecutionResult(
            ok=tool_result.ok,
            content=content,
            is_terminal=True,
            answer=pruned,
        )
