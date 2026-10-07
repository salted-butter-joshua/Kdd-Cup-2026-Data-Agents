"""Hypothesis + evidence layer for the ReAct loop.

Generates soft verification plans from schema links and question/knowledge text,
then scores probe traces so observations can nudge the model without hard-coding
question types.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from data_agent_baseline.agents.aggregate_grain import wants_dual_grain_check
from data_agent_baseline.agents.dual_probe import format_dual_probe_block
from data_agent_baseline.agents.predicate_scope import (
    wants_division_scope_check,
    wants_multi_avg_scope_check,
)
from data_agent_baseline.agents.question_route import classify_question
from data_agent_baseline.agents.runtime import StepRecord
from data_agent_baseline.agents.schema_link import AmbiguityGroup, SchemaLinkPlan
from data_agent_baseline.agents.value_normalize import (
    NormalizePlan,
    sql_is_unrelated_scan,
    sql_uses_coarse_time_match,
    sql_uses_exact_question_literal,
)

_AVG_Q_RE = re.compile(r"\b(?:average|mean|avg)\b", flags=re.IGNORECASE)
_SUM_FORMULA_RE = re.compile(
    r"(?:total\s+annual|/?\s*12|divided\s+by\s*12|/\s*12)",
    flags=re.IGNORECASE,
)
_TIME_LITERAL_RE = re.compile(
    r"\b\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?\b|\b\d{1,2}:\d{2}\.\d+\b"
)
_LIKE_SUFFIX_RE = re.compile(
    r"LIKE\s+'%[_\\]?(\d+)'|LIKE\s+\"%[_\\]?(\d+)\"",
    flags=re.IGNORECASE,
)
_LIST_Q_RE = re.compile(
    r"\b(?:which|what)\b.+\b(?:races?|names?|ids?|elements?|types?|items?)\b|"
    r"\blist\b|\btally\b|\ball\b",
    flags=re.IGNORECASE,
)
_RATIO_Q_RE = re.compile(
    r"\b(?:how\s+many\s+times|ratio|divide[sd]?|times\s+(?:as|larger|greater|more)|"
    r"relative\s+to|per)\b",
    flags=re.IGNORECASE,
)
_KNOWLEDGE_DIVIDE_RE = re.compile(r"\bDIVIDE\b|Count\s*\(", flags=re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class Hypothesis:
    hid: str
    title: str
    rationale: str
    probe_sql_hints: tuple[str, ...] = ()


@dataclass(slots=True)
class EvidenceState:
    consecutive_empty_probes: int = 0
    consecutive_errors: int = 0
    consecutive_exact_empty: int = 0
    coarse_nonempty_hits: int = 0
    last_coarse_sql: str | None = None
    last_row_counts: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def build_hypotheses(
    *,
    question: str,
    knowledge_text: str,
    link: SchemaLinkPlan | None,
) -> list[Hypothesis]:
    """Propose competing interpretations to verify with probes."""
    hyps: list[Hypothesis] = []
    q = question or ""
    knowledge = knowledge_text or ""
    kinds = classify_question(q)

    if link is not None:
        for index, group in enumerate(link.ambiguity_groups[:4]):
            cols = [f"{c.table}.{c.column}" for c in group.columns[:4]]
            probes = tuple(
                f"-- verify '{group.term}' via {col}\n"
                f'SELECT COUNT(*) AS n FROM "{col.split(".")[0]}" '
                f'WHERE "{col.split(".")[1]}" IS NOT NULL'
                for col in cols
            )
            same_name = len({c.column.casefold() for c in group.columns[:4]}) == 1
            project_hint = ""
            if link is not None and same_name:
                term = group.term.casefold()
                for hint in link.homonym_projections:
                    head = hint.split(":", 1)[0].strip().casefold()
                    if head == term:
                        project_hint = hint
                        break
            if same_name and project_hint:
                rationale = (
                    "The same column name exists on multiple tables — they are "
                    "different columns. Probe EACH table.column. Schema linking "
                    f"ranks them as: {project_hint}. Project the entity-table "
                    "copy; keep event-table columns in WHERE only. Fewest joins "
                    "is not a reason to project the filter table's copy."
                )
            elif same_name:
                rationale = (
                    "The same column name exists on multiple tables — they are "
                    "different columns. Probe EACH table.column (a probe on one "
                    "table does not cover the other). Then pick by: (1) knowledge "
                    "mapping; (2) the identifier on the entity table the question "
                    "names; (3) fewest joins / direct FK path."
                )
            else:
                rationale = (
                    "Multiple columns could match this term. Probe each with the "
                    "same filter and compare hit counts / result sets before choosing. "
                    "Then pick by: (1) knowledge mapping; (2) the identifier on the "
                    "entity table the question names; (3) fewest joins / direct FK path."
                )
            hyps.append(
                Hypothesis(
                    hid=f"ambig_col_{index}",
                    title=f"Ambiguous attribute '{group.term}'",
                    rationale=rationale,
                    probe_sql_hints=probes,
                )
            )

    if _AVG_Q_RE.search(q) and _SUM_FORMULA_RE.search(knowledge):
        measure_hint = ""
        if link is not None:
            measures = [
                f"{c.table}.{c.column}"
                for c in link.candidate_columns
                if c.role == "measure"
            ][:4]
            if measures:
                measure_hint = " Candidate measure columns: " + ", ".join(measures) + "."
        hyps.append(
            Hypothesis(
                hid="avg_grain",
                title="Average: SUM/N vs AVG(column)",
                rationale=(
                    "Knowledge may define Total/N while gold often matches AVG(column) "
                    "or per-entity averages. Probe BOTH and compare magnitudes before final."
                    + measure_hint
                ),
                probe_sql_hints=(
                    "-- fine: AVG of measure rows",
                    "-- coarse: SUM(measure)/divisor from knowledge",
                ),
            )
        )

    if "ratio" in kinds:
        hyps.append(
            Hypothesis(
                hid="ratio_direction",
                title="Ratio direction (numerator / denominator reading)",
                rationale=(
                    "Ratio questions can have two self-consistent readings "
                    "(A/B vs B/A, or votes BY an entity vs votes ON its rows). "
                    "Probe both counts, then pick by: (a) knowledge.md; "
                    "(b) direct foreign-key path; (c) fewest joins. Do not oscillate."
                ),
                probe_sql_hints=(
                    "-- reading A: COUNT numerator / COUNT denominator",
                    "-- reading B: invert the two counts",
                ),
            )
        )
    elif _RATIO_Q_RE.search(q) and _KNOWLEDGE_DIVIDE_RE.search(knowledge):
        hyps.append(
            Hypothesis(
                hid="ratio_direction",
                title="Ratio direction (numerator / denominator reading)",
                rationale=(
                    "Ratio questions can have two self-consistent readings "
                    "(e.g. votes cast BY an entity vs votes received ON its rows). "
                    "Probe both counts, then arbitrate by the fixed policy: "
                    "(a) the reading knowledge.md defines; (b) the reading via a "
                    "direct foreign-key path; (c) fewest joins. Do not oscillate."
                ),
                probe_sql_hints=(
                    "-- reading A: COUNT over the entity's own actions",
                    "-- reading B: COUNT over actions targeting the entity's rows",
                ),
            )
        )

    if "clinical" in kinds:
        hyps.append(
            Hypothesis(
                hid="clinical_range",
                title="Clinical normal range from knowledge/docs, not sample quantiles",
                rationale=(
                    "Do not treat Q1–Q3 / NTILE of this warehouse as 'normal'. "
                    "Use a knowledge threshold, or search_docs for a documented range. "
                    "If none exists, leave the bound UNDECIDED rather than inventing one."
                ),
                probe_sql_hints=(
                    "-- knowledge-pattern: WHERE lab.<col> > <documented bound>",
                    "-- forbidden: NTILE / percentile_cont as the normal band",
                ),
            )
        )

    if re.search(r"\b(?:type|category|description)\b", q, flags=re.IGNORECASE):
        hyps.append(
            Hypothesis(
                hid="type_grain",
                title="Type/category grain: entity vs line-item",
                rationale=(
                    "Questions that say type/category/description often have a coarse "
                    "entity dimension and a fine expense/line description. Probe both "
                    "GROUP BY grains and compare row counts before submitting."
                ),
                probe_sql_hints=(
                    "-- coarse: GROUP BY event/budget type",
                    "-- fine: GROUP BY description/line item",
                ),
            )
        )

    if _TIME_LITERAL_RE.search(q):
        hyps.append(
            Hypothesis(
                hid="time_granularity",
                title="Time literal vs stored precision",
                rationale=(
                    "Question time may be coarser than stored values. Normalize first "
                    "(parse → truncate to question grain), then compare. Do not require "
                    "exact string equality to the question text."
                ),
                probe_sql_hints=(
                    "SELECT DISTINCT <time_col> FROM <table> WHERE <time_col> IS NOT NULL "
                    "ORDER BY 1 LIMIT 30",
                    "-- then: WHERE <time_col> LIKE '<canonical>%'  (grain match)",
                ),
            )
        )

    if _LIST_Q_RE.search(q):
        hyps.append(
            Hypothesis(
                hid="list_vs_scalar",
                title="List answer vs single row",
                rationale=(
                    "Question may expect multiple rows. After filtering, COUNT(*) without "
                    "LIMIT; do not submit LIMIT 1 unless the question asks for a unique winner."
                ),
                probe_sql_hints=("SELECT COUNT(*) AS n FROM (<filtered query>)",),
            )
        )

    if wants_division_scope_check(q):
        hyps.append(
            Hypothesis(
                hid="pred_scope_div",
                title="Predicate scope: defined-domain division",
                rationale=(
                    "A filter a/b > threshold is only defined when b is non-zero. "
                    "Probe wide (bare division) and narrow (b <> 0 AND division). "
                    "Unless knowledge keeps zero-denominator rows, submit the defined "
                    "domain. Seeing inf in a probe is evidence, not optional style."
                ),
                probe_sql_hints=(
                    "-- wide: WHERE <numer>/<denom> > <threshold>",
                    "-- narrow: WHERE <denom> <> 0 AND <numer>/<denom> > <threshold>",
                ),
            )
        )

    if wants_multi_avg_scope_check(q) and _AVG_Q_RE.search(q):
        hyps.append(
            Hypothesis(
                hid="pred_scope_pop",
                title="Predicate scope: wide vs complete-case population",
                rationale=(
                    "Two+ averages on the same FROM share a population. Default is "
                    "wide: question filters only; each AVG skips its own nulls. "
                    "AND col IS NOT NULL is narrow (complete-case) — use it only if "
                    "the question or knowledge licenses known/have/complete for that "
                    "column. A CAST/empty-string probe error is not a license."
                ),
                probe_sql_hints=(
                    "-- wide: AVG(col_a), AVG(col_b) with question filters only",
                    "-- narrow: same plus col_b IS NOT NULL; compare col_a",
                ),
            )
        )

    if wants_dual_grain_check(q, knowledge) and re.search(
        r"\b(?:cost|spent|amount|consumption|value)\b", q, flags=re.IGNORECASE
    ):
        hyps.append(
            Hypothesis(
                hid="measure_identity",
                title="Measure identity: named column vs synonym KPI",
                rationale=(
                    "Lowest/highest/average words often match more than one numeric "
                    "column (cost vs spent, amount vs consumption). Probe the same "
                    "question on EACH measure and compare entity result sets. Prefer "
                    "the column whose name appears in the question, unless knowledge "
                    "binds that word to another measure in the same formula sentence. "
                    "A Total/SUM formula for a neighboring KPI is not a substitute."
                ),
                probe_sql_hints=(
                    "-- named: MIN/MAX(<question-column>) then the entities that hit it",
                    "-- synonym: SUM/MIN of the bag peer; compare the entity lists",
                ),
            )
        )

    return hyps


def format_hypothesis_plan(hypotheses: list[Hypothesis]) -> str:
    if not hypotheses:
        return ""
    lines = [
        "Evidence plan (hypotheses to verify — do not assume; probe, then choose):",
    ]
    for hyp in hypotheses:
        lines.append(f"- [{hyp.hid}] {hyp.title}: {hyp.rationale}")
        for hint in hyp.probe_sql_hints[:2]:
            lines.append(f"    probe hint: {hint}")
    return "\n".join(lines)


def _step_row_count(step: StepRecord) -> int | None:
    content = step.observation.get("content") if isinstance(step.observation, dict) else None
    if not isinstance(content, dict):
        return None
    row_count = content.get("row_count")
    if isinstance(row_count, int):
        return row_count
    return None


def _step_sql(step: StepRecord) -> str:
    if isinstance(step.action_input, dict):
        sql = step.action_input.get("sql")
        if isinstance(sql, str):
            return sql
    return ""


def update_evidence(
    state: EvidenceState,
    step: StepRecord,
    *,
    normalize_plan: NormalizePlan | None = None,
) -> EvidenceState:
    """Update empty/error streaks; keep exact-literal empty separate from unrelated hits."""
    plan = normalize_plan or NormalizePlan()
    if step.action == "run_sql":
        if not step.ok:
            state.consecutive_errors += 1
            return state
        state.consecutive_errors = 0
        count = _step_row_count(step)
        sql = _step_sql(step)
        exact = sql_uses_exact_question_literal(sql, plan) if plan.has_work else False
        coarse = sql_uses_coarse_time_match(sql, plan) if plan.has_work else False
        unrelated = sql_is_unrelated_scan(sql, plan) if plan.has_work else False
        if count is not None:
            state.last_row_counts.append(count)
            state.last_row_counts = state.last_row_counts[-8:]
            if count == 0:
                state.consecutive_empty_probes += 1
                if exact:
                    state.consecutive_exact_empty += 1
            else:
                # Non-empty: only clear general empty streak for relevant probes;
                # DISTINCT / catalog scans must not erase exact-literal thrashing.
                if not unrelated:
                    state.consecutive_empty_probes = 0
                if coarse:
                    state.coarse_nonempty_hits += 1
                    state.last_coarse_sql = sql
                    state.consecutive_exact_empty = 0
                    state.consecutive_empty_probes = 0
                elif exact:
                    state.consecutive_exact_empty = 0
                    state.consecutive_empty_probes = 0
                # unrelated nonempty leaves consecutive_exact_empty unchanged
    elif step.action == "__error__":
        state.consecutive_errors += 1
    return state


def evidence_guidance(
    *,
    evidence: EvidenceState,
    hypotheses: list[Hypothesis],
    question: str,
    steps: list[StepRecord],
    remaining_steps: int | None,
    normalize_plan: NormalizePlan | None = None,
    extract_incomplete: bool = False,
    knowledge_text: str = "",
) -> str:
    """Build an observation addendum when the loop is thrashing or near the end."""
    notes: list[str] = []
    plan = normalize_plan or NormalizePlan()

    if evidence.coarse_nonempty_hits >= 1 and (
        evidence.consecutive_exact_empty >= 1
        or (remaining_steps is not None and remaining_steps <= 6)
    ):
        notes.append(
            "Value alignment: a coarse/normalized probe already returned rows. "
            "Do NOT retry exact string equality to the question literal. "
            "Re-run that grain-matched predicate as run_sql with final=true; "
            "keep ALL matching rows (ties). Then call answer."
        )
        if evidence.last_coarse_sql:
            snippet = evidence.last_coarse_sql.strip().replace("\n", " ")
            if len(snippet) > 220:
                snippet = snippet[:220] + "…"
            notes.append(f"Prior nonempty coarse probe SQL: {snippet}")

    if evidence.consecutive_exact_empty >= 2:
        notes.append(
            "Exact equality to the question time/date literal is empty. "
            "Normalize: parse the question value, truncate stored values to the "
            "question's grain, then compare (prefix / floor seconds). "
            "Example: 0:01:54 (H:MM:SS) ≡ 1:54 / 1:54.xxx (M:SS.mmm)."
        )
        for hyp in hypotheses:
            if hyp.hid == "time_granularity":
                notes.append(f"Try hypothesis [{hyp.hid}]: {hyp.title}")
                break

    if evidence.consecutive_empty_probes >= 3 and evidence.consecutive_exact_empty < 2:
        notes.append(
            "Convergence warning: 3+ consecutive empty probes. Change predicates "
            "(distinct values / format / column choice). Do not retry the same literal."
        )
        for hyp in hypotheses:
            if hyp.hid in {"time_granularity", "ambig_col_0", "avg_grain"}:
                notes.append(f"Try hypothesis [{hyp.hid}]: {hyp.title}")

    if evidence.consecutive_errors >= 2:
        notes.append(
            "Recent steps failed (parse/JSON/SQL). Simplify the next action; "
            "prefer a short probe SQL or list_tables."
        )

    for step in steps[-4:]:
        sql = _step_sql(step)
        if _LIKE_SUFFIX_RE.search(sql):
            notes.append(
                "Pattern warning: LIKE '%_N' also matches _1N/_2N. Prefer exact suffix "
                "checks (e.g. split_part / regexp '_N$') before final."
            )
            break

    if extract_incomplete or evidence.consecutive_empty_probes >= 3:
        notes.append(
            "If a measure column is empty or a join key is missing, call "
            "search_docs with the question entities. Do not submit constant 0."
        )

    dual = format_dual_probe_block(
        question=question,
        knowledge_text=knowledge_text,
        steps=steps,
    )
    if dual and remaining_steps is not None and remaining_steps <= 8:
        notes.append(dual.replace("\n", " | "))

    if remaining_steps is not None and remaining_steps <= 3:
        notes.append(
            f"Only {remaining_steps} step(s) left. If a final=true result already "
            "answers the question, call answer now. Otherwise run one final=true SELECT."
        )
        if evidence.coarse_nonempty_hits >= 1:
            notes.append(
                "Coarse value match already succeeded earlier — prefer final=true on "
                "that predicate over more exploration."
            )
        if _LIST_Q_RE.search(question or ""):
            notes.append(
                "List-shaped question: avoid LIMIT 1 on the final SELECT unless uniqueness is proven."
            )

    scalars = _collect_numeric_scalars(steps)
    if len(scalars) >= 2:
        a, b = scalars[-2], scalars[-1]
        if a > 0 and b > 0:
            ratio = max(a, b) / min(a, b)
            if ratio >= 50:
                notes.append(
                    f"Knowledge alignment: two probed scalars differ by {ratio:.0f}× "
                    f"({a} vs {b}). Re-check SUM vs AVG / divisor before submitting."
                )
        if _RATIO_Q_RE.search(question or "") and len(set(scalars)) >= 2:
            distinct = sorted({round(v, 6) for v in scalars})
            notes.append(
                f"Ratio arbitration: several candidate values were probed "
                f"({distinct[:4]}). Pick ONE per the fixed policy — knowledge-defined "
                "reading > direct foreign-key path > fewest joins — and submit; "
                "do not re-probe the same reading."
            )

    return "\n".join(f"- {n}" for n in notes)


def _collect_numeric_scalars(steps: list[StepRecord]) -> list[float]:
    values: list[float] = []
    for step in steps:
        if step.action != "run_sql" or not step.ok:
            continue
        content = step.observation.get("content") if isinstance(step.observation, dict) else None
        if not isinstance(content, dict):
            continue
        rows = content.get("rows")
        if not isinstance(rows, list) or len(rows) != 1:
            continue
        row = rows[0]
        if not isinstance(row, (list, tuple)) or len(row) != 1:
            continue
        cell = row[0]
        try:
            if isinstance(cell, bool):
                continue
            values.append(float(cell))
        except (TypeError, ValueError):
            continue
    return values


def format_ambiguity_probe_block(groups: list[AmbiguityGroup]) -> str:
    if not groups:
        return ""
    lines = [
        "Required disambiguation probes (same filter on each table.column;",
        "a probe on one table does not cover the same name on another table):",
    ]
    for group in groups[:3]:
        lines.append(f"- term '{group.term}':")
        for col in group.columns[:4]:
            lines.append(
                f'  SELECT COUNT(*) AS n, MIN("{col.column}") AS sample '
                f'FROM "{col.table}" /* filter here */'
            )
    return "\n".join(lines)
