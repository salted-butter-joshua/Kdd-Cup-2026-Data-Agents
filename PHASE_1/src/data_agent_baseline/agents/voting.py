"""Vote among final candidate SQLs when the ReAct loop times out.

Picks by execution-result agreement, not max row count.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from data_agent_baseline.agents.answer_contract import (
    question_wants_list,
    question_wants_scalar_value,
    result_fingerprint,
)
from data_agent_baseline.agents.runtime import StepRecord


@dataclass(frozen=True, slots=True)
class VoteCandidate:
    sql: str
    score: float
    reason: str
    step_index: int


def _final_sql_from_step(step: StepRecord) -> str | None:
    if step.action != "run_sql" or not step.ok:
        return None
    if not isinstance(step.action_input, dict):
        return None
    if not bool(step.action_input.get("final")):
        return None
    sql = step.action_input.get("sql")
    return sql if isinstance(sql, str) and sql.strip() else None


def _row_count(step: StepRecord) -> int | None:
    content = step.observation.get("content") if isinstance(step.observation, dict) else None
    if not isinstance(content, dict):
        return None
    count = content.get("row_count")
    return count if isinstance(count, int) else None


def collect_final_candidates(steps: list[StepRecord]) -> list[tuple[int, StepRecord, str]]:
    out: list[tuple[int, StepRecord, str]] = []
    for index, step in enumerate(steps):
        sql = _final_sql_from_step(step)
        if sql is not None:
            out.append((index, step, sql))
    return out


def vote_best_final(
    *,
    question: str,
    steps: list[StepRecord],
) -> VoteCandidate | None:
    """Cluster successful finals by result fingerprint; take the largest cluster."""
    candidates = collect_final_candidates(steps)
    if not candidates:
        return None

    scalar = question_wants_scalar_value(question)
    list_shaped = question_wants_list(question)
    clusters: dict[tuple, list[tuple[int, StepRecord, str]]] = defaultdict(list)
    for index, step, sql in candidates:
        count = _row_count(step)
        if count == 0:
            continue
        obs = step.observation if isinstance(step.observation, dict) else {}
        clusters[result_fingerprint(obs)].append((index, step, sql))

    if not clusters:
        return None

    ranked = sorted(
        clusters.items(),
        key=lambda item: (
            len(item[1]),
            -item[0][0] if item[0][0] else 0,
            item[1][-1][0],
        ),
        reverse=True,
    )

    best: VoteCandidate | None = None
    for fingerprint, members in ranked:
        n_cols, n_rows, _cells = fingerprint
        cluster_n = len(members)
        index, step, sql = members[-1]
        score = cluster_n * 5.0
        reasons = [f"cluster={cluster_n}", f"rows={n_rows}", f"cols={n_cols}"]
        if n_rows is not None and n_rows <= 0:
            continue
        if scalar:
            if n_rows == 1 and n_cols <= 2:
                score += 4.0
                reasons.append("scalar_ok")
            else:
                score -= 12.0
                reasons.append("scalar_wide")
        elif list_shaped and isinstance(n_rows, int) and n_rows > 1:
            score += 1.0
            reasons.append("list_ok")
        if n_cols:
            score += 1.0 / n_cols
        score += index * 0.01
        cand = VoteCandidate(
            sql=sql,
            score=score,
            reason=",".join(reasons),
            step_index=index,
        )
        if best is None or cand.score > best.score:
            best = cand

    if best is None:
        return None
    if scalar and best.score < 0:
        return None
    return best
