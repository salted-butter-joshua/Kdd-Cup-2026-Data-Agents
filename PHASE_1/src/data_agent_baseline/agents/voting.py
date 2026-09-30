"""Soft voting among final candidate SQLs when the ReAct loop times out.

Picks the best last_final-style candidate using cheap heuristics — not gold-chasing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from data_agent_baseline.agents.runtime import StepRecord

_LIMIT_ONE_RE = re.compile(r"\bLIMIT\s+1\b", flags=re.IGNORECASE)
_LIST_Q_RE = re.compile(
    r"\b(?:which|what)\b.+\b(?:races?|names?|ids?|elements?|types?|items?)\b|"
    r"\blist\b|\btally\b|\ball\b",
    flags=re.IGNORECASE,
)


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
    """Score final=true successful runs; prefer non-empty, list-friendly shapes."""
    candidates = collect_final_candidates(steps)
    if not candidates:
        return None

    list_shaped = bool(_LIST_Q_RE.search(question or ""))
    best: VoteCandidate | None = None
    for index, step, sql in candidates:
        count = _row_count(step)
        score = 0.0
        reasons: list[str] = []
        if count is None:
            score -= 2.0
            reasons.append("unknown_rows")
        elif count == 0:
            score -= 5.0
            reasons.append("empty")
        else:
            score += 3.0
            reasons.append(f"rows={count}")
            if list_shaped and count > 1:
                score += 2.0
                reasons.append("list_ok")
            if list_shaped and count == 1 and _LIMIT_ONE_RE.search(sql):
                score -= 1.5
                reasons.append("suspicious_limit1")
        # Prefer later successful finals slightly (more informed).
        score += index * 0.05
        cand = VoteCandidate(
            sql=sql,
            score=score,
            reason=",".join(reasons),
            step_index=index,
        )
        if best is None or cand.score > best.score:
            best = cand
    return best
