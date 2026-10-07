"""Unified grain contract gate (架构设计 §13.2).

25 (extremum grain ≠ output grain) and 199 (filter grain ≠ output grain) look
like two diseases but share one root cause: the grain at which the SQL
aggregates does not match the grain the question asks about. Instead of two
independent gates with a seam between them, the contract funnels every
grain-consistency verdict through one place.

Evidence sources stay in their own modules:
- ``aggregate_grain.submit_grain_rejection`` — coarse final needs a knowledge
  definition for the coarse formula (pre-answer, SQL only);
- ``submit_validation.submit_membership_rejection`` — final dropped the
  aggregate-filter table and re-expanded via a group key (post-answer);
- ``submit_validation.submit_member_expansion_rejection`` — final expanded a
  small grouped-aggregate probe to member rows (post-answer).

The wrapper only orders the checks and tags payloads; all judgement logic lives
in the evidence sources. Missing evidence → pass (申报缺失时放行).
"""

from __future__ import annotations

from typing import Any

from data_agent_baseline.agents.aggregate_grain import submit_grain_rejection
from data_agent_baseline.agents.runtime import StepRecord
from data_agent_baseline.agents.submit_validation import (
    submit_member_expansion_rejection,
    submit_membership_rejection,
)
from data_agent_baseline.benchmark.schema import AnswerTable
from data_agent_baseline.tools.warehouse import WarehouseState


def submit_grain_contract_pre(
    question: str,
    steps: list[StepRecord],
    sql: str | None,
    knowledge_text: str | None,
) -> dict[str, Any] | None:
    """Pre-answer contract check: coarse final without knowledge definition.

    May return a REJECT dict or an UNDECIDED payload (§13.5) — callers must
    distinguish via ``gate_common.is_undecided``.
    """
    payload = submit_grain_rejection(question, steps, sql, knowledge_text)
    if payload is not None:
        payload.setdefault("grain_contract", True)
    return payload


def submit_grain_contract_post(
    question: str,
    steps: list[StepRecord],
    sql: str | None,
    answer: AnswerTable,
    state: WarehouseState | None = None,
) -> dict[str, Any] | None:
    """Post-answer contract check: member expansion beyond the filter grain."""
    payload = submit_membership_rejection(steps, sql, answer)
    if payload is not None:
        payload.setdefault("grain_contract", True)
        return payload
    payload = submit_member_expansion_rejection(question, steps, sql, answer)
    if payload is not None:
        payload.setdefault("grain_contract", True)
        return payload
    return None
