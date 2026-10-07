"""P2 dual-track probes: formula / ratio / grain. Soft hints; miss → fail-open.

Hard reject only when SQL itself proves the forbidden pattern:
sample quantile used as a clinical 'normal' bound.
"""

from __future__ import annotations

import re
from typing import Any

from data_agent_baseline.agents.aggregate_grain import grains_in_sql
from data_agent_baseline.agents.question_route import classify_question
from data_agent_baseline.agents.runtime import StepRecord
from data_agent_baseline.agents.sql_ir import normalize_sql

_SUM_DIV_RE = re.compile(r"\bSUM\s*\([^)]*\)\s*/", flags=re.IGNORECASE)
_AVG_RE = re.compile(r"\bAVG\s*\(", flags=re.IGNORECASE)
_QUANTILE_SQL_RE = re.compile(
    r"\b(?:percentile(?:_cont|_disc)?|ntile|quantile|quartile)\s*\(",
    flags=re.IGNORECASE,
)
_NORMAL_Q_RE = re.compile(r"\b(?:normal|abnormal)\b", flags=re.IGNORECASE)
_KNOWLEDGE_RANGE_RE = re.compile(
    r"percentile|quantile|quartile|\bq[13]\b|normal\s+range",
    flags=re.IGNORECASE,
)


def _step_sql(step: StepRecord) -> str:
    if isinstance(step.action_input, dict):
        sql = step.action_input.get("sql")
        if isinstance(sql, str):
            return sql
    return ""


def observed_formula_tracks(steps: list[StepRecord]) -> set[str]:
    found: set[str] = set()
    for step in steps:
        if step.action != "run_sql" or not step.ok:
            continue
        text = normalize_sql(_step_sql(step))
        if _AVG_RE.search(text):
            found.add("avg")
        if _SUM_DIV_RE.search(text):
            found.add("sum_div")
        grains = grains_in_sql(text)
        found.update(grains)
    return found


def format_dual_probe_block(
    *,
    question: str,
    knowledge_text: str,
    steps: list[StepRecord],
) -> str:
    """PROGRESS addendum: required tracks vs what the trace already ran."""
    kinds = classify_question(question)
    required: list[str] = []
    if "agg" in kinds or "ratio" in kinds:
        if re.search(r"/\s*12|divided\s+by\s*12", knowledge_text or "", flags=re.IGNORECASE):
            required.append("formula: AVG(col) AND SUM(col)/N (compare magnitude; do not auto-pick)")
        if "ratio" in kinds:
            required.append("ratio: probe A/B and B/A (pick by knowledge, then FK, then fewest joins)")
    if "extremum" in kinds or "agg" in kinds:
        if re.search(r"\b(?:type|category|description)\b", question or "", flags=re.IGNORECASE):
            required.append("grain: coarse entity type AND fine description/category")
    if "clinical" in kinds and _NORMAL_Q_RE.search(question or ""):
        required.append("clinical: use knowledge thresholds; do not use sample Q1/Q3 as 'normal'")
    if not required:
        return ""
    seen = observed_formula_tracks(steps)
    seen_txt = ",".join(sorted(seen)) if seen else "none"
    lines = ["P2 dual tracks (probe both; do not auto-rewrite the answer):"]
    for item in required:
        lines.append(f"- need {item}")
    lines.append(f"- seen_in_trace: {seen_txt}")
    return "\n".join(lines)


def submit_quantile_normal_rejection(
    question: str,
    knowledge_text: str | None,
    sql: str | None,
) -> dict[str, Any] | None:
    """Reject sample percentiles used as clinical normal bounds."""
    kinds = classify_question(question or "")
    if "clinical" not in kinds or not _NORMAL_Q_RE.search(question or ""):
        return None
    if not sql or not _QUANTILE_SQL_RE.search(sql):
        return None
    if _KNOWLEDGE_RANGE_RE.search(knowledge_text or ""):
        return None
    return {
        "ok": False,
        "error": (
            "answer rejected: the SQL uses a sample percentile/NTILE as a "
            "'normal' bound, but knowledge does not define that range."
        ),
        "hint": (
            "Use a threshold stated in knowledge.md (e.g. LDH > 500 style). "
            "If none exists, search_docs for the documented range. Do not treat "
            "Q1–Q3 of this warehouse sample as clinical normal."
        ),
        "quantile_normal_check": {"sql": sql},
    }
