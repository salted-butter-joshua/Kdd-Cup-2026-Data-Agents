"""Measure-identity gate: question token vs synonym measure column.

Task_25 class: 'lowest cost' binds to expense.cost, not budget.spent, even
though knowledge defines Total Expenditure as SUM(spent) and the two words
share a synonym bag. Dual-grain only asks how to aggregate, not which measure.

Arbitration (no gold, no task_id):
  1. If the question uses a column's exact name and that column exists, prefer it
     over a synonym in the same bag.
  2. Knowledge may override when a sentence binds the question term to another
     measure via SUM/AVG/MIN/MAX/DIVIDE (e.g. 'Cost: SUM(spent)').

L3 rejects a final SQL that aggregates a synonym while an exact-named peer
column is in the schema and knowledge does not bind the term to the synonym.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from data_agent_baseline.agents.aggregate_grain import wants_dual_grain_check
from data_agent_baseline.agents.schema_link import ColumnCandidate, _SYNONYM_BAGS
from data_agent_baseline.agents.sql_ir import normalize_sql

_AGG_MEASURE_RE = re.compile(
    r"\b(?:SUM|MIN|MAX|AVG)\s*\(\s*(?:DISTINCT\s+)?"
    r"(?:[A-Za-z_][A-Za-z0-9_]*\s*\.\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*\)",
    flags=re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]{2,}")


def _tokens(text: str) -> set[str]:
    return {m.group(0).casefold() for m in _TOKEN_RE.finditer(text or "")}


def _whole_word(text: str, word: str) -> bool:
    return bool(re.search(rf"\b{re.escape(word)}\b", text or "", flags=re.IGNORECASE))


def measures_in_sql(sql: str | None) -> list[str]:
    """Column names inside SUM/MIN/MAX/AVG, first-seen order."""
    text = normalize_sql(sql)
    seen: set[str] = set()
    ordered: list[str] = []
    for match in _AGG_MEASURE_RE.finditer(text):
        name = match.group(1)
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(name)
    return ordered


def _shared_bag(name_a: str, name_b: str) -> frozenset[str] | None:
    a, b = name_a.casefold(), name_b.casefold()
    for bag in _SYNONYM_BAGS:
        if a in bag and b in bag:
            return bag
    return None


def knowledge_binds_term_to_measure(
    knowledge_text: str,
    term: str,
    measure: str,
) -> bool:
    """True when a knowledge sentence ties the question term to this measure formula."""
    if not knowledge_text or not term or not measure:
        return False
    if term.casefold() == measure.casefold():
        return True
    for part in re.split(r"[.!?\n]", knowledge_text):
        if not _whole_word(part, term) or not _whole_word(part, measure):
            continue
        if re.search(r"\b(?:SUM|AVG|MIN|MAX|DIVIDE)\s*\(", part, flags=re.IGNORECASE):
            return True
    return False


def exact_named_schema_columns(
    question: str,
    candidates: Iterable[ColumnCandidate] | None,
) -> set[str]:
    """Question tokens that are also schema column names (casefolded)."""
    if not candidates:
        return set()
    col_names = {c.column.casefold() for c in candidates}
    return _tokens(question) & col_names


def submit_measure_identity_rejection(
    question: str,
    knowledge_text: str | None,
    sql: str | None,
    candidates: Iterable[ColumnCandidate] | None,
) -> dict[str, Any] | None:
    """Reject synonym-measure substitution when the question names a real column."""
    if not sql or not wants_dual_grain_check(question, knowledge_text):
        return None
    used = measures_in_sql(sql)
    if not used:
        return None
    exact = exact_named_schema_columns(question, candidates)
    if not exact:
        return None
    knowledge = knowledge_text or ""
    # If every aggregated measure is an exact question column, identity is resolved.
    used_fold = {u.casefold() for u in used}
    if used_fold <= exact:
        return None
    # Substitution: SQL uses a synonym of an exact-named column.
    for used_name in used:
        if used_name.casefold() in exact:
            continue
        for named in exact:
            bag = _shared_bag(named, used_name)
            if bag is None:
                continue
            if knowledge_binds_term_to_measure(knowledge, named, used_name):
                continue
            return {
                "ok": False,
                "error": (
                    "answer rejected: measure identity. The question names "
                    f"'{named}' which exists as a column, but the final SQL "
                    f"aggregates synonym '{used_name}'. These are different measures."
                ),
                "hint": (
                    "Probe the same question on EACH measure column (row-level "
                    f"MIN/MAX/AVG of '{named}', and the synonym aggregate). "
                    "Compare the entity result sets. Prefer the column whose name "
                    "matches the question, unless knowledge.md binds that word to "
                    "another measure in the same formula sentence. Do not substitute "
                    "a neighboring KPI (e.g. Total/SUM of a synonym) for the named "
                    "measure."
                ),
                "measure_identity_check": {
                    "question_column": named,
                    "sql_measure": used_name,
                    "bag": sorted(bag),
                },
            }
    return None
