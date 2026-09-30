"""Predicate-scope contract (L1.5 declare + L3 enforce).

WHERE/HAVING defines the *population* (which rows enter the answer). Extra
predicates that the question/knowledge did not license — or that are
undefined (division by zero) — must not silently rewrite that population.

Two mechanism rules, no task_id:

- Defined-domain division: ``a / b`` used as a filter is only meaningful when
  ``b`` is non-zero. Unguarded inf > threshold is not a licensed membership test.
- Multi-metric population: two+ aggregates on different columns share one FROM.
  An ``IS NOT NULL`` / ``<> ''`` on a *subset* of those columns is complete-case
  (narrow). Default for scalar side-by-side averages is wide (each AVG skips
  its own nulls) unless question/knowledge licenses complete-case.

Judgement consumes the SQL IR. Never execute normalized SQL.
"""

from __future__ import annotations

import re
from typing import Any

from data_agent_baseline.agents.gate_common import undecided_payload
from data_agent_baseline.agents.sql_ir import normalize_sql

_DIV_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*)\s*/\s*([A-Za-z_][A-Za-z0-9_]*)\b"
)
_AGG_RE = re.compile(
    r"\b(?:AVG|SUM|MIN|MAX)\s*\(\s*(?:DISTINCT\s+)?"
    r"(?:[A-Za-z_][A-Za-z0-9_]*\.)?([A-Za-z_][A-Za-z0-9_]*)\s*\)",
    flags=re.IGNORECASE,
)
_NULL_FILTER_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*)\s+IS\s+NOT\s+NULL\b|"
    r"""\b([A-Za-z_][A-Za-z0-9_]*)\s*(?:<>|!=)\s*(?:''|"")""",
    flags=re.IGNORECASE,
)
# Phrases that license shrinking the FROM to complete cases / known attributes.
_LICENSE_RE = re.compile(
    r"\b(?:with\s+known|who\s+have|who\s+has|that\s+have|that\s+has|"
    r"reported|not\s+missing|complete(?:-|\s+)case|complete\s+profiles?|"
    r"non-?null|not\s+null|both\s+present|must\s+have|provided)\b",
    flags=re.IGNORECASE,
)
_PER_UNIT_Q_RE = re.compile(
    r"\b(?:per\s+unit|unit\s+price|paid\s+more\s+than|price\s*/\s*amount)\b",
    flags=re.IGNORECASE,
)
_TWO_AVG_Q_RE = re.compile(
    r"\b(?:average|mean|avg)\b.{0,80}\b(?:and|,)\b.{0,80}\b(?:average|mean|avg)\b",
    flags=re.IGNORECASE,
)
_MAJOR_TAIL_RE = re.compile(
    r"(?:GROUP\s+BY|ORDER\s+BY|LIMIT|UNION|EXCEPT|INTERSECT)\b",
    flags=re.IGNORECASE,
)


def extract_predicate_bodies(sql: str | None) -> list[str]:
    """Outer and nested WHERE/HAVING bodies (paren-aware cut at query tails)."""
    text = normalize_sql(sql)
    if not text:
        return []
    bodies: list[str] = []
    for match in re.finditer(r"\b(?:WHERE|HAVING)\b", text, flags=re.IGNORECASE):
        start = match.end()
        depth = 0
        index = start
        while index < len(text):
            char = text[index]
            if char == "(":
                depth += 1
            elif char == ")":
                if depth == 0:
                    break
                depth -= 1
            elif depth == 0 and _MAJOR_TAIL_RE.match(text, index):
                break
            index += 1
        body = text[start:index].strip()
        if body:
            bodies.append(body)
    return bodies


def unguarded_div_filters(sql: str | None) -> list[tuple[str, str]]:
    """``(numer, denom)`` pairs used as WHERE/HAVING comparisons without a zero-guard."""
    found: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for body in extract_predicate_bodies(sql):
        for match in _DIV_RE.finditer(body):
            numer, denom = match.group(1), match.group(2)
            start, end = match.span()
            after = body[end : end + 24]
            before = body[max(0, start - 24) : start]
            used_as_cmp = bool(
                re.match(r"\s*(?:>=|<=|<>|!=|>|<|=)", after)
            ) or bool(re.search(r"(?:>=|<=|<>|!=|>|<|=)\s*$", before))
            if not used_as_cmp:
                continue
            if _denominator_guarded(body, denom):
                continue
            key = (numer.casefold(), denom.casefold())
            if key in seen:
                continue
            seen.add(key)
            found.append((numer, denom))
    return found


def _denominator_guarded(body: str, denom: str) -> bool:
    token = re.escape(denom)
    return bool(
        re.search(
            rf"\b{token}\s*(?:<>|!=|>|>=)\s*0\b|"
            rf"\bNULLIF\s*\(\s*{token}\s*,",
            body,
            flags=re.IGNORECASE,
        )
    )


def aggregated_columns(sql: str | None) -> list[str]:
    text = normalize_sql(sql)
    seen: list[str] = []
    keys: set[str] = set()
    for match in _AGG_RE.finditer(text):
        name = match.group(1)
        key = name.casefold()
        if key not in keys:
            keys.add(key)
            seen.append(name)
    return seen


def null_filter_columns(sql: str | None) -> list[str]:
    names: list[str] = []
    keys: set[str] = set()
    for body in extract_predicate_bodies(sql):
        for match in _NULL_FILTER_RE.finditer(body):
            name = match.group(1) or match.group(2)
            if not name:
                continue
            key = name.casefold()
            if key not in keys:
                keys.add(key)
                names.append(name)
    return names


def complete_case_licensed(question: str, knowledge_text: str, column: str) -> bool:
    """True when question/knowledge licenses shrinking the FROM onto this attribute."""
    blob = f"{question or ''}\n{knowledge_text or ''}"
    if not _LICENSE_RE.search(blob):
        return False
    return column.casefold() in {t.casefold() for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", blob)}


def wants_division_scope_check(question: str) -> bool:
    return bool(_PER_UNIT_Q_RE.search(question or ""))


def wants_multi_avg_scope_check(question: str) -> bool:
    return bool(_TWO_AVG_Q_RE.search(question or ""))


def submit_predicate_scope_rejection(
    question: str,
    knowledge_text: str | None,
    sql: str | None,
) -> dict[str, Any] | None:
    """REJECT unlicensed narrow predicates; UNDECIDED if SQL cannot be judged."""
    if not sql:
        return None
    knowledge = knowledge_text or ""

    divs = unguarded_div_filters(sql)
    if divs:
        numer, denom = divs[0]
        return {
            "ok": False,
            "error": (
                "answer rejected: predicate-scope (defined-domain). "
                f"Filter uses {numer}/{denom} without a non-zero guard on {denom}; "
                "undefined rows (div-by-zero / inf) must not enter the population."
            ),
            "hint": (
                "Probe two populations: (wide/undefined) the bare a/b comparison; "
                "(narrow/defined) AND {denom} <> 0 (or > 0 if the measure is a count). "
                "Unless knowledge says to keep zero-denominator rows, submit the "
                "defined-domain predicate. Probe errors must not be upgraded into "
                "this filter."
            ),
            "predicate_scope_check": {
                "kind": "division_domain",
                "numerator": numer,
                "denominator": denom,
            },
        }

    aggs = aggregated_columns(sql)
    filters = null_filter_columns(sql)
    if len(aggs) >= 2 and filters:
        agg_keys = {name.casefold() for name in aggs}
        extra = [name for name in filters if name.casefold() in agg_keys]
        if extra and set(n.casefold() for n in extra) != agg_keys:
            unlicensed = [
                name for name in extra if not complete_case_licensed(question, knowledge, name)
            ]
            if unlicensed:
                return {
                    "ok": False,
                    "error": (
                        "answer rejected: predicate-scope (population). "
                        f"WHERE/HAVING applies {unlicensed} IS NOT NULL (or <> '') "
                        "while other aggregates share the same FROM; that shrinks "
                        "every metric, not only the filtered column."
                    ),
                    "hint": (
                        "Probe two populations: (wide) question filters only — each "
                        "AVG/SUM skips its own nulls; (narrow) complete-case AND "
                        "col IS NOT NULL. Use narrow only if the question or "
                        "knowledge licenses 'known/have/complete' for that attribute. "
                        "A probe CAST error is not a license. Compare the metrics "
                        "that were not in the IS NOT NULL clause."
                    ),
                    "predicate_scope_check": {
                        "kind": "multi_metric_population",
                        "aggregates": aggs,
                        "null_filters": extra,
                    },
                }

    text = normalize_sql(sql)
    if "/" in text and "WHERE" in text.upper() and not extract_predicate_bodies(sql):
        return undecided_payload(
            check="predicate_scope",
            reason="division present but WHERE/HAVING bodies could not be parsed",
            suggestion="Rewrite predicates in canonical SQL so defined-domain can be checked.",
        )
    return None
