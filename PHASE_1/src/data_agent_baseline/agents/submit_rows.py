"""Submit gates: reject empty finals and extremum queries that hide tied rows.

Checks the last final SQL against the warehouse. Does not rewrite the question
or knowledge.
"""

from __future__ import annotations

import re
from typing import Any

from data_agent_baseline.benchmark.schema import AnswerTable
from data_agent_baseline.tools.warehouse import WarehouseState, quote_ident

_EXTREMUM_RE = re.compile(
    r"(?:"
    r"\b(?:lowest|highest|cheapest|smallest|largest|fewest|fastest|slowest|"
    r"minimum|maximum|least|best|worst)\b|"
    r"\bmost\s+(?:expensive|costly|cheap)\b"
    r")",
    flags=re.IGNORECASE,
)
_SINGLE_WINNER_RE = re.compile(
    r"\b(?:the one|top\s*1|only one|a single|unique winner)\b",
    flags=re.IGNORECASE,
)
_TOP_N_RE = re.compile(
    r"\btop\s+(\d+)\b|\b(\d+)\s+(?:highest|lowest|largest|smallest|best|worst)\b",
    flags=re.IGNORECASE,
)
_LIMIT_TAIL_RE = re.compile(
    r"\bLIMIT\s+(\d+)\s*(?:OFFSET\s+\d+)?\s*$",
    flags=re.IGNORECASE,
)
_IDENT_RE = re.compile(r'"([^"]+)"|`([^`]+)`|\b([A-Za-z_][A-Za-z0-9_]*)\b')


def question_is_open_extremum(question: str) -> bool:
    """Extremum question that must keep ties, unless it asks for one winner or top-N."""
    text = question or ""
    if not _EXTREMUM_RE.search(text):
        return False
    if _SINGLE_WINNER_RE.search(text):
        return False
    return _TOP_N_RE.search(text) is None


def explicit_top_n(question: str) -> int | None:
    match = _TOP_N_RE.search(question or "")
    if match is None:
        return None
    raw = match.group(1) or match.group(2)
    return int(raw)


def trailing_limit(sql: str) -> int | None:
    match = _LIMIT_TAIL_RE.search((sql or "").strip().rstrip(";"))
    if match is None:
        return None
    return int(match.group(1))


def sql_without_trailing_limit(sql: str) -> str:
    return _LIMIT_TAIL_RE.sub("", (sql or "").strip().rstrip(";")).strip()


def count_without_limit(state: WarehouseState, sql: str) -> int | None:
    if trailing_limit(sql) is None:
        return None
    inner = sql_without_trailing_limit(sql)
    if not inner:
        return None
    try:
        with state._lock:
            row = state.conn.execute(f"SELECT COUNT(*) FROM ({inner}) AS _hit").fetchone()
    except Exception:
        return None
    if row is None:
        return None
    return int(row[0])


def _sql_idents(sql: str) -> set[str]:
    found: set[str] = set()
    for match in _IDENT_RE.finditer(sql or ""):
        token = match.group(1) or match.group(2) or match.group(3) or ""
        if token:
            found.add(token.casefold())
    return found


def column_bounds(state: WarehouseState, sql: str, *, limit: int = 8) -> list[dict[str, Any]]:
    """MIN/MAX and row count for columns the final SQL mentions."""
    from data_agent_baseline.tools.warehouse import _cell, describe_tables

    mentioned = _sql_idents(sql)
    stats: list[dict[str, Any]] = []
    for table in describe_tables(state):
        for col in table["columns"]:
            name = str(col["name"])
            if name.casefold() not in mentioned:
                continue
            try:
                with state._lock:
                    row = state.conn.execute(
                        "SELECT MIN({col}), MAX({col}), COUNT(*), COUNT({col}) FROM {tbl}".format(
                            col=quote_ident(name),
                            tbl=quote_ident(str(table["name"])),
                        )
                    ).fetchone()
            except Exception:
                continue
            if row is None:
                continue
            stats.append(
                {
                    "table": table["name"],
                    "column": name,
                    "min": _cell(row[0]),
                    "max": _cell(row[1]),
                    "row_count": int(row[2]),
                    "non_null": int(row[3]),
                }
            )
            if len(stats) >= limit:
                return stats
    return stats


def submit_projection_rejection(
    answer: AnswerTable,
    *,
    sql: str | None,
    steps: list[Any],
) -> dict[str, Any] | None:
    """Placeholder: projection checks are not task-specific regexes.

    Avoids hardcoding column-name or value-shape rules that only fix individual
    tasks. Keep empty unless a genuinely general projection check is added.
    """
    return None


def submit_row_rejection(
    question: str,
    answer: AnswerTable,
    *,
    sql: str | None,
    state: WarehouseState | None,
) -> dict[str, Any] | None:
    """Return a rejection observation, or None when the answer may be submitted."""
    row_count = len(answer.rows)
    if row_count == 0:
        bounds = column_bounds(state, sql or "") if state is not None and sql else []
        return {
            "ok": False,
            "error": (
                "answer rejected: the final SQL returned 0 rows. "
                "Do not submit an empty table."
            ),
            "hint": (
                "Read min/max below. If the filter is outside the column range, "
                "drop or relax that predicate and run_sql final=true again."
            ),
            "empty_check": {
                "row_count": 0,
                "sql": sql,
                "column_bounds": bounds,
            },
        }

    if not sql or state is None or not question_is_open_extremum(question):
        return None
    limit = trailing_limit(sql)
    if limit is None:
        return None
    top_n = explicit_top_n(question)
    if top_n is not None and top_n == limit:
        return None
    hit_count = count_without_limit(state, sql)
    if hit_count is None:
        if limit == 1:
            return _tie_rejection(sql, limit, None, row_count)
        return None
    if hit_count > row_count:
        return _tie_rejection(sql, limit, hit_count, row_count)
    return None


def _tie_rejection(
    sql: str,
    limit: int,
    hit_count: int | None,
    submitted: int,
) -> dict[str, Any]:
    return {
        "ok": False,
        "error": (
            "answer rejected: extremum query uses LIMIT and returns fewer rows "
            "than the same query without LIMIT."
        ),
        "hint": (
            "Use WHERE col = (SELECT MIN(col) ...) or MAX, and return every tied row. "
            "Do not use ORDER BY … LIMIT 1 unless the question asks for one winner."
        ),
        "tie_check": {
            "limit": limit,
            "hit_count": hit_count,
            "submitted_rows": submitted,
            "sql": sql,
        },
    }
