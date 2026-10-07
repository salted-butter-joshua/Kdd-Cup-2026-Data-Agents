"""Shared submit-time contracts for P0/P1: cutoff, scalar shape, fake-zero, projection.

Mechanism-level only. Evidence missing → do not reject.
"""

from __future__ import annotations

import re
from typing import Any

from data_agent_baseline.agents.sql_ir import normalize_sql
from data_agent_baseline.benchmark.schema import AnswerTable
from data_agent_baseline.tools.warehouse import WarehouseState, describe_tables, quote_ident

_LATEST_RE = re.compile(
    r"\b(?:most\s+recent|latest|last\s+(?:one|payment|date)|"
    r"only\s+(?:the\s+)?(?:one|date|winner)|top\s*1|unique\s+winner|"
    r"a\s+single)\b",
    flags=re.IGNORECASE,
)
_SCALAR_Q_RE = re.compile(
    r"(?:"
    r"\bhow\s+many\b|"
    r"\bhow\s+much\b|"
    r"\bcalculate\b|"
    r"what(?:'s|\s+is)\s+the\s+(?:total|average|avg|mean|count|number|ratio|share|percentage)|"
    r"what\s+percentage|"
    r"percentage\s+of|"
    r"\bpercent(?:age)?\b|"
    r"\bratio\b|"
    r"how\s+many\s+times|"
    r"more\s+than\b"
    r")",
    flags=re.IGNORECASE,
)
_LIST_Q_RE = re.compile(
    r"\b(?:which|what)\b.+\b(?:races?|names?|ids?|elements?|types?|items?)\b|"
    r"\blist\b|\btally\b|\ball\s+the\b",
    flags=re.IGNORECASE,
)
_AGG_ONLY_RE = re.compile(
    r"^\s*SELECT\s+(?:DISTINCT\s+)?(?:(?:COUNT|SUM|AVG|MIN|MAX)\s*\([^)]*\)(?:\s+AS\s+\w+)?\s*,?\s*)+\s*(?:FROM\b|$)",
    flags=re.IGNORECASE,
)
_GROUP_BY_RE = re.compile(r"\bGROUP\s+BY\b", flags=re.IGNORECASE)
_CONST_ZERO_RE = re.compile(
    r"^\s*SELECT\s+(?:DISTINCT\s+)?(?:CAST\s*\(\s*)?0(?:\.0+)?(?:\s*\)\s*)?"
    r"(?:\s+AS\s+[A-Za-z_][\w]*)?\s*;?\s*$",
    flags=re.IGNORECASE,
)
_LIMIT_TAIL_RE = re.compile(
    r"\bLIMIT\s+(\d+)\s*(?:OFFSET\s+\d+)?\s*$",
    flags=re.IGNORECASE,
)
_MAXMIN_EQ_HEAD_RE = re.compile(
    r"(?P<conj>\b(?:AND|WHERE)\b)\s+"
    r"(?:\"[^\"]+\"|`[^`]+`|[A-Za-z_][\w\.]*)"
    r"\s*=\s*\(\s*SELECT\s+(?:MAX|MIN)\s*\(",
    flags=re.IGNORECASE,
)
_EMPTY_STRINGS = frozenset({"", "-", "none", "null", "nan"})
_SIDECAR_EXACT = frozenset(
    {
        "year",
        "q1",
        "q2",
        "q3",
        "tally",
        "cnt",
        "count",
        "creationdate",
        "score",
    }
)
_DUMP_COL_RE = re.compile(
    r"^(?:date|type|operation|amount|balance|k_symbol|bank|account|account_id)$",
    flags=re.IGNORECASE,
)
_ID_KEEP_RE = re.compile(
    r"^(?:trans_id|post_id|comment_id|id)$",
    flags=re.IGNORECASE,
)
_TEXT_COL_RE = re.compile(r"^(?:text|body|content|comment)$", flags=re.IGNORECASE)


def question_allows_single_row_cutoff(question: str) -> bool:
    """True when the question itself asks for one latest/only row."""
    return bool(_LATEST_RE.search(question or ""))


def question_wants_scalar_value(question: str) -> bool:
    """Count / percent / ratio / how-many-times — not a list of entities."""
    text = question or ""
    if _LIST_Q_RE.search(text) and not re.search(r"\bhow\s+many\b", text, flags=re.IGNORECASE):
        return False
    return bool(_SCALAR_Q_RE.search(text))


def question_wants_list(question: str) -> bool:
    return bool(_LIST_Q_RE.search(question or ""))


def sql_is_global_aggregate(sql: str) -> bool:
    """SELECT COUNT/SUM/AVG/... with no GROUP BY — naturally one row."""
    text = normalize_sql(sql)
    if not text or _GROUP_BY_RE.search(text):
        return False
    return bool(_AGG_ONLY_RE.match(text))


def sql_without_trailing_limit(sql: str) -> str:
    return _LIMIT_TAIL_RE.sub("", (sql or "").strip().rstrip(";")).strip()


def sql_has_maxmin_equality(sql: str) -> bool:
    """True when SQL pins a column to (SELECT MIN/MAX …)."""
    return _MAXMIN_EQ_HEAD_RE.search(sql or "") is not None


def sql_without_singleton_cutoff(sql: str) -> str:
    """Strip trailing LIMIT only.

    Do not strip col=(SELECT MIN/MAX …). That predicate *defines* the
    extremum set; removing it counts every sibling row and false-rejects
    unique winners.
    """
    return sql_without_trailing_limit(sql)


def count_without_cutoff(state: WarehouseState, sql: str) -> int | None:
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


def is_constant_zero_sql(sql: str) -> bool:
    text = normalize_sql(sql)
    if _CONST_ZERO_RE.match(text):
        return True
    if re.match(r"^SELECT(?: DISTINCT)?(?: CAST\()?0(?:\.0+)?", text, flags=re.IGNORECASE):
        if not re.search(r"\bFROM\b", text, flags=re.IGNORECASE):
            return True
    return False


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text.casefold() in _EMPTY_STRINGS


def column_fill_rate(state: WarehouseState, table: str, column: str) -> float | None:
    try:
        with state._lock:
            row = state.conn.execute(
                "SELECT COUNT(*), COUNT({col}) FROM {tbl}".format(
                    col=quote_ident(column),
                    tbl=quote_ident(table),
                )
            ).fetchone()
            sample = state.conn.execute(
                "SELECT {col} FROM {tbl} LIMIT 80".format(
                    col=quote_ident(column),
                    tbl=quote_ident(table),
                )
            ).fetchall()
    except Exception:
        return None
    if row is None or not row[0]:
        return None
    n_total = int(row[0])
    nonempty_typed = int(row[1] or 0)
    if nonempty_typed == 0:
        return 0.0
    if not sample:
        return nonempty_typed / n_total
    blank = sum(1 for item in sample if _is_blank(item[0] if item else None))
    if blank == len(sample):
        return 0.0
    return nonempty_typed / n_total


def empty_columns_mentioned(
    state: WarehouseState | None,
    sql: str | None,
) -> list[str]:
    if state is None or not sql:
        return []
    text = normalize_sql(sql).casefold()
    found: list[str] = []
    for table in describe_tables(state):
        tname = str(table["name"])
        for col in table["columns"]:
            cname = str(col["name"])
            if cname.casefold() not in text:
                continue
            rate = column_fill_rate(state, tname, cname)
            if rate is not None and rate <= 1e-12:
                found.append(f"{tname}.{cname}")
    return found


def answer_is_numeric_zero(answer: AnswerTable) -> bool:
    if len(answer.columns) != 1 or len(answer.rows) != 1:
        return False
    value = answer.rows[0][0]
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return float(value) == 0.0
    if isinstance(value, str):
        try:
            return float(value.strip()) == 0.0
        except ValueError:
            return False
    return False


def submit_fake_zero_rejection(
    question: str,
    answer: AnswerTable,
    *,
    sql: str | None,
    state: WarehouseState | None,
) -> dict[str, Any] | None:
    """Reject SELECT 0 / empty-column 0% when the warehouse is incomplete."""
    if not answer_is_numeric_zero(answer):
        return None
    sql_text = sql or ""
    constant = is_constant_zero_sql(sql_text)
    empty_cols = empty_columns_mentioned(state, sql_text)
    if not constant and not empty_cols:
        return None
    if not question_wants_scalar_value(question) and not constant:
        return None
    return {
        "ok": False,
        "error": (
            "answer rejected: submitted 0 but the warehouse is missing the "
            "measured column (empty or constant-zero SQL). Do not treat a missing "
            "extract as a true zero."
        ),
        "hint": (
            "Extract doc/*.md (extract_docs / complete the warehouse) so the "
            "metric column is populated, then recompute. Do not SELECT 0."
        ),
        "incomplete_schema_check": {
            "constant_zero_sql": constant,
            "empty_columns": empty_cols,
            "sql": sql_text,
        },
    }


def result_fingerprint(step_observation: dict[str, Any] | None) -> tuple[Any, ...]:
    """Execution-result identity: column count, row count, cell bag (names ignored)."""
    content = step_observation or {}
    if "content" in content and isinstance(content.get("content"), dict):
        content = content["content"]
    columns = content.get("columns") or []
    rows = content.get("rows") or []
    n_rows = content.get("row_count")
    if not isinstance(n_rows, int):
        n_rows = len(rows) if isinstance(rows, list) else -1
    cells: list[tuple[str, ...]] = []
    if isinstance(rows, list):
        for row in rows[:400]:
            if isinstance(row, (list, tuple)):
                cells.append(tuple(str(item).strip() if item is not None else "" for item in row))
    return (len(columns) if isinstance(columns, list) else 0, n_rows, tuple(sorted(cells)))


def drop_unasked_sidecar_columns(question: str, columns: list[str]) -> list[str]:
    """Drop probe-only extras (year/q3/tally) and wide SELECT * dumps on list asks."""
    if len(columns) <= 1:
        return list(columns)
    q = (question or "").casefold()
    q_tokens = {tok for tok in re.split(r"[^a-z0-9]+", q) if len(tok) > 1}

    def col_tokens(name: str) -> set[str]:
        return {tok for tok in re.split(r"[^a-z0-9]+", name.casefold()) if tok}

    if question_wants_list(question) and len(columns) >= 6:
        id_hits = [name for name in columns if _ID_KEEP_RE.search(name)]
        dump_n = sum(1 for name in columns if _DUMP_COL_RE.search(name))
        if id_hits and dump_n >= 3:
            return id_hits[:1]

    if "comment" in q_tokens:
        text_cols = [name for name in columns if _TEXT_COL_RE.search(name)]
        if text_cols:
            return text_cols

    kept: list[str] = []
    dropped_sidecar = False
    for name in columns:
        leaf = name.rsplit(".", 1)[-1].casefold()
        tokens = col_tokens(name)
        asked = bool(tokens & q_tokens)
        if leaf in _SIDECAR_EXACT and not asked:
            dropped_sidecar = True
            continue
        kept.append(name)
    if dropped_sidecar and kept:
        return kept
    return list(columns)
