"""Canonical SQL IR for L3 gates (架构设计 §13.1).

Gates must analyze semantics, not surface spelling. DuckDB accepts double-quoted,
back-quoted, or bracketed identifiers, so the same logical SQL can wear several
"outfits" — and a regex written against one outfit is bypassed by another
(task_25: ``SUM(e."cost")`` slipped past a gate written for ``SUM(e.cost)``).

``normalize_sql`` produces the canonical judgement copy every gate consumes.

IMPORTANT: the IR is a *judgement copy only*. Never execute normalized SQL —
stripping quotes from identifiers that contain spaces (e.g. "School Name")
would produce invalid SQL.
"""

from __future__ import annotations

import re

# DuckDB identifier quoting: "double", `backtick`, [bracket].
# Single quotes are string literals and are left untouched.
_QUOTED_IDENT_RE = re.compile(r'"([^"]+)"|`([^`]+)`|\[([^\]]+)\]')
_WS_RE = re.compile(r"\s+")
_FROM_JOIN_RE = re.compile(
    r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)"
    r"(?:\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*))?",
    flags=re.IGNORECASE,
)
_QUALIFIED_COL_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b"
)
_SELECT_STAR_RE = re.compile(
    r"\bSELECT\s+(?:DISTINCT\s+)?\*",
    flags=re.IGNORECASE,
)

# Words that may legitimately follow "FROM t" without being an alias.
_SQL_KEYWORDS = frozenset(
    {
        "where", "group", "order", "having", "limit", "offset", "on", "and",
        "or", "not", "left", "right", "inner", "outer", "full", "cross",
        "join", "union", "select", "from", "as", "in", "is", "null",
        "between", "like", "case", "when", "then", "else", "end", "distinct",
        "all", "asc", "desc", "by", "with", "natural", "using", "except",
        "intersect", "fetch", "for", "into", "values", "set",
    }
)


def normalize_sql(sql: str | None) -> str:
    """Return the canonical judgement copy of a SQL string.

    - strips identifier quotes / backticks / brackets (string literals untouched)
    - collapses every whitespace run to a single space
    """
    if not sql:
        return ""
    text = _QUOTED_IDENT_RE.sub(
        lambda m: m.group(1) or m.group(2) or m.group(3), sql
    )
    return _WS_RE.sub(" ", text).strip()


def table_scope(sql: str | None) -> dict[str, str]:
    """Map alias or table name (casefolded keys) to the base table spelling.

    Unaliased ``FROM t`` registers ``t -> t``. Subquery-derived tables
    (``FROM (SELECT ...) x``) are not resolved.
    """
    text = normalize_sql(sql)
    mapping: dict[str, str] = {}
    for match in _FROM_JOIN_RE.finditer(text):
        table = match.group(1)
        alias = match.group(2)
        mapping[table.casefold()] = table
        if alias is None or alias.casefold() in _SQL_KEYWORDS:
            continue
        mapping[alias.casefold()] = table
    return mapping


def alias_map(sql: str | None) -> dict[str, str]:
    """Best-effort ``alias -> base table`` mapping from FROM/JOIN clauses.

    Input is normalized first, so quoted table names work too. Aliases of
    subquery-derived tables (``FROM (SELECT ...) t``) are not resolved and
    simply absent from the map. Unaliased tables are not entries (see
    ``table_scope`` / ``from_tables``).
    """
    text = normalize_sql(sql)
    mapping: dict[str, str] = {}
    for match in _FROM_JOIN_RE.finditer(text):
        table = match.group(1)
        alias = match.group(2)
        if alias is None:
            continue
        if alias.casefold() in _SQL_KEYWORDS:
            continue
        mapping[alias.casefold()] = table
    return mapping


def from_tables(sql: str | None) -> list[str]:
    """Unique base tables in FROM/JOIN, first-seen order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for base in table_scope(sql).values():
        key = base.casefold()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(base)
    return ordered


def sql_mentions_table_column(sql: str | None, table: str, column: str) -> bool:
    """Whether this SQL scopes ``table`` and refers to ``table.column``.

    A probe that only uses another table's same-named column does not count.
    ``SELECT *`` from a query whose only scoped table is ``table`` counts.
    Unqualified ``column`` counts only when that table is the unique FROM/JOIN
    base table — a join of two homonym tables leaves the reference unresolved.
    """
    if not sql or not table or not column:
        return False
    text = normalize_sql(sql)
    scope = table_scope(sql)
    bases = {base.casefold() for base in scope.values()}
    if table.casefold() not in bases:
        return False

    unique_base = len(bases) == 1
    if unique_base and _SELECT_STAR_RE.search(text):
        return True

    for match in _QUALIFIED_COL_RE.finditer(text):
        left, right = match.group(1), match.group(2)
        if right.casefold() != column.casefold():
            continue
        resolved = scope.get(left.casefold())
        if resolved is not None and resolved.casefold() == table.casefold():
            return True

    if unique_base and next(iter(bases)) == table.casefold():
        if re.search(rf"\b{re.escape(column)}\b", text, flags=re.IGNORECASE):
            return True
    return False


def strip_subqueries(sql: str | None) -> str:
    """Remove all parenthesized blocks, leaving the outermost query skeleton.

    Used to inspect the OUTER query grain (e.g. does the outer SELECT carry its
    own HAVING) without being confused by filter subqueries.
    """
    text = normalize_sql(sql)
    out: list[str] = []
    depth = 0
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(char)
    return "".join(out)
