"""Deterministic submit-time validation gates.

These checks run after the model proposes an answer table. They are meant to be
conservative: when the rule cannot be evaluated deterministically, the check
returns None and lets the answer through.
"""

from __future__ import annotations

import re
from typing import Any

from data_agent_baseline.agents.aggregate_grain import wants_member_expansion_check
from data_agent_baseline.agents.gate_common import undecided_payload
from data_agent_baseline.agents.runtime import StepRecord
from data_agent_baseline.agents.sql_ir import (
    from_tables,
    normalize_sql,
    sql_mentions_table_column,
    strip_subqueries,
)
from data_agent_baseline.benchmark.schema import AnswerTable
from data_agent_baseline.tools.warehouse import WarehouseState, quote_ident

_HAVING_RE = re.compile(r"\bHAVING\b", flags=re.IGNORECASE)
_GROUP_BY_RE = re.compile(r"\bGROUP\s+BY\b", flags=re.IGNORECASE)
_FROM_JOIN_RE = re.compile(
    r"\b(?:FROM|JOIN)\s+((?:\"[^\"]+\"|`[^`]+`|[A-Za-z_][A-Za-z0-9_]*)"
    r"(?:\s+(?:AS\s+)?(?:\"[^\"]+\"|`[^`]+`|[A-Za-z_][A-Za-z0-9_]*))?)",
    flags=re.IGNORECASE,
)
_IDENT_RE = re.compile(r'"([^"]+)"|`([^`]+)`|([A-Za-z_][A-Za-z0-9_]*)')
_REL_ID_COL_RE = re.compile(
    r"\b(bond_id|relation_id|rel_id|edge_id|conn_id|link_id)\b",
    flags=re.IGNORECASE,
)
_COUNT_STAR_RE = re.compile(r"\bCOUNT\s*\(\s*\*\s*\)", flags=re.IGNORECASE)
_COUNT_COL_RE = re.compile(
    r"\bCOUNT\s*\(\s*(?:DISTINCT\s+)?(?:\"[^\"]+\"|`[^`]+`|[A-Za-z_][A-Za-z0-9_\.]*\s*,?\s*)+\s*\)",
    flags=re.IGNORECASE,
)
_COUNT_DISTINCT_HEAD_RE = re.compile(
    r"\bCOUNT\s*\(\s*DISTINCT\b",
    flags=re.IGNORECASE,
)
_PERCENT_RE = re.compile(
    r"percent|percentage|\bpct\b|\*\s*100|\*\s*100\.0",
    flags=re.IGNORECASE,
)


def _first_ident(text: str) -> str | None:
    match = _IDENT_RE.search(text)
    if not match:
        return None
    return match.group(1) or match.group(2) or match.group(3)


def _extract_from_tables(sql: str) -> set[str]:
    """Best-effort set of base table names referenced after FROM/JOIN.

    Consumes the canonical IR (§13.1) so quoted table names cannot bypass.
    """
    found: set[str] = set()
    for match in _FROM_JOIN_RE.finditer(normalize_sql(sql)):
        clause = match.group(1)
        first = _first_ident(clause)
        if first and first.upper() not in {"SELECT", "WITH"}:
            found.add(first.casefold())
    return found


def _step_sql(step: StepRecord) -> str:
    if isinstance(step.action_input, dict):
        sql = step.action_input.get("sql")
        if isinstance(sql, str) and sql.strip():
            return sql
    content = step.observation.get("content") if isinstance(step.observation, dict) else None
    if isinstance(content, dict):
        sql = content.get("sql")
        if isinstance(sql, str):
            return sql
    return ""


def _step_row_count(step: StepRecord) -> int | None:
    content = step.observation.get("content") if isinstance(step.observation, dict) else None
    if not isinstance(content, dict):
        return None
    row_count = content.get("row_count")
    if isinstance(row_count, int):
        return row_count
    return None


def _is_aggregate_filter_probe(step: StepRecord) -> bool:
    """True for a probe that likely produced a small member set via aggregation."""
    if step.action != "run_sql" or not step.ok:
        return False
    sql = normalize_sql(_step_sql(step))
    if not sql:
        return False
    return bool(_HAVING_RE.search(sql) and _GROUP_BY_RE.search(sql))


_AGG_FUNC_RE = re.compile(r"\b(?:MIN|MAX|AVG|SUM|COUNT)\s*\(", flags=re.IGNORECASE)


def _is_grouped_aggregate_probe(step: StepRecord) -> bool:
    """True for a probe that aggregated at some group grain (HAVING optional)."""
    if step.action != "run_sql" or not step.ok:
        return False
    sql = normalize_sql(_step_sql(step))
    if not sql:
        return False
    return bool(_GROUP_BY_RE.search(sql) and _AGG_FUNC_RE.search(sql))


def submit_membership_rejection(
    steps: list[StepRecord],
    sql: str | None,
    answer: AnswerTable,
) -> dict[str, Any] | None:
    """Reject when the final answer re-expands beyond an aggregate-filter member set.

    Looks for small HAVING probes in the trace. If the final SQL no longer
    references any of the tables used by those probes and the final answer is
    much larger, the agent likely dropped the filtered aggregate table and
    re-expanded via a group key.
    """
    row_count = len(answer.rows)
    if row_count == 0:
        return None

    final_tables = _extract_from_tables(sql or "")
    probes: list[dict[str, Any]] = []
    for step in steps:
        if not _is_aggregate_filter_probe(step):
            continue
        probe_count = _step_row_count(step)
        if probe_count is None or probe_count > 10:
            continue
        probe_sql = _step_sql(step)
        probe_tables = _extract_from_tables(probe_sql)
        if not probe_tables:
            continue
        probes.append(
            {
                "row_count": probe_count,
                "tables": sorted(probe_tables),
                "sql": probe_sql,
            }
        )

    if not probes:
        return None

    for probe in probes:
        overlap = final_tables & {t.casefold() for t in probe["tables"]}
        if overlap:
            continue
        if row_count <= probe["row_count"] * 5:
            continue
        return {
            "ok": False,
            "error": (
                "answer rejected: final SQL dropped tables used by an earlier "
                f"aggregate filter probe (member set size={probe['row_count']}), "
                f"but the submitted answer has {row_count} rows."
            ),
            "hint": (
                "The rows that passed the aggregate filter are the ones that "
                "should be returned. Keep the aggregate-filter table in the final "
                "query instead of re-expanding through a group key. If the filter "
                "produced the correct members, SELECT from that result directly "
                "or join it back on the member key only."
            ),
            "membership_check": {
                "probe_row_count": probe["row_count"],
                "probe_tables": probe["tables"],
                "probe_sql": probe["sql"],
                "final_row_count": row_count,
                "final_tables": sorted(final_tables),
            },
        }
    return None


def submit_member_expansion_rejection(
    question: str,
    steps: list[StepRecord],
    sql: str | None,
    answer: AnswerTable,
) -> dict[str, Any] | None:
    """Reject member re-expansion of a small grouped aggregate result (§13.2).

    Pattern (task_199): a probe aggregated at group grain and returned a handful
    of rows (2 districts); the final SQL then listed the *members* of those
    groups (57 schools), while the question's aggregate condition belongs to the
    output entity itself (schools whose average > 400).

    Fires only when ALL structural signals hold:
    - the question is extremum/average-flagged (member-expansion class; not
      dual-grain — average still qualifies here even without a SUM formula);
    - a grouped-aggregate probe returned <= 10 rows and shares a table with the
      final SQL;
    - the final answer is much larger than the probe (> max(10, 5x probe rows));
    - the OUTER final query carries no HAVING of its own (the aggregate filter
      was not applied at the output-row grain).

    Conservative escapes: correct small answers (row-count guard) and finals
    that filter at the output grain (outer-HAVING guard) never fire. Intended
    member listings may false-fire and are bounded by the L4 escalation cap.
    """
    if not wants_member_expansion_check(question or ""):
        return None
    row_count = len(answer.rows)
    if row_count <= 10:
        return None
    outer = strip_subqueries(sql)
    if _HAVING_RE.search(outer):
        return None
    final_tables = _extract_from_tables(sql or "")
    if not final_tables:
        return None

    for step in steps:
        if not _is_grouped_aggregate_probe(step):
            continue
        probe_count = _step_row_count(step)
        if probe_count is None or probe_count > 10:
            continue
        if row_count <= max(10, probe_count * 5):
            continue
        probe_sql = _step_sql(step)
        probe_tables = _extract_from_tables(probe_sql)
        if not (probe_tables & final_tables):
            continue
        return {
            "ok": False,
            "error": (
                "answer rejected: a grouped aggregate probe returned "
                f"{probe_count} group(s), but the final answer expands to "
                f"{row_count} member rows without an outer HAVING filter. The "
                "aggregate condition was applied at the group grain and the "
                "result was re-expanded to members."
            ),
            "hint": (
                "If the question's aggregate condition describes the OUTPUT rows "
                "themselves (e.g. 'schools whose average exceeds X'), apply it at "
                "the output-row grain: GROUP BY <output entity> HAVING <condition>, "
                "and SELECT from that result. Only expand to group members when the "
                "question explicitly asks for members of the filtered groups."
            ),
            "member_expansion_check": {
                "probe_row_count": probe_count,
                "probe_tables": sorted(probe_tables),
                "probe_sql": probe_sql,
                "final_row_count": row_count,
                "final_tables": sorted(final_tables),
                "outer_has_having": False,
            },
        }
    return None


def _find_relation_tables(state: WarehouseState) -> list[str]:
    """Tables that look like symmetric relationship tables (have a rel_id column)."""
    try:
        from data_agent_baseline.tools.warehouse import describe_tables

        tables = describe_tables(state)
    except Exception:
        return []
    rel_tables: list[str] = []
    for table in tables:
        name = str(table.get("name", ""))
        cols = {str(c.get("name", "")).casefold() for c in table.get("columns", [])}
        # A relationship table typically has an explicit relationship id plus
        # two entity keys.
        if _REL_ID_COL_RE.search(name) or any(_REL_ID_COL_RE.match(c) for c in cols):
            rel_tables.append(name)
    return rel_tables


def _relation_is_symmetric(state: WarehouseState, table: str) -> tuple[bool, int | None]:
    """Check whether the table stores each relationship multiple times."""
    try:
        rel_id_col = None
        from data_agent_baseline.tools.warehouse import describe_tables

        for tbl in describe_tables(state):
            if str(tbl.get("name", "")).casefold() != table.casefold():
                continue
            cols = [str(c.get("name", "")).casefold() for c in tbl.get("columns", [])]
            for c in cols:
                if _REL_ID_COL_RE.match(c):
                    rel_id_col = c
                    break
            if rel_id_col is None and "id" in cols:
                rel_id_col = "id"
            break
        if rel_id_col is None:
            return False, None
        with state._lock:
            row = state.conn.execute(
                f"SELECT {quote_ident(rel_id_col)}, COUNT(*) AS n "
                f"FROM {quote_ident(table)} "
                f"GROUP BY {quote_ident(rel_id_col)} "
                f"HAVING COUNT(*) > 1 "
                f"LIMIT 1"
            ).fetchone()
        if row is None:
            return False, None
        return True, int(row[1])
    except Exception:
        return False, None


def _counts_distinct_relation_id(sql: str) -> bool:
    """True when SQL already DISTINCT-counts a relationship-id column.

    Design intent of the relation gate: reject COUNT(*) / COUNT(entity) on
    symmetric relationship tables. COUNT(DISTINCT bond_id) (or relation_id /
    edge_id / …), including CASE expressions that select those ids, is the
    recommended fix and must be allowed through.
    """
    text = normalize_sql(sql)
    for match in _COUNT_DISTINCT_HEAD_RE.finditer(text):
        open_paren = text.find("(", match.start())
        if open_paren < 0:
            continue
        depth = 0
        close = None
        for index, char in enumerate(text[open_paren:], open_paren):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    close = index
                    break
        if close is None:
            continue
        if _REL_ID_COL_RE.search(text[open_paren : close + 1]):
            return True
    return False


def submit_symmetric_relation_rejection(
    sql: str | None,
    state: WarehouseState | None,
) -> dict[str, Any] | None:
    """Reject row counts on symmetric relationship tables.

    If the final SQL counts rows of a relationship table that stores each
    relationship in both directions, it should count DISTINCT relationship ids.
    Already-correct COUNT(DISTINCT rel_id) forms are allowed through.
    """
    if not sql or state is None:
        return None
    sql_text = sql or ""
    if not (_COUNT_STAR_RE.search(sql_text) or _COUNT_COL_RE.search(sql_text)):
        return None
    # P0: do not reject SQL that already DISTINCT-counts the relationship id.
    if _counts_distinct_relation_id(sql_text):
        return None
    final_tables = _extract_from_tables(sql_text)
    if not final_tables:
        return None

    rel_tables = _find_relation_tables(state)
    for rel_table in rel_tables:
        if rel_table.casefold() not in final_tables:
            continue
        is_sym, dup_count = _relation_is_symmetric(state, rel_table)
        if not is_sym:
            continue
        return {
            "ok": False,
            "error": (
                f"answer rejected: '{rel_table}' stores each relationship multiple "
                f"times (example relationship id appears {dup_count} times)."
            ),
            "hint": (
                "Do not COUNT(*) or COUNT(entity_column) on a symmetric relationship "
                "table. Use COUNT(DISTINCT relationship_id) instead, where "
                "relationship_id is the identifier of the bond/edge/connection itself."
            ),
            "relation_check": {
                "relation_table": rel_table,
                "duplicate_example_count": dup_count,
            },
        }
    return None


def submit_sanity_rejection(
    sql: str | None,
    answer: AnswerTable,
) -> dict[str, Any] | None:
    """Reject obvious out-of-range numeric answers.

    Currently only percentage/ratio values outside [0, 100] are blocked. This is
    intentionally conservative to avoid false positives.
    """
    if len(answer.rows) != 1 or len(answer.columns) != 1:
        return None
    value = answer.rows[0][0]
    if not isinstance(value, (int, float)):
        return None
    if not _PERCENT_RE.search(sql or ""):
        return None
    if 0 <= float(value) <= 100:
        return None
    return {
        "ok": False,
        "error": (
            f"answer rejected: the result is a percentage/ratio ({value}) outside "
            "the range [0, 100]."
        ),
        "hint": (
            "Check numerator and denominator. Make sure you are computing "
            "part / whole * 100 and not inverting the ratio. If the value is a "
            "share, it may already be in [0, 1] and should not be multiplied by 100."
        ),
        "sanity_check": {
            "value": value,
            "range": "[0, 100]",
        },
    }


# --- Answer shape locks (B2 / B3) -------------------------------------------------

_SCALAR_SHAPE_RE = re.compile(
    r"(?:"
    r"\bcalculate\b|"
    r"\bhow\s+many\b|"
    r"\bhow\s+much\b|"
    r"what(?:'s|\s+is)\s+the\s+(?:total|average|avg|mean|count|number|ratio|share|percentage)|"
    r"what\s+percentage|"
    r"percentage\s+of|"
    r"\bpercent(?:age)?\b"
    r")",
    flags=re.IGNORECASE,
)
_STATUS_METRIC_Q_RE = re.compile(
    r"(?:"
    r"consumption\s+status|"
    r"\bstatus\s+of\b|"
    r"what\s+is\s+(?:the\s+)?(?:consumption|status)|"
    r"give\s+(?:me\s+)?(?:the\s+)?(?:status|consumption|balance|amount)|"
    r"report\s+(?:the\s+)?(?:status|consumption)"
    r")",
    flags=re.IGNORECASE,
)
_ENTITY_ID_COL_RE = re.compile(
    r"(?:^|_)(?:.+_)?id$|customerid|userid|accountid|memberid|patientid|clientid",
    flags=re.IGNORECASE,
)
_METRIC_COL_HINT_RE = re.compile(
    r"status|amount|balance|price|cost|total|count|qty|quantity|score|value|rate|percent",
    flags=re.IGNORECASE,
)


def wants_scalar_shape(question: str) -> bool:
    """True when the question asks for a single scalar (count / total / percent)."""
    return bool(_SCALAR_SHAPE_RE.search(question or ""))


def wants_status_metric_shape(question: str) -> bool:
    """True when the question asks for a status/metric value, not entity keys."""
    return bool(_STATUS_METRIC_Q_RE.search(question or ""))


def _is_numeric_cell(value: Any) -> bool:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return True
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if not text:
            return False
        try:
            float(text)
            return True
        except ValueError:
            return False
    return False


def is_scalar_answer_table(answer: AnswerTable) -> bool:
    """Single-column, single-row, numeric-like answer."""
    if len(answer.columns) != 1 or len(answer.rows) != 1:
        return False
    return _is_numeric_cell(answer.rows[0][0])


def _final_answer_from_step(step: StepRecord) -> AnswerTable | None:
    if step.action != "run_sql" or not step.ok:
        return None
    content = step.observation.get("content") if isinstance(step.observation, dict) else None
    if not isinstance(content, dict) or not content.get("full_scan"):
        return None
    columns = content.get("columns")
    rows = content.get("rows")
    if not isinstance(columns, list) or not isinstance(rows, list):
        return None
    col_names = [str(c) for c in columns]
    normalized_rows: list[list[Any]] = []
    for row in rows:
        if isinstance(row, (list, tuple)):
            normalized_rows.append(list(row))
    return AnswerTable(columns=col_names, rows=normalized_rows)


def submit_shape_rejection(
    question: str,
    answer: AnswerTable,
    steps: list[StepRecord],
) -> dict[str, Any] | None:
    """Reject detail finals that overwrite a prior scalar final on scalar questions.

    Conservative: only fires when the question looks scalar AND a prior final=true
    already produced a 1×1 numeric table, but the submitted answer is wider/taller.
    """
    if not wants_scalar_shape(question):
        return None
    if is_scalar_answer_table(answer):
        return None

    prior_scalar: AnswerTable | None = None
    for step in steps:
        prior = _final_answer_from_step(step)
        if prior is not None and is_scalar_answer_table(prior):
            prior_scalar = prior

    # Overwrite case (task_200 pattern): had a good COUNT, then submitted detail.
    if prior_scalar is not None and (
        len(answer.columns) > 1 or len(answer.rows) > 1
    ):
        return {
            "ok": False,
            "error": (
                "answer rejected: this question asks for a single scalar value, but "
                "a later final query replaced an earlier scalar result with a "
                f"detail table ({len(answer.columns)} columns × {len(answer.rows)} rows)."
            ),
            "hint": (
                "Re-run the successful aggregate/COUNT as run_sql with final=true "
                "(one metric column, typically one row), then call answer. Do not "
                "submit entity-level detail rows for a how-many / calculate / total "
                "question."
            ),
            "shape_check": {
                "question_wants_scalar": True,
                "prior_scalar_columns": list(prior_scalar.columns),
                "prior_scalar_rows": prior_scalar.rows,
                "current_columns": list(answer.columns),
                "current_row_count": len(answer.rows),
            },
        }

    # Clear multi-column detail on an unambiguous scalar ask, even without prior.
    if len(answer.columns) > 1 and len(answer.rows) > 1:
        return {
            "ok": False,
            "error": (
                "answer rejected: scalar question submitted a multi-column detail "
                f"table ({len(answer.columns)} columns × {len(answer.rows)} rows)."
            ),
            "hint": (
                "SELECT only the metric (COUNT / SUM / percentage). Use a single "
                "output column; do not include entity id or name columns."
            ),
            "shape_check": {
                "question_wants_scalar": True,
                "prior_scalar_columns": None,
                "current_columns": list(answer.columns),
                "current_row_count": len(answer.rows),
            },
        }
    return None


def submit_ambiguity_probe_rejection(
    *,
    ambiguity_groups: list[Any] | None,
    steps: list[StepRecord],
    sql: str | None,
) -> dict[str, Any] | None:
    """Reject when final SQL picks one of several ambiguous columns without probing peers.

    Identity is ``table.column`` (not the leaf name). Same-named columns on
    different tables are distinct candidates; a probe on one table does not
    cover the other.

    Consumes the canonical IR (§13.1): quoted identifiers are unquoted, and
    FROM/JOIN + alias resolution decide which table an unqualified column
    belongs to. Unresolved homonyms (join of two tables that share the leaf,
    unqualified projection) are UNDECIDED — not a silent pass-as-clean, and
    not a reject.
    """
    if not ambiguity_groups or not sql:
        return None
    probe_sqls = [
        _step_sql(step)
        for step in steps
        if step.action == "run_sql" and step.ok
    ]

    for group in ambiguity_groups:
        columns = list(getattr(group, "columns", None) or [])
        if len(columns) < 2:
            continue
        if _homonym_unresolved(sql, columns):
            names = [f"{c.table}.{c.column}" for c in columns[:4]]
            return undecided_payload(
                check="ambiguity",
                reason=(
                    "unqualified column on a join of tables that share this name; "
                    f"candidates {names}"
                ),
                suggestion=(
                    "Qualify the projected column as table.column, probe EACH "
                    "table.column, then pick by knowledge / named entity table / "
                    "fewest joins."
                ),
            )
        mentioned = [
            c for c in columns if sql_mentions_table_column(sql, c.table, c.column)
        ]
        if len(mentioned) != 1:
            continue
        chosen = mentioned[0]
        peers = [
            c
            for c in columns
            if f"{c.table}.{c.column}".casefold()
            != f"{chosen.table}.{chosen.column}".casefold()
        ]
        if not peers:
            continue
        same_leaf = len({c.column.casefold() for c in columns}) == 1

        def _peer_probed(peer: Any) -> bool:
            return any(
                sql_mentions_table_column(probe, peer.table, peer.column)
                for probe in probe_sqls
            )

        if same_leaf:
            unprobed = [p for p in peers if not _peer_probed(p)]
        else:
            unprobed = [] if any(_peer_probed(p) for p in peers) else list(peers)
        if not unprobed:
            continue
        peer_names = [f"{p.table}.{p.column}" for p in unprobed[:4]]
        return {
            "ok": False,
            "error": (
                f"answer rejected: ambiguous term '{getattr(group, 'term', '?')}' "
                f"resolved to {chosen.table}.{chosen.column} without probing peer "
                f"columns {peer_names}."
            ),
            "hint": (
                "Run the same filter on EACH table.column via run_sql (a probe "
                "on one table does not cover the same column name on another "
                "table). Compare hit counts / row sets, then choose: (1) the "
                "column knowledge maps; (2) the identifier on the entity table "
                "the question names; (3) fewest joins / direct FK path."
            ),
            "ambiguity_check": {
                "term": getattr(group, "term", None),
                "chosen": f"{chosen.table}.{chosen.column}",
                "peers": peer_names,
            },
        }
    return None


def _homonym_unresolved(sql: str | None, columns: list[Any]) -> bool:
    """Unqualified leaf name on a join of two homonym tables — cannot tell which."""
    leaves = {str(c.column).casefold() for c in columns}
    if len(leaves) != 1:
        return False
    leaf = next(iter(leaves))
    text = normalize_sql(sql)
    if not re.search(rf"\b{re.escape(leaf)}\b", text, flags=re.IGNORECASE):
        return False
    if any(sql_mentions_table_column(sql, c.table, c.column) for c in columns):
        return False
    group_tables = {str(c.table).casefold() for c in columns}
    scoped = {name.casefold() for name in from_tables(sql)}
    return len(group_tables & scoped) >= 2


def submit_id_column_rejection(
    question: str,
    answer: AnswerTable,
) -> dict[str, Any] | None:
    """Reject entity-id columns on status/metric questions when a metric col exists.

    Uncertain cases (no clear metric column, or no id column) are allowed through.
    """
    if not wants_status_metric_shape(question):
        return None
    if len(answer.columns) <= 1:
        return None

    id_cols = [c for c in answer.columns if _ENTITY_ID_COL_RE.search(c)]
    metric_cols = [
        c
        for c in answer.columns
        if _METRIC_COL_HINT_RE.search(c) and c not in id_cols
    ]
    if not id_cols or not metric_cols:
        return None

    return {
        "ok": False,
        "error": (
            "answer rejected: status/metric question includes entity id column(s) "
            f"{id_cols} alongside metric column(s) {metric_cols}."
        ),
        "hint": (
            "SELECT only the metric/status column(s) the question asks for. "
            f"Drop id columns such as {id_cols[0]} from the final query."
        ),
        "id_column_check": {
            "id_columns": id_cols,
            "metric_columns": metric_cols,
            "all_columns": list(answer.columns),
        },
    }
