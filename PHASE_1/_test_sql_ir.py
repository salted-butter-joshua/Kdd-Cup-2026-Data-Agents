"""Unit tests for the canonical SQL IR (架构设计 §13.1)."""

from __future__ import annotations

from data_agent_baseline.agents.sql_ir import (
    alias_map,
    from_tables,
    normalize_sql,
    sql_mentions_table_column,
    strip_subqueries,
)


def test_normalize_strips_identifier_quotes() -> None:
    assert (
        normalize_sql('SELECT e."cost" FROM "expense" AS e')
        == "SELECT e.cost FROM expense AS e"
    )
    assert normalize_sql("SELECT `cost` FROM `expense`") == "SELECT cost FROM expense"
    assert normalize_sql("SELECT [cost] FROM [expense]") == "SELECT cost FROM expense"


def test_normalize_keeps_string_literals() -> None:
    sql = "SELECT * FROM frpm WHERE \"District Name\" LIKE '%Riverside%'"
    assert normalize_sql(sql) == (
        "SELECT * FROM frpm WHERE District Name LIKE '%Riverside%'"
    )


def test_normalize_collapses_whitespace() -> None:
    assert normalize_sql("SELECT  a\n  FROM   t\t\tWHERE x = 1") == (
        "SELECT a FROM t WHERE x = 1"
    )
    assert normalize_sql("") == ""
    assert normalize_sql(None) == ""


def test_alias_map() -> None:
    mapping = alias_map(
        'SELECT * FROM "expense" e JOIN budget AS b ON e.id = b.id '
        "WHERE b.x IN (SELECT y FROM event)"
    )
    assert mapping == {"e": "expense", "b": "budget"}


def test_alias_map_ignores_keywords() -> None:
    mapping = alias_map("SELECT a FROM yearmonth GROUP BY a")
    assert mapping == {}


def test_from_tables_includes_unaliased() -> None:
    assert from_tables('SELECT number FROM "qualifying"') == ["qualifying"]
    assert from_tables(
        "SELECT n FROM qualifying q JOIN drivers d ON q.driverId = d.driverId"
    ) == ["qualifying", "drivers"]


def test_sql_mentions_table_column_is_table_scoped() -> None:
    q_sql = "SELECT number FROM qualifying WHERE q3 LIKE '1:54%'"
    assert sql_mentions_table_column(q_sql, "qualifying", "number")
    assert not sql_mentions_table_column(q_sql, "drivers", "number")
    d_sql = 'SELECT number FROM "drivers" WHERE driverId = 20'
    assert sql_mentions_table_column(d_sql, "drivers", "number")
    assert not sql_mentions_table_column(d_sql, "qualifying", "number")
    qualified = "SELECT d.number FROM qualifying q JOIN drivers d ON q.driverId = d.driverId"
    assert sql_mentions_table_column(qualified, "drivers", "number")
    assert not sql_mentions_table_column(qualified, "qualifying", "number")
    star = "SELECT * FROM drivers LIMIT 5"
    assert sql_mentions_table_column(star, "drivers", "number")


def test_strip_subqueries() -> None:
    sql = (
        "SELECT s FROM frpm f WHERE d IN (SELECT d FROM frpm GROUP BY d "
        "HAVING AVG(x) > 400) ORDER BY s"
    )
    outer = strip_subqueries(sql)
    assert "HAVING" not in outer
    assert "ORDER BY" in outer
    # Outer HAVING survives stripping.
    sql2 = "SELECT d FROM t WHERE x IN (SELECT y FROM u) GROUP BY d HAVING AVG(x) > 1"
    assert "HAVING" in strip_subqueries(sql2)


if __name__ == "__main__":
    test_normalize_strips_identifier_quotes()
    test_normalize_keeps_string_literals()
    test_normalize_collapses_whitespace()
    test_alias_map()
    test_alias_map_ignores_keywords()
    test_from_tables_includes_unaliased()
    test_sql_mentions_table_column_is_table_scoped()
    test_strip_subqueries()
    print("ok")
