"""Gates: empty final and extremum LIMIT that hides ties. Majority vote keeps one table."""

from __future__ import annotations

from pathlib import Path

import duckdb

from data_agent_baseline.benchmark.schema import AnswerTable
from data_agent_baseline.agents.submit_rows import (
    submit_projection_rejection,
    submit_row_rejection,
)
from data_agent_baseline.run.runner import select_majority_answer
from data_agent_baseline.tools.warehouse import RegisteredTable, WarehouseState


def _state() -> WarehouseState:
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE cost(spent INTEGER)")
    conn.execute("INSERT INTO cost VALUES (1), (1), (5)")
    state = WarehouseState(conn=conn, context_dir=Path("."))
    state.tables.append(
        RegisteredTable(
            canonical="cost",
            aliases=[],
            source_type="csv",
            source_rel="context/csv/cost.csv",
        )
    )
    return state


def test_limit_hides_tie() -> None:
    state = _state()
    sql = 'SELECT spent FROM cost ORDER BY spent LIMIT 1'
    answer = AnswerTable(columns=["spent"], rows=[[1]])
    rejection = submit_row_rejection(
        "Which row has the lowest spent?",
        answer,
        sql=sql,
        state=state,
    )
    assert rejection is not None
    assert rejection["tie_check"]["hit_count"] == 3
    state.close()


def test_equality_keeps_ties() -> None:
    state = _state()
    sql = "SELECT spent FROM cost WHERE spent = (SELECT MIN(spent) FROM cost)"
    answer = AnswerTable(columns=["spent"], rows=[[1], [1]])
    rejection = submit_row_rejection(
        "Which row has the lowest spent?",
        answer,
        sql=sql,
        state=state,
    )
    assert rejection is None
    state.close()


def test_empty_includes_bounds() -> None:
    state = _state()
    sql = "SELECT spent FROM cost WHERE spent = 9"
    answer = AnswerTable(columns=["spent"], rows=[])
    rejection = submit_row_rejection(
        "List spent equal to 9",
        answer,
        sql=sql,
        state=state,
    )
    assert rejection is not None
    bounds = rejection["empty_check"]["column_bounds"]
    assert bounds and bounds[0]["min"] == 1 and bounds[0]["max"] == 5
    assert bounds[0]["row_count"] == 3
    state.close()


def test_projection_rejection_is_not_task_specific() -> None:
    """Projection rejection no longer hardcodes time-prefix / id-only / category rules."""
    answer = AnswerTable(columns=["number"], rows=[[3], [1]])
    rejection = submit_projection_rejection(
        answer,
        sql="SELECT number FROM qualifying WHERE q3 LIKE '1:54%'",
        steps=[],
    )
    assert rejection is None


def test_majority_does_not_union() -> None:
    a = {"answer": {"columns": ["id"], "rows": [[1], [2]]}}
    b = {"answer": {"columns": ["id"], "rows": [[1], [2]]}}
    c = {"answer": {"columns": ["id"], "rows": [[9]]}}
    chosen = select_majority_answer([c, a, b])
    assert chosen["answer"]["rows"] == [[1], [2]]
    assert [9] not in chosen["answer"]["rows"] or chosen["answer"]["rows"] == [[1], [2]]


if __name__ == "__main__":
    test_limit_hides_tie()
    test_equality_keeps_ties()
    test_empty_includes_bounds()
    test_projection_rejection_is_not_task_specific()
    test_majority_does_not_union()
    print("ALL_OK")
