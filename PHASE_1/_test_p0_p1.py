"""Mechanism tests for P0 cutoff / fake-zero / scalar promote and P1 voting / sidecars."""

from __future__ import annotations

from pathlib import Path

import duckdb

from data_agent_baseline.agents.answer_contract import (
    drop_unasked_sidecar_columns,
    is_constant_zero_sql,
    question_allows_single_row_cutoff,
    question_wants_scalar_value,
    sql_is_global_aggregate,
    sql_without_singleton_cutoff,
    submit_fake_zero_rejection,
)
from data_agent_baseline.agents.episode import ConfirmedProbe, EpisodeState
from data_agent_baseline.agents.runtime import StepRecord
from data_agent_baseline.agents.submit_rows import submit_row_rejection
from data_agent_baseline.agents.voting import vote_best_final
from data_agent_baseline.benchmark.schema import AnswerTable
from data_agent_baseline.tools.warehouse import WarehouseState


def _final_step(index: int, sql: str, columns: list[str], rows: list[list[object]]) -> StepRecord:
    return StepRecord(
        step_index=index,
        thought="",
        action="run_sql",
        action_input={"sql": sql, "final": True},
        raw_response="{}",
        observation={
            "ok": True,
            "content": {
                "columns": columns,
                "rows": rows,
                "row_count": len(rows),
            },
        },
        ok=True,
    )


def test_cutoff_keeps_maxmin_predicate() -> None:
    sql = (
        "SELECT name FROM lab WHERE species = 'dog' AND date = "
        "(SELECT MAX(date) FROM lab)"
    )
    kept = sql_without_singleton_cutoff(sql)
    assert "MAX" in kept.upper()
    assert "species" in kept


def test_unique_min_not_rejected() -> None:
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE qualifying (driver VARCHAR, raceId INT, q2 VARCHAR)")
    conn.execute(
        "INSERT INTO qualifying VALUES "
        "('slow', 19, '1:20.0'), ('fast', 19, '1:10.0'), ('mid', 19, '1:15.0')"
    )
    state = WarehouseState(conn=conn, context_dir=Path("."))
    sql = (
        "SELECT driver FROM qualifying WHERE raceId = 19 AND q2 = "
        "(SELECT MIN(q2) FROM qualifying WHERE raceId = 19 AND q2 IS NOT NULL)"
    )
    answer = AnswerTable(columns=["driver"], rows=[["fast"]])
    assert (
        submit_row_rejection(
            "Who had the fastest Q2 time in race 19?",
            answer,
            sql=sql,
            state=state,
        )
        is None
    )


def test_limit_10_not_treated_as_cutoff() -> None:
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE t (id INT)")
    conn.execute("INSERT INTO t VALUES (1), (2), (3)")
    state = WarehouseState(conn=conn, context_dir=Path("."))
    sql = "SELECT id FROM t LIMIT 10"
    answer = AnswerTable(columns=["id"], rows=[[1], [2], [3]])
    assert submit_row_rejection("List the ids.", answer, sql=sql, state=state) is None


def test_min_plus_limit_1_rejects_hidden_ties() -> None:
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE t (name VARCHAR, score INT)")
    conn.execute("INSERT INTO t VALUES ('a', 10), ('b', 10), ('c', 8)")
    state = WarehouseState(conn=conn, context_dir=Path("."))
    sql = (
        "SELECT name FROM t WHERE score = (SELECT MAX(score) FROM t) LIMIT 1"
    )
    answer = AnswerTable(columns=["name"], rows=[["a"]])
    rejected = submit_row_rejection(
        "Who has the highest score?",
        answer,
        sql=sql,
        state=state,
    )
    assert rejected is not None


def test_open_question_rejects_limit_1() -> None:
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE lab (name VARCHAR, date DATE)")
    conn.execute("INSERT INTO lab VALUES ('a', DATE '2020-01-01'), ('b', DATE '2020-01-02')")
    state = WarehouseState(conn=conn, context_dir=Path("."))
    question = "What are the names of dogs in the lab?"
    sql = "SELECT name FROM lab ORDER BY date DESC LIMIT 1"
    answer = AnswerTable(columns=["name"], rows=[["b"]])
    rejected = submit_row_rejection(question, answer, sql=sql, state=state)
    assert rejected is not None
    assert "cutoff" in str(rejected.get("error", "")).lower() or "LIMIT" in str(
        rejected.get("hint", "")
    )


def test_latest_question_allows_limit_1() -> None:
    assert question_allows_single_row_cutoff("What is the most recent payment date?")
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE p (d DATE)")
    conn.execute("INSERT INTO p VALUES (DATE '2020-01-01'), (DATE '2020-01-02')")
    state = WarehouseState(conn=conn, context_dir=Path("."))
    sql = "SELECT d FROM p ORDER BY d DESC LIMIT 1"
    answer = AnswerTable(columns=["d"], rows=[["2020-01-02"]])
    assert submit_row_rejection(
        "What is the most recent payment date?",
        answer,
        sql=sql,
        state=state,
    ) is None


def test_count_aggregate_not_treated_as_cutoff() -> None:
    assert sql_is_global_aggregate("SELECT COUNT(*) AS n FROM t")
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE t (id INT)")
    conn.execute("INSERT INTO t VALUES (1)")
    state = WarehouseState(conn=conn, context_dir=Path("."))
    sql = "SELECT COUNT(*) AS n FROM t LIMIT 1"
    answer = AnswerTable(columns=["n"], rows=[[1]])
    assert submit_row_rejection("How many rows?", answer, sql=sql, state=state) is None


def test_constant_zero_rejected() -> None:
    assert is_constant_zero_sql("SELECT 0")
    assert question_wants_scalar_value("What percentage of comments were deleted?")
    answer = AnswerTable(columns=["pct"], rows=[[0]])
    rejected = submit_fake_zero_rejection(
        "What percentage of comments were deleted?",
        answer,
        sql="SELECT 0 AS pct",
        state=None,
    )
    assert rejected is not None


def test_scalar_does_not_promote_name_list() -> None:
    episode = EpisodeState()
    episode.confirmed.append(
        ConfirmedProbe(
            sql="SELECT name FROM races",
            fingerprint="select name from races",
            row_count=14,
            columns=["name"],
        )
    )
    episode.confirmed.append(
        ConfirmedProbe(
            sql="SELECT COUNT(*) AS n FROM races",
            fingerprint="select count(*) as n from races",
            row_count=1,
            columns=["n"],
        )
    )
    q = "How many times did the champion win?"
    best = episode.best_probe_for(q)
    assert best is not None
    assert best.row_count == 1
    assert best.columns == ["n"]
    episode_only_list = EpisodeState()
    episode_only_list.confirmed.append(episode.confirmed[0])
    assert episode_only_list.best_probe_for(q) is None


def test_vote_prefers_execution_cluster_not_max_rows() -> None:
    q = "How many times did A win?"
    steps = [
        _final_step(1, "SELECT name FROM t", ["name"], [["a"], ["b"], ["c"]]),
        _final_step(2, "SELECT COUNT(*) AS n FROM t", ["n"], [[3]]),
        _final_step(3, "SELECT COUNT(*) AS cnt FROM t", ["cnt"], [[3]]),
    ]
    voted = vote_best_final(question=q, steps=steps)
    assert voted is not None
    assert "COUNT" in voted.sql.upper()


def test_sidecar_year_and_q3_dropped() -> None:
    q = "Which races did the driver win?"
    assert drop_unasked_sidecar_columns(q, ["name", "year"]) == ["name"]
    assert drop_unasked_sidecar_columns(
        "How many posts mention Q3?",
        ["n", "q3"],
    ) == ["n", "q3"]
    dumped = drop_unasked_sidecar_columns(
        "Which ids match the filter?",
        ["trans_id", "date", "type", "operation", "amount", "balance", "k_symbol"],
    )
    assert dumped == ["trans_id"]


def test_message_text_falls_back_to_reasoning() -> None:
    from types import SimpleNamespace

    from data_agent_baseline.agents.model import _message_text_content

    empty = SimpleNamespace(content=None, reasoning="{\"thought\":\"x\"}")
    assert "{\"thought\"" in _message_text_content(empty)
    filled = SimpleNamespace(content="{\"ok\":1}", reasoning="ignore")
    assert _message_text_content(filled) == "{\"ok\":1}"


if __name__ == "__main__":
    test_cutoff_keeps_maxmin_predicate()
    test_unique_min_not_rejected()
    test_limit_10_not_treated_as_cutoff()
    test_min_plus_limit_1_rejects_hidden_ties()
    test_open_question_rejects_limit_1()
    test_latest_question_allows_limit_1()
    test_count_aggregate_not_treated_as_cutoff()
    test_constant_zero_rejected()
    test_scalar_does_not_promote_name_list()
    test_vote_prefers_execution_cluster_not_max_rows()
    test_sidecar_year_and_q3_dropped()
    test_message_text_falls_back_to_reasoning()
    print("OK")
