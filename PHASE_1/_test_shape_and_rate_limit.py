"""Unit tests for A2 rate-limit hard-stop, B2 shape lock, B3 id columns, C1 registry names."""

from __future__ import annotations

from data_agent_baseline.agents.react import is_rate_limit_error, parse_model_step
from data_agent_baseline.agents.runtime import AgentRuntimeState, StepRecord
from data_agent_baseline.agents.submit_validation import (
    is_scalar_answer_table,
    submit_id_column_rejection,
    submit_shape_rejection,
    wants_scalar_shape,
)
from data_agent_baseline.benchmark.schema import AnswerTable
from data_agent_baseline.tools.doc_extract import apply_registry_official_names


def test_rate_limit_detection():
    assert is_rate_limit_error("Error code: 429 - rate_limit_exceeded")
    assert is_rate_limit_error("Model request failed: Rate limit reached")
    assert is_rate_limit_error("insufficient_quota")
    assert not is_rate_limit_error("Binder Error: table missing")


def test_wants_scalar_shape():
    assert wants_scalar_shape("Calculate how many elements match?")
    assert wants_scalar_shape("How many customers paid?")
    assert wants_scalar_shape("What is the percentage of overdue invoices?")
    assert not wants_scalar_shape("List every customer who paid.")


def test_shape_rejects_detail_overwrite():
    prior = StepRecord(
        step_index=1,
        thought="count",
        action="run_sql",
        action_input={"sql": "SELECT COUNT(*) AS n FROM t", "final": True},
        raw_response="",
        observation={
            "ok": True,
            "content": {
                "full_scan": True,
                "columns": ["n"],
                "rows": [[42]],
                "row_count": 1,
            },
        },
        ok=True,
    )
    detail = AnswerTable(
        columns=["element", "value"],
        rows=[["a", 1], ["b", 2], ["c", 3]],
    )
    result = submit_shape_rejection(
        "Calculate how many elements match the filter?",
        detail,
        [prior],
    )
    assert result is not None
    assert "shape_check" in result
    assert result["shape_check"]["prior_scalar_columns"] == ["n"]


def test_shape_allows_scalar_submit():
    answer = AnswerTable(columns=["n"], rows=[[42]])
    assert is_scalar_answer_table(answer)
    result = submit_shape_rejection("How many rows?", answer, [])
    assert result is None


def test_id_column_rejection_for_status():
    answer = AnswerTable(
        columns=["CustomerID", "ConsumptionStatus"],
        rows=[["C1", "High"], ["C2", "Low"]],
    )
    result = submit_id_column_rejection(
        "What is the consumption status of each segment?",
        answer,
    )
    assert result is not None
    assert "CustomerID" in result["id_column_check"]["id_columns"]


def test_id_column_allows_when_uncertain():
    answer = AnswerTable(columns=["name", "city"], rows=[["a", "b"]])
    result = submit_id_column_rejection("What is the consumption status?", answer)
    assert result is None


def test_registry_official_names():
    paragraphs = [
        "The program for Business (Registry ID: recABC) launched in 2020.",
        "Later the general Business program (Registry ID: recABC) expanded.",
    ]
    rows = [
        {"registry_id": "recABC", "program_name": "General Business"},
    ]
    out = apply_registry_official_names(
        paragraphs, rows, columns=["registry_id", "program_name"]
    )
    assert out[0]["program_name"] == "Business"


def test_registry_keeps_general_motors_alone():
    paragraphs = [
        "General Motors (Registry ID: gm1) is listed once.",
    ]
    rows = [{"registry_id": "gm1", "program_name": "General Motors"}]
    out = apply_registry_official_names(
        paragraphs, rows, columns=["registry_id", "program_name"]
    )
    assert out[0]["program_name"] == "General Motors"


def test_parse_model_step_still_works():
    step = parse_model_step(
        '```json\n{"thought":"t","action":"answer","action_input":{}}\n```'
    )
    assert step.action == "answer"


if __name__ == "__main__":
    test_rate_limit_detection()
    test_wants_scalar_shape()
    test_shape_rejects_detail_overwrite()
    test_shape_allows_scalar_submit()
    test_id_column_rejection_for_status()
    test_id_column_allows_when_uncertain()
    test_registry_official_names()
    test_registry_keeps_general_motors_alone()
    test_parse_model_step_still_works()
    print("ok")
