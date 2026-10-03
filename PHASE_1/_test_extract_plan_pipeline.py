"""Smoke tests for plan → segment merge → prune → subprocess normalize."""

from __future__ import annotations

from data_agent_baseline.agents.model import ScriptedModelAdapter
from data_agent_baseline.tools.doc_extract import (
    ExtractionBudget,
    _run_segment_workers,
    _stub_missing_records,
    merge_rows,
    parse_document_plan,
    prune_extract_columns,
)
from data_agent_baseline.tools.extract_normalize import (
    apply_column_converters,
    builtin_converter_code,
    parse_format_declarations,
    run_converter_code,
)


def test_parse_plan_1based_segments() -> None:
    plan = parse_document_plan(
        {
            "columns": [{"name": "registry_id"}, {"name": "name"}, {"name": "memo"}],
            "primary_key": ["registry_id"],
            "records": ["TR1", "TR2"],
            "segments": [{"start": 1, "end": 4}, {"start": 5, "end": 8}],
        },
        8,
    )
    assert plan.columns == ["registry_id", "name", "memo"]
    assert plan.primary_key == ["registry_id"]
    assert plan.record_keys == ["TR1", "TR2"]
    assert plan.segments == [(0, 4), (4, 8)]


def test_prune_chatter_and_superseded() -> None:
    columns = ["id", "name", "memo", "original_name", "status"]
    rows = [{"id": "1", "name": "A", "memo": "x", "original_name": "old", "status": "ok"}]
    kept, pruned = prune_extract_columns(columns, rows, primary_key=["id"])
    assert "memo" not in kept
    assert "original_name" not in kept
    assert kept == ["id", "name", "status"]
    assert pruned[0] == {"id": "1", "name": "A", "status": "ok"}


def test_stub_missing_records() -> None:
    rows = [{"id": "TR1", "name": "A"}]
    out = _stub_missing_records(
        rows, columns=["id", "name"], primary_key=["id"], record_keys=["TR1", "TR2"]
    )
    assert {row["id"] for row in out} == {"TR1", "TR2"}
    stub = next(row for row in out if row["id"] == "TR2")
    assert stub["name"] == ""


def test_merge_wide_table() -> None:
    rows = [
        {"id": "1", "name": "A", "status": ""},
        {"id": "1", "name": "", "status": "active"},
    ]
    merged = merge_rows(rows, columns=["id", "name", "status"], primary_key=["id"])
    assert merged == [{"id": "1", "name": "A", "status": "active"}]


def test_normalize_subprocess_upper() -> None:
    code = builtin_converter_code("upper")
    out = run_converter_code(code, ["ab", "Cd"])
    assert out == ["AB", "CD"]


def test_normalize_timeout_keeps_original() -> None:
    code = "def convert(value):\n    import time\n    time.sleep(30)\n    return value\n"
    out = run_converter_code(code, ["keep"], timeout_seconds=0.5)
    assert out == ["keep"]


def test_parse_format_declarations() -> None:
    payload = {"formats": {"name": "upper", "when": {"format": "iso_date"}}}
    conv = parse_format_declarations(payload, ["name", "when"])
    assert "name" in conv and "when" in conv
    rows = apply_column_converters(
        [{"name": "ab", "when": "2020/1/2"}],
        columns=["name", "when"],
        converters=conv,
    )
    assert rows[0]["name"] == "AB"
    assert rows[0]["when"] == "2020-01-02"


def test_segment_worker_keeps_parseable() -> None:
    import json

    plan = parse_document_plan(
        {
            "columns": [{"name": "id"}, {"name": "name"}],
            "primary_key": ["id"],
            "records": ["1"],
            "segments": [{"start": 1, "end": 1}, {"start": 2, "end": 2}],
        },
        2,
    )
    # first segment ok; second invalid json then still invalid -> empty rows, still complete
    responses = [
        json.dumps({"rows": [{"id": "1", "name": "A"}]}),
        "not-json",
        "still-not-json",
    ]
    from data_agent_baseline.tools import doc_extract as de

    old = de.EXTRACT_CONCURRENCY
    de.EXTRACT_CONCURRENCY = 1
    try:
        model = ScriptedModelAdapter(responses)
        budget = ExtractionBudget.from_env(seconds=30)
        rows, complete = _run_segment_workers(
            model,
            paragraphs=["Entity 1 is A.", "Noise paragraph here without structure."],
            plan=plan,
            knowledge="",
            budget=budget,
            source_rel="doc/x.md",
        )
    finally:
        de.EXTRACT_CONCURRENCY = old
    assert complete is True
    merged = merge_rows(rows, columns=["id", "name"], primary_key=["id"])
    assert merged[0]["id"] == "1"
    assert merged[0]["name"] == "A"


if __name__ == "__main__":
    tests = [
        test_parse_plan_1based_segments,
        test_prune_chatter_and_superseded,
        test_stub_missing_records,
        test_merge_wide_table,
        test_normalize_subprocess_upper,
        test_normalize_timeout_keeps_original,
        test_parse_format_declarations,
        test_segment_worker_keeps_parseable,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            raise
    raise SystemExit(failed)
