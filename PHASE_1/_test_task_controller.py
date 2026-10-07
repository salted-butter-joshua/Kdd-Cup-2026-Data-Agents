"""Mechanism tests for TaskBudget, SQL fingerprint bans, and message windowing."""

from __future__ import annotations

import tempfile
from pathlib import Path

from data_agent_baseline.agents.episode import (
    EpisodeState,
    fingerprint_sql,
    illegal_sql_reason,
    pack_step_window,
)
from data_agent_baseline.agents.runtime import StepRecord
from data_agent_baseline.run.task_budget import (
    TaskBudget,
    classify_task_budget,
    process_kill_timeout,
)


def _step(index: int, sql: str = "SELECT 1") -> StepRecord:
    return StepRecord(
        step_index=index,
        thought="",
        action="run_sql",
        action_input={"sql": sql},
        raw_response="{}",
        observation={"ok": True},
        ok=True,
    )


def test_easy_budget_no_docs() -> None:
    with tempfile.TemporaryDirectory() as raw:
        context = Path(raw)
        (context / "csv").mkdir()
        budget = classify_task_budget(context)
        assert budget.tier == "easy"
        assert budget.extract_max == 0
        assert budget.task_timeout <= 180
        assert process_kill_timeout(budget) <= 180
        assert budget.max_steps <= 16


def test_hard_budget_two_docs() -> None:
    with tempfile.TemporaryDirectory() as raw:
        context = Path(raw)
        doc = context / "doc"
        doc.mkdir()
        (doc / "a.md").write_text("x" * 100, encoding="utf-8")
        (doc / "b.md").write_text("y" * 100, encoding="utf-8")
        budget = classify_task_budget(context)
        assert budget.tier == "hard"
        assert budget.extract_max >= 180
        assert budget.task_timeout >= 420
        assert process_kill_timeout(budget) >= 600
        assert budget.extract_deadline <= budget.task_deadline - budget.solve_reserve + 1e-6


def test_medium_upgrade_extends_extract() -> None:
    budget = TaskBudget(
        tier="medium",
        task_timeout=420,
        extract_max=180,
        solve_reserve=180,
        max_steps=24,
        n_doc_files=1,
        started_at=0.0,
    )
    assert budget.maybe_upgrade_to_hard()
    assert budget.tier == "hard"
    assert budget.task_timeout >= 420
    assert budget.max_steps >= 24


def test_illegal_and_repeat_fingerprint() -> None:
    assert illegal_sql_reason("ATTACH 'x.db'")
    assert illegal_sql_reason("SELECT * FROM glob('*.csv')")
    assert fingerprint_sql("SELECT  a  FROM t LIMIT 3") == fingerprint_sql("select a from t")
    ep = EpisodeState()
    assert ep.gate_sql("ATTACH 'x.db' AS ext")
    reason = ep.gate_sql("select a from t")
    assert reason is None
    ep.record_sql_result(sql="select a from t", ok=True, row_count=0, columns=["a"], error=None)
    banned = ep.gate_sql("SELECT a FROM t LIMIT 9")
    assert banned is not None


def test_pack_window_keeps_last_three() -> None:
    steps = [_step(i) for i in range(1, 10)]
    folded, recent = pack_step_window(steps, keep_full=3, skip_fold_below=6)
    assert len(recent) == 3
    assert len(folded) == 6
    short = [_step(i) for i in range(1, 5)]
    folded2, recent2 = pack_step_window(short, keep_full=3, skip_fold_below=6)
    assert folded2 == []
    assert len(recent2) == 4


def test_empty_warehouse_stops_after_two_catalog_looks() -> None:
    ep = EpisodeState()
    ep.observe_list_tables(0)
    assert not ep.should_stop_empty_warehouse()
    ep.observe_list_tables(0)
    assert ep.should_stop_empty_warehouse()


if __name__ == "__main__":
    test_easy_budget_no_docs()
    test_hard_budget_two_docs()
    test_medium_upgrade_extends_extract()
    test_illegal_and_repeat_fingerprint()
    test_pack_window_keeps_last_three()
    test_empty_warehouse_stops_after_two_catalog_looks()
    print("OK")
