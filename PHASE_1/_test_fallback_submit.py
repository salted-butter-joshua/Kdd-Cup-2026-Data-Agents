"""Unit tests for max_steps fallback submit."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from data_agent_baseline.agents.model import ScriptedModelAdapter
from data_agent_baseline.agents.react import ReActAgent, ReActAgentConfig
from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from data_agent_baseline.tools.registry import create_default_tool_registry


def _make_task(tmpdir: str) -> PublicTask:
    root = Path(tmpdir)
    task_dir = root / "task_fallback"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    (context_dir / "knowledge.md").write_text("Names of items.\n", encoding="utf-8")
    (context_dir / "data.csv").write_text("name\nalice\nbob\n", encoding="utf-8")
    return PublicTask(
        record=TaskRecord(
            task_id="task_fallback",
            difficulty="easy",
            question="What are the names?",
        ),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def test_fallback_submit_after_final_without_answer():
    """max_steps exhausted after final=true → auto-submit through L3 gates."""
    with tempfile.TemporaryDirectory() as tmpdir:
        task = _make_task(tmpdir)
        response = json.dumps(
            {
                "thought": "This SELECT is the answer table.",
                "action": "run_sql",
                "action_input": {
                    "sql": 'SELECT name FROM "data"',
                    "final": True,
                },
            }
        )
        model = ScriptedModelAdapter([response])
        tools = create_default_tool_registry(model=model)
        agent = ReActAgent(
            model=model,
            tools=tools,
            config=ReActAgentConfig(max_steps=1),
        )
        result = agent.run(task)
        assert result.answer is not None
        assert result.failure_reason is None
        assert result.answer.columns == ["name"]
        assert [row[0] for row in result.answer.rows] == ["alice", "bob"]
        assert any(
            step.raw_response == "__fallback_submit__" for step in result.steps
        )


def test_no_fallback_without_final_result():
    """Probe-only traces must not invent an answer at max_steps."""
    with tempfile.TemporaryDirectory() as tmpdir:
        task = _make_task(tmpdir)
        response = json.dumps(
            {
                "thought": "Just listing tables.",
                "action": "list_tables",
                "action_input": {},
            }
        )
        model = ScriptedModelAdapter([response])
        tools = create_default_tool_registry(model=model)
        agent = ReActAgent(
            model=model,
            tools=tools,
            config=ReActAgentConfig(max_steps=1),
        )
        result = agent.run(task)
        assert result.answer is None
        assert result.failure_reason is not None
        assert "max_steps" in result.failure_reason


if __name__ == "__main__":
    test_fallback_submit_after_final_without_answer()
    test_no_fallback_without_final_result()
    print("fallback submit tests passed")
