"""Smoke test: knowledge snippet retrieve (no semantic layer)."""

from __future__ import annotations

from pathlib import Path

from data_agent_baseline.agents.prompt import build_task_prompt, retrieve_knowledge_snippets
from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord


def test_retrieve_snippets() -> None:
    knowledge = (
        "Constructors score points.\n\n"
        "## Qualifying\n\nq3 stores lap times in M:SS.mmm format for qualifying.\n\n"
        "Circuits have latitude."
    )
    snippets = retrieve_knowledge_snippets(
        knowledge, "Q3 qualifying time 0:01:54", max_chars=500
    )
    assert "q3" in snippets.casefold()
    assert snippets == knowledge.strip()


def test_long_knowledge_keeps_whole_section() -> None:
    filler = "Unrelated budget notes. " * 80
    knowledge = filler + "\n\n## Qualifying\n\nq3 stores lap times in M:SS.mmm format for qualifying.\n"
    snippets = retrieve_knowledge_snippets(knowledge, "Q3 qualifying time", max_chars=400)
    assert snippets.startswith("## Qualifying")
    assert "M:SS.mmm" in snippets
    assert "Unrelated budget" not in snippets


def test_task_prompt_has_question_and_knowledge_only() -> None:
    q = "Which event has the lowest cost?"
    task = PublicTask(
        record=TaskRecord(task_id="t", difficulty="easy", question=q),
        assets=TaskAssets(task_dir=Path("."), context_dir=Path(".")),
    )
    prompt = build_task_prompt(task, knowledge_text="cost uses SUM(spent)")
    assert f"Question: {q}" in prompt
    assert "cost uses SUM(spent)" in prompt
    assert "Semantic hypothesis" not in prompt
    assert "semantic" not in prompt.casefold()


if __name__ == "__main__":
    test_retrieve_snippets()
    test_task_prompt_has_question_and_knowledge_only()
    print("ALL_OK")
