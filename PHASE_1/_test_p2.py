"""Mechanism tests for P2 dual-track probes, doc search, and grain expansion."""

from __future__ import annotations

from pathlib import Path

import duckdb

from data_agent_baseline.agents.dual_probe import (
    format_dual_probe_block,
    submit_quantile_normal_rejection,
)
from data_agent_baseline.agents.question_route import classify_question
from data_agent_baseline.agents.submit_validation import submit_symmetric_relation_rejection
from data_agent_baseline.benchmark.schema import (
    PublicTask,
    TaskAssets,
    TaskRecord,
)
from data_agent_baseline.tools.doc_search import search_docs
from data_agent_baseline.tools.warehouse import RegisteredTable, WarehouseState


def test_classify_ratio_and_clinical() -> None:
    kinds = classify_question("How many times larger is A compared to B?")
    assert "ratio" in kinds
    kinds2 = classify_question("How many male patients have abnormal fibrinogen?")
    assert "clinical" in kinds2
    assert "agg" in kinds2


def test_dual_block_asks_formula_tracks() -> None:
    block = format_dual_probe_block(
        question="What is the average monthly consumption?",
        knowledge_text="Average Monthly = Total Annual / 12",
        steps=[],
    )
    assert "AVG" in block
    assert "SUM" in block


def test_quantile_normal_rejected_without_knowledge() -> None:
    rejected = submit_quantile_normal_rejection(
        "How many male patients have abnormal FG?",
        knowledge_text="Some lab notes.",
        sql="SELECT COUNT(*) FROM lab WHERE FG < percentile_cont(0.25) WITHIN GROUP (ORDER BY FG)",
    )
    assert rejected is not None
    assert (
        submit_quantile_normal_rejection(
            "How many male patients have abnormal FG?",
            knowledge_text="Normal range uses the 25th percentile.",
            sql="SELECT COUNT(*) FROM lab WHERE FG < percentile_cont(0.25) WITHIN GROUP (ORDER BY FG)",
        )
        is None
    )


def test_count_star_on_symmetric_rejected() -> None:
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE connected (bond_id INT, atom_id INT, atom_id2 INT)")
    conn.execute(
        "INSERT INTO connected VALUES (1,10,11),(1,11,10),(2,12,13),(2,13,12)"
    )
    state = WarehouseState(
        conn=conn,
        context_dir=Path("."),
        tables=[
            RegisteredTable(
                canonical="connected",
                aliases=[],
                source_type="csv",
                source_rel="connected.csv",
            )
        ],
    )
    rejected = submit_symmetric_relation_rejection(
        "SELECT COUNT(*) AS n FROM connected",
        state,
    )
    assert rejected is not None


def test_legal_or_join_distinct_allowed() -> None:
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE connected (bond_id INT, atom_id INT, atom_id2 INT)")
    conn.execute(
        "INSERT INTO connected VALUES (1,10,11),(1,11,10),(2,12,13),(2,13,12)"
    )
    conn.execute("CREATE TABLE atom (atom_id INT, element VARCHAR)")
    conn.execute("INSERT INTO atom VALUES (10,'i'),(11,'c')")
    state = WarehouseState(
        conn=conn,
        context_dir=Path("."),
        tables=[
            RegisteredTable("connected", [], "csv", "connected.csv"),
            RegisteredTable("atom", [], "csv", "atom.csv"),
        ],
    )
    sql = (
        "SELECT AVG(bond_count) AS avg_bonds FROM ("
        "SELECT a.atom_id, COUNT(DISTINCT c.bond_id) AS bond_count "
        "FROM atom a LEFT JOIN connected c "
        "ON a.atom_id = c.atom_id OR a.atom_id = c.atom_id2 "
        "WHERE a.element = 'i' GROUP BY a.atom_id)"
    )
    assert submit_symmetric_relation_rejection(sql, state) is None


def test_search_docs_skips_knowledge_and_hits_doc() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        doc = root / "doc"
        doc.mkdir()
        (root / "knowledge.md").write_text("ignore me height_cm", encoding="utf-8")
        (doc / "heroes.md").write_text(
            "Marvel Comics publishes many heroes.\n\n"
            "The height_cm of Storm is 180 and publisher is Marvel Comics.\n",
            encoding="utf-8",
        )
        task = PublicTask(
            record=TaskRecord(
                task_id="t",
                difficulty="easy",
                question="What percentage of Marvel Comics heroes are 150 to 180 cm?",
            ),
            assets=TaskAssets(task_dir=root, context_dir=root),
        )
        result = search_docs(task, "Marvel height_cm")
        assert result["hits"]
        assert all(hit["path"] != "knowledge.md" for hit in result["hits"])
        assert any("180" in hit["snippet"] for hit in result["hits"])


if __name__ == "__main__":
    test_classify_ratio_and_clinical()
    test_dual_block_asks_formula_tracks()
    test_quantile_normal_rejected_without_knowledge()
    test_count_star_on_symmetric_rejected()
    test_legal_or_join_distinct_allowed()
    test_search_docs_skips_knowledge_and_hits_doc()
    print("OK")
