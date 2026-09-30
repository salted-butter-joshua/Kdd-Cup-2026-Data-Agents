"""Unit tests for submit-time validation gates."""

from __future__ import annotations

import tempfile
from pathlib import Path

from data_agent_baseline.agents.runtime import StepRecord
from data_agent_baseline.agents.submit_validation import (
    submit_member_expansion_rejection,
    submit_membership_rejection,
    submit_sanity_rejection,
    submit_symmetric_relation_rejection,
)
from data_agent_baseline.agents.grain_contract import submit_grain_contract_post
from data_agent_baseline.benchmark.schema import AnswerTable
from data_agent_baseline.tools.warehouse import build_warehouse


def _make_step(*, sql: str, row_count: int | None = None, ok: bool = True) -> StepRecord:
    content: dict[str, object] = {
        "sql": sql,
        "row_count": row_count,
        "columns": ["c"],
        "rows": [],
    }
    return StepRecord(
        step_index=1,
        thought="probe",
        action="run_sql",
        action_input={"sql": sql, "final": False},
        raw_response="",
        observation={"ok": ok, "content": content},
        ok=ok,
    )


def test_membership_rejection_when_expanding():
    probe = _make_step(
        sql="SELECT district, AVG(sat_math) FROM satscores s JOIN frpm f ON s.district = f.district GROUP BY district HAVING AVG(sat_math) > 450",
        row_count=2,
    )
    # Final drops BOTH probe tables (satscores and frpm) and re-expands from a
    # wider table — the documented trigger for this gate.
    final_sql = "SELECT school FROM schools WHERE district IN ('Riverside Unified', 'Desert Sands Unified')"
    answer = AnswerTable(columns=["school"], rows=[[f"school_{i}"] for i in range(57)])
    result = submit_membership_rejection([probe], final_sql, answer)
    assert result is not None
    assert "dropped tables" in result["error"].lower()
    assert result["membership_check"]["probe_row_count"] == 2
    assert result["membership_check"]["final_row_count"] == 57


def test_membership_no_rejection_when_small_result():
    probe = _make_step(
        sql="SELECT district, AVG(sat_math) FROM satscores GROUP BY district HAVING AVG(sat_math) > 450",
        row_count=2,
    )
    final_sql = "SELECT district, avg_math FROM satscores"
    answer = AnswerTable(columns=["district"], rows=[["d1"], ["d2"]])
    result = submit_membership_rejection([probe], final_sql, answer)
    assert result is None


def test_symmetric_relation_rejection():
    with tempfile.TemporaryDirectory() as tmpdir:
        context_dir = Path(tmpdir)
        (context_dir / "knowledge.md").write_text("")
        db_path = context_dir / "db"
        db_path.mkdir(parents=True)
        import sqlite3

        conn = sqlite3.connect(db_path / "atoms.sqlite")
        conn.execute("CREATE TABLE connected (bond_id INTEGER, atom_id INTEGER, atom_id2 INTEGER)")
        conn.executemany(
            "INSERT INTO connected VALUES (?, ?, ?)",
            [(1, 10, 20), (1, 20, 10), (2, 10, 30), (2, 30, 10)],
        )
        conn.commit()
        conn.close()

        state = build_warehouse(context_dir, model=None)
        result = submit_symmetric_relation_rejection(
            "SELECT COUNT(*) FROM connected WHERE atom_id = 10",
            state,
        )
        assert result is not None
        assert "COUNT(DISTINCT" in result["hint"]


def test_symmetric_relation_allows_count_distinct_rel_id():
    """P0: COUNT(DISTINCT bond_id) is the recommended fix and must not be rejected."""
    with tempfile.TemporaryDirectory() as tmpdir:
        context_dir = Path(tmpdir)
        (context_dir / "knowledge.md").write_text("")
        db_path = context_dir / "db"
        db_path.mkdir(parents=True)
        import sqlite3

        conn = sqlite3.connect(db_path / "atoms.sqlite")
        conn.execute("CREATE TABLE connected (bond_id INTEGER, atom_id INTEGER, atom_id2 INTEGER)")
        conn.executemany(
            "INSERT INTO connected VALUES (?, ?, ?)",
            [(1, 10, 20), (1, 20, 10), (2, 10, 30), (2, 30, 10)],
        )
        conn.commit()
        conn.close()

        state = build_warehouse(context_dir, model=None)
        for sql in (
            "SELECT COUNT(DISTINCT bond_id) FROM connected WHERE atom_id = 10",
            "SELECT AVG(n) FROM (SELECT a.atom_id, COUNT(DISTINCT c.bond_id) AS n "
            "FROM atom a LEFT JOIN connected c ON a.atom_id = c.atom_id "
            "OR a.atom_id = c.atom_id2 GROUP BY a.atom_id)",
            "SELECT AVG(unique_bonds) FROM ("
            "SELECT a.atom_id, COUNT(DISTINCT CASE WHEN c.atom_id = a.atom_id "
            "THEN c.bond_id END) AS unique_bonds FROM atom a "
            "LEFT JOIN connected c ON a.atom_id = c.atom_id GROUP BY a.atom_id)",
        ):
            result = submit_symmetric_relation_rejection(sql, state)
            assert result is None, f"unexpected rejection for: {sql}"


def test_sanity_rejection_for_percentage():
    answer = AnswerTable(columns=["pct"], rows=[[150]])
    result = submit_sanity_rejection("SELECT COUNT(*)*100.0/total AS percentage FROM t", answer)
    assert result is not None
    assert result["sanity_check"]["value"] == 150


def test_sanity_no_rejection_for_in_range_percentage():
    answer = AnswerTable(columns=["pct"], rows=[[45.5]])
    result = submit_sanity_rejection("SELECT COUNT(*)*100.0/total FROM t", answer)
    assert result is None


# --- §13.2: member expansion (task_199 pattern) ---------------------------------

_Q199 = (
    "List the school name and funding type of schools in Riverside-related "
    "districts whose average SAT math score exceeds 400"
)
_PROBE_199_SQL = (
    'SELECT f."District Name", AVG(s.AvgScrMath) AS avg_math '
    'FROM frpm f JOIN satscores s ON f."School Name" = s.sname '
    "WHERE s.rtype = 'S' GROUP BY f.\"District Name\""
)
_FINAL_199_SQL = (
    'SELECT DISTINCT f."School Name", f."Charter Funding Type" FROM frpm f '
    'WHERE f."District Name" IN ('
    'SELECT f2."District Name" FROM frpm f2 JOIN satscores s '
    'ON f2."School Name" = s.sname WHERE s.rtype = \'S\' '
    'GROUP BY f2."District Name" HAVING AVG(s.AvgScrMath) > 400) '
    'ORDER BY f."School Name"'
)


def _big_answer(n: int) -> AnswerTable:
    return AnswerTable(columns=["school"], rows=[[f"school_{i}"] for i in range(n)])


def test_member_expansion_fires_on_199_pattern():
    probe = _make_step(sql=_PROBE_199_SQL, row_count=2)
    result = submit_member_expansion_rejection(
        _Q199, [probe], _FINAL_199_SQL, _big_answer(57)
    )
    assert result is not None
    check = result["member_expansion_check"]
    assert check["probe_row_count"] == 2
    assert check["final_row_count"] == 57


def test_member_expansion_passes_output_grain_having():
    """Correct school-level HAVING final (103145Z path) must not fire."""
    probe = _make_step(sql=_PROBE_199_SQL, row_count=2)
    final_sql = (
        "SELECT sname FROM satscores WHERE rtype = 'S' "
        "GROUP BY sname HAVING AVG(AvgScrMath) > 400"
    )
    result = submit_member_expansion_rejection(
        _Q199, [probe], final_sql, _big_answer(57)
    )
    assert result is None


def test_member_expansion_passes_small_answer():
    probe = _make_step(sql=_PROBE_199_SQL, row_count=2)
    result = submit_member_expansion_rejection(
        _Q199, [probe], _FINAL_199_SQL, _big_answer(6)
    )
    assert result is None


def test_member_expansion_passes_non_dual_question():
    probe = _make_step(sql=_PROBE_199_SQL, row_count=2)
    result = submit_member_expansion_rejection(
        "List all schools in Riverside districts", [probe], _FINAL_199_SQL,
        _big_answer(57),
    )
    assert result is None


def test_member_expansion_passes_unrelated_probe():
    probe = _make_step(
        sql="SELECT kind, COUNT(*) FROM other_table GROUP BY kind", row_count=3
    )
    result = submit_member_expansion_rejection(
        _Q199, [probe], _FINAL_199_SQL, _big_answer(57)
    )
    assert result is None


def test_grain_contract_post_orders_membership_then_expansion():
    # Old membership rule (dropped tables) wins first when both could fire:
    # final shares NO table with the probe → membership_check.
    probe = _make_step(
        sql="SELECT district, AVG(sat_math) FROM satscores s JOIN frpm f "
        "ON s.district = f.district GROUP BY district HAVING AVG(sat_math) > 450",
        row_count=2,
    )
    final_sql = "SELECT school FROM schools WHERE district IN ('A', 'B')"
    payload = submit_grain_contract_post(_Q199, [probe], final_sql, _big_answer(57))
    assert payload is not None
    assert "membership_check" in payload
    assert payload["grain_contract"] is True
    # Same-table 199 pattern (final still references probe tables) falls
    # through to the expansion rule.
    probe2 = _make_step(sql=_PROBE_199_SQL, row_count=2)
    payload2 = submit_grain_contract_post(
        _Q199, [probe2], _FINAL_199_SQL, _big_answer(57)
    )
    assert payload2 is not None
    assert "member_expansion_check" in payload2


if __name__ == "__main__":
    test_membership_rejection_when_expanding()
    test_membership_no_rejection_when_small_result()
    test_symmetric_relation_rejection()
    test_symmetric_relation_allows_count_distinct_rel_id()
    test_sanity_rejection_for_percentage()
    test_sanity_no_rejection_for_in_range_percentage()
    test_member_expansion_fires_on_199_pattern()
    test_member_expansion_passes_output_grain_having()
    test_member_expansion_passes_small_answer()
    test_member_expansion_passes_non_dual_question()
    test_member_expansion_passes_unrelated_probe()
    test_grain_contract_post_orders_membership_then_expansion()
    print("submit_validation tests passed")
