"""Unit tests for L1.0 value normalization and evidence convergence."""

from __future__ import annotations

from data_agent_baseline.agents.hypothesis import (
    EvidenceState,
    evidence_guidance,
    update_evidence,
)
from data_agent_baseline.agents.runtime import StepRecord
from data_agent_baseline.agents.value_normalize import (
    build_normalize_plan,
    find_promotable_probe,
    format_normalize_plan,
    parse_time_token,
    sql_uses_coarse_time_match,
    sql_uses_exact_question_literal,
)


def _step(sql: str, *, rows: int, final: bool = False) -> StepRecord:
    return StepRecord(
        step_index=1,
        thought="",
        action="run_sql",
        action_input={"sql": sql, "final": final},
        raw_response="",
        observation={"ok": True, "content": {"row_count": rows, "rows": []}},
        ok=True,
    )


def test_parse_hms_to_canonical_mmss():
    lit = parse_time_token("0:01:54")
    assert lit is not None
    assert lit.total_seconds == 114
    assert lit.grain == "second"
    assert lit.canonical_mmss == "1:54"


def test_parse_mss_with_millis():
    lit = parse_time_token("1:54.455")
    assert lit is not None
    assert abs(lit.total_seconds - 114.455) < 1e-9
    assert lit.canonical_mmss == "1:54"


def test_normalize_plan_for_task80_style_question():
    plan = build_normalize_plan(
        question="What is his number of the driver who finished 0:01:54 in the Q3?",
        knowledge_text="Time metrics are standardized to 'MM:SS.mmm' format.",
    )
    assert plan.has_work
    assert plan.time_literals[0].canonical_mmss == "1:54"
    text = format_normalize_plan(plan)
    assert "1:54" in text
    assert "raw strings" in text.lower() or "semantic" in text.lower()


def test_exact_vs_coarse_sql_detection():
    plan = build_normalize_plan(question="finished 0:01:54 in Q3", knowledge_text="")
    exact = 'SELECT * FROM qualifying WHERE q3 = \'0:01:54\''
    coarse = "SELECT * FROM qualifying WHERE q3 LIKE '1:54%' AND raceId = 903"
    assert sql_uses_exact_question_literal(exact, plan)
    assert not sql_uses_coarse_time_match(exact, plan)
    assert sql_uses_coarse_time_match(coarse, plan)
    assert not sql_uses_exact_question_literal(coarse, plan)


def test_exact_empty_not_cleared_by_distinct():
    plan = build_normalize_plan(question="time 0:01:54", knowledge_text="")
    evidence = EvidenceState()
    update_evidence(
        evidence,
        _step("SELECT * FROM t WHERE q3 = '0:01:54'", rows=0),
        normalize_plan=plan,
    )
    assert evidence.consecutive_exact_empty == 1
    update_evidence(
        evidence,
        _step("SELECT DISTINCT q3 FROM qualifying", rows=11),
        normalize_plan=plan,
    )
    # unrelated DISTINCT must not wipe exact-empty streak
    assert evidence.consecutive_exact_empty == 1


def test_coarse_hit_clears_exact_empty_and_guides():
    plan = build_normalize_plan(question="finished 0:01:54", knowledge_text="")
    evidence = EvidenceState()
    update_evidence(
        evidence,
        _step("SELECT * FROM t WHERE q3 = '0:01:54'", rows=0),
        normalize_plan=plan,
    )
    update_evidence(
        evidence,
        _step("SELECT * FROM t WHERE q3 LIKE '1:54%'", rows=2),
        normalize_plan=plan,
    )
    assert evidence.coarse_nonempty_hits == 1
    assert evidence.consecutive_exact_empty == 0
    update_evidence(
        evidence,
        _step("SELECT * FROM t WHERE q3 = '0:01:54'", rows=0),
        normalize_plan=plan,
    )
    notes = evidence_guidance(
        evidence=evidence,
        hypotheses=[],
        question="finished 0:01:54",
        steps=[],
        remaining_steps=5,
        normalize_plan=plan,
    )
    assert "Value alignment" in notes
    assert "final=true" in notes


def test_promotable_probe_prefers_coarse_nonempty():
    plan = build_normalize_plan(question="0:01:54 in race", knowledge_text="")
    steps = [
        _step("SELECT * FROM qualifying WHERE q3 = '0:01:54'", rows=0),
        _step(
            "SELECT * FROM qualifying WHERE raceId = 903 AND q3 LIKE '1:54%' LIMIT 20",
            rows=2,
        ),
    ]
    promoted = find_promotable_probe(question="0:01:54", steps=steps, plan=plan)
    assert promoted is not None
    sql, reason = promoted
    assert "LIKE" in sql.upper()
    assert "LIMIT" not in sql.upper()
    assert "coarse_probe" in reason
