"""Unit tests for schema linking, hypotheses, voting, and ambiguity gate."""

from __future__ import annotations

from data_agent_baseline.agents.gate_common import is_undecided
from data_agent_baseline.agents.hypothesis import (
    EvidenceState,
    build_hypotheses,
    evidence_guidance,
    format_hypothesis_plan,
    update_evidence,
)
from data_agent_baseline.agents.runtime import StepRecord
from data_agent_baseline.agents.schema_link import infer_column_role, link_schema
from data_agent_baseline.agents.submit_validation import submit_ambiguity_probe_rejection
from data_agent_baseline.agents.voting import vote_best_final


def _tables():
    return [
        {
            "name": "results",
            "n_rows": 100,
            "columns": [
                {"name": "raceId", "type": "INTEGER"},
                {"name": "position", "type": "INTEGER"},
                {"name": "positionOrder", "type": "INTEGER"},
                {"name": "round", "type": "INTEGER"},
                {"name": "milliseconds", "type": "INTEGER"},
            ],
        },
        {
            "name": "races",
            "n_rows": 20,
            "columns": [
                {"name": "raceId", "type": "INTEGER"},
                {"name": "name", "type": "VARCHAR"},
                {"name": "year", "type": "INTEGER"},
            ],
        },
    ]


def test_schema_link_finds_position_ambiguity():
    plan = link_schema(
        question="Which driver finished in position 2?",
        knowledge_text="position is finishing place; positionOrder is grid order.",
        tables=_tables(),
    )
    assert plan.candidate_tables
    assert any(c.column == "position" for c in plan.candidate_columns)
    assert plan.ambiguity_groups
    terms = {g.term for g in plan.ambiguity_groups}
    assert terms  # synonym bag hit on position/rank/order


def test_hypothesis_plan_for_avg_and_list():
    plan = link_schema(
        question="What is the average monthly consumption?",
        knowledge_text="Total annual consumption / 12.",
        tables=_tables(),
    )
    hyps = build_hypotheses(
        question="What is the average monthly consumption?",
        knowledge_text="Total annual consumption / 12.",
        link=plan,
    )
    ids = {h.hid for h in hyps}
    assert "avg_grain" in ids
    text = format_hypothesis_plan(hyps)
    assert "avg_grain" in text


def test_evidence_empty_probe_guidance():
    evidence = EvidenceState(consecutive_empty_probes=3)
    hyps = build_hypotheses(
        question="What happened at 1:54?",
        knowledge_text="",
        link=None,
    )
    notes = evidence_guidance(
        evidence=evidence,
        hypotheses=hyps,
        question="What happened at 1:54?",
        steps=[],
        remaining_steps=4,
    )
    assert "empty probes" in notes.lower() or "3+" in notes


def test_update_evidence_tracks_empty():
    evidence = EvidenceState()
    step = StepRecord(
        step_index=1,
        thought="",
        action="run_sql",
        action_input={"sql": "SELECT 1", "final": False},
        raw_response="",
        observation={"ok": True, "content": {"row_count": 0}},
        ok=True,
    )
    update_evidence(evidence, step)
    assert evidence.consecutive_empty_probes == 1


def test_vote_prefers_nonempty_list():
    steps = [
        StepRecord(
            step_index=1,
            thought="",
            action="run_sql",
            action_input={"sql": "SELECT name FROM t LIMIT 1", "final": True},
            raw_response="",
            observation={"ok": True, "content": {"row_count": 1, "rows": [["a"]]}},
            ok=True,
        ),
        StepRecord(
            step_index=2,
            thought="",
            action="run_sql",
            action_input={"sql": "SELECT name FROM t", "final": True},
            raw_response="",
            observation={
                "ok": True,
                "content": {"row_count": 3, "rows": [["a"], ["b"], ["c"]]},
            },
            ok=True,
        ),
    ]
    voted = vote_best_final(
        question="Which races are listed?",
        steps=steps,
    )
    assert voted is not None
    assert "LIMIT" not in voted.sql.upper()
    assert voted.score > 0


def test_ambiguity_gate_rejects_unprobed_peer():
    plan = link_schema(
        question="Who finished in position 2 on the track?",
        knowledge_text="",
        tables=_tables(),
    )
    assert plan.ambiguity_groups
    steps = [
        StepRecord(
            step_index=1,
            thought="",
            action="run_sql",
            action_input={
                "sql": 'SELECT COUNT(*) FROM results WHERE "position" = 2',
                "final": False,
            },
            raw_response="",
            observation={"ok": True, "content": {"row_count": 1}},
            ok=True,
        ),
    ]
    rejection = submit_ambiguity_probe_rejection(
        ambiguity_groups=plan.ambiguity_groups,
        steps=steps,
        sql='SELECT driver FROM results WHERE "position" = 2',
    )
    assert rejection is not None
    assert "ambiguity_check" in rejection


def test_ambiguity_gate_allows_when_peers_probed():
    plan = link_schema(
        question="Who finished in position 2 on the track?",
        knowledge_text="",
        tables=_tables(),
    )
    steps = [
        StepRecord(
            step_index=1,
            thought="",
            action="run_sql",
            action_input={
                "sql": 'SELECT COUNT(*) FROM results WHERE "position" = 2',
                "final": False,
            },
            raw_response="",
            observation={"ok": True, "content": {"row_count": 1}},
            ok=True,
        ),
        StepRecord(
            step_index=2,
            thought="",
            action="run_sql",
            action_input={
                "sql": 'SELECT COUNT(*) FROM results WHERE "positionOrder" = 2',
                "final": False,
            },
            raw_response="",
            observation={"ok": True, "content": {"row_count": 0}},
            ok=True,
        ),
    ]
    rejection = submit_ambiguity_probe_rejection(
        ambiguity_groups=plan.ambiguity_groups,
        steps=steps,
        sql='SELECT driver FROM results WHERE "position" = 2',
    )
    assert rejection is None


def _number_tables():
    return [
        {
            "name": "qualifying",
            "n_rows": 20,
            "columns": [
                {"name": "raceId", "type": "INTEGER"},
                {"name": "driverId", "type": "INTEGER"},
                {"name": "number", "type": "INTEGER"},
                {"name": "q3", "type": "VARCHAR"},
            ],
        },
        {
            "name": "drivers",
            "n_rows": 10,
            "columns": [
                {"name": "driverId", "type": "INTEGER"},
                {"name": "number", "type": "INTEGER"},
                {"name": "forename", "type": "VARCHAR"},
            ],
        },
    ]


def test_schema_link_homonym_number_group():
    plan = link_schema(
        question="What is his number of the driver who finished in Q3?",
        knowledge_text="",
        tables=_number_tables(),
    )
    homonyms = [
        g
        for g in plan.ambiguity_groups
        if {c.column.casefold() for c in g.columns} == {"number"}
        and {c.table.casefold() for c in g.columns} >= {"qualifying", "drivers"}
    ]
    assert homonyms, plan.ambiguity_groups
    hyps = build_hypotheses(
        question="What is his number of the driver who finished in Q3?",
        knowledge_text="",
        link=plan,
    )
    text = format_hypothesis_plan(hyps)
    assert "table.column" in text or "multiple tables" in text


def test_schema_link_no_homonym_when_name_absent():
    plan = link_schema(
        question="Which driver finished in position 2?",
        knowledge_text="",
        tables=_number_tables() + _tables(),
    )
    number_only = [
        g
        for g in plan.ambiguity_groups
        if {c.column.casefold() for c in g.columns} == {"number"}
        and len({c.table.casefold() for c in g.columns}) >= 2
    ]
    assert not number_only


def test_ambiguity_gate_rejects_unprobed_homonym_table():
    plan = link_schema(
        question="What is his number of the driver who finished in Q3?",
        knowledge_text="",
        tables=_number_tables(),
    )
    steps = [
        StepRecord(
            step_index=1,
            thought="",
            action="run_sql",
            action_input={
                "sql": "SELECT number FROM qualifying WHERE q3 LIKE '1:54%'",
                "final": False,
            },
            raw_response="",
            observation={"ok": True, "content": {"row_count": 2}},
            ok=True,
        ),
    ]
    rejection = submit_ambiguity_probe_rejection(
        ambiguity_groups=plan.ambiguity_groups,
        steps=steps,
        sql="SELECT number FROM qualifying WHERE q3 LIKE '1:54%'",
    )
    assert rejection is not None
    assert not is_undecided(rejection)
    assert rejection["ambiguity_check"]["chosen"] == "qualifying.number"
    assert "drivers.number" in rejection["ambiguity_check"]["peers"]


def test_ambiguity_gate_allows_when_homonym_peer_table_probed():
    plan = link_schema(
        question="What is his number of the driver who finished in Q3?",
        knowledge_text="",
        tables=_number_tables(),
    )
    steps = [
        StepRecord(
            step_index=1,
            thought="",
            action="run_sql",
            action_input={
                "sql": "SELECT number FROM qualifying WHERE q3 LIKE '1:54%'",
                "final": False,
            },
            raw_response="",
            observation={"ok": True, "content": {"row_count": 2}},
            ok=True,
        ),
        StepRecord(
            step_index=2,
            thought="",
            action="run_sql",
            action_input={
                "sql": "SELECT number FROM drivers WHERE driverId IN (20, 817)",
                "final": False,
            },
            raw_response="",
            observation={"ok": True, "content": {"row_count": 2}},
            ok=True,
        ),
    ]
    rejection = submit_ambiguity_probe_rejection(
        ambiguity_groups=plan.ambiguity_groups,
        steps=steps,
        sql="SELECT number FROM qualifying WHERE q3 LIKE '1:54%'",
    )
    assert rejection is None or is_undecided(rejection)
    assert rejection is None


def test_ambiguity_gate_undecided_unqualified_homonym_join():
    plan = link_schema(
        question="What is his number of the driver who finished in Q3?",
        knowledge_text="",
        tables=_number_tables(),
    )
    payload = submit_ambiguity_probe_rejection(
        ambiguity_groups=plan.ambiguity_groups,
        steps=[],
        sql=(
            "SELECT number FROM qualifying q "
            "JOIN drivers d ON q.driverId = d.driverId"
        ),
    )
    assert is_undecided(payload)


# --- §13.3: column roles --------------------------------------------------------


def test_infer_column_role():
    assert infer_column_role("CustomerID", "BIGINT") == "identifier"
    assert infer_column_role("bond_id", "INTEGER") == "identifier"
    assert infer_column_role("id", "INTEGER") == "identifier"
    assert infer_column_role("cost", "DOUBLE") == "measure"
    assert infer_column_role("AvgScrMath", "INTEGER") == "measure"
    assert infer_column_role("name", "VARCHAR") == "dimension"
    # 'paid' ends with 'id' but is not an identifier.
    assert infer_column_role("paid", "BOOLEAN") == "dimension"


def test_schema_link_populates_roles():
    plan = link_schema(
        question="Which driver finished in position 2?",
        knowledge_text="raceId links results to races; position is the finish place.",
        tables=_tables(),
    )
    roles = {c.column: c.role for c in plan.candidate_columns}
    assert roles.get("raceId") == "identifier"
    assert roles.get("position") == "measure"


# --- §13.4: ratio-direction interpretation --------------------------------------


def test_ratio_direction_hypothesis():
    hyps = build_hypotheses(
        question="How many times more posts than votes does user 24 have?",
        knowledge_text="Ratio: DIVIDE(Count(post.Id), Count(votes.Id)).",
        link=None,
    )
    ids = {h.hid for h in hyps}
    assert "ratio_direction" in ids
    text = format_hypothesis_plan(hyps)
    assert "arbitrate" in text


def test_ratio_hypothesis_needs_divide_knowledge():
    hyps = build_hypotheses(
        question="How many times did the user post?",
        knowledge_text="",
        link=None,
    )
    ids = {h.hid for h in hyps}
    assert "ratio_direction" not in ids


def _code_year_tables():
    return [
        {
            "name": "orders",
            "n_rows": 8,
            "columns": [
                {"name": "customer_id", "type": "INTEGER"},
                {"name": "code", "type": "INTEGER"},
                {"name": "year", "type": "INTEGER"},
            ],
        },
        {
            "name": "customers",
            "n_rows": 8,
            "columns": [
                {"name": "customer_id", "type": "INTEGER"},
                {"name": "code", "type": "INTEGER"},
                {"name": "name", "type": "VARCHAR"},
            ],
        },
    ]


def test_homonym_ranks_entity_column_above_filter_table():
    plan = link_schema(
        question="What is the code of the customer who ordered in year 2012?",
        knowledge_text="",
        tables=_code_year_tables(),
    )
    by_key = {f"{c.table}.{c.column}".casefold(): c.score for c in plan.candidate_columns}
    assert by_key["customers.code"] > by_key["orders.code"]
    assert plan.homonym_projections
    assert "project customers.code" in plan.homonym_projections[0]
    assert "filter on orders" in plan.homonym_projections[0]
    text = plan.format_for_prompt()
    assert "project customers.code" in text
    hyps = build_hypotheses(
        question="What is the code of the customer who ordered in year 2012?",
        knowledge_text="",
        link=plan,
    )
    hyp_text = format_hypothesis_plan(hyps)
    assert "project customers.code" in hyp_text


def test_homonym_no_rerank_without_event_column():
    tables = [
        {
            "name": "lefts",
            "n_rows": 2,
            "columns": [{"name": "code", "type": "INTEGER"}],
        },
        {
            "name": "rights",
            "n_rows": 2,
            "columns": [{"name": "code", "type": "INTEGER"}],
        },
    ]
    plan = link_schema(
        question="What is the code?",
        knowledge_text="",
        tables=tables,
    )
    assert plan.homonym_projections == []
    scores = {
        f"{c.table}.{c.column}".casefold(): c.score
        for c in plan.candidate_columns
        if c.column.casefold() == "code"
    }
    assert scores["lefts.code"] == scores["rights.code"]


def test_homonym_skips_rerank_when_leaf_is_event_column():
    tables = [
        {
            "name": "qualifying",
            "n_rows": 4,
            "columns": [
                {"name": "q3", "type": "VARCHAR"},
                {"name": "raceId", "type": "INTEGER"},
            ],
        },
        {
            "name": "results",
            "n_rows": 4,
            "columns": [
                {"name": "q3", "type": "VARCHAR"},
                {"name": "raceId", "type": "INTEGER"},
            ],
        },
    ]
    plan = link_schema(
        question="What is the q3 in race 903?",
        knowledge_text="",
        tables=tables,
    )
    assert plan.homonym_projections == []


def test_driver_number_prompt_prefers_entity_table():
    plan = link_schema(
        question="What is his number of the driver who finished in Q3?",
        knowledge_text="",
        tables=_number_tables(),
    )
    by_key = {f"{c.table}.{c.column}".casefold(): c.score for c in plan.candidate_columns}
    assert by_key["drivers.number"] > by_key["qualifying.number"]
    assert any("project drivers.number" in h for h in plan.homonym_projections)


def _place_tables():
    return [
        {
            "name": "results",
            "n_rows": 100,
            "columns": [
                {"name": "raceId", "type": "INTEGER"},
                {"name": "rank", "type": "INTEGER"},
                {"name": "positionOrder", "type": "INTEGER"},
                {"name": "time", "type": "VARCHAR"},
            ],
        },
        {
            "name": "races",
            "n_rows": 20,
            "columns": [
                {"name": "raceId", "type": "INTEGER"},
                {"name": "name", "type": "VARCHAR"},
                {"name": "year", "type": "INTEGER"},
                {"name": "time", "type": "VARCHAR"},
            ],
        },
    ]


_RANK_VS_ORDER_KNOWLEDGE = (
    "rank vs positionOrder: Use positionOrder for final race finishing order; "
    "rank represents the ranking based on fastestLapTime, not overall finishing position."
)


def test_generic_id_is_not_a_homonym_group():
    tables = [
        {
            "name": "posts",
            "n_rows": 3,
            "columns": [
                {"name": "Id", "type": "INTEGER"},
                {"name": "OwnerUserId", "type": "INTEGER"},
            ],
        },
        {
            "name": "posthistory",
            "n_rows": 3,
            "columns": [
                {"name": "Id", "type": "INTEGER"},
                {"name": "PostId", "type": "INTEGER"},
            ],
        },
    ]
    plan = link_schema(
        question="Which post by slashnick has the most answers? State the post ID.",
        knowledge_text="",
        tables=tables,
    )
    id_groups = [
        g
        for g in plan.ambiguity_groups
        if {c.column.casefold() for c in g.columns} == {"id"}
    ]
    assert not id_groups
    number_plan = link_schema(
        question="What is his number of the driver who finished in Q3?",
        knowledge_text="",
        tables=_number_tables(),
    )
    number_groups = [
        g
        for g in number_plan.ambiguity_groups
        if {c.column.casefold() for c in g.columns} == {"number"}
    ]
    assert number_groups


def test_knowledge_boosts_rank_when_question_says_ranked():
    plan = link_schema(
        question=(
            "What's the finish time for the driver who ranked second "
            "in 2008's Chinese Grand Prix?"
        ),
        knowledge_text=_RANK_VS_ORDER_KNOWLEDGE,
        tables=_place_tables(),
    )
    by_key = {f"{c.table}.{c.column}".casefold(): c for c in plan.candidate_columns}
    assert by_key["results.rank"].score > by_key["results.positionorder"].score
    assert "knowledge-rank" in by_key["results.rank"].reasons


def test_knowledge_boosts_position_order_for_finishing_position():
    plan = link_schema(
        question="Who has the finishing position 2?",
        knowledge_text=_RANK_VS_ORDER_KNOWLEDGE,
        tables=_place_tables(),
    )
    by_key = {f"{c.table}.{c.column}".casefold(): c for c in plan.candidate_columns}
    assert by_key["results.positionorder"].score > by_key["results.rank"].score
    assert "knowledge-positionOrder" in by_key["results.positionorder"].reasons


def test_knowledge_place_fail_open_when_both_sides():
    plan = link_schema(
        question="Who ranked second in finishing position?",
        knowledge_text=_RANK_VS_ORDER_KNOWLEDGE,
        tables=_place_tables(),
    )
    by_key = {f"{c.table}.{c.column}".casefold(): c for c in plan.candidate_columns}
    assert "knowledge-rank" not in by_key["results.rank"].reasons
    assert "knowledge-positionOrder" not in by_key["results.positionorder"].reasons


def test_knowledge_place_fail_open_without_both_columns_in_knowledge():
    plan = link_schema(
        question="Who ranked second?",
        knowledge_text="rank is fastest-lap ranking.",
        tables=_place_tables(),
    )
    by_key = {f"{c.table}.{c.column}".casefold(): c for c in plan.candidate_columns}
    assert "knowledge-rank" not in by_key["results.rank"].reasons


def test_homonym_reranks_time_onto_entity_table():
    plan = link_schema(
        question="What's the finish time in year 2008?",
        knowledge_text="",
        tables=_place_tables(),
    )
    by_key = {f"{c.table}.{c.column}".casefold(): c.score for c in plan.candidate_columns}
    assert by_key["results.time"] > by_key["races.time"]
    assert any("project results.time" in h for h in plan.homonym_projections)
