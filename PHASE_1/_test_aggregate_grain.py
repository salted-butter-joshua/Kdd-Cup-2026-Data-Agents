from data_agent_baseline.agents.aggregate_grain import (
    collect_aggregate_grains,
    dual_grain_satisfied,
    grains_in_sql,
    knowledge_defines_coarse,
    pre_submit_grain_notes,
    submit_grain_rejection,
    wants_dual_grain_check,
    wants_member_expansion_check,
)
from data_agent_baseline.agents.gate_common import is_undecided
from data_agent_baseline.agents.runtime import StepRecord


def _sql_step(sql: str, *, ok: bool = True, grain: str | None = None) -> StepRecord:
    action_input: dict = {"sql": sql, "final": False}
    if grain is not None:
        action_input["grain"] = grain
    content: dict = {"sql": sql, "full_scan": False}
    if grain is not None:
        content["grain"] = grain
    return StepRecord(
        step_index=1,
        thought="",
        action="run_sql",
        action_input=action_input,
        raw_response="",
        observation={"ok": ok, "content": content},
        ok=ok,
    )


def test_wants_dual_grain() -> None:
    assert wants_dual_grain_check("Which event has the lowest cost?")
    assert not wants_dual_grain_check("What was the average monthly consumption?")
    assert wants_dual_grain_check(
        "What was the average monthly consumption?",
        "Average monthly consumption = SUM(Consumption) / 12.",
    )
    assert not wants_dual_grain_check("What is the average number of bonds?")
    assert not wants_dual_grain_check("List all the withdrawals in cash transactions.")


def test_wants_member_expansion_keeps_average() -> None:
    assert wants_member_expansion_check(
        "List schools whose average SAT math score exceeds 400"
    )
    assert wants_member_expansion_check("Which event has the lowest cost?")
    assert not wants_member_expansion_check("List all schools in Riverside")


def test_grains_in_sql() -> None:
    assert grains_in_sql("SELECT MIN(cost) FROM expense") == {"fine"}
    assert grains_in_sql(
        "SELECT event_name, SUM(spent) AS t FROM budget GROUP BY event_name"
    ) == {"coarse"}
    # SUM + MIN of totals → coarse only (not row-level fine)
    assert grains_in_sql(
        "WITH c AS (SELECT SUM(spent) AS t FROM budget GROUP BY link_to_event) "
        "SELECT MIN(t) FROM c"
    ) == {"coarse"}
    assert grains_in_sql("SELECT AVG(Consumption) / 12 FROM yearmonth") == {"fine"}
    assert grains_in_sql("SELECT SUM(Consumption) / 12 FROM yearmonth") == {"coarse"}


def test_dual_grain_gate_task25_style() -> None:
    q = "Which event has the lowest cost?"
    coarse_only = [
        _sql_step(
            "SELECT event_name FROM budget GROUP BY event_name "
            "HAVING SUM(spent) = (SELECT MIN(s) FROM ("
            "SELECT SUM(spent) AS s FROM budget GROUP BY link_to_event))"
        )
    ]
    assert collect_aggregate_grains(coarse_only) == {"coarse"}
    assert not dual_grain_satisfied(q, coarse_only)

    both = coarse_only + [
        _sql_step(
            "SELECT e.event_name FROM expense x "
            "JOIN budget b ON x.link_to_budget = b.budget_id "
            "JOIN event e ON b.link_to_event = e.event_id "
            "WHERE x.cost = (SELECT MIN(cost) FROM expense)",
            grain="fine",
        )
    ]
    assert dual_grain_satisfied(q, both)


def test_dual_grain_tag_alone_counts() -> None:
    q = "Which event has the lowest cost?"
    steps = [
        _sql_step("SELECT 1", grain="fine"),
        _sql_step("SELECT 2", grain="coarse"),
    ]
    assert dual_grain_satisfied(q, steps)


def test_non_dual_question_always_ok() -> None:
    q = "List Connor Hilton dues dates."
    assert dual_grain_satisfied(q, [])


_TASK25_KNOWLEDGE = """
Average Cost: AVG(cost). Total Expenditure: SUM(spent).
"""

_TASK169_KNOWLEDGE = """
Average Monthly Consumption = SUM(Consumption) / 12.
Total Annual Consumption / 12.
"""


def test_knowledge_defines_coarse() -> None:
    # 25: knowledge does not define a coarse formula for the measure "cost".
    assert not knowledge_defines_coarse(_TASK25_KNOWLEDGE, "cost")
    # 169: knowledge defines coarse (Total) for the measure "Consumption".
    assert knowledge_defines_coarse(_TASK169_KNOWLEDGE, "Consumption")


def test_submit_grain_rejection_rejects_coarse_when_not_defined() -> None:
    q = "Which event has the lowest cost?"
    fine_sql = "SELECT e.event_name FROM expense e JOIN budget b ON e.link_to_budget = b.budget_id JOIN event e2 ON b.link_to_event = e2.event_id WHERE e.cost = (SELECT MIN(cost) FROM expense)"
    coarse_sql = "SELECT ev.event_name FROM (SELECT b.link_to_event, SUM(e.cost) AS total_spent FROM expense e JOIN budget b ON e.link_to_budget = b.budget_id GROUP BY b.link_to_event HAVING SUM(e.cost) = (SELECT MIN(total_spent) FROM (SELECT SUM(e.cost) AS total_spent FROM expense e JOIN budget b ON e.link_to_budget = b.budget_id GROUP BY b.link_to_event))) t JOIN event ev ON t.link_to_event = ev.event_id"
    steps = [
        _sql_step(fine_sql, grain="fine"),
        _sql_step(coarse_sql, grain="coarse"),
    ]
    rejection = submit_grain_rejection(q, steps, coarse_sql, _TASK25_KNOWLEDGE)
    assert rejection is not None
    assert rejection["grain_check"]["measure_col"] == "cost"


def test_submit_grain_rejection_allows_coarse_when_defined() -> None:
    q = "What was the average monthly consumption of customers in SME for the year 2013?"
    fine_sql = "SELECT AVG(customer_total / 12) FROM (SELECT CustomerID, SUM(Consumption) AS customer_total FROM yearmonth GROUP BY CustomerID)"
    coarse_sql = "SELECT SUM(Consumption) / 12 FROM yearmonth"
    steps = [
        _sql_step(fine_sql, grain="fine"),
        _sql_step(coarse_sql, grain="coarse"),
    ]
    rejection = submit_grain_rejection(q, steps, coarse_sql, _TASK169_KNOWLEDGE)
    assert rejection is None


def test_submit_grain_rejection_ignores_fine_final() -> None:
    q = "Which event has the lowest cost?"
    fine_sql = "SELECT e.event_name FROM expense e WHERE e.cost = (SELECT MIN(cost) FROM expense)"
    coarse_sql = "SELECT event_name FROM budget GROUP BY event_name"
    steps = [
        _sql_step(fine_sql, grain="fine"),
        _sql_step(coarse_sql, grain="coarse"),
    ]
    rejection = submit_grain_rejection(q, steps, fine_sql, _TASK25_KNOWLEDGE)
    assert rejection is None


# --- §13.1 IR: quoted identifiers cannot bypass the grain gate -----------------


def test_grains_in_sql_quoted_identifiers() -> None:
    assert grains_in_sql('SELECT MIN("cost") FROM "expense"') == {"fine"}
    assert grains_in_sql(
        'SELECT event_name, SUM("spent") AS t FROM "budget" GROUP BY event_name'
    ) == {"coarse"}


def test_submit_grain_rejection_quoted_sum_measure() -> None:
    """task_25 hole: SUM(e."cost") must still yield measure 'cost' and reject."""
    q = "Which event has the lowest cost?"
    fine_sql = "SELECT e.event_name FROM expense e WHERE e.cost = (SELECT MIN(cost) FROM expense)"
    coarse_sql = (
        'SELECT ev.event_name FROM (SELECT b.link_to_event, SUM(e."cost") AS t '
        'FROM "expense" e JOIN budget b ON e.link_to_budget = b.budget_id '
        'GROUP BY b.link_to_event) x JOIN event ev ON x.link_to_event = ev.event_id'
    )
    steps = [
        _sql_step(fine_sql, grain="fine"),
        _sql_step(coarse_sql, grain="coarse"),
    ]
    rejection = submit_grain_rejection(q, steps, coarse_sql, _TASK25_KNOWLEDGE)
    assert rejection is not None
    assert not is_undecided(rejection)
    assert rejection["grain_check"]["measure_col"] == "cost"


# --- §13.3: question-anchored fine evidence ------------------------------------


def test_profiling_probe_is_not_fine_evidence() -> None:
    """MIN(spent)/MAX(spent) range profiling must not count as fine for 'cost'."""
    q = "Which event has the lowest cost?"
    profiling = _sql_step('SELECT MIN("spent"), MAX("spent") FROM expense')
    coarse = _sql_step("SELECT event_name, SUM(spent) FROM budget GROUP BY event_name")
    grains = collect_aggregate_grains([profiling, coarse], q)
    assert grains == {"coarse"}
    assert not dual_grain_satisfied(q, [profiling, coarse])


def test_anchored_probe_counts_as_fine() -> None:
    q = "Which event has the lowest cost?"
    fine_probe = _sql_step("SELECT MIN(cost) FROM expense")
    coarse = _sql_step("SELECT event_name, SUM(spent) FROM budget GROUP BY event_name")
    grains = collect_aggregate_grains([fine_probe, coarse], q)
    assert grains == {"fine", "coarse"}
    assert dual_grain_satisfied(q, [fine_probe, coarse])


def test_collect_without_question_keeps_legacy_behavior() -> None:
    profiling = _sql_step("SELECT MIN(spent), MAX(spent) FROM expense")
    assert collect_aggregate_grains([profiling]) == {"fine"}


# --- §13.5: UNDECIDED third state + pre-submit notes ---------------------------


def test_submit_grain_rejection_undecided_when_measure_unextractable() -> None:
    q = "Which school has the lowest score?"
    fine_sql = "SELECT AVG(score) FROM results"
    coarse_sql = "SELECT school, AVG(score) FROM results GROUP BY school"
    steps = [
        _sql_step(fine_sql, grain="fine"),
        _sql_step(coarse_sql, grain="coarse"),
    ]
    # Coarse final via GROUP BY + AVG (no SUM) → measure not extractable.
    payload = submit_grain_rejection(q, steps, coarse_sql, "")
    assert payload is not None
    assert is_undecided(payload)
    assert payload["check"] == "grain"


def test_pre_submit_grain_notes_warns_before_reject() -> None:
    q = "Which event has the lowest cost?"
    fine_sql = "SELECT MIN(cost) FROM expense"
    coarse_sql = "SELECT event_name, SUM(cost) AS t FROM expense GROUP BY event_name"
    steps = [_sql_step(fine_sql, grain="fine")]
    notes = pre_submit_grain_notes(q, steps, coarse_sql, _TASK25_KNOWLEDGE)
    assert notes and "REJECTED" in notes[0]


def test_pre_submit_grain_notes_undecided_note() -> None:
    q = "Which event has the lowest cost?"
    coarse_no_sum = "SELECT event_name, AVG(cost) FROM expense GROUP BY event_name"
    notes = pre_submit_grain_notes(q, [], coarse_no_sum, _TASK25_KNOWLEDGE)
    assert notes and "UNDECIDED" in notes[0]


def test_pre_submit_grain_notes_silent_for_fine() -> None:
    q = "Which event has the lowest cost?"
    assert pre_submit_grain_notes(q, [], "SELECT MIN(cost) FROM expense", "") == []


if __name__ == "__main__":
    test_wants_dual_grain()
    test_wants_member_expansion_keeps_average()
    test_grains_in_sql()
    test_dual_grain_gate_task25_style()
    test_dual_grain_tag_alone_counts()
    test_non_dual_question_always_ok()
    test_knowledge_defines_coarse()
    test_submit_grain_rejection_rejects_coarse_when_not_defined()
    test_submit_grain_rejection_allows_coarse_when_defined()
    test_submit_grain_rejection_ignores_fine_final()
    test_grains_in_sql_quoted_identifiers()
    test_submit_grain_rejection_quoted_sum_measure()
    test_profiling_probe_is_not_fine_evidence()
    test_anchored_probe_counts_as_fine()
    test_collect_without_question_keeps_legacy_behavior()
    test_submit_grain_rejection_undecided_when_measure_unextractable()
    test_pre_submit_grain_notes_warns_before_reject()
    test_pre_submit_grain_notes_undecided_note()
    test_pre_submit_grain_notes_silent_for_fine()
    print("ok")
