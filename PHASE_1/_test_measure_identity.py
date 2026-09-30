"""Measure-identity tests (cost vs spent class, no task_id)."""

from __future__ import annotations

from data_agent_baseline.agents.hypothesis import build_hypotheses
from data_agent_baseline.agents.measure_identity import (
    knowledge_binds_term_to_measure,
    measures_in_sql,
    submit_measure_identity_rejection,
)
from data_agent_baseline.agents.schema_link import ColumnCandidate


def _candidates() -> list[ColumnCandidate]:
    return [
        ColumnCandidate("expense", "cost", 3.0, role="measure"),
        ColumnCandidate("budget", "spent", 2.0, role="measure"),
        ColumnCandidate("event", "event_name", 1.0, role="dimension"),
    ]


def test_measures_extracted_from_sum_spent() -> None:
    sql = (
        'SELECT e.event_name FROM "event" e LEFT JOIN "budget" b '
        "ON e.event_id = b.link_to_event GROUP BY e.event_id, e.event_name "
        "HAVING SUM(b.spent) = 0"
    )
    assert "spent" in [m.casefold() for m in measures_in_sql(sql)]


def test_reject_spent_when_question_names_cost() -> None:
    sql = (
        "SELECT e.event_name FROM event e LEFT JOIN budget b "
        "ON e.event_id = b.link_to_event GROUP BY e.event_id "
        "HAVING SUM(b.spent) = (SELECT MIN(s) FROM ("
        "SELECT SUM(spent) AS s FROM event e JOIN budget b "
        "ON e.event_id = b.link_to_event GROUP BY e.event_id))"
    )
    knowledge = (
        "Average Cost: AVG(cost)\n"
        "Total Expenditure: SUM(spent)\n"
    )
    rejection = submit_measure_identity_rejection(
        "Which event has the lowest cost?",
        knowledge,
        sql,
        _candidates(),
    )
    assert rejection is not None
    assert rejection["measure_identity_check"]["question_column"] == "cost"
    assert rejection["measure_identity_check"]["sql_measure"].casefold() == "spent"


def test_allow_min_cost() -> None:
    sql = (
        "SELECT e.event_name FROM event e JOIN expense x "
        "ON e.event_id = x.link_to_event "
        "WHERE x.cost = (SELECT MIN(cost) FROM expense)"
    )
    assert (
        submit_measure_identity_rejection(
            "Which event has the lowest cost?",
            "Average Cost: AVG(cost)\nTotal Expenditure: SUM(spent)",
            sql,
            _candidates(),
        )
        is None
    )


def test_allow_when_knowledge_binds_cost_to_spent() -> None:
    sql = "SELECT event_name FROM event JOIN budget GROUP BY event_name HAVING SUM(spent) = 0"
    knowledge = "Event cost: SUM(spent) per event."
    assert knowledge_binds_term_to_measure(knowledge, "cost", "spent")
    assert (
        submit_measure_identity_rejection(
            "Which event has the lowest cost?",
            knowledge,
            sql,
            _candidates(),
        )
        is None
    )


def test_no_reject_when_named_column_absent() -> None:
    sql = "SELECT event_name FROM event JOIN budget GROUP BY 1 HAVING SUM(spent) = 0"
    only_spent = [ColumnCandidate("budget", "spent", 2.0, role="measure")]
    assert (
        submit_measure_identity_rejection(
            "Which event has the lowest cost?",
            "",
            sql,
            only_spent,
        )
        is None
    )


def test_hypothesis_emitted() -> None:
    hyps = build_hypotheses(
        question="Which event has the lowest cost?",
        knowledge_text="",
        link=None,
    )
    assert "measure_identity" in {h.hid for h in hyps}
