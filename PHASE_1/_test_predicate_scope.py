"""Unit tests for the predicate-scope contract (180/249 class, mechanism-level)."""

from __future__ import annotations

from data_agent_baseline.agents.gate_common import is_undecided
from data_agent_baseline.agents.hypothesis import build_hypotheses
from data_agent_baseline.agents.predicate_scope import (
    complete_case_licensed,
    submit_predicate_scope_rejection,
    unguarded_div_filters,
    wants_division_scope_check,
    wants_multi_avg_scope_check,
)


def test_unguarded_div_in_where_detected() -> None:
    sql = (
        'SELECT ym.Consumption FROM yearmonth ym WHERE ym.Date = 201208 '
        'AND ym.CustomerID IN (SELECT CustomerID FROM transactions_1k '
        'WHERE ProductID = 5 AND Price / Amount > 29.00)'
    )
    pairs = unguarded_div_filters(sql)
    assert pairs
    assert pairs[0][1].casefold() == "amount"


def test_guarded_div_not_detected() -> None:
    sql = (
        "SELECT Consumption FROM yearmonth WHERE CustomerID IN ("
        "SELECT CustomerID FROM transactions_1k "
        "WHERE ProductID = 5 AND Amount > 0 AND Price / Amount > 29.00)"
    )
    assert unguarded_div_filters(sql) == []


def test_div_in_select_not_a_filter() -> None:
    sql = "SELECT Price / Amount AS unit FROM transactions_1k WHERE ProductID = 5"
    assert unguarded_div_filters(sql) == []


def test_reject_unguarded_unit_price_filter() -> None:
    sql = (
        "SELECT Consumption FROM yearmonth WHERE CustomerID IN ("
        "SELECT CustomerID FROM t WHERE ProductID = 5 AND Price / Amount > 29)"
    )
    rejection = submit_predicate_scope_rejection(
        "For people who paid more than 29.00 per unit of product 5.",
        "",
        sql,
    )
    assert rejection is not None
    assert not is_undecided(rejection)
    assert rejection["predicate_scope_check"]["kind"] == "division_domain"


def test_allow_guarded_unit_price_filter() -> None:
    sql = (
        "SELECT Consumption FROM yearmonth WHERE CustomerID IN ("
        "SELECT CustomerID FROM t WHERE ProductID = 5 "
        "AND Amount > 0 AND Price / Amount > 29)"
    )
    assert (
        submit_predicate_scope_rejection(
            "paid more than 29 per unit",
            "",
            sql,
        )
        is None
    )


def test_reject_unlicensed_complete_case_on_second_avg() -> None:
    sql = (
        "SELECT AVG(UpVotes) AS avg_upvotes, AVG(Age) AS avg_age "
        "FROM users u JOIN (SELECT OwnerUserId FROM posts "
        "GROUP BY OwnerUserId HAVING COUNT(*) > 10) p ON u.Id = p.OwnerUserId "
        "WHERE u.Age IS NOT NULL"
    )
    question = (
        "What is the average of the up votes and the average user age "
        "for users creating more than 10 posts?"
    )
    rejection = submit_predicate_scope_rejection(question, "", sql)
    assert rejection is not None
    assert rejection["predicate_scope_check"]["kind"] == "multi_metric_population"


def test_allow_licensed_complete_case() -> None:
    sql = (
        "SELECT AVG(UpVotes) AS avg_upvotes, AVG(Age) AS avg_age "
        "FROM users WHERE Age IS NOT NULL"
    )
    question = (
        "What is the average of the up votes and the average age "
        "for users with known age?"
    )
    assert complete_case_licensed(question, "", "Age")
    assert submit_predicate_scope_rejection(question, "", sql) is None


def test_single_avg_null_filter_allowed() -> None:
    sql = "SELECT AVG(Age) AS avg_age FROM users WHERE Age IS NOT NULL"
    assert (
        submit_predicate_scope_rejection(
            "What is the average age?",
            "",
            sql,
        )
        is None
    )


def test_hypotheses_declared_for_both_classes() -> None:
    assert wants_division_scope_check(
        "paid more than 29.00 per unit of product id 5"
    )
    assert wants_multi_avg_scope_check(
        "average of the up votes and the average user age"
    )
    div_hyps = build_hypotheses(
        question="Who paid more than 29 per unit?",
        knowledge_text="",
        link=None,
    )
    pop_hyps = build_hypotheses(
        question="What is the average upvotes and the average age?",
        knowledge_text="",
        link=None,
    )
    assert "pred_scope_div" in {h.hid for h in div_hyps}
    assert "pred_scope_pop" in {h.hid for h in pop_hyps}


if __name__ == "__main__":
    test_unguarded_div_in_where_detected()
    test_guarded_div_not_detected()
    test_div_in_select_not_a_filter()
    test_reject_unguarded_unit_price_filter()
    test_allow_guarded_unit_price_filter()
    test_reject_unlicensed_complete_case_on_second_avg()
    test_allow_licensed_complete_case()
    test_single_avg_null_filter_allowed()
    test_hypotheses_declared_for_both_classes()
    print("ok")
