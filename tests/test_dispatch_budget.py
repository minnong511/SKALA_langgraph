from concurrent.futures import ThreadPoolExecutor

import pytest

from kv_cache_agent.schemas.research import BudgetExceeded, BudgetLedger, BudgetLimits


def test_group_allocation_is_atomic_and_actual_calls_are_not_double_charged():
    budget = BudgetLedger(BudgetLimits(model_calls=5, finish_reserve=2, search_calls=3))
    workers = budget.allocate_many(
        {
            "a": {"model_calls": 1, "search_calls": 2},
            "b": {"model_calls": 1, "search_calls": 1},
        },
        retain={"model_calls": 1},
    )
    assert (
        budget.snapshot()["model_calls"] == 0
        and budget.held_snapshot()["model_calls"] == 2
    )
    workers["a"].reserve(model_calls=1, search_calls=1)
    assert (
        budget.snapshot()["search_calls"] == 1
        and budget.held_snapshot()["search_calls"] == 2
    )
    with pytest.raises(BudgetExceeded):
        budget.reserve(model_calls=2)
    workers["a"].close()
    workers["b"].close()
    assert not any(budget.held_snapshot().values())
    assert budget.snapshot()["model_calls"] == 1
    budget.reserve(model_calls=2, finishing=True)


def test_rejected_allocation_holds_nothing_and_keeps_finishing_reserve():
    budget = BudgetLedger(BudgetLimits(model_calls=4, finish_reserve=2))
    with pytest.raises(BudgetExceeded):
        budget.allocate_many({"a": {"model_calls": 2}}, retain={"model_calls": 1})
    assert not any(budget.held_snapshot().values()) and not any(
        budget.snapshot().values()
    )
    budget.reserve(model_calls=2, finishing=True)


def test_parallel_attempts_cannot_spend_sibling_capacity():
    budget = BudgetLedger(BudgetLimits(model_calls=5, finish_reserve=1))
    pool = budget.allocate_many({"a": {"model_calls": 2}, "b": {"model_calls": 1}})

    def attempt(_):
        try:
            pool["a"].reserve(model_calls=1)
            return 1
        except BudgetExceeded:
            return 0

    with ThreadPoolExecutor(max_workers=8) as threads:
        assert sum(threads.map(attempt, range(20))) == 2
    assert budget.held_snapshot()["model_calls"] == 1
    pool["b"].reserve(model_calls=1)
    assert budget.snapshot()["model_calls"] == 3
    for reservation in pool.values():
        reservation.close()


def test_close_returns_only_unused_capacity_and_cannot_refund_failed_attempt():
    budget = BudgetLedger()
    reservation = budget.allocate_many({"a": {"extract_calls": 2, "extract_urls": 6}})[
        "a"
    ]
    reservation.reserve(extract_calls=1, extract_urls=3)
    reservation.close()
    reservation.close()
    assert (
        budget.snapshot()["extract_calls"] == 1
        and budget.snapshot()["extract_urls"] == 3
    )
    assert not any(budget.held_snapshot().values())
    with pytest.raises(BudgetExceeded):
        reservation.reserve(extract_calls=1)
