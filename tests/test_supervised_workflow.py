"""Closed-loop dispatch, selected joins, partial replacement and bounded termination."""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import Mock

import pytest

from kv_cache_agent.agents.supervisor_control import SupervisorController
from kv_cache_agent.graph.progress import (
    pending_claim_ids,
    ready_to_finalize,
)
from kv_cache_agent.graph.workflow import build_supervised_workflow
from kv_cache_agent.observability.logger import RunSession
from kv_cache_agent.observability.tracing import TracingSettings
from kv_cache_agent.schemas.research import BudgetLimits
from kv_cache_agent.schemas.supervision import Assignment, ReportOutline, RouteProposal
from tests.supervised_fixtures import (
    QUERY,
    FixtureResearch,
    choose_missing,
    controller,
    plan,
    raw_plan,
)


def workflow(research=None, **kwargs):
    research = research or FixtureResearch()
    return build_supervised_workflow(
        supervisor=kwargs.pop("supervisor", controller()),
        workers=research.workers,
        verifier_factory=research.verifier,
        **kwargs,
    ), research


def test_plan_is_created_before_workers_and_normal_path_reaches_finalize(tmp_path):
    instance, research = workflow()
    finalizer = Mock(return_value={"mock_artifact": "ready"})
    instance.finalizer = finalizer
    with RunSession(
        tmp_path, tracing=TracingSettings(enabled=False), console=False
    ) as run:
        result = instance.invoke({"user_query": QUERY})
    assert result["status"] == "ready_for_finalization", result.get("fatal_errors")
    assert ready_to_finalize(result) and not pending_claim_ids(result)
    assert result["round_id"] == 2
    assert [r.task.agent for r in research.calls].count("technical") == 1
    assert len(research.calls) == 4
    assert finalizer.call_count == 1
    assert all(d.status == "approved" for d in result["finalization_input"]["drafts"])
    events = [
        json.loads(row)
        for row in (run.directory / "events.jsonl").read_text().splitlines()
    ]
    starts = [e["node_path"] for e in events if e["event"] == "node_start"]
    assert starts[0] == "supervised.plan_report"
    assert starts.index("supervised.verify_claims") < starts.index(
        "supervised.worker", starts.index("supervised.worker") + 1
    )
    joined = [e for e in events if e["event"] == "round_results_collected"]
    assert len(joined) == 2
    assert (
        len(joined[1]["details"]["completed"]) == 3
        and not joined[1]["details"]["outstanding"]
    )
    assert result["budget_usage"]["model_calls"] == 10


def test_only_missing_cxl_market_cell_is_reassigned_and_turboquant_is_preserved():
    research = FixtureResearch(
        behavior=lambda req, cell, spec: (
            {"skip": True}
            if req.task.agent == "market"
            and req.task.round_id == 2
            and cell[1] == "CXL-based"
            else {}
        )
    )
    instance, _ = workflow(research)
    result = instance.invoke({"user_query": QUERY})
    assert result["status"] == "ready_for_finalization", result.get("fatal_errors")
    market = [r.task for r in research.calls if r.task.agent == "market"]
    assert len(market) == 2 and market[1].technologies == ("CXL-based",)
    assert market[1].criteria == ("adoption",)
    draft = result["drafts_by_section"]["market"]
    assert draft.version == 2 and {c.technology for c in draft.claims} == {
        "TurboQuant",
        "CXL-based",
    }
    turbo = next(c for c in draft.claims if c.technology == "TurboQuant")
    cxl = next(c for c in draft.claims if c.technology == "CXL-based")
    assert turbo.version == 1 and cxl.version == 2
    assert all(
        sum(r.task.agent == role for r in research.calls) == 1
        for role in ("technical", "stakeholder", "cloud_domain")
    )
    assert result["round_id"] == 3


def test_semantic_overstatement_triggers_revision_without_new_search():
    def behavior(req, cell, spec):
        if req.task.agent == "cloud_domain" and cell[1] == "CXL-based":
            return {
                "body": "CXL-based may lower latency under suitable conditions.",
                "text": "CXL-based always lowers latency."
                if req.task.action == "research"
                else "CXL-based may lower latency under suitable conditions.",
                "claim_type": "fact" if req.task.action == "research" else "inference",
                "caveat": ""
                if req.task.action == "research"
                else "Assumes the stated operating conditions.",
            }
        return {}

    research = FixtureResearch(behavior)
    instance, _ = workflow(research)
    result = instance.invoke({"user_query": QUERY})
    assert result["status"] == "ready_for_finalization", result.get("fatal_errors")
    calls = [r for r in research.calls if r.task.agent == "cloud_domain"]
    assert len(calls) == 2
    assert calls[-1].task.action == "revise" and calls[-1].task.max_search_calls == 0
    assert calls[-1].task.feedback and calls[-1].existing_evidence
    cell = next(
        c
        for c in result["coverage"]
        if c.section_id == "cloud_domain" and c.technology == "CXL-based"
    )
    assert cell.status == "inferred"


@pytest.mark.parametrize("parallel", (False, True))
def test_sequential_and_parallel_modes_have_same_round_boundaries(parallel):
    instance, research = workflow(parallel=parallel)
    result = instance.invoke({"user_query": QUERY})
    assert result["status"] == "ready_for_finalization", result.get("fatal_errors")
    assert result["round_id"] == 2 and len(research.calls) == 4
    assert len(result["processed_task_ids"]) == 4


def test_parallel_join_waits_for_selected_delayed_worker_only(tmp_path):
    research = FixtureResearch()
    done = Event()
    original = research.workers["stakeholder"]

    def delayed(req, **kwargs):
        time.sleep(0.03)
        result = original(req, **kwargs)
        done.set()
        return result

    research.workers["stakeholder"] = delayed

    def router(context, config=None):
        if context["round_id"] == 2:
            assert done.is_set()
        return choose_missing(context, config)

    instance, _ = workflow(research, supervisor=controller(router))
    with RunSession(tmp_path, tracing=TracingSettings(enabled=False), console=False):
        result = instance.invoke({"user_query": QUERY})
    assert result["status"] == "ready_for_finalization"
    assert done.is_set()
    assert len(result["active_task_ids"]) == 3


def test_worker_exception_is_collected_and_other_branches_survive():
    research = FixtureResearch()
    original = research.workers["market"]

    def fail_once(req, **kwargs):
        if req.task.round_id == 2:
            raise RuntimeError("temporary worker failure")
        return original(req, **kwargs)

    research.workers["market"] = fail_once
    instance, _ = workflow(research)
    result = instance.invoke({"user_query": QUERY})
    assert result["status"] == "ready_for_finalization", result.get("fatal_errors")
    assert any(r.status == "failed" for r in result["task_results"].values())
    assert {
        c.technology for c in result["drafts_by_section"]["stakeholder"].claims
    } == {"TurboQuant", "CXL-based"}
    assert any("temporary worker failure" in e for e in result["errors"])


def test_worker_protocol_failure_is_failed_not_provisional():
    research = FixtureResearch()
    original = research.workers["market"]
    research.workers["market"] = lambda req, **kwargs: original(
        req, **kwargs
    ).model_copy(update={"task_id": "wrong-task"})
    instance, _ = workflow(research)
    finalizer = Mock()
    instance.finalizer = finalizer
    result = instance.invoke({"user_query": QUERY})
    assert result["status"] == "failed" and result["fatal_errors"]
    finalizer.assert_not_called()
    assert len(result["deliveries"]) == 4


def test_stale_delivery_is_discarded_and_current_task_has_failure_result():
    research = FixtureResearch()
    original = research.workers["market"]

    def stale(req, **kwargs):
        result = original(req, **kwargs)
        return (
            result.model_copy(update={"round_id": 1})
            if req.task.round_id == 2
            else result
        )

    research.workers["market"] = stale
    instance, _ = workflow(research)
    result = instance.invoke({"user_query": QUERY})
    assert result["status"] == "ready_for_finalization", result.get("fatal_errors")
    assert any(d.issue_kind == "stale" for d in result["deliveries"].values())
    assert result["drafts_by_section"]["market"].version == 1


def test_two_no_progress_rounds_stop_without_waiting_for_other_agents():
    research = FixtureResearch(lambda *args: {"skip": True})
    instance, _ = workflow(research)
    result = instance.invoke({"user_query": QUERY})
    assert result["status"] == "provisional"
    assert result["termination_reason"] == "no_progress_limit"
    assert result["round_id"] == 2 and result["no_progress_rounds"] == 2
    assert len(research.calls) == 2 and all(
        r.task.agent == "technical" for r in research.calls
    )
    assert len(result["finalization_input"]["missing_items"]) == 8


def test_round_limit_preserves_missing_cells_and_safe_finalization_payload():
    research = FixtureResearch(
        lambda req, cell, spec: {"skip": True} if req.task.agent == "market" else {}
    )
    instance, _ = workflow(research, budget_limits=BudgetLimits(research_rounds=2))
    result = instance.invoke({"user_query": QUERY})
    assert (
        result["status"] == "provisional"
        and result["termination_reason"] == "research_round_limit"
    )
    assert {
        (c.section_id, c.technology)
        for c in result["finalization_input"]["missing_items"]
    } == {("market", "TurboQuant"), ("market", "CXL-based")}
    assert all(
        "always" not in c.text
        for d in result["finalization_input"]["drafts"]
        for c in d.claims
    )


def test_budget_exhaustion_preserves_finalization_reserve():
    instance, research = workflow(
        budget_limits=BudgetLimits(model_calls=6, finish_reserve=2)
    )

    def finalize(payload, *, budget, config=None):
        assert budget.held_snapshot() == dict.fromkeys(budget.snapshot(), 0)
        budget.reserve(model_calls=2, finishing=True)
        return {"finalization_hook_invoked": True}

    instance.finalizer = finalize
    result = instance.invoke({"user_query": QUERY, "report_plan": plan()})
    assert result["status"] == "provisional"
    assert result["termination_reason"] == "research_model_budget_exhausted"
    assert result["finalization_result"]["finalization_hook_invoked"]
    assert result["budget_usage"]["model_calls"] == 5
    assert len(research.calls) == 1


def test_finisher_exception_changes_ready_outcome_to_failed():
    instance, _ = workflow(
        finalizer=Mock(side_effect=RuntimeError("rendering hook failed"))
    )
    result = instance.invoke({"user_query": QUERY})
    assert (
        result["status"] == "failed"
        and result["termination_reason"] == "finalization_failed"
    )


def test_low_recursion_limit_stops_and_preserves_partial_state():
    instance, _ = workflow()
    result = instance.invoke({"user_query": QUERY}, config={"recursion_limit": 9})
    assert result["status"] in {"provisional", "failed"}
    assert result["termination_reason"] == "recursion_limit"
    assert result["report_plan"] and result["task_results"]


def test_reused_workflow_has_independent_budgets_and_task_ids():
    instance, _ = workflow()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(lambda _: instance.invoke({"user_query": QUERY}), range(2))
        )
    assert all(r["status"] == "ready_for_finalization" for r in results)
    assert results[0]["budget_usage"] == results[1]["budget_usage"]
    assert not set(results[0]["tasks_by_id"]) & set(results[1]["tasks_by_id"])


def test_enabled_tracing_missing_configuration_fails_before_planner(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_PROJECT", "")
    planner = Mock()
    instance, _ = workflow(supervisor=SupervisorController(planner=planner))
    result = instance.invoke({"user_query": QUERY})
    assert (
        result["status"] == "failed"
        and result["termination_reason"] == "configuration_failed"
    )
    assert result["budget_usage"]["model_calls"] == 0
    planner.assert_not_called()


def test_multiple_disjoint_tasks_in_one_section_merge_at_one_version():
    def split_router(context, config=None):
        proposal = choose_missing(context, config)
        tasks = []
        for assignment in proposal.assignments:
            if assignment.agent == "technical":
                tasks.extend(
                    assignment.model_copy(update={"technologies": (technology,)})
                    for technology in assignment.technologies
                )
            else:
                tasks.append(assignment)
        return proposal.model_copy(update={"assignments": tuple(tasks)})

    instance, research = workflow(supervisor=controller(split_router))
    result = instance.invoke({"user_query": QUERY})
    assert result["status"] == "ready_for_finalization", result.get("fatal_errors")
    assert len(research.calls) == 5
    assert result["drafts_by_section"]["technical"].version == 1
    assert len(result["drafts_by_section"]["technical"].claims) == 2


def test_budget_admits_only_selected_subset_and_does_not_wait_for_deferred_agents():
    instance, research = workflow(
        budget_limits=BudgetLimits(model_calls=8, finish_reserve=1)
    )
    result = instance.invoke({"user_query": QUERY, "report_plan": plan()})
    assert result["status"] == "provisional"
    assert result["budget_usage"]["model_calls"] <= 7
    assert len(research.calls) == 3
    assert len(result["active_task_ids"]) == 2
    assert result["finalization_input"]["missing_items"]


def test_failed_delivery_is_logged_as_failed_node_result(tmp_path):
    research = FixtureResearch()
    research.workers["technical"] = Mock(side_effect=RuntimeError("failed research"))
    instance, _ = workflow(research)
    with RunSession(
        tmp_path, tracing=TracingSettings(enabled=False), console=False
    ) as run:
        result = instance.invoke({"user_query": QUERY})
    records = [
        json.loads(line)
        for line in (run.directory / "events.jsonl").read_text().splitlines()
    ]
    ends = [
        r
        for r in records
        if r["node_path"] == "supervised.worker" and r["event"] == "node_end"
    ]
    assert ends and all(r["status"] == "failed" and r["level"] == "ERROR" for r in ends)
    assert result["status"] == "provisional"


def test_verification_contract_mismatch_is_failed_and_never_approved():
    research = FixtureResearch()

    def factory(budget):
        original = research.verifier(budget)

        def verify(claims, evidence, config=None):
            result = original.verify(claims, evidence, config=config)
            decisions = tuple(
                d.model_copy(update={"policy_version": "outdated-policy"})
                for d in result.decisions
            )
            return result.model_copy(update={"decisions": decisions})

        return Mock(verify=verify)

    instance = build_supervised_workflow(
        supervisor=controller(), workers=research.workers, verifier_factory=factory
    )
    result = instance.invoke({"user_query": QUERY})
    assert result["status"] == "failed"
    assert any("stale claim decision" in error for error in result["fatal_errors"])


@pytest.mark.parametrize("round_limit", (2, 3))
def test_semantic_review_distinguishes_true_prototype_fact_from_production_answer(
    round_limit,
):
    outline = raw_plan()
    sections = tuple(
        s.model_copy(
            update={"questions": ("Which named operator deployed CXL in production?",)}
        )
        if s.owner == "market"
        else s
        for s in outline.sections
    )

    def behavior(req, cell, spec):
        if req.task.agent == "market" and cell[1] == "CXL-based":
            text = (
                "CXL-based has only a documented prototype."
                if req.task.round_id == 2
                else "AcmeOperator deployed CXL-based KV cache in production."
            )
            return {"text": text, "body": text}
        return {}

    research = FixtureResearch(behavior)

    def router(context, config=None):
        if context.get("completion_review") and context["round_id"] == 3:
            assert any(
                "AcmeOperator" in c["text"]
                for d in context["current_drafts"]
                for c in d["claims"]
            )
            return RouteProposal(
                action="finalize",
                reason="The planned production question now has a supported named answer",
            )
        if context.get("completion_review") and context["round_id"] == 2:
            target = next(
                t
                for t in context["reviewable_targets"]
                if t["section_id"] == "market" and t["technology"] == "CXL-based"
            )
            return RouteProposal(
                action="research",
                reason="Prototype truth does not answer production adoption",
                assignments=(
                    Assignment(
                        agent="market",
                        section_ids=("market",),
                        technologies=("CXL-based",),
                        criteria=("adoption",),
                        objective="Find actual named production deployment",
                        questions=("Named CXL KV-cache production deployment",),
                        trigger="semantic_gap",
                        unanswered_question_ids=(
                            target["questions"][0]["question_id"],
                        ),
                        feedback=(
                            "The verified prototype is not a production operator deployment.",
                        ),
                    ),
                ),
            )
        return choose_missing(context, config)

    service = SupervisorController(
        planner=lambda q, config=None: ReportOutline(sections=sections), router=router
    )
    instance, _ = workflow(
        research,
        supervisor=service,
        budget_limits=BudgetLimits(research_rounds=round_limit),
    )
    result = instance.invoke({"user_query": QUERY})
    market = [req for req in research.calls if req.task.agent == "market"]
    if round_limit == 3:
        assert result["status"] == "ready_for_finalization", result.get("fatal_errors")
        assert len(market) == 2 and market[-1].task.technologies == ("CXL-based",)
        assert market[-1].task.unanswered_questions == (
            "Which named operator deployed CXL in production?",
        )
        assert not result["semantic_gaps"]
        assert any(
            "AcmeOperator" in c.text
            for c in result["drafts_by_section"]["market"].claims
        )
    else:
        assert (
            result["status"] == "provisional"
            and result["termination_reason"] == "research_round_limit"
        )
        assert len(market) == 1 and result["finalization_input"]["unresolved_questions"]


def test_last_completion_review_budget_is_not_silently_treated_as_approval():
    instance, _ = workflow(budget_limits=BudgetLimits(model_calls=9, finish_reserve=1))
    result = instance.invoke({"user_query": QUERY, "report_plan": plan()})
    assert all(c.status == "verified" for c in result["coverage"])
    assert result["status"] == "provisional"
    assert result["termination_reason"] == "completion_review_budget_exhausted"
