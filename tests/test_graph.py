"""Exercise the real scheduler and supervisor with deterministic injected workers."""

import json
from collections import Counter
from threading import Barrier, Lock

from src.agents import supervisor
from src.common.artifacts import ArtifactStore
from src.common.events import EventLogger
from src.common.runtime import AgentRuntime, CallBudget
from src.demo import demo_agents
from src.graph import build_graph, initial_state
from src.schemas import RESEARCH_AGENTS, AgentContext, AgentRequest, FollowUpRequest


def execute_graph(tmp_path, *, replacements=None, limits=None, recorder=None):
    request = AgentRequest(run_id="test-graph", task_id="workflow", **({"limits": limits} if limits else {}))
    store = ArtifactStore(tmp_path, request.run_id)
    events = EventLogger(store.run_dir, request.run_id, quiet=True)
    context = AgentContext(
        demo=True,
        events=events,
        budget=CallBudget(request.limits.max_total_calls, request.limits.max_run_seconds),
    )
    runtime = AgentRuntime(context, store)
    implementations = demo_agents()
    implementations.update(replacements or {})
    calls = Counter()
    call_lock = Lock()

    def track(name, implementation):
        def run(request, context):
            with call_lock:
                calls[name] += 1
                if recorder is not None:
                    recorder.append((name, request, dict(context.results)))
            return implementation(request, context)

        return run

    graph = build_graph(
        runtime, {name: track(name, implementation) for name, implementation in implementations.items()}
    )
    state = graph.invoke(initial_state(request), config={"recursion_limit": 100})
    return state, calls, store, context


def test_workers_return_to_supervisor_and_parallel_results_merge(tmp_path):
    workers = demo_agents()
    barrier = Barrier(3, timeout=3)
    observed = []

    def parallel_worker(request, context):
        assert "technical" in context.results
        assert not {"market", "stakeholder", "domain"}.intersection(context.results)
        barrier.wait()  # Sequential execution would break this barrier and fail the result.
        return workers[context.agent](request, context)

    state, calls, store, _ = execute_graph(
        tmp_path,
        recorder=observed,
        replacements={name: parallel_worker for name in ("market", "stakeholder", "domain")},
    )
    assert all(state[f"{name}_result"].status == "completed" for name in RESEARCH_AGENTS)
    supervisor_calls = [
        (request.context["phase"], results) for name, request, results in observed if name == "supervisor"
    ]
    assert [phase for phase, _ in supervisor_calls] == [
        "start",
        "technical",
        "evaluations",
        "verification",
        "synthesis",
        "report",
    ]
    assert set(supervisor_calls[2][1]) == set(RESEARCH_AGENTS)
    assert all(calls[name] == 1 for name in workers if name != "supervisor")
    assert state["review"]["passed"] is True
    assert state["status"] == "needs_review"  # Demo evidence is never accepted as a real final report.
    events = [json.loads(line) for line in (store.run_dir / "events.jsonl").read_text().splitlines()]
    progress = [event["details"]["current"] for event in events if event["event"] == "parallel_progress"]
    assert progress == [1, 2, 3]
    assert all(
        event["details"]["elapsed_seconds"] >= 0 for event in events if event["event"] == "agent_return"
    )


def test_verification_retries_only_the_affected_perspective(tmp_path):
    baseline = demo_agents()["verification"]

    def verification(request, context):
        result = baseline(request, context)
        if request.attempt == 1:
            result.verification[2].status = "uncertain"
            result.verification[2].reason = "시장 채택 사례의 적용 조건을 확인하세요."
        return result

    observed = []
    state, calls, _, _ = execute_graph(
        tmp_path, replacements={"verification": verification}, recorder=observed
    )
    assert calls["market"] == 2
    assert calls["technical"] == calls["stakeholder"] == calls["domain"] == 1
    second_market = [request for name, request, _ in observed if name == "market"][1]
    assert "시장 채택 사례의 적용 조건을 확인하세요." in second_market.feedback
    assert state["retry_counts"]["market"] == 1
    assert state["market_result"].attempt == 2
    assert state["stakeholder_result"].attempt == 1
    assert state["review"]["passed"] is True


def test_technical_revision_invalidates_every_dependent_result(tmp_path):
    baseline = demo_agents()["verification"]

    def verification(request, context):
        result = baseline(request, context)
        if request.attempt == 1:
            result.follow_up_requests = [
                FollowUpRequest(target_agent="technical", reason="기술 실험 조건 변경")
            ]
        return result

    observed = []
    state, calls, _, _ = execute_graph(
        tmp_path, replacements={"verification": verification}, recorder=observed
    )
    second_technical = [(request, results) for name, request, results in observed if name == "technical"][1]
    assert set(second_technical[1]) == {"technical"}
    for name, request, results in observed:
        if name in {"market", "stakeholder", "domain"} and request.attempt == 2:
            assert set(results) == {"technical"}
            assert results["technical"].attempt == 2
    assert all(calls[name] == 2 for name in RESEARCH_AGENTS)
    assert all(state[f"{name}_result"].attempt == 2 for name in RESEARCH_AGENTS)
    assert all("-2-" in identifier for identifier in state["report_result"].used_evidence_ids)


def test_research_retries_are_capped_and_uncertainty_survives(tmp_path):
    baseline = demo_agents()["verification"]

    def uncertain_market(request, context):
        result = baseline(request, context)
        for verdict in result.verification:
            if verdict.target_agent == "market":
                verdict.status = "uncertain"
                verdict.reason = "원문으로 채택 사례를 확인할 수 없습니다."
        return result

    state, calls, _, _ = execute_graph(
        tmp_path, replacements={"verification": uncertain_market}, limits={"max_research_retries": 1}
    )
    assert calls["market"] == 2
    assert calls["verification"] == 2
    assert calls["technical"] == calls["stakeholder"] == calls["domain"] == 1
    assert state["status"] == "needs_review"
    assert any(verdict.status == "uncertain" for verdict in state["verification_result"].verification)


def test_report_review_failure_is_revised_then_stops_at_limit(tmp_path):
    baseline = demo_agents()["report"]

    def invalid_report(request, context):
        result = baseline(request, context)
        result.report_markdown = (
            "# SUMMARY\n검증되지 않은 단정입니다. [invented]\n\n# REFERENCE\n[invented] 없는 출처"
        )
        result.used_evidence_ids = ["invented"]
        return result

    state, calls, _, _ = execute_graph(
        tmp_path, replacements={"report": invalid_report}, limits={"max_report_revisions": 1}
    )
    assert calls["report"] == 2
    assert state["retry_counts"]["report"] == 1
    assert state["status"] == "needs_review"
    assert state["review"]["passed"] is False
    assert any("목차" in issue for issue in state["review"]["issues"])
    assert any("invented" in issue for issue in state["review"]["issues"])


def test_total_budget_stops_work_without_exceeding_limit(tmp_path):
    state, calls, store, context = execute_graph(tmp_path, limits={"max_total_calls": 2})
    assert sum(context.budget.counts.values()) == 2
    assert calls == {"supervisor": 1, "technical": 1}
    assert state["status"] == "needs_review"
    assert state["budget_exhausted"] is True
    assert not state.get("report_result")
    rows = [json.loads(line) for line in (store.run_dir / "events.jsonl").read_text().splitlines()]
    assert any(row["event"] == "limit_reached" for row in rows)


def test_elapsed_budget_can_end_before_any_agent_is_called(tmp_path):
    request = AgentRequest(run_id="expired", task_id="workflow")
    store = ArtifactStore(tmp_path, request.run_id)
    budget = CallBudget(100, 1)
    budget.started -= 2
    context = AgentContext(
        demo=True, events=EventLogger(store.run_dir, request.run_id, quiet=True), budget=budget
    )
    graph = build_graph(AgentRuntime(context, store), demo_agents())
    state = graph.invoke(initial_state(request))
    assert state["status"] == "needs_review"
    assert state["budget_exhausted"] is True
    assert not budget.counts
    assert state["attempts"] == {"supervisor": 1}


def test_supervisor_rejects_unverified_report_citations():
    from src.schemas import AgentResult

    report = AgentResult(
        task_id="report",
        agent="report",
        status="completed",
        summary="bad report",
        report_markdown="# SUMMARY\n본문 [not-verified]\n# REFERENCE\n[not-verified] 출처",
        used_evidence_ids=["not-verified"],
    )
    assert any("not-verified" in issue for issue in supervisor.review_report(report, {}))


def test_synthesis_limit_survives_cycles_through_research(tmp_path):
    baseline = demo_agents()["synthesis"]

    def synthesis(request, context):
        result = baseline(request, context)
        result.follow_up_requests = [FollowUpRequest(target_agent="market", reason="추가 시장 근거 필요")]
        return result

    state, calls, _, _ = execute_graph(
        tmp_path,
        replacements={"synthesis": synthesis},
        limits={"max_synthesis_retries": 0, "max_research_retries": 2},
    )
    assert calls["synthesis"] == 1
    assert calls["market"] == 2
    assert state["status"] == "needs_review"
    assert "한도" in state["review"]["issues"][0]
