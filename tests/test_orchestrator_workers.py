"""Dynamic planning, concurrent fan-out, recovery, quality loops and bounds."""

import json
import logging
from collections import Counter
from copy import deepcopy
from threading import Barrier
from unittest.mock import Mock

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Send

from kv_cache_agent.agents import orchestrator, quality_evaluator, report_writer
from kv_cache_agent.agents.orchestrator import orchestrator_agent
from kv_cache_agent.agents.quality_evaluator import quality_evaluator_agent
from kv_cache_agent.agents.research_worker import _research_once
from kv_cache_agent.config import WorkflowLimits
from kv_cache_agent.graph.routing import assign_workers
from kv_cache_agent.graph.workflow import _initialize, build_workflow
from kv_cache_agent.mock_run import mock_plan, mock_services
from kv_cache_agent.schemas.tasks import REQUIRED_PERSPECTIVES, ResearchPlan

INPUT = {"payload": {"user_query": "두 기술을 평가해줘"}}


def initial_state(query="두 기술을 평가해줘", limits=None):
    return _initialize({"payload": {"user_query": query}}, limits or WorkflowLimits())


def run(**kwargs):
    with mock_services():
        return build_workflow(
            limits=kwargs.pop("limits", WorkflowLimits()), **kwargs
        ).invoke(INPUT)


def test_dynamic_planning_varies_with_input():
    with mock_services():
        simple = orchestrator_agent(initial_state())["payload"]["tasks"]
        complex_tasks = orchestrator_agent(initial_state("공급망과 위험도 평가"))[
            "payload"
        ]["tasks"]
    assert len(simple) != len(complex_tasks)
    assert set(REQUIRED_PERSPECTIVES) <= {t["perspective"] for t in simple}
    assert len([t for t in simple if t["perspective"] == "technical_maturity"]) > 1


@pytest.mark.parametrize("perspective", ["market", "stakeholder"])
def test_empty_web_heuristics_fall_back_to_source_bound_extraction(
    monkeypatch, perspective
):
    from kv_cache_agent.agents import research_worker as module
    from kv_cache_agent.schemas.tasks import SubTask
    from kv_cache_agent.schemas.technical import TechnicalExtraction, TechnicalFinding

    web = [
        {
            "title": "Supported memory expansion",
            "url": "https://example.com/support",
            "content": "Short search summary",
            "raw_content": "Vendor supports memory expansion under specified hardware conditions.",
        }
    ]
    monkeypatch.setattr(module, "search_web", lambda *a, **k: web)
    monkeypatch.setattr(module.market, "_build_evidence_cards", lambda _: ([], []))
    monkeypatch.setattr(
        module.stakeholder,
        "_build_stakeholder_evidence",
        lambda _: {"evidence_cards": []},
    )
    observed = []

    def extract(local, chunks):
        observed.append((local, chunks))
        return TechnicalExtraction(
            summary="지원 조건",
            limitations=[],
            findings=[
                TechnicalFinding(
                    technology="CXL-based",
                    claim="공급사의 메모리 확장 지원 조건",
                    evidence_text=chunks[0]["content"],
                    source_chunk_ids=["web-0"],
                    claim_type="fact",
                    confidence=0.8,
                )
            ],
        )

    monkeypatch.setattr(module.technical, "_extract_findings", extract)
    task = SubTask(
        task_id="fallback",
        perspective=perspective,
        technology="CXL-based",
        objective="공식 지원 근거",
        query="hardware support",
        preferred_source="web",
        priority=1,
    )
    result = module.research_worker(
        {
            "task": task.model_dump(),
            "trace_id": "trace",
            "user_query": "근거 조사",
            "target_domain": "서빙",
            "limits": {"max_worker_retries": 0, "max_cards_per_task": 24},
        }
    )
    card = result["payload"]["evidence_cards"][0]
    assert card["source_url"] == web[0]["url"]
    assert card["perspective"] == perspective
    assert card["evidence_text"] == web[0]["raw_content"]
    assert task.objective in observed[0][0]["user_query"]
    assert result["payload"]["worker_results"][0]["retry_count"] == 0


def test_planner_uses_structured_output_and_repairs_missing_perspectives(monkeypatch):
    plan = mock_plan("simple")
    plan.tasks = plan.tasks[:1]
    llm = Mock()
    llm.with_structured_output.return_value.invoke.return_value = plan
    monkeypatch.setattr(orchestrator, "get_llm", lambda: llm)
    state = initial_state()
    before = deepcopy(state)
    result = orchestrator_agent(state)
    llm.with_structured_output.assert_called_once_with(ResearchPlan)
    assert set(REQUIRED_PERSPECTIVES) <= {
        t["perspective"] for t in result["payload"]["tasks"]
    }
    assert state == before


@pytest.mark.parametrize("count", [1, 4, 7, 13])
def test_dynamic_fan_out_sends_exactly_n_tasks(count):
    state = initial_state()
    task = mock_plan("simple").tasks[0]
    state["payload"]["tasks"] = [
        task.model_copy(update={"task_id": f"t{i}"}).model_dump() for i in range(count)
    ]
    state["control"]["task_status"] = {f"t{i}": "pending" for i in range(count)}
    sends = assign_workers(state)
    assert len(sends) == count
    assert all(
        isinstance(send, Send) and send.node == "research_worker" for send in sends
    )
    assert {send.arg["task"]["task_id"] for send in sends} == set(
        state["control"]["task_status"]
    )
    state["control"]["task_status"]["t0"] = "success"
    remaining = assign_workers(state)
    assert len(remaining) == count - 1 if count > 1 else remaining == "reduce_results"


def test_parallel_worker_reducer_preserves_every_result_and_card():
    barrier = Barrier(5)
    calls = []

    def worker(task, request):
        calls.append(task.task_id)
        barrier.wait(timeout=5)  # Fails if this is a disguised serial fixed flow.
        return _research_once(task, request)

    result = run(worker_service=worker)
    payload = result["payload"]
    assert len(calls) == len(set(calls)) == len(payload["tasks"]) == 5
    assert len(payload["worker_results"]) == 5
    expected = {
        c["evidence_id"] for r in payload["worker_results"] for c in r["evidence_cards"]
    }
    assert expected == {c["evidence_id"] for c in payload["evidence_cards"]}
    assert all(c["task_id"] in calls for c in payload["evidence_cards"])
    assert result["control"]["status"] == "completed"


def test_worker_retries_same_task_and_recovers():
    attempts = Counter()

    def worker(task, request):
        attempts[task.task_id] += 1
        if task.task_id.startswith("r0-0") and task.retry_count == 0:
            raise RuntimeError("SECRET request payload")
        return _research_once(task, request)

    result = run(worker_service=worker)
    retried = next(r for r in result["payload"]["worker_results"] if r["retry_count"])
    assert attempts[retried["task_id"]] == 2
    assert retried["status"] == "success"
    assert result["control"]["retry_count"][retried["task_id"]] == 1
    assert result["control"]["status"] == "completed"
    assert "SECRET" not in str(result)


def test_exhausted_worker_is_excluded_while_workflow_continues():
    attempts = Counter()

    def worker(task, request):
        attempts[task.task_id] += 1
        if task.task_id.startswith("r0-0"):
            raise RuntimeError("permanent failure")
        return _research_once(task, request)

    result = run(worker_service=worker)
    failed = next(
        r for r in result["payload"]["worker_results"] if r["status"] == "failed"
    )
    assert attempts[failed["task_id"]] == 3
    assert failed["retry_count"] == 2
    assert failed["evidence_cards"] == failed["findings"] == []
    assert failed["limitations"]
    assert result["control"]["failed_tasks"] == [failed["task_id"]]
    assert result["control"]["status"] == "completed"
    assert failed["task_id"] not in {
        c["task_id"] for c in result["payload"]["evidence_cards"]
    }


def test_entire_mandatory_perspective_failure_cannot_pass():
    def worker(task, request):
        if task.perspective == "technical_maturity":
            raise RuntimeError("unavailable papers")
        return _research_once(task, request)

    result = run(worker_service=worker, limits=WorkflowLimits(max_report_revisions=0))
    assert result["control"]["status"] == "failed"
    assert (
        "technical_maturity" in result["payload"]["evaluation"]["missing_perspectives"]
    )
    assert result["control"]["termination_reason"] == "max_report_revisions"


def test_quality_pass_routes_to_end_without_revisions():
    result = run()
    assert result["control"]["decision"] == "pass"
    assert result["control"]["revision_count"] == 0
    assert result["control"]["planning_round"] == 1


@pytest.mark.parametrize("failure", ["coverage", "groundedness"])
def test_quality_fail_replans_adds_tasks_and_re_evaluates(failure):
    writer_calls = []

    def writer(state):
        writer_calls.append(state["payload"].get("evaluation"))
        update = report_writer.report_writer_agent(state)
        if len(writer_calls) == 1:
            update["payload"]["report"] = update["payload"]["report"].replace(
                "4.2 시장성" if failure == "coverage" else "# REFERENCE",
                "4.2 기타" if failure == "coverage" else "# 자료",
            )
        return update

    result = run(report_node=writer)
    assert result["control"]["status"] == "completed"
    assert result["control"]["planning_round"] == 2
    assert result["control"]["revision_count"] == 1
    assert len(result["payload"]["worker_results"]) > len(mock_plan("simple").tasks)
    assert len(writer_calls) == 2
    assert writer_calls[1]["overall_pass"] is False
    assert result["payload"]["evaluation"]["overall_pass"] is True


@pytest.mark.parametrize(
    ("flag", "syntheses", "reports"),
    [
        ("neutrality", 1, 2),
        ("bias_control", 2, 2),
    ],
)
def test_quality_fail_rewrites_or_resynthesizes_without_new_workers(
    flag, syntheses, reports
):
    from kv_cache_agent.agents.synthesis import synthesis_agent

    counts = Counter()

    def judge(state):
        counts["judge"] += 1
        update = quality_evaluator_agent(state)
        if counts["judge"] == 1:
            update["payload"]["evaluation"][flag] = False
            update["payload"]["evaluation"]["overall_pass"] = False
        return update

    def synthesize(state):
        counts["synthesis"] += 1
        return synthesis_agent(state)

    def write(state):
        counts["writer"] += 1
        return report_writer.report_writer_agent(state)

    result = run(evaluator_node=judge, synthesis_node=synthesize, report_node=write)
    assert result["control"]["status"] == "completed"
    assert counts == {"judge": 2, "synthesis": syntheses, "writer": reports}
    assert result["control"]["planning_round"] == 1
    assert len(result["payload"]["worker_results"]) == 5


@pytest.mark.parametrize(
    "limits, reason",
    [
        (WorkflowLimits(max_steps=8, max_report_revisions=100), "max_steps"),
        (WorkflowLimits(max_steps=100, max_report_revisions=2), "max_report_revisions"),
        (WorkflowLimits(max_steps=1), "max_steps"),
    ],
)
def test_continuous_quality_failure_always_terminates(limits, reason):
    def failing_judge(state):
        update = quality_evaluator_agent(state)
        update["payload"]["evaluation"].update(groundedness=False, overall_pass=False)
        return update

    result = run(evaluator_node=failing_judge, limits=limits)
    assert result["control"]["status"] in {"best_effort", "failed"}
    assert result["control"]["termination_reason"] == reason
    assert result["control"]["step_count"] <= limits.max_steps
    assert result["control"]["revision_count"] <= limits.max_report_revisions


def test_checkpoint_resume_keeps_trace_and_pending_task_ids():
    saver = InMemorySaver()
    workflow = build_workflow(
        limits=WorkflowLimits(),
        checkpointer=saver,
        interrupt_before=["research_worker"],
    )
    config = {"configurable": {"thread_id": "recovery-test"}}
    with mock_services():
        paused = workflow.invoke(INPUT, config)
        assert len(workflow.get_state(config).next) == len(paused["payload"]["tasks"])
        trace_id = paused["control"]["trace_id"]
        task_ids = {t["task_id"] for t in paused["payload"]["tasks"]}
        restored = workflow.invoke(None, config)
    assert restored["control"]["status"] == "completed"
    assert restored["control"]["trace_id"] == trace_id
    assert {r["task_id"] for r in restored["payload"]["worker_results"]} == task_ids
    assert len(restored["payload"]["worker_results"]) == len(task_ids)


def test_planner_exception_is_redacted_and_terminates():
    def fail(_state):
        raise RuntimeError("SECRET")

    result = run(planner_node=fail)
    assert result["control"]["status"] == "failed"
    assert result["control"]["last_error"] == "orchestrator:RuntimeError"
    assert "SECRET" not in str(result)


def test_state_is_compact_and_events_identify_fan_out_and_judge(caplog):
    caplog.set_level(logging.INFO, logger="kv_cache_agent.events")
    result = run()
    assert set(result) == {"payload", "control"}
    assert not any(
        s in json.dumps(result)
        for s in (
            "retrieved_chunks",
            "source_documents",
            "raw_content",
            "SystemMessage",
        )
    )
    events = [
        json.loads(r.message)
        for r in caplog.records
        if r.name == "kv_cache_agent.events"
    ]
    sends = [e for e in events if e["node"] == "dynamic_fan_out"]
    assert len(sends) == len(result["payload"]["tasks"])
    assert all(e["trace_id"] == result["control"]["trace_id"] for e in events)
    assert all(
        {
            "task_id",
            "node",
            "perspective",
            "decision",
            "decision_reason",
            "retry_count",
            "timestamp",
        }
        <= e.keys()
        for e in events
    )
    assert any(
        e["node"] == "quality_routing" and e["decision"] == "pass" for e in events
    )


@pytest.mark.parametrize(
    "mutation", ["unknown_id", "wrong_source", "unmapped", "headers_only"]
)
def test_hybrid_rules_override_permissive_llm_judge(mutation):
    result = run()
    state = deepcopy(result)
    report = state["payload"]["report"]
    if mutation == "unknown_id":
        key = state["payload"]["usable_evidence_cards"][0]["evidence_id"]
        state["payload"]["report"] = report.replace(
            f"근거 ID: {key}", "근거 ID: fabricated"
        )
    elif mutation == "wrong_source":
        url = state["payload"]["usable_evidence_cards"][0]["source_url"]
        state["payload"]["report"] = report.replace(url, "https://forged.example")
    elif mutation == "unmapped":
        state["payload"]["report"] = report.replace("[1]", "[999]", 1)
    else:
        state["payload"]["report"] = (
            "# SUMMARY\n## 기술 성숙도\n## 시장성\n## 이해관계자\n## 도메인 적용성\n## 한계\n# REFERENCE\n"
        )
        state["payload"]["usable_evidence_cards"] = []
    with mock_services():
        evaluation = quality_evaluator_agent(state)["payload"]["evaluation"]
    assert evaluation["overall_pass"] is False
    assert evaluation["groundedness"] is False
    assert evaluation["recommended_action"] == "additional_research"


def test_sqlite_recovery_after_reopening_database(tmp_path):
    sqlite = pytest.importorskip("langgraph.checkpoint.sqlite")
    db_path = str(tmp_path / "recovery.sqlite")
    config = {"configurable": {"thread_id": "restart-test"}}
    with mock_services(), sqlite.SqliteSaver.from_conn_string(db_path) as saver:
        paused = build_workflow(
            limits=WorkflowLimits(),
            checkpointer=saver,
            interrupt_before=["research_worker"],
        ).invoke(INPUT, config)
        trace_id = paused["control"]["trace_id"]
        tasks = paused["payload"]["tasks"]
    # New connection and compiled graph, as after a process restart.
    with mock_services(), sqlite.SqliteSaver.from_conn_string(db_path) as saver:
        workflow = build_workflow(limits=WorkflowLimits(), checkpointer=saver)
        result = workflow.invoke(None, config)
    assert result["control"]["trace_id"] == trace_id
    assert result["control"]["status"] == "completed"
    assert result["control"]["planning_round"] == 1
    assert {t["task_id"] for t in tasks} == {
        r["task_id"] for r in result["payload"]["worker_results"]
    }
    import sqlite3

    with sqlite3.connect(db_path) as connection:
        namespaces = connection.execute(
            "SELECT DISTINCT checkpoint_ns FROM checkpoints"
        ).fetchall()
    assert namespaces == [("",)]


def test_oversized_or_out_of_scope_plan_fails_before_fan_out(monkeypatch):
    llm = Mock()
    plan = mock_plan("공급망")
    llm.with_structured_output.return_value.invoke.return_value = plan
    monkeypatch.setattr(orchestrator, "get_llm", lambda: llm)
    with pytest.raises(ValueError, match="MAX_TASKS_PER_PLAN"):
        orchestrator_agent(initial_state(limits=WorkflowLimits(max_tasks_per_plan=4)))
    plan.tasks[0].technology = "unselected"
    with pytest.raises(ValueError, match="outside selected scope"):
        orchestrator_agent(initial_state())


@pytest.mark.parametrize("perspective", [*REQUIRED_PERSPECTIVES, "risk", "additional"])
@pytest.mark.parametrize("source", ["paper", "web", "hybrid"])
def test_generic_worker_uses_only_requested_sources(monkeypatch, perspective, source):
    from kv_cache_agent.agents import research_worker as module
    from kv_cache_agent.mock_run import _mock_retrieve
    from kv_cache_agent.schemas.tasks import SubTask

    task = SubTask(
        task_id="source-test",
        perspective=perspective,
        technology=["TurboQuant", "CXL-based"],
        objective="배정된 관점의 근거 수집",
        query="TurboQuant CXL-based evidence",
        preferred_source=source,
        priority=1,
    )
    with mock_services():
        paper = Mock(side_effect=_mock_retrieve)
        web = Mock(wraps=module.search_web)
        monkeypatch.setattr(module, "retrieve_paper_chunks", paper)
        monkeypatch.setattr(module, "search_web", web)
        result = module.research_worker(
            {
                "task": task.model_dump(),
                "trace_id": "source-test",
                "user_query": "배정된 관점 평가",
                "target_domain": "클라우드",
                "limits": {"max_worker_retries": 0, "max_cards_per_task": 24},
            }
        )["payload"]["worker_results"][0]
    assert paper.call_count == int(source in {"paper", "hybrid"})
    assert web.call_count == int(source in {"web", "hybrid"})
    assert result["status"] != "failed"
    assert result["evidence_cards"]
    assert all(
        c["perspective"] == module.LEGACY_PERSPECTIVES[perspective]
        for c in result["evidence_cards"]
    )
    assert {c["retrieval_method"] for c in result["evidence_cards"]} <= (
        {"faiss"}
        if source == "paper"
        else {"tavily"}
        if source == "web"
        else {"faiss", "tavily"}
    )


def test_scoped_task_ids_remain_valid_with_long_planner_ids(monkeypatch):
    plan = mock_plan("simple")
    plan.tasks[0].task_id = "a" * 100
    llm = Mock()
    llm.with_structured_output.return_value.invoke.return_value = plan
    monkeypatch.setattr(orchestrator, "get_llm", lambda: llm)
    result = orchestrator_agent(initial_state())
    parsed = ResearchPlan.model_validate(result["payload"]["research_plan"])
    assert all(len(task.task_id) <= 100 for task in parsed.tasks)


def test_judge_exception_never_returns_pass(monkeypatch):
    result = run()
    llm = Mock()
    llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("SECRET")
    monkeypatch.setattr(quality_evaluator, "get_llm", lambda: llm)
    evaluation = quality_evaluator_agent(result)["payload"]["evaluation"]
    assert evaluation["overall_pass"] is False
    assert all(
        not evaluation[flag]
        for flag in (
            "groundedness",
            "neutrality",
            "bias_control",
            "perspective_coverage",
        )
    )
    assert "SECRET" not in str(evaluation)
