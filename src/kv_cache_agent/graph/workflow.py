"""Runtime Orchestrator-Workers graph with bounded research/revision loops."""

from dataclasses import asdict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from kv_cache_agent.agents.orchestrator import orchestrator_agent
from kv_cache_agent.agents.quality_evaluator import quality_evaluator_agent
from kv_cache_agent.agents.report_writer import report_writer_agent
from kv_cache_agent.agents.research_worker import research_worker
from kv_cache_agent.agents.synthesis import synthesis_agent
from kv_cache_agent.agents.verifier import evidence_verification_agent
from kv_cache_agent.config import WorkflowLimits
from kv_cache_agent.graph.routing import (
    assign_workers,
    route_after_quality_check,
    route_next,
)
from kv_cache_agent.graph.state import GlobalState
from kv_cache_agent.observability import node_span, record_event
from kv_cache_agent.schemas.evaluation import QualityEvaluation
from kv_cache_agent.schemas.tasks import LEGACY_PERSPECTIVES, REQUIRED_PERSPECTIVES


def _initialize(state, limits):
    payload = state.get("payload", {})
    control = state.get("control", {})
    query = payload.get("user_query", state.get("user_query", "")).strip()
    if not query:
        raise ValueError("user_query must not be blank")
    technologies = payload.get("selected_technologies", ["TurboQuant", "CXL-based"])
    if set(technologies) != {"TurboQuant", "CXL-based"}:
        raise ValueError("selected_technologies must contain TurboQuant and CXL-based")
    return {
        "payload": {
            "user_query": query,
            "selected_technologies": technologies,
            "target_domain": payload.get("target_domain", "클라우드 LLM 서빙"),
        },
        "control": {
            "trace_id": control.get("trace_id") or str(uuid4()),
            "step_count": 0,
            "revision_count": 0,
            "planning_round": 0,
            "task_status": {},
            "retry_count": {},
            "failed_tasks": [],
            "last_error": None,
            "status": "planning",
            "decision": "plan",
            "termination_reason": None,
            "limits": asdict(limits),
            **control,
        },
    }


def _bounded_node(node, name):
    def run(state: GlobalState):
        control = state["control"]
        if control["step_count"] >= control["limits"]["max_steps"]:
            return {
                "control": {
                    **control,
                    "status": "stopping",
                    "termination_reason": "max_steps",
                }
            }
        with node_span(
            control["trace_id"],
            name,
            revision_count=control["revision_count"],
            step_count=control["step_count"] + 1,
        ):
            try:
                update = node(state)
            except Exception as error:  # noqa: BLE001 - bounded node boundary
                update = {
                    "control": {
                        **control,
                        "status": "stopping",
                        "last_error": f"{name}:{type(error).__name__}",
                        "termination_reason": f"{name}_failed",
                    }
                }
            updated_control = {
                **control,
                "decision": "next",
                "decision_reason": "",
                **update.get("control", {}),
                "step_count": control["step_count"] + 1,
            }
            update["control"] = updated_control
            record_event(
                control["trace_id"],
                name,
                updated_control.get("decision", "next"),
                updated_control.get("decision_reason", ""),
                step_count=updated_control["step_count"],
                revision_count=updated_control["revision_count"],
            )
            return update

    # Do not copy a legacy node's input annotation: LangGraph would register
    # its fixed result fields as channels and filter out the dynamic payload.
    run.__name__ = name
    return run


def _reduce_results(state):
    control = state["control"]
    results = state["payload"].get("worker_results", [])
    statuses = {
        **control["task_status"],
        **{r["task_id"]: r["status"] for r in results},
    }
    retries = {r["task_id"]: r["retry_count"] for r in results}
    failed = [r["task_id"] for r in results if r["status"] == "failed"]
    limitations = list(dict.fromkeys(s for r in results for s in r["limitations"]))
    return {
        "payload": {"limitations": limitations},
        "control": {
            **control,
            "status": "verifying",
            "task_status": statuses,
            "retry_count": retries,
            "failed_tasks": failed,
            "decision": "verify",
            "decision_reason": "parallel batch reduced",
        },
    }


def _evaluate(state, evaluator):
    update = evaluator(state)
    evaluation = QualityEvaluation.model_validate(update["payload"]["evaluation"])
    covered = {
        c.get("perspective") for c in state["payload"].get("usable_evidence_cards", [])
    }
    missing = [
        p for p in REQUIRED_PERSPECTIVES if LEGACY_PERSPECTIVES[p] not in covered
    ]
    if missing:
        evaluation.perspective_coverage = False
        evaluation.missing_perspectives = list(
            dict.fromkeys(
                [
                    *evaluation.missing_perspectives,
                    *missing,
                ]
            )
        )
        evaluation.coverage_reason += "; mandatory perspective evidence missing"
    # A custom judge cannot bypass fail flags by returning overall_pass=True.
    evaluation.overall_pass = all(
        (
            evaluation.groundedness,
            evaluation.neutrality,
            evaluation.bias_control,
            evaluation.perspective_coverage,
        )
    )
    control = dict(state["control"])
    reason = (
        f"{evaluation.groundedness_reason}; {evaluation.neutrality_reason}; "
        f"{evaluation.bias_reason}; {evaluation.coverage_reason}"
    )[:3000]
    if evaluation.overall_pass:
        decision = "pass"
        control["termination_reason"] = "quality_pass"
    elif control["step_count"] + 1 >= control["limits"]["max_steps"]:
        decision = "stop"
        control["termination_reason"] = "max_steps"
    elif control["revision_count"] >= control["limits"]["max_report_revisions"]:
        decision = "stop"
        control["termination_reason"] = "max_report_revisions"
    else:
        if not evaluation.groundedness or not evaluation.perspective_coverage:
            decision = "additional_research"
        elif not evaluation.bias_control:
            decision = (
                "additional_research"
                if evaluation.recommended_action == "additional_research"
                else "resynthesis"
            )
        else:
            decision = "rewrite"
        control["revision_count"] += 1
    evaluation.recommended_action = (
        "pass"
        if decision == "pass"
        else (decision if decision != "stop" else evaluation.recommended_action)
    )
    control.update(
        decision=decision,
        decision_reason=reason,
        status="stopping" if decision == "stop" else "evaluating",
    )
    record_event(
        control["trace_id"],
        "quality_routing",
        decision,
        reason,
        revision_count=control["revision_count"],
    )
    return {"payload": {"evaluation": evaluation.model_dump()}, "control": control}


def _finalize(state):
    payload, control = state["payload"], dict(state["control"])
    if control.get("termination_reason") == "quality_pass":
        status = "completed"
    else:
        covered = {
            c.get("perspective") for c in payload.get("usable_evidence_cards", [])
        }
        mandatory = {LEGACY_PERSPECTIVES[p] for p in REQUIRED_PERSPECTIVES}
        valid_report = payload.get("report", "").startswith("# SUMMARY")
        status = "best_effort" if valid_report and mandatory <= covered else "failed"
        control["termination_reason"] = (
            control.get("termination_reason") or "incomplete_research"
        )
    control["status"] = status
    record_event(control["trace_id"], "finalize", status, control["termination_reason"])
    update = {"control": control}
    if status != "completed" and payload.get("report"):
        update["payload"] = {
            "report": payload["report"]
            + (
                f"\n한계: 실행 상태={status}, 종료 이유={control['termination_reason']}. "
                "품질 평가를 통과하지 못한 보고서입니다.\n"
            )
        }
    return update


def build_workflow(
    *,
    limits: WorkflowLimits | None = None,
    planner_node=None,
    worker_service=None,
    verification_node=None,
    synthesis_node=None,
    report_node=None,
    evaluator_node=None,
    checkpointer=None,
    interrupt_before=None,
):
    """Inject services for offline tests; production defaults use preserved RAG.

    A checkpointer plus stable configurable.thread_id enables invoke(None) resume.
    MAX_STEPS counts sequential controller nodes (Worker attempts have their own
    retry bound). Every parallel batch converges at reduce_results exactly once.
    """
    limits = limits or WorkflowLimits.from_env()
    graph = StateGraph(GlobalState)

    def initialize(state: GlobalState):
        return _initialize(state, limits)

    def worker(request):
        return research_worker(request, researcher=worker_service)

    def evaluate(state: GlobalState):
        return _evaluate(state, evaluator_node or quality_evaluator_agent)

    graph.add_node("initialize", initialize)
    graph.add_node(
        "orchestrator",
        _bounded_node(planner_node or orchestrator_agent, "orchestrator"),
    )
    graph.add_node("research_worker", worker)
    graph.add_node("reduce_results", _bounded_node(_reduce_results, "reduce_results"))
    graph.add_node(
        "verifier",
        _bounded_node(verification_node or evidence_verification_agent, "verifier"),
    )
    graph.add_node(
        "synthesis", _bounded_node(synthesis_node or synthesis_agent, "synthesis")
    )
    graph.add_node(
        "report_writer",
        _bounded_node(report_node or report_writer_agent, "report_writer"),
    )
    graph.add_node("quality_evaluator", _bounded_node(evaluate, "quality_evaluator"))
    graph.add_node("finalize", _finalize)
    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "orchestrator")
    graph.add_conditional_edges(
        "orchestrator",
        assign_workers,
        ["research_worker", "reduce_results", "finalize"],
    )
    graph.add_edge("research_worker", "reduce_results")
    for source, target in (
        ("reduce_results", "verifier"),
        ("verifier", "synthesis"),
        ("synthesis", "report_writer"),
        ("report_writer", "quality_evaluator"),
    ):
        graph.add_conditional_edges(
            source, route_next, {"next": target, "stop": "finalize"}
        )
    graph.add_conditional_edges(
        "quality_evaluator",
        route_after_quality_check,
        {
            "pass": "finalize",
            "stop": "finalize",
            "additional_research": "orchestrator",
            "resynthesis": "synthesis",
            "rewrite": "report_writer",
        },
    )
    graph.add_edge("finalize", END)
    # Ensure LangGraph's defensive recursion bound exceeds our controller budget.
    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before,
        name="kv_cache_orchestrator_workers",
    ).with_config(
        {
            "recursion_limit": limits.max_steps * 3 + 10,
        }
    )
