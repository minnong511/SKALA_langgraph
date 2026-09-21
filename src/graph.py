"""All workers return to the supervisor; conditional fan-out merges separate keys."""

from __future__ import annotations

from importlib import import_module
from typing import Callable

from langgraph.graph import END, START, StateGraph

from src.common.runtime import AgentRuntime
from src.schemas import RESEARCH_AGENTS, AgentRequest, AgentResult
from src.state import AgentState

AGENTS = (*RESEARCH_AGENTS, "verification", "synthesis", "report")


def get_results(state: AgentState) -> dict[str, AgentResult]:
    return {name: state[f"{name}_result"] for name in AGENTS if state.get(f"{name}_result") is not None}


def initial_state(request: AgentRequest, config: dict | None = None) -> AgentState:
    return {
        "request": request,
        "config": config or {},
        "phase": "start",
        "status": "running",
        "plan": {},
        "pending_tasks": [],
        "attempts": {},
        "retry_counts": {},
        "feedback": {},
        "review_history": [],
        "final_artifacts": {},
        "research_round": 0,
        "review": {},
    }


def build_graph(runtime: AgentRuntime, agents: dict[str, Callable] | None = None):
    implementations = agents or {
        name: import_module(f"src.agents.{name}").run for name in ("supervisor", *AGENTS)
    }
    builder = StateGraph(AgentState)

    def supervisor_node(state):
        attempts = dict(state.get("attempts", {}))
        attempts["supervisor"] = attempts.get("supervisor", 0) + 1
        context_data = {
            **state["request"].context,
            **{key: state.get(key) for key in ("phase", "attempts", "feedback", "plan", "review")},
        }
        context_data["budget_exhausted"] = bool(runtime.context.budget and runtime.context.budget.exhausted)
        request = state["request"].model_copy(
            update={"task_id": "supervisor", "attempt": attempts["supervisor"], "context": context_data}
        )
        if context_data["budget_exhausted"]:
            decision = AgentResult(
                task_id="supervisor",
                agent="supervisor",
                attempt=request.attempt,
                status="partial",
                summary="전체 실행 한도 도달",
                data={"targets": [], "phase": "done", "run_status": "needs_review"},
            )
            path = runtime.artifacts.save_result(decision)
            runtime.context.events.emit(
                "limit_reached",
                decision.summary,
                task_id="supervisor",
                agent="supervisor",
                attempt=request.attempt,
                path=str(path),
            )
        else:
            decision = runtime.execute(
                "supervisor", implementations["supervisor"], request, get_results(state)
            )
        if decision.status == "failed":
            data = {
                "targets": [],
                "phase": "done",
                "run_status": "failed",
                "review": {"issues": decision.errors},
            }
        else:
            data = decision.data
        targets = data.get("targets", [])
        if any(name not in AGENTS for name in targets):
            raise ValueError("슈퍼바이저가 알 수 없는 노드를 배정했습니다.")
        for target in targets:
            attempts[target] = attempts.get(target, 0) + 1
        feedback = {**state.get("feedback", {}), **data.get("feedback", {})}
        update = {
            "attempts": attempts,
            "retry_counts": {k: max(v - 1, 0) for k, v in attempts.items() if k != "supervisor"},
            "pending_tasks": targets,
            "phase": data.get("phase", "done"),
            "status": data.get("run_status", "running"),
            "feedback": feedback,
            "plan": data.get("plan", state.get("plan", {})),
            "review": data.get("review", state.get("review", {})),
            "review_history": [
                *state.get("review_history", []),
                {"phase": state.get("phase"), "reason": decision.summary, "targets": targets},
            ],
            "budget_exhausted": context_data["budget_exhausted"],
        }
        if "technical" in targets and attempts["technical"] > 1:
            # A revised technical basis invalidates all dependent evaluations and downstream work.
            update.update(
                {
                    f"{name}_result": None
                    for name in ("market", "stakeholder", "domain", "verification", "synthesis", "report")
                }
            )
        if targets and all(name in ("market", "stakeholder", "domain") for name in targets):
            update["research_round"] = state.get("research_round", 0) + 1
            runtime.start_round(targets, update["research_round"])
        runtime.context.events.emit(
            "review_complete",
            decision.summary,
            agent="supervisor",
            task_id="supervisor",
            attempt=request.attempt,
            targets=targets,
            status=update["status"],
        )
        return update

    builder.add_node("supervisor", supervisor_node)
    for name in AGENTS:

        def worker(state, name=name):
            request = state["request"].model_copy(
                update={
                    "task_id": name,
                    "attempt": state["attempts"][name],
                    "questions": state.get("plan", {})
                    .get("questions", {})
                    .get(name, state["request"].questions),
                    "feedback": state.get("feedback", {}).get(name, []),
                    "context": {**state["request"].context, "plan": state.get("plan", {})},
                }
            )
            result = runtime.execute(name, implementations[name], request, get_results(state))
            return {f"{name}_result": result}

        builder.add_node(name, worker)
        builder.add_edge(name, "supervisor")
    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        lambda state: state["pending_tasks"] or END,
        {name: name for name in AGENTS} | {END: END},
    )
    return builder.compile()


build_full_graph = build_graph
