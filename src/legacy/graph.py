"""Complete LangGraph wiring for all eight project agents."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from langgraph.graph import END, START, StateGraph

from state import AgentState

from .routing import route_after_validation
from .agent_01_supervisor import route_after_report_quality


AGENT_KEYS = {
    "technical": (
        "technical_result",
        "technical_evidence_cards",
        "technical_retrieved_context",
    ),
    "market": (
        "market_result",
        "market_evidence_cards",
        "market_retrieved_context",
    ),
    "stakeholder": (
        "stakeholder_result",
        "stakeholder_evidence_cards",
        "stakeholder_retrieved_context",
    ),
    "domain": (
        "domain_result",
        "domain_evidence_cards",
        "domain_retrieved_context",
    ),
}


def build_full_graph(
    *,
    supervisor_plan_node: Callable[[AgentState], dict[str, Any]],
    technical_node: Callable[[AgentState], dict[str, Any]],
    market_node: Callable[[AgentState], dict[str, Any]],
    stakeholder_node: Callable[[AgentState], dict[str, Any]],
    domain_node: Callable[[AgentState], dict[str, Any]],
    validation_node: Callable[[AgentState], dict[str, Any]],
    synthesis_node: Callable[[AgentState], dict[str, Any]],
    report_node: Callable[[AgentState], dict[str, Any]],
    quality_node: Callable[[AgentState], dict[str, Any]],
):
    research_nodes = {
        "technical": technical_node,
        "market": market_node,
        "stakeholder": stakeholder_node,
        "domain": domain_node,
    }

    builder = StateGraph(AgentState)
    builder.add_node("supervisor_plan", supervisor_plan_node)
    builder.add_node("technical", technical_node)
    builder.add_node("market", market_node)
    builder.add_node("stakeholder", stakeholder_node)
    builder.add_node("domain", domain_node)
    builder.add_node("join_results", join_results_node)
    builder.add_node(
        "targeted_retry",
        build_targeted_retry_node(research_nodes),
    )
    builder.add_node("validation", validation_node)
    builder.add_node("synthesis", synthesis_node)
    builder.add_node("report", report_node)
    builder.add_node("quality_review", quality_node)

    builder.add_edge(START, "supervisor_plan")
    builder.add_edge("supervisor_plan", "technical")

    # The technical evidence base is created first; the three perspective
    # agents then fan out in parallel and write separate State keys.
    builder.add_edge("technical", "market")
    builder.add_edge("technical", "stakeholder")
    builder.add_edge("technical", "domain")
    builder.add_edge(["market", "stakeholder", "domain"], "join_results")

    builder.add_edge("join_results", "validation")
    builder.add_conditional_edges(
        "validation",
        route_after_validation,
        {
            "retry": "targeted_retry",
            "synthesis": "synthesis",
            "failed": END,
        },
    )
    builder.add_edge("targeted_retry", "join_results")
    builder.add_edge("synthesis", "report")
    builder.add_edge("report", "quality_review")
    builder.add_conditional_edges(
        "quality_review",
        route_after_report_quality,
        {
            "revise": "report",
            "end": END,
            "stop": END,
        },
    )
    return builder.compile()


def join_results_node(state: AgentState) -> dict[str, Any]:
    cards: list[dict[str, Any]] = []
    status = dict(state.get("agent_status", {}))
    for agent_name, (_, evidence_key, _) in AGENT_KEYS.items():
        cards.extend(state.get(evidence_key, []) or [])
        result_key = AGENT_KEYS[agent_name][0]
        result = state.get(result_key, {}) or {}
        status[agent_name] = str(result.get("status", "completed"))
    return {
        "evidence_cards": cards,
        "agent_status": status,
    }


def build_targeted_retry_node(
    research_nodes: Mapping[str, Callable[[AgentState], dict[str, Any]]]
) -> Callable[[AgentState], dict[str, Any]]:
    """Re-run only agents requested by Agent 5, preserving unrelated results."""

    def targeted_retry_node(state: AgentState) -> dict[str, Any]:
        targets = list(dict.fromkeys(state.get("retry_targets", [])))
        counts = dict(state.get("retry_counts", {}))
        combined: dict[str, Any] = {"errors": [], "warnings": []}
        working_state = dict(state)

        for target in targets:
            node = research_nodes.get(target)
            if node is None:
                combined["errors"].append(f"알 수 없는 재조사 대상: {target}")
                continue
            update = node(working_state)
            for key, value in update.items():
                if key in {"errors", "warnings"}:
                    combined[key].extend(value)
                else:
                    combined[key] = value
                    working_state[key] = value
            counts[target] = counts.get(target, 0) + 1

        combined["retry_counts"] = counts
        combined["retry_targets"] = []
        return combined

    return targeted_retry_node
