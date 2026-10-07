"""Supervisor 중심의 상위 LangGraph.

관점별 에이전트는 서로 직접 연결되지 않고 매번 Supervisor로 결과를 돌려준다.
하위 에이전트 내부의 기존 LangGraph 도식은 변경하지 않는다.
"""

from collections.abc import Callable
from copy import deepcopy
from typing import Any

from langgraph.graph import END, START, StateGraph

from kv_cache_agent.agents.cloud_domain import cloud_domain_agent
from kv_cache_agent.agents.market import market_evaluation_agent
from kv_cache_agent.agents.report_quality import evaluate_report_quality
from kv_cache_agent.agents.report_writer import report_writer_agent
from kv_cache_agent.agents.stakeholder import stakeholder_evaluation_agent
from kv_cache_agent.agents.supervisor import supervisor_agent
from kv_cache_agent.agents.synthesis import synthesis_agent
from kv_cache_agent.agents.technical import technical_research_agent
from kv_cache_agent.agents.verifier import evidence_verification_agent
from kv_cache_agent.graph.state import GlobalState
from kv_cache_agent.tools.source_metadata import normalize_source

REWORK_QUERIES = {
    "technical": [
        "TurboQuant KV cache quantization limitations experiments",
        "ITME CXL hybrid memory FPGA evaluation limitations",
    ],
    "market": [
        "TurboQuant KV cache deployment adoption limitations",
        "CXL memory expansion cloud adoption cost barriers",
    ],
    "stakeholder": [
        "TurboQuant KV cache Google developer quality concerns",
        "CXL memory expansion SK hynix cloud operators deployment concerns",
    ],
    "cloud_domain": [
        "TurboQuant KV cache long context inference quality latency",
        "ITME CXL KV cache concurrent inference latency limitations",
    ],
}

Node = Callable[[GlobalState], dict[str, Any]]
WORKER_NODES: dict[str, Node] = {
    "technical": technical_research_agent,
    "market": market_evaluation_agent,
    "stakeholder": stakeholder_evaluation_agent,
    "cloud_domain": cloud_domain_agent,
}


def _merge_cards(previous: list[dict], incoming: list[dict]) -> list[dict]:
    """재조사한 카드가 같은 ID이면 교체하여 State의 무한 증가를 방지한다."""
    merged = {
        str(card.get("evidence_id", f"old-{index}")): card
        for index, card in enumerate(previous)
    }
    for index, card in enumerate(incoming):
        key = str(card.get("evidence_id") or f"new-{len(merged) + index}")
        merged[key] = card
    return list(merged.values())


def _worker_node(name: str, node: Node) -> Node:
    def run(state: GlobalState) -> dict[str, Any]:
        # 부족 사유는 검색 질의가 아니다. 기술명을 포함한 재조사 질의와 분리한다.
        worker_input = deepcopy(state)
        gaps = state.get("control", {}).get("gap_requests", {}).get(name, [])
        if gaps:
            plan = deepcopy(worker_input.get("research_plan", {}))
            queries = dict(plan.get("search_questions", {}))
            queries[name] = list(
                dict.fromkeys([*REWORK_QUERIES[name], *queries.get(name, [])])
            )
            plan["search_questions"] = queries
            worker_input["research_plan"] = plan
        try:
            output = node(worker_input)
        except Exception as error:  # noqa: BLE001 - 다른 관점은 계속 평가할 수 있어야 한다.
            output = {
                f"{name}_result": {
                    "agent_name": name,
                    "status": "failed",
                    "summary": "담당 작업 실패",
                    "evidence_ids": [],
                    "limitations": [],
                    "errors": [f"{type(error).__name__}: {error}"],
                    "payload": {},
                },
                "evidence_cards": [],
            }
        control = deepcopy(state.get("control", {}))
        attempts = dict(control.get("retry_count", {}))
        attempts[name] = attempts.get(name, 0) + 1
        control["retry_count"] = attempts
        control["evidence_revision"] = control.get("evidence_revision", 0) + 1
        result_status = output.get(f"{name}_result", {}).get("status", "failed")
        control["node_status"] = {**control.get("node_status", {}), name: result_status}
        if result_status == "failed":
            error = (
                "; ".join(output.get(f"{name}_result", {}).get("errors", []))
                or f"{name} failed"
            )
            control["last_error"] = error
            control["errors"] = [*control.get("errors", []), error][-8:]
        if "evidence_cards" in output:
            # 재조사는 해당 관점의 이전 카드를 대체한다. 오래된 카드가 남아
            # 최신 평가에 섞이거나 State가 계속 커지는 일을 방지한다.
            output["evidence_cards"] = _merge_cards(
                [
                    c
                    for c in state.get("evidence_cards", [])
                    if c.get("perspective") != name
                ],
                [normalize_source(c) for c in output.get("evidence_cards", [])],
            )
        return {**output, "control": control}

    return run


def _verifier_node(node: Node) -> Node:
    def run(state: GlobalState) -> dict[str, Any]:
        try:
            output = node(state)
        except Exception as error:  # noqa: BLE001 - 실패도 Supervisor의 판단 입력이다.
            output = {
                "verification_result": {
                    "agent_name": "verifier",
                    "status": "failed",
                    "summary": "근거 검증 실패",
                    "limitations": [],
                    "errors": [f"{type(error).__name__}: {error}"],
                    "payload": {},
                },
                "verified_evidence_cards": [],
                "usable_evidence_cards": [],
            }
        control = deepcopy(state.get("control", {}))
        control["verified_revision"] = control.get("evidence_revision", 0)
        control["node_status"] = {
            **control.get("node_status", {}),
            "verifier": output.get("verification_result", {}).get("status", "failed"),
        }
        if control["node_status"]["verifier"] == "failed":
            error = (
                "; ".join(output.get("verification_result", {}).get("errors", []))
                or "verifier failed"
            )
            control["last_error"] = error
            control["errors"] = [*control.get("errors", []), error][-8:]
        return {**output, "control": control}

    return run


def _synthesis_node(node: Node) -> Node:
    def run(state: GlobalState) -> dict[str, Any]:
        output = node(state)
        control = deepcopy(state.get("control", {}))
        control["node_status"] = {
            **control.get("node_status", {}),
            "synthesis": output.get("synthesis_result", {}).get("status", "failed"),
        }
        return {**output, "control": control}

    return run


def _writer_node(node: Node) -> Node:
    def run(state: GlobalState) -> dict[str, Any]:
        output = node(state)
        control = deepcopy(state.get("control", {}))
        control["report_revision"] = control.get("report_revision", 0) + 1
        control["node_status"] = {
            **control.get("node_status", {}),
            "report_writer": "completed" if output.get("final_report") else "failed",
        }
        return {**output, "control": control}

    return run


def _route_supervisor(state: GlobalState) -> str:
    """분기 함수에는 모델 호출을 넣지 않는다. 결정된 허용 대상만 반환한다."""
    allowed = {*WORKER_NODES, "verifier", "synthesis", "report_writer", "end"}
    choice = state.get("control", {}).get("next_agent", "end")
    return choice if choice in allowed else "end"


def _route_quality(state: GlobalState) -> str:
    verdict = state.get("quality_result", {})
    control = state.get("control", {})
    if verdict.get("status") == "revise" and control.get(
        "report_revision", 0
    ) < control.get("max_report_revisions", 2):
        return "rewrite"
    return "end"


def build_workflow(
    technical_node: Node | None = None,
    *,
    node_overrides: dict[str, Node] | None = None,
):
    """테스트에서는 노드를 주입하고, 실제 실행에서는 기존 에이전트를 사용한다."""
    overrides = node_overrides or {}
    workers = {
        **WORKER_NODES,
        **{name: overrides[name] for name in WORKER_NODES if name in overrides},
    }
    if technical_node is not None:
        workers["technical"] = technical_node

    graph = StateGraph(GlobalState)
    graph.add_node("supervisor", overrides.get("supervisor", supervisor_agent))
    for name, node in workers.items():
        graph.add_node(name, _worker_node(name, node))
        graph.add_edge(name, "supervisor")
    graph.add_node(
        "verifier",
        _verifier_node(overrides.get("verifier", evidence_verification_agent)),
    )
    graph.add_node(
        "synthesis", _synthesis_node(overrides.get("synthesis", synthesis_agent))
    )
    graph.add_node(
        "report_writer",
        _writer_node(overrides.get("report_writer", report_writer_agent)),
    )
    graph.add_node(
        "quality_eval", overrides.get("quality_eval", evaluate_report_quality)
    )

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        _route_supervisor,
        {
            **{name: name for name in workers},
            "verifier": "verifier",
            "synthesis": "synthesis",
            "report_writer": "report_writer",
            "end": END,
        },
    )
    graph.add_edge("verifier", "supervisor")
    graph.add_edge("synthesis", "supervisor")
    graph.add_edge("report_writer", "quality_eval")
    graph.add_conditional_edges(
        "quality_eval", _route_quality, {"rewrite": "report_writer", "end": END}
    )
    # 한 번의 Supervisor 단계에는 워커/검증기 노드가 동반된다.
    # LangGraph 기본 재귀 상한보다 넉넉하게 잡되, 실제 종료는 State의 step_count가 보장한다.
    return graph.compile().with_config({"recursion_limit": 80})
