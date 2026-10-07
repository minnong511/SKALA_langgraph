from collections.abc import Callable
from typing import Any

from langgraph.graph import END, START, StateGraph

from kv_cache_agent.agents.cloud_domain import cloud_domain_agent
from kv_cache_agent.agents.market import market_evaluation_agent
from kv_cache_agent.agents.report_writer import report_writer_agent
from kv_cache_agent.agents.stakeholder import stakeholder_evaluation_agent
from kv_cache_agent.agents.supervisor import supervisor_agent
from kv_cache_agent.agents.synthesis import synthesis_agent
from kv_cache_agent.agents.technical import technical_research_agent
from kv_cache_agent.agents.verifier import evidence_verification_agent
from kv_cache_agent.graph.state import GlobalState
from kv_cache_agent.observability.nodes import add_logged_node


def build_workflow(
    technical_node: Callable[[GlobalState], dict[str, Any]] | None = None,
):
    """Supervisor 기반 전체 실행 그래프를 생성한다.

    현재 작업자 노드는 가벼운 Stub으로 연결되어 있으며, 각 에이전트의
    내부 RAG와 평가 로직은 이 오케스트레이션 그래프를 바꾸지 않고 구현한다.
    """
    graph = StateGraph(GlobalState)

    add_logged_node(graph, "supervisor", supervisor_agent)
    # 테스트에서는 실제 BGE-M3를 로드하지 않도록 기술 노드를 주입할 수 있다.
    add_logged_node(graph, "technical", technical_node or technical_research_agent)
    add_logged_node(graph, "market", market_evaluation_agent)
    add_logged_node(graph, "stakeholder", stakeholder_evaluation_agent)
    add_logged_node(graph, "cloud_domain", cloud_domain_agent)
    add_logged_node(graph, "verifier", evidence_verification_agent)
    add_logged_node(graph, "synthesis", synthesis_agent)
    add_logged_node(graph, "report_writer", report_writer_agent)

    graph.add_edge(START, "supervisor")
    graph.add_edge("supervisor", "technical")

    # 공통 기술 조사 결과가 준비되면 세 관점 평가를 병렬로 실행한다.
    graph.add_edge("technical", "market")
    graph.add_edge("technical", "stakeholder")
    graph.add_edge("technical", "cloud_domain")

    # 세 시작 노드를 지정하면 세 분기가 모두 끝난 뒤 검증기가 실행된다.
    graph.add_edge(["market", "stakeholder", "cloud_domain"], "verifier")
    graph.add_edge("verifier", "synthesis")
    graph.add_edge("synthesis", "report_writer")
    graph.add_edge("report_writer", END)

    return graph.compile()


def build_supervised_workflow(**dependencies):
    """Build the stages 9–10 loop; the CLI switches after reporting migration."""
    from kv_cache_agent.graph.supervised import SupervisedWorkflow

    return SupervisedWorkflow(**dependencies)
