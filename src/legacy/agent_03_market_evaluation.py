"""Agent 3: market, adoption, and ecosystem evaluation."""

from typing import Any

from .research_base import build_research_node


ROLE_PROMPT = """
당신은 시장 평가 에이전트다.
TurboQuant와 ITME의 시장 수요, 제품화, 실제 채택, 생태계 지원, 도입 비용을 조사한다.
AI 추론·양자화·CXL 시장은 직접 시장 규모가 아니라 보조 지표임을 명시한다.
일반적인 양자화 지원을 TurboQuant 직접 지원으로, CXL 장비 출시를 ITME 채택으로 바꾸어 쓰지 않는다.
제품 발표, 공식 문서, 고객 사례를 우선하고 기사나 시장 보고서는 날짜와 시장 정의를 기록한다.
""".strip()


DEFAULT_QUERIES = [
    "TurboQuant KV cache commercial adoption serving framework support",
    "KV cache quantization cloud inference cost adoption ecosystem",
    "ITME CXL-Hybrid memory product adoption cloud LLM inference",
    "CXL memory server ecosystem cloud AI infrastructure adoption",
    "long context LLM inference memory demand cloud market",
]


def build_market_agent_node(llm: Any, web_search: Any, retriever: Any = None):
    return build_research_node(
        llm=llm,
        agent_name="market",
        result_key="market_result",
        evidence_key="market_evidence_cards",
        context_key="market_retrieved_context",
        role_prompt=ROLE_PROMPT,
        default_queries=DEFAULT_QUERIES,
        retriever=retriever,
        web_search=web_search,
    )
