"""Agent 4: stakeholder positions and adoption barriers."""

from typing import Any

from .research_base import build_research_node


ROLE_PROMPT = """
당신은 이해관계자 평가 에이전트다.
클라우드 사업자, 모델·서빙 개발자, GPU·CPU·메모리·서버 공급자,
도입 고객 및 투자·산업 분석 관점의 기대 효과와 부담을 조사한다.
직접 발언과 조사자의 해석을 구분하고, 경쟁 기술의 존재를 경쟁사의 반응으로 오해하지 않는다.
직접적인 TurboQuant 또는 ITME 의견이 없으면 관련 산업 전반의 의견이라고 표시하고 특정 입장을 추정하지 않는다.
""".strip()


DEFAULT_QUERIES = [
    "cloud provider developer views KV cache quantization accuracy deployment barriers",
    "LLM serving developer KV cache compression integration maintenance concerns",
    "cloud operator CXL memory expansion cost latency operational complexity",
    "GPU CPU memory vendors CXL strategy AI inference memory",
    "investment analyst view inference optimization memory expansion AI infrastructure",
]


def build_stakeholder_agent_node(llm: Any, web_search: Any):
    return build_research_node(
        llm=llm,
        agent_name="stakeholder",
        result_key="stakeholder_result",
        evidence_key="stakeholder_evidence_cards",
        context_key="stakeholder_retrieved_context",
        role_prompt=ROLE_PROMPT,
        default_queries=DEFAULT_QUERIES,
        web_search=web_search,
    )
