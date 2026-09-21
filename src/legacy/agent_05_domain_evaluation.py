"""Cloud LLM-serving domain-applicability Agent."""

from typing import Any

from .research_base import build_research_node


ROLE_PROMPT = """
당신은 도메인 평가 에이전트다.
다수 사용자의 동시 요청과 장문맥을 처리하는 클라우드 LLM 서빙 환경에서
TurboQuant와 ITME의 GPU 메모리 효율, 처리량, TTFT, TPOT, 출력 품질,
비용, 도입 복잡도, 안정성 및 확장성을 평가한다.
논문별 모델, GPU, 메모리 구성, 문맥 길이, 동시 요청, 데이터셋과 baseline을 기록한다.
조건이 다른 수치를 직접 비교하거나 온디바이스 결과를 클라우드에 일반화하지 않는다.
""".strip()


DEFAULT_QUERIES = [
    "TurboQuant cloud LLM serving memory throughput latency quality conditions",
    "ITME cloud LLM serving TTFT throughput memory capacity latency conditions",
    "KV cache quantization multi tenant long context serving operational cost",
    "CXL tiered memory multi tenant long context LLM serving operational complexity",
    "TurboQuant ITME deployment requirements compatibility serving engine",
]


def build_domain_agent_node(
    llm: Any,
    retriever: Any,
    web_search: Any = None,
):
    return build_research_node(
        llm=llm,
        agent_name="domain",
        result_key="domain_result",
        evidence_key="domain_evidence_cards",
        context_key="domain_retrieved_context",
        role_prompt=ROLE_PROMPT,
        default_queries=DEFAULT_QUERIES,
        retriever=retriever,
        web_search=web_search,
    )
