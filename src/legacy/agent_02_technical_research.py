"""Agent 2: primary-paper RAG research for TurboQuant and ITME."""

from typing import Any

from .research_base import build_research_node


ROLE_PROMPT = """
당신은 기술 조사 에이전트다.
TurboQuant와 CXL-Hybrid 메모리 기반 ITME의 원문을 조사해
기술 원리, 구현 범위, 성능 주장, 실험 조건, 한계 및 공개 정보 기반 TRL 근거를 추출한다.
일반 양자화 또는 일반 CXL의 성숙도를 선택 기술 자체의 성숙도로 대체하지 않는다.
TurboQuant의 비트 수와 ITME의 처리량 수치는 baseline과 실험 시나리오를 함께 기록한다.
""".strip()


DEFAULT_QUERIES = [
    "TurboQuant KV cache quantization mechanism distortion guarantee experiments limitations",
    "TurboQuant LongBench needle in a haystack bit width baseline experimental conditions",
    "ITME CXL-Hybrid memory architecture KV cache prefetch prototype limitations",
    "ITME throughput baseline ShareGPT FPGA prototype evaluation conditions",
    "TurboQuant ITME TRL evidence code prototype production deployment",
]


def build_technical_agent_node(llm: Any, retriever: Any):
    return build_research_node(
        llm=llm,
        agent_name="technical",
        result_key="technical_result",
        evidence_key="technical_evidence_cards",
        context_key="technical_retrieved_context",
        role_prompt=ROLE_PROMPT,
        default_queries=DEFAULT_QUERIES,
        retriever=retriever,
    )
