"""논문 FAISS 검색 결과를 기술 근거로 변환하는 에이전트."""

from pathlib import Path
from typing import Any

import yaml
from langchain_core.messages import HumanMessage, SystemMessage

from kv_cache_agent.config import OPENAI_API_KEY
from kv_cache_agent.graph.state import GlobalState
from kv_cache_agent.llm import get_llm
from kv_cache_agent.schemas.technical import TechnicalExtraction
from kv_cache_agent.tools.paper_retriever import retrieve_paper_chunks

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "technical.yaml"

DEFAULT_TECHNICAL_QUERIES = [
    "TurboQuant KV Cache compression mechanism and memory reduction",
    "CXL-based KV Cache memory expansion and access mechanism",
    "KV Cache bottleneck addressed by TurboQuant and CXL-based approaches",
    "performance benefits and limitations of TurboQuant KV Cache",
    "performance benefits and limitations of CXL-based KV Cache",
    "hardware and software requirements for both approaches",
]


def _load_system_prompt() -> str:
    """YAML 파일에서 기술 조사 시스템 프롬프트를 읽는다."""
    with PROMPT_PATH.open(encoding="utf-8") as prompt_file:
        prompt_config = yaml.safe_load(prompt_file) or {}
    return str(prompt_config.get("system_prompt", ""))


def _build_queries(state: GlobalState) -> list[str]:
    """ResearchPlan의 질문을 사용하고 없으면 기본 질문을 사용한다."""
    research_plan = state.get("research_plan", {})
    search_questions = research_plan.get("search_questions", {})
    planned_queries = search_questions.get("technical", [])

    if planned_queries:
        return planned_queries
    return DEFAULT_TECHNICAL_QUERIES


def _build_context(retrieved_chunks: list[dict[str, Any]]) -> str:
    """검색 chunk를 LLM이 출처를 추적할 수 있는 문맥으로 만든다."""
    context_parts: list[str] = []
    for chunk in retrieved_chunks:
        metadata = chunk["metadata"]
        context_parts.append(
            "\n".join(
                [
                    f"[source_chunk_id={chunk['chunk_id']}]",
                    f"source_title={metadata.get('source_title', '')}",
                    f"source_locator={metadata.get('source_locator', '')}",
                    f"content={chunk['content']}",
                ]
            )
        )
    return "\n\n---\n\n".join(context_parts)


def _extract_findings(
    state: GlobalState,
    retrieved_chunks: list[dict[str, Any]],
) -> TechnicalExtraction:
    """검색된 논문 문맥을 GPT-4o-mini의 구조화 출력으로 변환한다."""
    llm = get_llm().with_structured_output(TechnicalExtraction)
    user_query = state.get("user_query", "")
    prompt = (
        f"사용자 질문: {user_query}\n\n"
        "아래 논문 검색 결과만 사용하여 기술 조사 결과를 작성하라.\n"
        "각 finding에는 실제 검색 결과의 source_chunk_id를 반드시 넣어라.\n\n"
        f"논문 검색 결과:\n{_build_context(retrieved_chunks)}"
    )
    response = llm.invoke(
        [
            SystemMessage(content=_load_system_prompt()),
            HumanMessage(content=prompt),
        ]
    )

    if isinstance(response, TechnicalExtraction):
        return response
    return TechnicalExtraction.model_validate(response)


def _build_evidence_cards(
    extraction: TechnicalExtraction,
    retrieved_chunks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """LLM 결과를 출처가 연결된 EvidenceCard로 변환한다."""
    chunk_by_id = {chunk["chunk_id"]: chunk for chunk in retrieved_chunks}
    cards: list[dict[str, Any]] = []
    missing_source_findings: list[str] = []

    for finding_number, finding in enumerate(extraction.findings, start=1):
        source_chunks = [
            chunk_by_id[chunk_id]
            for chunk_id in finding.source_chunk_ids
            if chunk_id in chunk_by_id
        ]
        if not source_chunks:
            missing_source_findings.append(finding.claim)
            continue

        source_titles = list(
            dict.fromkeys(
                str(chunk["metadata"].get("source_title", ""))
                for chunk in source_chunks
            )
        )
        source_locators = list(
            dict.fromkeys(
                str(chunk["metadata"].get("source_locator", ""))
                for chunk in source_chunks
            )
        )
        first_metadata = source_chunks[0]["metadata"]
        technology_slug = finding.technology.lower().replace("-", "_")
        evidence_id = f"technical-{technology_slug}-{finding_number:03d}"

        cards.append(
            {
                "evidence_id": evidence_id,
                "technology": finding.technology,
                "perspective": "technical",
                "claim": finding.claim,
                "evidence_text": finding.evidence_text,
                "source_title": "; ".join(source_titles),
                "source_url": str(
                    first_metadata.get("source_url")
                    or first_metadata.get("source_path", "")
                ),
                "source_type": "paper",
                "source_locator": "; ".join(source_locators),
                "retrieval_method": "faiss",
                "published_date": str(first_metadata.get("published_date", "")),
                "claim_type": finding.claim_type,
                "confidence": finding.confidence,
                "caveat": finding.caveat,
                "verification_status": "unverified",
            }
        )

    return cards, missing_source_findings


def technical_research_agent(state: GlobalState) -> dict[str, Any]:
    """논문 FAISS 검색과 기술 근거 추출을 수행한다."""
    queries = _build_queries(state)

    try:
        retrieved_chunks = retrieve_paper_chunks(queries, top_k=4)
    except FileNotFoundError as error:
        return {
            "technical_result": {
                "agent_name": "technical_research",
                "status": "insufficient_evidence",
                "summary": "FAISS 인덱스가 없어 기술 조사를 수행하지 못했습니다.",
                "evidence_ids": [],
                "limitations": ["논문 FAISS 인덱스를 먼저 생성해야 합니다."],
                "errors": [str(error)],
                "payload": {"queries": queries, "retrieved_chunks": []},
            },
            "evidence_cards": [],
        }

    if not retrieved_chunks:
        return {
            "technical_result": {
                "agent_name": "technical_research",
                "status": "insufficient_evidence",
                "summary": "질문과 관련된 논문 chunk를 찾지 못했습니다.",
                "evidence_ids": [],
                "limitations": ["검색 결과가 비어 있습니다."],
                "errors": [],
                "payload": {"queries": queries, "retrieved_chunks": []},
            },
            "evidence_cards": [],
        }

    base_payload = {
        "queries": queries,
        "retrieved_chunks": retrieved_chunks,
    }

    if not OPENAI_API_KEY:
        return {
            "technical_result": {
                "agent_name": "technical_research",
                "status": "insufficient_evidence",
                "summary": (
                    "FAISS 검색은 완료했지만 OPENAI_API_KEY가 없어 "
                    "근거 추출을 생략했습니다."
                ),
                "evidence_ids": [],
                "limitations": ["GPT-4o-mini 호출 설정이 필요합니다."],
                "errors": [],
                "payload": base_payload,
            },
            "evidence_cards": [],
        }

    try:
        extraction = _extract_findings(state, retrieved_chunks)
        evidence_cards, missing_source_findings = _build_evidence_cards(
            extraction,
            retrieved_chunks,
        )
    except Exception as error:  # noqa: BLE001
        return {
            "technical_result": {
                "agent_name": "technical_research",
                "status": "failed",
                "summary": "논문 검색 결과에서 기술 근거를 추출하지 못했습니다.",
                "evidence_ids": [],
                "limitations": [],
                "errors": [str(error)],
                "payload": base_payload,
            },
            "evidence_cards": [],
        }

    limitations = list(extraction.limitations)
    if missing_source_findings:
        limitations.append("일부 주장은 검색 chunk와 출처 연결에 실패했습니다.")

    return {
        "technical_result": {
            "agent_name": "technical_research",
            "status": "ok" if evidence_cards else "insufficient_evidence",
            "summary": extraction.summary,
            "evidence_ids": [card["evidence_id"] for card in evidence_cards],
            "limitations": limitations,
            "errors": [],
            "payload": {
                **base_payload,
                "missing_source_findings": missing_source_findings,
            },
        },
        "evidence_cards": evidence_cards,
    }
