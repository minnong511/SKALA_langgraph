"""Offline smoke run using real graph/agents with deterministic API fixtures.

Every fixture is synthetic. Passing this run demonstrates execution and schema
contracts, not live research accuracy or actual quality of the two technologies.
"""

import json
import re
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

from kv_cache_agent.agents import (
    cloud_domain,
    orchestrator,
    quality_evaluator,
    report_writer,
    research_worker,
    synthesis,
    technical,
    verifier,
)
from kv_cache_agent.schemas.evaluation import QualityEvaluation
from kv_cache_agent.schemas.tasks import ResearchPlan, SubTask
from kv_cache_agent.tools import source_fetcher, tavily_search

MOCK_CLAIMS = {
    "TurboQuant": "TurboQuant는 KV Cache의 저장 표현을 압축한다.",
    "CXL-based": "CXL-based 방식은 KV Cache의 메모리 계층을 확장한다.",
}


def mock_plan(query, followup=False):
    specifications = [
        ("technical_maturity", "TurboQuant", "paper", "SW 구현과 실험 조건"),
        ("technical_maturity", "CXL-based", "paper", "HW 구현과 실험 조건"),
        ("market", ["TurboQuant", "CXL-based"], "web", "상용화와 시장 근거"),
        ("stakeholder", ["TurboQuant", "CXL-based"], "web", "도입 기업의 기대와 우려"),
        (
            "domain_application",
            ["TurboQuant", "CXL-based"],
            "hybrid",
            "클라우드 운영 조건",
        ),
    ]
    if "공급망" in query:
        specifications += [
            ("market", "CXL-based", "web", "공급망과 조달 조건"),
            ("risk", ["TurboQuant", "CXL-based"], "hybrid", "반대 근거와 위험"),
        ]
    if followup:
        specifications = [
            ("market", ["TurboQuant", "CXL-based"], "web", "누락 주장 추가 근거")
        ]
    return ResearchPlan(
        tasks=[
            SubTask(
                task_id=f"task_{index}",
                perspective=p,
                technology=t,
                objective=objective,
                query=f"{t} {objective}",
                preferred_source=source,
                priority=1,
            )
            for index, (p, t, source, objective) in enumerate(specifications)
        ],
        planning_reason="Mock fixture: 질문에 따라 독립 조사 목표를 분리함",
    )


class FakeTavilyClient:
    def __init__(self, **_kwargs):
        pass

    def search(self, **_kwargs):
        return {
            "results": [
                {
                    "title": f"Mock {technology} cloud providers market evidence",
                    "url": f"https://mock-{index}.example/evidence",
                    "content": (
                        f"{claim} Cloud providers and hardware manufacturers describe "
                        "market adoption benefits and integration concerns."
                    ),
                    "score": 0.9,
                    "raw_content": claim,
                    "published_date": "2026-10-07",
                }
                for index, (technology, claim) in enumerate(MOCK_CLAIMS.items())
            ]
        }


def _mock_retrieve(queries, **_kwargs):
    text = " ".join(queries)
    selected = [t for t in MOCK_CLAIMS if t in text]
    return [
        {
            "chunk_id": f"mock-{t}",
            "content": MOCK_CLAIMS[t],
            "metadata": {
                "source_title": f"Mock {t} paper",
                "source_path": f"data/papers/{t}.pdf",
                "source_locator": "p. 1",
            },
        }
        for t in selected or MOCK_CLAIMS
    ]


def _mock_source(card):
    return {
        "fetch_status": "ok",
        "content": card["evidence_text"],
        "source_type": "web",
        "url": card["source_url"],
    }


class FakeLLM:
    def with_structured_output(self, schema):
        self.schema = schema
        return self

    def batch(self, inputs, config=None, return_exceptions=False):
        return [self.invoke(messages) for messages in inputs]

    def invoke(self, messages):
        content = messages[-1].content
        name = self.schema.__name__
        if name == "ResearchPlan":
            context = json.loads(content)
            return mock_plan(context["user_query"], bool(context["planning_round"]))
        if name == "TechnicalExtraction":
            blocks = re.findall(r"\[source_chunk_id=([^\]]+)\]([^\[]*)", content)
            technologies = {
                source_id: "TurboQuant" if "TurboQuant" in block else "CXL-based"
                for source_id, block in blocks
            }
            return {
                "summary": "Mock 논문 조사",
                "limitations": [],
                "findings": [
                    {
                        "technology": technologies[source_id],
                        "claim": MOCK_CLAIMS[technologies[source_id]],
                        "evidence_text": MOCK_CLAIMS[technologies[source_id]],
                        "source_chunk_ids": [source_id],
                        "claim_type": "fact",
                        "confidence": 0.9,
                    }
                    for source_id in technologies
                ],
            }
        if name == "CloudDomainExtraction":
            urls = re.findall(r"source_url=(.+)", content)

            def source_url(technology, index):
                web_url = f"https://mock-{index}.example/evidence"
                return (
                    web_url
                    if web_url in urls
                    else next(url for url in urls if technology in url)
                )

            return {
                "summary": "Mock 클라우드 조사",
                "comparison": {
                    "turboquant": "SW 압축",
                    "cxl_based": "계층 확장",
                    "trade_off": "실험 조건 차이로 직접 우열 비교 제한",
                },
                "findings": [
                    {
                        "technology": t,
                        "criterion": "cloud_llm_serving",
                        "claim": claim,
                        "evidence_text": claim,
                        "source_url": source_url(t, i),
                        "source_type": "official",
                        "claim_type": "inference",
                        "confidence": 0.8,
                        "caveat": "Mock, 실제 환경 검증 필요",
                    }
                    for i, (t, claim) in enumerate(MOCK_CLAIMS.items())
                ],
                "limitations": ["Mock 시나리오 평가"],
            }
        if name == "ClaimComparisonBatch":
            ids = re.findall(r"\[evidence_id=([^\]]+)\]", content)
            quote = content.split("original_source=", 1)[-1].strip()
            return {
                "decisions": [
                    {
                        "evidence_id": i,
                        "support_level": "full",
                        "matched_text": quote,
                        "rationale": "Mock 원문 지원",
                        "claim_type_assessment": "correct",
                    }
                    for i in ids
                ]
            }
        context = json.loads(content)
        if name == "SynthesisDraft":
            cards = context["evidence_cards"]
            item = {
                "text": "Mock 조건부 평가이며 실험 조건 차이로 직접 우열 비교를 제한한다.",
                "claim_type": "inference",
                "evidence_ids": [c["evidence_id"] for c in cards],
            }
            return {
                "summary": [item],
                "comparison_rows": [
                    {
                        **item,
                        "perspective": p,
                        "evidence_ids": [
                            c["evidence_id"] for c in cards if c["perspective"] == p
                        ],
                    }
                    for p in synthesis.PERSPECTIVES
                ],
                "agreements": [item],
                "conflicts": [],
                "conditional_recommendations": [],
                "limitations": ["Mock 보고서, 실제 기술 평가 결과가 아님"],
            }
        if name == "ReportDraft":
            cards = context["evidence_cards"]
            perspectives = dict(
                zip(
                    ("section_4_1", "section_4_2", "section_4_3", "section_4_4"),
                    synthesis.PERSPECTIVES,
                    strict=True,
                )
            )
            return {
                "sections": [
                    {
                        "section_id": section_id,
                        "paragraphs": [
                            {
                                "text": "Mock 평가에서는 적용 조건과 공개 근거의 한계를 함께 검토한다.",
                                "claim_type": "inference",
                                "evidence_ids": [
                                    c["evidence_id"]
                                    for c in cards
                                    if c["perspective"]
                                    == perspectives.get(section_id, "technical")
                                ],
                            }
                        ]
                        * 2,
                    }
                    for section_id in context["required_section_ids"]
                ]
            }
        if name == "QualityEvaluation":
            return QualityEvaluation(
                groundedness=True,
                neutrality=True,
                bias_control=True,
                perspective_coverage=True,
                overall_pass=True,
                recommended_action="pass",
                groundedness_reason="Mock judge: citation fixture supported",
                neutrality_reason="Mock judge: no winner",
                bias_reason="Mock judge: balanced fixture",
                coverage_reason="Mock judge: four perspectives",
            )
        raise AssertionError(f"unexpected schema {name}")


@contextmanager
def mock_services():
    """Mock API boundaries only; run real Send, reducer, verifier and writers."""
    with ExitStack() as stack:
        # Do not upload synthetic smoke traces to a connected LangSmith project.
        stack.enter_context(
            patch.dict(
                "os.environ",
                {
                    "LANGSMITH_TRACING": "false",
                    "LANGCHAIN_TRACING_V2": "false",
                },
            )
        )
        for module in (
            orchestrator,
            quality_evaluator,
            report_writer,
            cloud_domain,
            synthesis,
            technical,
            verifier,
        ):
            stack.enter_context(patch.object(module, "get_llm", side_effect=FakeLLM))
        stack.enter_context(patch.object(tavily_search, "TAVILY_API_KEY", "mock-only"))
        stack.enter_context(
            patch.object(tavily_search, "TavilyClient", FakeTavilyClient)
        )
        stack.enter_context(
            patch.object(research_worker, "retrieve_paper_chunks", _mock_retrieve)
        )
        stack.enter_context(
            patch.object(verifier, "_fetch_original_source", _mock_source)
        )
        stack.enter_context(
            patch.object(
                source_fetcher,
                "fetch_source",
                side_effect=lambda url, **kwargs: {
                    "fetch_status": "ok",
                    "content": " ".join(MOCK_CLAIMS.values()),
                    "url": url,
                },
            )
        )
        yield
