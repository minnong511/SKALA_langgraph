"""보고서 생성 *후* 실행하는 품질 평가 노드.

형식·출처 수는 코드가 검사하고, 내용 판단은 구조화된 LLM Judge가 맡는다.
Judge의 판정은 State에 저장하지만 실제 재작성 분기는 그래프의 조건부 엣지가 한다.
"""

import json
import re
from pathlib import Path

import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from kv_cache_agent.config import OPENAI_API_KEY
from kv_cache_agent.graph.state import GlobalState
from kv_cache_agent.llm import get_llm
from kv_cache_agent.schemas.quality import ReportQualityResult

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "report_quality.yaml"
PERSPECTIVE_HEADINGS = (
    "4.1 기술 성숙도",
    "4.2 시장성",
    "4.3 이해관계자",
    "4.4 도메인 적용",
)


class QualityJudgement(BaseModel):
    """자유형 dict 대신 네 평가 항목을 빠짐없이 받기 위한 모델."""

    groundedness: bool
    neutrality: bool
    bias_control: bool
    perspective_coverage: bool
    feedback: list[str] = Field(default_factory=list)


def evaluate_report_quality(state: GlobalState) -> dict[str, object]:
    """기존 보고서와 검증 카드로 품질을 평가하고 수정 지시를 반환한다."""
    report = state.get("final_report", "")
    cards = state.get("verified_evidence_cards", [])
    reference = report.split("# REFERENCE", 1)[-1] if "# REFERENCE" in report else ""
    body = report.split("# REFERENCE", 1)[0]
    cited_numbers = set(re.findall(r"\[(\d+)\]", body))
    reference_numbers = set(re.findall(r"(?m)^\[(\d+)\]", reference))
    used_urls = {
        str(card.get("source_url"))
        for card in cards
        if card.get("source_url")
        and (
            str(card["source_url"]) in reference
            or str(card.get("canonical_url") or card["source_url"]) in reference
        )
    }

    # 단순 목차 존재만으로는 내용의 품질을 보장하지 못하므로 Judge와 함께 검사한다.
    structure_ok = "# SUMMARY" in report and "# REFERENCE" in report
    coverage_ok = all(heading in report for heading in PERSPECTIVE_HEADINGS)
    citations_ok = bool(cited_numbers) and cited_numbers <= reference_numbers
    source_diversity_ok = len(used_urls) >= 2
    feedback: list[str] = []
    if not structure_ok:
        feedback.append("SUMMARY와 REFERENCE 필수 목차를 복원하세요.")
    if not coverage_ok:
        feedback.append(
            "기술 성숙도·시장성·이해관계자·도메인 적용 네 관점을 모두 채우세요."
        )
    if not citations_ok:
        feedback.append("본문 인용 번호를 실제 REFERENCE 출처에 연결하세요.")
    if not source_diversity_ok:
        feedback.append(
            "두 개 이상의 독립된 검증 출처를 반영하거나 자료 한계를 명시하세요."
        )
    if not state.get("control", {}).get("evidence_ready", False):
        feedback.append(
            "Supervisor가 근거 충분성을 승인하지 않았습니다. 제한적 보고서로 표시하세요."
        )

    if OPENAI_API_KEY and structure_ok:
        prompt = yaml.safe_load(PROMPT_PATH.read_text(encoding="utf-8"))[
            "system_prompt"
        ]
        judge_context = {
            "report": report[:30_000],
            "verified_sources": [
                {
                    "evidence_id": card.get("evidence_id"),
                    "source_url": card.get("source_url"),
                    "canonical_url": card.get("canonical_url"),
                    "perspective": card.get("perspective"),
                    "claim": card.get("claim"),
                    "evidence_text": card.get("evidence_text"),
                    "source_locator": card.get("source_locator"),
                    "verification_status": card.get("verification_status"),
                }
                for card in cards
            ],
        }
        try:
            response = (
                get_llm()
                .with_structured_output(QualityJudgement)
                .invoke(
                    [
                        SystemMessage(content=prompt),
                        HumanMessage(
                            content=json.dumps(judge_context, ensure_ascii=False)
                        ),
                    ]
                )
            )
            judgement = QualityJudgement.model_validate(response)
        except Exception as error:  # noqa: BLE001 - Judge 실패는 통과로 간주하지 않는다.
            judgement = QualityJudgement(
                groundedness=False,
                neutrality=False,
                bias_control=False,
                perspective_coverage=False,
                feedback=[f"품질 평가 모델 오류: {type(error).__name__}"],
            )
    elif not structure_ok:
        judgement = QualityJudgement(
            groundedness=False,
            neutrality=False,
            bias_control=False,
            perspective_coverage=False,
            feedback=[
                "보고서 작성이 완료되지 않았습니다. 작성 오류를 수정하고 다시 생성하세요."
            ],
        )
    else:
        judgement = QualityJudgement(
            groundedness=False,
            neutrality=False,
            bias_control=False,
            perspective_coverage=False,
            feedback=["OPENAI_API_KEY가 없어 내용 평가를 완료하지 못했습니다."],
        )

    excluded_numeric_claims = "수치·인용 검사 미통과 단락" in report
    groundedness = (
        citations_ok and judgement.groundedness and not excluded_numeric_claims
    )
    if excluded_numeric_claims:
        feedback.append(
            "수치·인용 검사에서 제외된 단락을 원문과 다시 연결하여 작성하세요."
        )
    coverage = coverage_ok and judgement.perspective_coverage
    bias_control = source_diversity_ok and judgement.bias_control
    neutrality = judgement.neutrality
    feedback.extend(judgement.feedback)
    passed = (
        structure_ok
        and state.get("control", {}).get("evidence_ready", False)
        and all((groundedness, coverage, bias_control, neutrality))
    )
    revision = state.get("control", {}).get("report_revision", 1)
    max_revisions = state.get("control", {}).get("max_report_revisions", 2)
    status = (
        "passed" if passed else "revise" if revision < max_revisions else "needs_review"
    )
    result: ReportQualityResult = {
        "passed": passed,
        "groundedness": groundedness,
        "neutrality": neutrality,
        "bias_control": bias_control,
        "perspective_coverage": coverage,
        "feedback": list(dict.fromkeys(feedback)),
        "status": status,
    }
    control = dict(state.get("control", {}))
    control["status"] = (
        "completed" if passed else "writing" if status == "revise" else "needs_review"
    )
    control["node_status"] = {**control.get("node_status", {}), "quality_eval": status}
    return {
        "quality_result": result,
        "quality_feedback": result["feedback"],
        "control": control,
    }
