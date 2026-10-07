"""Hybrid rules + structured LLM judge, after every generated report."""

import json
import re
from collections import Counter
from pathlib import Path

import yaml
from langchain_core.messages import HumanMessage, SystemMessage

from kv_cache_agent.evidence import comparison_context, source_identity
from kv_cache_agent.llm import get_llm
from kv_cache_agent.observability import record_event
from kv_cache_agent.schemas.evaluation import QualityEvaluation
from kv_cache_agent.schemas.tasks import LEGACY_PERSPECTIVES, REQUIRED_PERSPECTIVES

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "quality_evaluator.yaml"
SECTION_LABELS = {
    "technical_maturity": "기술 성숙도",
    "market": "시장성",
    "stakeholder": "이해관계자",
    "domain_application": "도메인 적용",
}


def check_report_rules(payload):
    report = payload.get("report", "")
    cards = {
        c["evidence_id"]: c
        for c in payload.get("usable_evidence_cards", [])
        if c.get("verification_status") in {"verified", "partially_verified"}
    }
    headings = re.findall(r"^#{1,3}\s+(.+)$", report, re.MULTILINE)
    missing = [
        p
        for p in REQUIRED_PERSPECTIVES
        if not (
            any(SECTION_LABELS[p] in heading for heading in headings)
            and any(
                c.get("perspective") == LEGACY_PERSPECTIVES[p] for c in cards.values()
            )
        )
    ]
    failures = []
    if "SUMMARY" not in headings:
        failures.append("missing_summary")
    if not any("한계" in heading for heading in headings):
        failures.append("missing_limitations")
    if "REFERENCE" not in headings:
        failures.append("missing_reference")
    body, _, references = report.partition("# REFERENCE")
    # The preserved renderer records [number] -> Evidence ID -> original source.
    citation_ids = re.findall(r"\[(\d+)\]", body)
    reference_pairs = re.findall(
        r"^\[(\d+)\].*?근거 ID:\s*(\S+)",
        references,
        re.MULTILINE,
    )
    reference_map = dict(reference_pairs)
    if not citation_ids:
        failures.append("missing_body_citations")
    if len(reference_map) != len(reference_pairs):
        failures.append("duplicate_reference_number")
    if not reference_pairs:
        failures.append("missing_evidence_ids")
    if any(number not in reference_map for number in citation_ids):
        failures.append("unmapped_citation")
    if any(evidence_id not in cards for evidence_id in reference_map.values()):
        failures.append("unknown_or_unverified_evidence_id")
    for number, evidence_id in reference_pairs:
        card = cards.get(evidence_id)
        line = next(
            (
                line
                for line in references.splitlines()
                if line.startswith(f"[{number}] ")
            ),
            "",
        )
        if card and (not card.get("source_url") or card["source_url"] not in line):
            failures.append("reference_source_mismatch")
    sources = Counter(source_identity(c) for c in cards.values())
    missing_technologies = [
        t
        for t in payload.get("selected_technologies", [])
        if not any(c.get("technology") in {t, "both"} for c in cards.values())
    ]
    return {
        "failures": list(dict.fromkeys(failures)),
        "missing_perspectives": missing,
        "source_counts": dict(sources),
        "missing_technologies": missing_technologies,
        "single_source": len(sources) < 2,
        "citation_map": reference_map,
    }


def quality_evaluator_agent(state):
    payload = state["payload"]
    rules = check_report_rules(payload)
    prompt = yaml.safe_load(PROMPT_PATH.read_text(encoding="utf-8"))
    context = {
        "user_query": payload["user_query"],
        "selected_technologies": payload["selected_technologies"],
        "target_domain": payload["target_domain"],
        "report": payload.get("report", ""),
        "synthesis_analysis": payload.get("synthesis", {}).get("payload", {}),
        "evidence_cards": payload.get("usable_evidence_cards", []),
        "excluded_evidence": [
            {
                key: card.get(key)
                for key in (
                    "evidence_id",
                    "technology",
                    "claim",
                    "source_url",
                    "caveat",
                )
            }
            for card in payload.get("evidence_cards", [])
            if card["evidence_id"]
            not in {c["evidence_id"] for c in payload.get("usable_evidence_cards", [])}
        ],
        "limitations": payload.get("limitations", []),
        "rules": rules,
        "measurement_comparisons": comparison_context(
            payload.get("usable_evidence_cards", [])
        ),
    }
    try:
        response = (
            get_llm()
            .with_structured_output(QualityEvaluation)
            .invoke(
                [
                    SystemMessage(content=prompt["system_prompt"]),
                    HumanMessage(content=json.dumps(context, ensure_ascii=False)),
                ]
            )
        )
        evaluation = QualityEvaluation.model_validate(response)
    except Exception as error:  # noqa: BLE001 - fail closed if judge unavailable
        evaluation = QualityEvaluation(
            groundedness=False,
            neutrality=False,
            bias_control=False,
            perspective_coverage=False,
            overall_pass=False,
            groundedness_reason=f"judge_error:{type(error).__name__}",
            neutrality_reason="Judge 평가 미완료",
            bias_reason="Judge 평가 미완료",
            coverage_reason="Judge 평가 미완료",
            recommended_action="rewrite",
        )
    evaluation.groundedness &= not rules["failures"]
    evaluation.perspective_coverage &= not rules["missing_perspectives"]
    evaluation.bias_control &= not (
        rules["single_source"] or rules["missing_technologies"]
    )
    evaluation.rule_failures = rules["failures"]
    evaluation.missing_perspectives = list(
        dict.fromkeys(
            [
                *evaluation.missing_perspectives,
                *rules["missing_perspectives"],
            ]
        )
    )
    if rules["failures"]:
        evaluation.groundedness_reason += "; rules: " + ", ".join(rules["failures"])
    if rules["missing_perspectives"]:
        evaluation.coverage_reason += "; 근거 또는 필수 절 누락: " + ", ".join(
            rules["missing_perspectives"],
        )
    if rules["single_source"] or rules["missing_technologies"]:
        evaluation.bias_reason += "; 독립 출처 또는 기술별 근거 부족"
        evaluation.missing_evidence_topics.append("독립 출처와 양 기술의 반대 근거")
    evaluation.overall_pass = all(
        (
            evaluation.groundedness,
            evaluation.neutrality,
            evaluation.bias_control,
            evaluation.perspective_coverage,
        )
    )
    if evaluation.overall_pass:
        evaluation.recommended_action = "pass"
    elif (
        not evaluation.groundedness
        or not evaluation.perspective_coverage
        or rules["single_source"]
        or rules["missing_technologies"]
    ):
        evaluation.recommended_action = "additional_research"
    elif not evaluation.bias_control:
        evaluation.recommended_action = "resynthesis"
    else:
        evaluation.recommended_action = "rewrite"
    record_event(
        state["control"]["trace_id"],
        "quality_evaluator",
        evaluation.recommended_action,
        f"{evaluation.groundedness_reason}; {evaluation.neutrality_reason}; "
        f"{evaluation.bias_reason}; {evaluation.coverage_reason}",
    )
    return {"payload": {"evaluation": evaluation.model_dump()}}
