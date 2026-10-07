"""Real LLM judge calibration fixtures, separate from production reports.

Uses the two source-audit claims already verified against local CXL pages and
one verified TurboQuant card from a saved run. Fixtures intentionally leave
other perspectives incomplete: only the specified criterion is asserted.
"""

import argparse
import json
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from langsmith import Client, trace

from kv_cache_agent.agents.quality_evaluator import quality_evaluator_agent
from kv_cache_agent.config import PAPERS_DIR
from kv_cache_agent.graph.workflow import build_workflow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-db", type=Path, required=True)
    parser.add_argument("--thread-id", required=True)
    parser.add_argument("--audit-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.audit_file.read_text())
    ids = {"manual-source-audit-p2", "manual-source-audit-p10"}
    assert ids <= {
        c["evidence_id"] for c in audit["cases"] if c["after_status"] == "verified"
    }
    with SqliteSaver.from_conn_string(str(args.checkpoint_db)) as cp:
        saved = (
            build_workflow(checkpointer=cp)
            .get_state({"configurable": {"thread_id": args.thread_id}})
            .values
        )
    turbo = deepcopy(
        next(
            c
            for c in saved["payload"]["usable_evidence_cards"]
            if c["technology"] == "TurboQuant"
            and c["verification_status"] == "verified"
            and c["claim_type"] == "fact"
        )
    )
    turbo["evidence_id"] = "calibration-tq"
    template = next(
        c
        for c in saved["payload"]["evidence_cards"]
        if "cxl_based_kv_cache.pdf" in c["source_url"]
    )
    cards = [turbo]
    for evidence_id, value, unit, baseline, quote in [
        (
            "manual-source-audit-p2",
            "1.80",
            "x",
            "NVMe-oF",
            "Our evaluation shows that ITME achieves a1 .80× throughput improvement over NVMe-oF-based disaggregated storage baselines in large-scale LLM inference.",
        ),
        (
            "manual-source-audit-p10",
            "35.7",
            "%",
            "CPU-offload",
            "Consequently, ITME achieves up to a 35.7% throughput improvement over the CPU-offload baseline in these extended turns.",
        ),
    ]:
        audited = next(c for c in audit["cases"] if c["evidence_id"] == evidence_id)
        cards.append(
            {
                **template,
                "evidence_id": evidence_id,
                "claim": audited["claim"],
                "source_url": str(PAPERS_DIR / "cxl_based_kv_cache.pdf"),
                "source_locator": audited["source_locator"],
                "perspective": "technical",
                "evidence_text": quote,
                "claim_type": "fact",
                "verification_status": "verified",
                "metric": "throughput improvement",
                "value": value,
                "unit": unit,
                "baseline": baseline,
                "conditions": {},
            }
        )
    report = (
        "# SUMMARY\n원문이 보고한 연구 결과를 조건부로 정리한다.\n## 기술 성숙도\n"
        + "\n".join(f"{c['claim']} [{i}]" for i, c in enumerate(cards, 1))
    )
    report += "\nCXL의 두 수치는 기준선과 실험 범위가 다르므로 상충이나 직접 우열로 판단하지 않는다.\n## 시장성\n현재 자료만으로 도입률과 가격을 확인할 수 없다.\n## 이해관계자\n운영자별 요구를 확인할 자료가 부족하다.\n## 도메인 적용성\n다른 클라우드 환경으로의 일반화는 보류한다.\n## 한계 및 불확실성\n보고된 연구 환경에 한정하며 실험 조건의 동일성이 확인되지 않았다.\n# REFERENCE\n"
    report += "\n".join(
        f"[{i}] {c['source_title']}. {c['source_url']} {c['source_locator']} 근거 ID: {c['evidence_id']}"
        for i, c in enumerate(cards, 1)
    )
    cases = [
        (
            "different_baselines_are_not_conflict",
            report,
            {"groundedness": True, "neutrality": True},
        ),
        (
            "unsupported_customer_adoption",
            report.replace(
                "현재 자료만으로 도입률과 가격을 확인할 수 없다.",
                "ITME는 AWS와 Azure 전체 운영 환경에 이미 상용 도입되어 있다. [2]",
            ),
            {"groundedness": False},
        ),
        (
            "unsupported_winner",
            report.replace(
                "원문이 보고한 연구 결과를 조건부로 정리한다.",
                "CXL이 모든 환경에서 우월하므로 반드시 CXL을 선택해야 한다.",
            ),
            {"neutrality": False},
        ),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace_id = str(uuid4())
    client = Client(timeout_ms=10_000)
    rows = []
    with trace(
        "quality-judge-calibration-fixtures",
        inputs={},
        client=client,
        metadata={
            "trace_id": trace_id,
            "purpose": "criterion calibration, not production report PASS",
        },
    ) as root:
        for name, candidate, expected in cases:
            state = {
                "control": {"trace_id": trace_id},
                "payload": {
                    "user_query": "근거에 연결된 기술 평가",
                    "target_domain": "클라우드 LLM 서빙",
                    "selected_technologies": ["TurboQuant", "CXL-based"],
                    "report": candidate,
                    "evidence_cards": cards,
                    "usable_evidence_cards": cards,
                },
            }
            evaluation = quality_evaluator_agent(state)["payload"]["evaluation"]
            rows.append(
                {
                    "name": name,
                    "expected_criteria": expected,
                    "criteria_passed": all(
                        evaluation[k] == v for k, v in expected.items()
                    ),
                    "evaluation": evaluation,
                }
            )
        root.end(
            outputs={"criteria_passed": all(row["criteria_passed"] for row in rows)}
        )
    client.flush(timeout=10)
    result = {
        "mode": "real_llm_judge_calibration_fixtures",
        "full_report_reevaluated": False,
        "passed": all(r["criteria_passed"] for r in rows),
        "trace_id": trace_id,
        "run_url": client.get_run_url(run=root),
        "cases": rows,
    }
    (args.output_dir / "judge-calibration.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
