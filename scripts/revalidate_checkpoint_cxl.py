"""Focused real-source audit of CXL numbers from an existing checkpoint.

This preserves the old workflow result and never labels a full report as PASS.
"""

import argparse
import json
import logging
from pathlib import Path
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from langsmith import Client, trace

from kv_cache_agent.agents.verifier import (
    _is_retrieval_limitation,
    evidence_verification_agent,
)
from kv_cache_agent.config import PAPERS_DIR
from kv_cache_agent.graph.workflow import build_workflow
from kv_cache_agent.observability import node_span


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-db", type=Path, required=True)
    parser.add_argument("--thread-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.getLogger("langsmith").setLevel(logging.ERROR)
    with SqliteSaver.from_conn_string(str(args.checkpoint_db)) as cp:
        state = (
            build_workflow(checkpointer=cp)
            .get_state({"configurable": {"thread_id": args.thread_id}})
            .values
        )
    originals = state["payload"]["evidence_cards"]
    cards = [
        c
        for c in originals
        if c.get("technology") in {"CXL-based", "both"}
        and any(
            n in c.get("claim", "") + c.get("evidence_text", "")
            for n in ("1.80", "35.7", "30%")
        )
    ]
    template = next(
        c for c in originals if "cxl_based_kv_cache.pdf" in c.get("source_url", "")
    )
    # A separate manual source-audit input, explicitly not a Worker output.
    cards.append(
        {
            **template,
            "evidence_id": "manual-source-audit-p2",
            "source_locator": "p. 2",
            "source_url": str(PAPERS_DIR / "cxl_based_kv_cache.pdf"),
            "claim": "ITME는 해당 대규모 LLM 추론 평가에서 NVMe-oF 기반 분리 스토리지 기준선 대비 처리량 1.80배 개선을 보고한다.",
            "evidence_text": "Our evaluation shows that ITME achieves a1 .80× throughput improvement over NVMe-oF-based disaggregated storage baselines in large-scale LLM inference.",
            "retrieval_method": "manual_source_audit",
            "claim_type": "fact",
        }
    )
    cards.append(
        {
            **template,
            "evidence_id": "manual-source-audit-p10",
            "perspective": "technical",
            "source_locator": "p. 10",
            "source_url": str(PAPERS_DIR / "cxl_based_kv_cache.pdf"),
            "claim": "ITME는 해당 실험의 확장된 대화 턴에서 CPU-offload 기준선 대비 최대 35.7% 처리량 개선을 보고한다.",
            "evidence_text": "Consequently, ITME achieves up to a 35.7% throughput improvement over the CPU-offload baseline in these extended turns.",
            "retrieval_method": "manual_source_audit",
            "claim_type": "fact",
            "caveat": "원문의 해당 실험 조건에 한정",
        }
    )
    old = {
        c["evidence_id"]: c for c in state["payload"].get("usable_evidence_cards", [])
    }
    trace_id = str(uuid4())
    client = Client(timeout_ms=10_000)
    with trace(
        "cxl-original-source-revalidation",
        client=client,
        inputs={},
        metadata={
            "trace_id": trace_id,
            "source_trace_id": state["control"]["trace_id"],
            "purpose": "focused source audit, not full report evaluation",
        },
    ) as root:
        with node_span(trace_id, "verifier", evidence_count=len(cards)):
            update = evidence_verification_agent({"evidence_cards": cards})
        verified = update["verification_result"]["payload"]["all_verified_cards"]
        rows = [
            {
                "evidence_id": c["evidence_id"],
                "claim": c["claim"],
                "source_url": c["source_url"],
                "source_locator": c["source_locator"],
                "before_status": old.get(c["evidence_id"], {}).get(
                    "verification_status", "excluded_or_manual_audit"
                ),
                "after_status": c["verification_status"],
                "reason": c.get("caveat", ""),
            }
            for c in verified
        ]
        batch_card = next(
            c
            for c in cards
            if "paper17.pdf" in c["source_url"]
            and "30%" in c["claim"]
            and not _is_retrieval_limitation(c)
        )
        positive_ids = {
            "manual-source-audit-p2",
            "manual-source-audit-p10",
            batch_card["evidence_id"],
        }
        positive_cards = [c for c in verified if c["evidence_id"] in positive_ids]
        passed = len(positive_cards) == len(positive_ids) and all(
            c["verification_status"] == "verified" for c in positive_cards
        )
        result = {
            "trace_id": trace_id,
            "run_id": str(root.id),
            "source_thread": args.thread_id,
            "mode": "real_api_original_source_audit",
            "full_report_reevaluated": False,
            "passed": passed,
            "cases": rows,
        }
        root.end(outputs={"passed": passed, "evidence_count": len(rows)})
    client.flush(timeout=10)
    result["run_url"] = client.get_run_url(run=root)
    (args.output_dir / "cxl-revalidation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
