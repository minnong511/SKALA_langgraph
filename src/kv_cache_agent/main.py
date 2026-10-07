import argparse
import json
import logging
import traceback
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pypdf import PdfReader

from kv_cache_agent.config import OUTPUTS_DIR
from kv_cache_agent.graph.workflow import build_workflow
from kv_cache_agent.tools.pdf_writer import write_pdf

DEFAULT_QUERY = (
    "클라우드 LLM 서빙에서 TurboQuant와 CXL-based KV Cache 최적화 기술을 "
    "기술, 시장, 이해관계자 관점에서 비교 평가해줘."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help="평가 질문. 생략하면 기본 KV Cache 비교 질문을 사용합니다.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="상세 실행 로그 경로. 생략하면 outputs/logs에 자동 저장합니다.",
    )
    parser.add_argument(
        "--resume-log",
        type=Path,
        default=None,
        help="이 프로젝트의 이전 workflow JSON을 읽어 근거 검수·종합·보고서 단계부터 재실행합니다.",
    )
    args = parser.parse_args()
    query = args.query.strip() or DEFAULT_QUERY
    trace_id = str(uuid4())
    initial_state = {"user_query": query, "trace_id": trace_id}
    if args.resume_log:
        saved = json.loads(args.resume_log.read_text(encoding="utf-8"))
        if not isinstance(saved.get("workflow_result"), dict):
            raise ValueError(
                "workflow_result가 포함된 이 프로젝트의 실행 로그가 필요합니다."
            )
        initial_state = saved["workflow_result"]
        query = str(initial_state.get("user_query") or query)
        initial_state["trace_id"] = trace_id
        # 기존 근거와 검증 버전/재조사 횟수는 보존한다. 재조사 한도는 늘리지 않는다.
        # 새 Supervisor가 검토한 뒤 종합·보고서·품질 평가를 다시 수행한다.
        for field in (
            "synthesis_result",
            "final_report",
            "quality_result",
            "quality_feedback",
        ):
            initial_state.pop(field, None)
        control = dict(initial_state.get("control", {}))
        control["report_revision"] = 0
        control.pop("evidence_review", None)
        initial_state["control"] = control

    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    log_path = args.log_file or OUTPUTS_DIR / "logs" / f"workflow_{timestamp}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path = log_path.with_name(f"{log_path.stem}_decisions.jsonl")
    decision_logger = logging.getLogger("kv_cache_agent.agents.supervisor")
    decision_logger.setLevel(logging.INFO)
    decision_handler = logging.FileHandler(decision_path, encoding="utf-8")
    decision_handler.setFormatter(logging.Formatter("%(message)s"))
    decision_logger.addHandler(decision_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    decision_logger.addHandler(console_handler)
    print(
        "평가 시작: 조사 → 원문 검증 → Supervisor 검수 → 보고서 작성 → 품질 평가",
        flush=True,
    )

    try:
        result = build_workflow().invoke(
            initial_state,
            config={"metadata": {"trace_id": trace_id}, "tags": ["supervisor-agent"]},
        )
    except Exception as error:
        log_path.write_text(
            json.dumps(
                {
                    "query": query,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"실행 실패. 상세 로그 저장: {log_path}")
        raise
    finally:
        decision_logger.removeHandler(decision_handler)
        decision_logger.removeHandler(console_handler)
        decision_handler.close()

    output_dir = OUTPUTS_DIR / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not result.get("final_report"):
        raise RuntimeError(
            "보고서가 생성되지 않았습니다. 실행 로그와 제어 상태를 확인하세요."
        )

    report_path = output_dir / f"kv_cache_evaluation_{timestamp}.md"
    report_path.write_text(result["final_report"], encoding="utf-8")
    pdf_path = output_dir / f"kv_cache_evaluation_{timestamp}.pdf"
    pdf_error = None
    pdf_page_count = None
    try:
        if result["final_report"].startswith("# 보고서 생성 실패"):
            raise RuntimeError(
                "작성 실패 안내는 평가 보고서 PDF로 내보내지 않습니다. Markdown과 로그를 확인하세요."
            )
        write_pdf(result["final_report"], pdf_path)
        pdf_page_count = len(PdfReader(str(pdf_path)).pages)
    except Exception as error:  # noqa: BLE001 - Markdown 저장은 유지하고 PDF 오류를 로그화
        pdf_error = {
            "error_type": type(error).__name__,
            "error": str(error),
        }

    quality_status = result.get("quality_result", {}).get("status", "needs_review")
    delivery_ready = (
        quality_status == "passed"
        and pdf_page_count is not None
        and pdf_page_count <= 10
    )
    log_path.write_text(
        json.dumps(
            {
                "query": query,
                "status": "completed" if delivery_ready else "needs_review",
                "trace_id": trace_id,
                "decision_log_path": str(decision_path),
                "quality_status": quality_status,
                "quality_feedback": result.get("quality_feedback", []),
                "pdf_page_count": pdf_page_count,
                "pdf_page_limit": 10,
                "report_path": str(report_path),
                "pdf_path": str(pdf_path) if pdf_error is None else None,
                "pdf_status": "ok" if pdf_error is None else "failed",
                "pdf_error": pdf_error,
                "verification_status": result.get("verification_result", {}).get(
                    "status"
                ),
                "synthesis_status": result.get("synthesis_result", {}).get("status"),
                "verified_evidence_card_ids": [
                    card.get("evidence_id")
                    for card in result.get("verified_evidence_cards", [])
                ],
                "usable_evidence_card_ids": [
                    card.get("evidence_id")
                    for card in result.get("usable_evidence_cards", [])
                ],
                "final_report_length": len(result.get("final_report", "")),
                "workflow_result": result,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(f"질문: {query}")
    print(f"보고서 저장 완료: {report_path}")
    if pdf_error is None:
        print(f"PDF 저장 완료: {pdf_path}")
    else:
        print(f"PDF 저장 실패: {pdf_error['error_type']} - {pdf_error['error']}")
    print(f"상세 실행 로그 저장: {log_path}")
    print(f"Supervisor 결정 로그: {decision_path}")
    print("검증 상태:", result.get("verification_result", {}).get("status"))
    print("종합 상태:", result.get("synthesis_result", {}).get("status"))
    print("보고서 품질 상태:", quality_status)
    print("PDF 페이지 수:", pdf_page_count, "(제출 한도: 10)")
    if not delivery_ready:
        print("제출 전 검토 필요: 품질 판정, PDF 생성 여부와 10쪽 한도를 확인하세요.")
    print("보고서 길이:", len(result.get("final_report", "")))


if __name__ == "__main__":
    main()
