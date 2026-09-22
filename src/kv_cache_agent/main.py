import argparse
import json
import traceback
from datetime import UTC, datetime
from pathlib import Path

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
    args = parser.parse_args()
    query = args.query.strip() or DEFAULT_QUERY

    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    log_path = args.log_file or OUTPUTS_DIR / "logs" / f"workflow_{timestamp}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        result = build_workflow().invoke({"user_query": query})
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

    output_dir = OUTPUTS_DIR / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / f"kv_cache_evaluation_{timestamp}.md"
    report_path.write_text(result["final_report"], encoding="utf-8")
    pdf_path = output_dir / f"kv_cache_evaluation_{timestamp}.pdf"
    pdf_error = None
    try:
        write_pdf(result["final_report"], pdf_path)
    except Exception as error:  # noqa: BLE001 - Markdown 저장은 유지하고 PDF 오류를 로그화
        pdf_error = {
            "error_type": type(error).__name__,
            "error": str(error),
        }

    log_path.write_text(
        json.dumps(
            {
                "query": query,
                "status": "completed",
                "report_path": str(report_path),
                "pdf_path": str(pdf_path) if pdf_error is None else None,
                "pdf_status": "ok" if pdf_error is None else "failed",
                "pdf_error": pdf_error,
                "verification_status": result.get("verification_result", {}).get(
                    "status"
                ),
                "synthesis_status": result.get("synthesis_result", {}).get(
                    "status"
                ),
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
    print("검증 상태:", result.get("verification_result", {}).get("status"))
    print("종합 상태:", result.get("synthesis_result", {}).get("status"))
    print("보고서 길이:", len(result.get("final_report", "")))

if __name__ == "__main__":
    main()
