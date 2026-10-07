"""CLI for live research, offline smoke runs and checkpoint recovery."""

import argparse
import json
import logging
from contextlib import ExitStack
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from kv_cache_agent.config import OUTPUTS_DIR, ReportBudget, WorkflowLimits
from kv_cache_agent.graph.workflow import build_workflow
from kv_cache_agent.observability import flush_traces
from kv_cache_agent.tools.pdf_writer import write_pdf

DEFAULT_QUERY = (
    "클라우드 LLM 서빙에서 TurboQuant와 CXL-based KV Cache 최적화 기술을 "
    "기술 성숙도, 시장성, 이해관계자, 도메인 적용성 관점에서 평가해줘."
)


def main() -> None:
    parser = argparse.ArgumentParser(description="KV Cache Orchestrator-Workers 평가")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--target-domain", default="클라우드 LLM 서빙")
    parser.add_argument(
        "--mock", action="store_true", help="외부 API 없이 합성 자료로 전체 실행"
    )
    parser.add_argument("--no-pdf", action="store_true")
    parser.add_argument(
        "--log-file", type=Path, default=None, help="최소 실행 요약 JSON 경로"
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-report-revisions", type=int)
    parser.add_argument("--max-worker-retries", type=int)
    parser.add_argument(
        "--checkpoint-db",
        type=Path,
        help="SQLite 체크포인트 경로 (recovery extra 필요)",
    )
    parser.add_argument("--thread-id", help="체크포인트 실행 ID, --resume 시 필수")
    parser.add_argument(
        "--resume", action="store_true", help="같은 thread의 체크포인트에서 재개"
    )
    args = parser.parse_args()
    if args.resume and not (args.checkpoint_db and args.thread_id):
        parser.error("--resume에는 --checkpoint-db와 --thread-id가 필요합니다.")
    limits = WorkflowLimits.from_env()
    overrides = {
        k: getattr(args, k)
        for k in (
            "max_steps",
            "max_report_revisions",
            "max_worker_retries",
        )
        if getattr(args, k) is not None
    }
    limits = replace(limits, **overrides)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    trace_id = str(uuid4())
    log_path = args.log_file or OUTPUTS_DIR / "logs" / f"workflow_{timestamp}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    events_path = log_path.with_suffix(".events.jsonl")
    event_logger = logging.getLogger("kv_cache_agent.events")
    handler = logging.FileHandler(events_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    event_logger.addHandler(handler)
    event_logger.setLevel(logging.INFO)

    try:
        with ExitStack() as stack:
            if args.mock:
                from kv_cache_agent.mock_run import mock_services

                stack.enter_context(mock_services())
            checkpointer = None
            if args.checkpoint_db:
                try:
                    from langgraph.checkpoint.sqlite import SqliteSaver
                except ImportError:
                    parser.error(
                        "SQLite 복구 사용 전 uv sync --extra recovery를 실행하세요."
                    )
                args.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
                checkpointer = stack.enter_context(
                    SqliteSaver.from_conn_string(str(args.checkpoint_db))
                )
            workflow = build_workflow(limits=limits, checkpointer=checkpointer)
            thread_id = args.thread_id or trace_id
            config = {
                "configurable": {"thread_id": thread_id},
                "metadata": {"trace_id": trace_id},
            }
            input_state = {
                "payload": {
                    "user_query": args.query.strip() or DEFAULT_QUERY,
                    "target_domain": args.target_domain,
                },
                "control": {"trace_id": trace_id},
            }
            if checkpointer:
                saved = workflow.get_state(config)
                if args.resume:
                    if not saved.values:
                        parser.error("지정한 thread에 체크포인트가 없습니다.")
                    trace_id = saved.values["control"]["trace_id"]
                    config["metadata"]["trace_id"] = trace_id
                    config["recursion_limit"] = (
                        saved.values["control"]["limits"]["max_steps"] * 3 + 10
                    )
                    input_state = None
                elif saved.values:
                    parser.error(
                        "기존 thread입니다. --resume 또는 새 --thread-id를 사용하세요."
                    )
            result = workflow.invoke(input_state, config)
    except Exception as error:  # noqa: BLE001 - keep raw provider payloads out of logs
        log_path.write_text(
            json.dumps(
                {
                    "trace_id": trace_id,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "events_path": str(events_path),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"실행 실패: {type(error).__name__}. 실행 요약: {log_path}")
        raise SystemExit(2) from None
    finally:
        if not args.mock:
            flush_traces()
        event_logger.removeHandler(handler)
        handler.close()

    payload, control = result["payload"], result["control"]
    report = payload.get("report", "")
    output_dir = OUTPUTS_DIR / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"kv_cache_evaluation_{timestamp}.md"
    pdf_path = report_path.with_suffix(".pdf")
    pdf_status = "skipped"
    if report:
        report_path.write_text(report, encoding="utf-8")
        if not args.no_pdf:
            try:
                write_pdf(report, pdf_path, max_pages=ReportBudget.from_env().max_pdf_pages)
                pdf_status = "ok"
            except Exception as error:  # noqa: BLE001 - preserve Markdown output
                pdf_status = f"failed:{type(error).__name__}"
    summary = {
        "trace_id": control["trace_id"],
        "status": "failed" if pdf_status.startswith("failed:") else control["status"],
        "workflow_status": control["status"],
        "termination_reason": (
            "pdf_export_failed" if pdf_status.startswith("failed:")
            else control["termination_reason"]
        ),
        "mock": args.mock,
        "thread_id": thread_id,
        "step_count": control["step_count"],
        "revision_count": control["revision_count"],
        "worker_count": len(payload.get("worker_results", [])),
        "failed_tasks": control["failed_tasks"],
        "last_error": control["last_error"],
        "evaluation": payload.get("evaluation"),
        "report_path": str(report_path) if report else None,
        "pdf_path": str(pdf_path) if pdf_status == "ok" else None,
        "pdf_status": pdf_status,
        "report_metrics": payload.get("report_metrics", {}),
        "events_path": str(events_path),
    }
    log_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"실행 상태: {summary['status']} / 종료 이유: {summary['termination_reason']}"
    )
    print(f"trace_id: {control['trace_id']}")
    print(
        f"Worker: {summary['worker_count']}개 / Revision: {control['revision_count']}"
    )
    print(f"보고서: {report_path}" if report else "보고서 미생성")
    print(f"PDF: {pdf_status}")
    print(f"실행 요약: {log_path}")
    print(f"이벤트 로그: {events_path}")
    if control["status"] != "completed" or pdf_status.startswith("failed:"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
