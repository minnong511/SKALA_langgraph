"""Re-synthesize verified checkpoint evidence; preserve the original run/artifacts."""

import argparse
import json
import logging
import re
import shutil
from copy import deepcopy
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from langsmith import Client, trace
from langsmith.utils import tracing_is_enabled

from kv_cache_agent.agents.quality_evaluator import quality_evaluator_agent
from kv_cache_agent.agents.report_writer import report_writer_agent
from kv_cache_agent.agents.synthesis import synthesis_agent
from kv_cache_agent.config import OPENAI_MODEL, OUTPUTS_DIR, ReportBudget
from kv_cache_agent.graph.state import merge_payload
from kv_cache_agent.graph.workflow import build_workflow
from kv_cache_agent.observability import node_span, record_event
from kv_cache_agent.tools.pdf_writer import pdf_page_count, write_pdf


def write_viewer(report, summary, output_dir):
    """Local, responsive reader; statistics describe editing, not new research."""
    lines, references, index = [], False, 0
    for line in report.splitlines():
        if not line.strip():
            continue
        if line.startswith("# "):
            index += 1
            references = line == "# REFERENCE"
            lines.append(f'<h2 id="chapter-{index}">{escape(line[2:])}</h2>')
        elif line.startswith("## "):
            lines.append(f"<h3>{escape(line[3:])}</h3>")
        else:
            content = escape(line)
            if references:
                match = re.match(r"\[(\d+)\]", line)
                anchor = f' id="ref-{match[1]}"' if match else ""
                lines.append(f"<p class=reference{anchor}>{content}</p>")
            else:
                content = re.sub(
                    r"\[(\d+)\]", r'<a href="#ref-\1" class=citation>[\1]</a>', content
                )
                lines.append(f"<p>{content}</p>")
    evaluation = summary["evaluation"]
    badges = "".join(
        f'<span>{label}: {"PASS" if evaluation[key] else "FAIL"}</span>'
        for key, label in (
            ("groundedness", "근거 연결"), ("neutrality", "중립성"),
            ("bias_control", "편향 통제"), ("perspective_coverage", "관점 커버리지"),
        )
    )
    page = """<!doctype html><html lang=ko><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><link rel=icon href="data:,">
<title>KV Cache 평가 | 해석을 보존한 보고서</title><style>
*{box-sizing:border-box}body{margin:0;color:#24334a;background:#f3f5f8;font:16px/1.9 -apple-system,BlinkMacSystemFont,'Apple SD Gothic Neo',sans-serif}
header{background:#17324d;color:white;padding:32px max(22px,calc((100vw - 920px)/2))}header p{margin:8px 0;font-size:14px}h1{font-size:30px;line-height:1.4;margin:0 0 12px}a{color:#245ccd}header a{color:#c7deff}nav{display:flex;gap:20px;flex-wrap:wrap}
main{max-width:980px;margin:24px auto;padding:0 22px}.badges{display:flex;gap:10px;flex-wrap:wrap}.badges span{background:#e4f3eb;color:#146441;border-radius:7px;padding:6px 12px;font-size:13px}
article,details{background:white;border:1px solid #dce4ec;border-radius:10px;padding:32px;margin:20px 0}h2{font-size:23px;color:#17324d;margin:32px 0 16px;border-bottom:1px solid #dce4ec;padding-bottom:8px;scroll-margin-top:20px}h2:first-child{margin-top:0}h3{font-size:18px;line-height:1.5;margin:28px 0 12px}p{margin:12px 0;overflow-wrap:anywhere}.reference{font-size:12px;color:#5b6880}.citation{font-size:12px;text-decoration:none}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}summary{cursor:pointer}.scope{color:#617085;font-size:13px}
@media(max-width:600px){body{font-size:15px}h1{font-size:24px}main{padding:0 12px}article,details{padding:20px}h2{font-size:21px}h3{font-size:17px}}
</style></head><body><header><h1>TurboQuant, CXL 기반 KV Cache 평가</h1>"""
    page += f'<p>기존 검증 근거를 다시 통합하고 문장을 편집한 보고서 · 참고문헌 포함 {summary["pdf_page_count"]}페이지</p>'
    page += '<nav><a href="report.pdf" download>PDF 다운로드</a><a href="report.md" download>Markdown 다운로드</a>'
    if summary.get("run_url"):
        page += f'<a href="{escape(summary["run_url"], quote=True)}" target=_blank rel=noopener>LangSmith Trace</a>'
    page += '</nav></header><main><div class=badges>' + badges + '</div>'
    page += '<p class=scope>재실행 범위: Synthesizer → Report Writer → Quality Evaluator. Worker와 검색은 다시 실행하지 않았습니다. 원본 근거와 상세 한계 기록은 보존했습니다.</p>'
    page += '<details><summary>평가 이유와 실행 결과</summary><pre>' + escape(json.dumps(summary, ensure_ascii=False, indent=2)) + '</pre></details>'
    page += '<article>' + '\n'.join(lines) + '</article></main></body></html>'
    (output_dir / "index.html").write_text(page, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="검증된 근거로 가독성과 분량을 개선하고 실제 품질 재평가")
    parser.add_argument("--checkpoint-db", type=Path, required=True)
    parser.add_argument("--thread-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-revisions", type=int, default=2)
    args = parser.parse_args()
    if not 0 <= args.max_revisions <= 3:
        parser.error("--max-revisions must be 0..3")
    with SqliteSaver.from_conn_string(str(args.checkpoint_db)) as cp:
        saved = build_workflow(checkpointer=cp).get_state(
            {"configurable": {"thread_id": args.thread_id}}
        )
        if not saved.values or not saved.values.get("payload", {}).get("usable_evidence_cards"):
            parser.error("검증된 근거를 가진 체크포인트가 필요합니다.")
        if saved.next or saved.values["control"]["status"] not in {"completed", "best_effort"}:
            parser.error("완료된 실행의 체크포인트를 사용하세요.")
        state = deepcopy(saved.values)
    source_trace_id = state["control"]["trace_id"]
    before_pages = pdf_page_count(state["payload"]["report"])
    before_chars = len(state["payload"]["report"])
    trace_id = str(uuid4())
    state["control"] = {
        **state["control"], "trace_id": trace_id,
        "step_count": 0, "revision_count": 0, "status": "editing",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    event_logger = logging.getLogger("kv_cache_agent.events")
    handler = logging.FileHandler(args.output_dir / "events.jsonl", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    event_logger.addHandler(handler)
    event_logger.setLevel(logging.INFO)
    client = Client(timeout_ms=10_000)
    steps, attempts = [], []
    with trace("readable-report-revision", client=client, inputs={}, metadata={
        "trace_id": trace_id, "source_trace_id": source_trace_id,
        "source_thread": args.thread_id, "purpose": "re-synthesis and editing; no new Workers or search",
    }) as root:
        for revision in range(args.max_revisions + 1):
            state["control"]["revision_count"] = revision
            for name, agent in (
                ("synthesis", synthesis_agent), ("report_writer", report_writer_agent),
                ("quality_evaluator", quality_evaluator_agent),
            ):
                print(f"revision={revision} node={name}", flush=True)
                with node_span(trace_id, name, revision_count=revision):
                    update = agent(state)
                state["payload"] = merge_payload(state["payload"], update["payload"])
                state["control"]["step_count"] += 1
                steps.append(name)
            evaluation = state["payload"]["evaluation"]
            attempts.append({
                "revision": revision, "evaluation": evaluation,
                "synthesis_status": state["payload"]["synthesis"]["status"],
                "synthesis_errors": state["payload"]["synthesis"].get("errors", []),
                "report_metrics": state["payload"].get("report_metrics"),
            })
            passed = evaluation["overall_pass"]
            record_event(trace_id, "editorial_quality_routing", "pass" if passed else "revise",
                         evaluation["groundedness_reason"], revision_count=revision)
            if passed:
                break
        state["control"]["status"] = "completed" if passed else "best_effort"
        state["control"]["termination_reason"] = "quality_pass" if passed else "max_editorial_revisions"
        root.end(outputs={"status": state["control"]["status"], "quality_pass": passed})
    event_logger.removeHandler(handler)
    handler.close()
    if tracing_is_enabled():
        client.flush(timeout=10)
    report = state["payload"]["report"]
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    report_path = OUTPUTS_DIR / "reports" / f"kv_cache_readable_{timestamp}.md"
    report_path.write_text(report, encoding="utf-8")
    pdf_path = write_pdf(report, report_path.with_suffix(".pdf"), max_pages=ReportBudget.from_env().max_pdf_pages)
    summary = {
        "mode": "real_api_report_revision", "model": OPENAI_MODEL,
        "status": state["control"]["status"], "trace_id": trace_id,
        "source_trace_id": source_trace_id, "source_thread": args.thread_id,
        "workers_executed": 0, "searches_executed": 0,
        "reused_usable_evidence_count": len(state["payload"]["usable_evidence_cards"]),
        "before_pdf_pages": before_pages, "pdf_page_count": pdf_page_count(report),
        "before_chars": before_chars, "report_chars": len(report),
        "report_path": str(report_path), "pdf_path": str(pdf_path),
        "evaluation": state["payload"]["evaluation"], "steps": steps, "attempts": attempts,
        "run_url": client.get_run_url(run=root) if tracing_is_enabled() else None,
    }
    (args.output_dir / "results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "revised-state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(report_path, args.output_dir / "report.md")
    shutil.copy2(pdf_path, args.output_dir / "report.pdf")
    write_viewer(report, summary, args.output_dir)
    print(json.dumps({key: summary[key] for key in ("status", "pdf_page_count", "report_path", "pdf_path", "run_url")}, ensure_ascii=False), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
