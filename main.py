"""CLI: configure dependencies, execute the graph, and save reviewable artifacts."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
from pathlib import Path
from time import monotonic
from uuid import uuid4

from src.common.artifacts import ArtifactStore, redact
from src.common.events import EventLogger
from src.common.runtime import AgentRuntime, CallBudget
from src.config import Settings, create_llm
from src.exporters.pdf import export_pdf
from src.graph import build_graph, get_results, initial_state
from src.schemas import AgentContext, AgentRequest


def run_workflow(settings: Settings, request: AgentRequest, *, demo=False, agents=None, context=None):
    if not demo and context is None:
        settings.validate_live()
    store = ArtifactStore(settings.output_dir, request.run_id, secrets=settings.secrets)
    events = EventLogger(store.run_dir, request.run_id, secrets=settings.secrets, quiet=settings.quiet)
    started = monotonic()
    events.emit("run_start", "예시 실행 시작" if demo else "실행 시작")
    state = initial_state(request, settings.public_config())
    state["final_artifacts"]["events"] = str(store.run_dir / "events.jsonl")
    try:
        if context is None:
            context = AgentContext(demo=demo)
            if not demo:
                from src.tools.retriever import FaissRetriever
                from src.tools.source_reader import SourceReader
                from src.tools.web_search import TavilySearch

                context.llm = create_llm(settings)
                context.structured_output_method = "function_calling"
                context.retriever = FaissRetriever(
                    settings.index_path, embedding_model=settings.embedding_model
                )
                context.web_search = TavilySearch(
                    settings.tavily_api_key.get_secret_value(),
                    timeout_seconds=settings.search_timeout_seconds,
                )
                context.source_reader = SourceReader(
                    settings.raw_dir, timeout_seconds=settings.search_timeout_seconds
                )
        context.events = events
        context.demo = demo
        context.market_rag = settings.market_rag
        context.stakeholder_rag = settings.stakeholder_rag
        context.pdf_font_path = settings.pdf_font_path
        context.budget = CallBudget(request.limits.max_total_calls, request.limits.max_run_seconds)
        runtime = AgentRuntime(context, store, heartbeat_seconds=settings.heartbeat_seconds)
        if demo and agents is None:
            from src.demo import demo_agents

            agents = demo_agents()
        graph = build_graph(runtime, agents)
        # Values mode retains the most recent complete state if a later node fails.
        for value in graph.stream(
            state,
            config={"recursion_limit": max(100, request.limits.max_total_calls * 3)},
            stream_mode="values",
        ):
            state = dict(value)
        state["call_counts"] = dict(context.budget.counts)
    except Exception as exc:
        state["status"] = "failed"
        state["run_error"] = f"{type(exc).__name__}: {exc}"
        events.emit("error", "전체 실행 실패", error=state["run_error"])
    report = state.get("report_result")
    if report and report.report_markdown:
        # A draft is published first and only promoted after the final PDF layout check.
        markdown = redact(report.report_markdown, settings.secrets)
        try:
            pdf_info = export_pdf(
                markdown, store.run_dir / "draft-report.pdf", font_path=settings.pdf_font_path
            )
            if not pdf_info["summary_within_half_page"]:
                state["status"] = "needs_review" if state["status"] != "failed" else "failed"
                state.setdefault("review", {}).setdefault("issues", []).append(
                    "최종 PDF SUMMARY가 반 페이지를 초과하거나 누락되었습니다."
                )
            state["pdf_layout"] = pdf_info
            if state["status"] == "completed":
                destination = store.run_dir / "report.pdf"
                (store.run_dir / "draft-report.pdf").rename(destination)
                pdf_info["path"] = str(destination)
            state.setdefault("final_artifacts", {})["pdf"] = pdf_info["path"]
            events.emit("file_saved", "PDF 저장", path=pdf_info["path"])
        except Exception as exc:
            state["status"] = "needs_review" if state["status"] != "failed" else "failed"
            state.setdefault("review", {}).setdefault("issues", []).append(
                f"PDF 저장 실패: {type(exc).__name__}: {exc}"
            )
            events.emit("error", "PDF 저장 실패", error=str(exc))
        name = "report.md" if state["status"] == "completed" else "draft-report.md"
        try:
            path = store.save_text(name, markdown)
            state.setdefault("final_artifacts", {})["markdown"] = str(path)
            events.emit("file_saved", "Markdown 저장", path=str(path))
        except Exception as exc:
            state["status"] = "failed"
            events.emit("error", "Markdown 저장 실패", error=str(exc))
    elif state["status"] != "failed":
        state["status"] = "needs_review"
        state.setdefault("review", {}).setdefault("issues", []).append("저장 가능한 보고서가 없습니다.")
    state.setdefault("final_artifacts", {})["events"] = str(store.run_dir / "events.jsonl")
    state["elapsed_seconds"] = round(monotonic() - started, 3)
    gaps = list(
        dict.fromkeys(
            [gap for result in get_results(state).values() for gap in result.gaps]
            + state.get("review", {}).get("issues", [])
        )
    )
    if demo:
        gaps.insert(0, "예시 데이터입니다. 실제 기술 조사 결과가 아닙니다.")
    state["unresolved"] = gaps
    try:
        state["final_artifacts"]["result"] = str(store.run_dir / "result.json")
        path = store.save_json("result.json", state)
        state["final_artifacts"]["result"] = str(path)
        events.emit("file_saved", "실행 상태 저장", path=str(path))
    except Exception as exc:
        state["final_artifacts"].pop("result", None)
        state["status"] = "failed"
        events.emit("error", "실행 상태 저장 실패", error=str(exc))
    events.emit(
        "run_end",
        {"completed": "완료", "needs_review": "검토 필요", "failed": "실패"}.get(
            state["status"], state["status"]
        ),
        status=state["status"],
        elapsed_seconds=state["elapsed_seconds"],
        unresolved=gaps,
        paths=state["final_artifacts"],
    )
    return state


def cli(argv=None):
    parser = argparse.ArgumentParser(description="TurboQuant / ITME 근거 기반 다관점 평가")
    parser.add_argument("--demo", action="store_true", help="키·원문·네트워크 없이 합성 예시로 통합 실행")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--technologies", nargs="+", default=["TurboQuant", "ITME"])
    parser.add_argument("--domain", default="클라우드 기반 LLM 서빙")
    parser.add_argument("--as-of-date", type=date.fromisoformat, default=date.today())
    parser.add_argument("--quiet", action="store_true", default=None)
    args = parser.parse_args(argv)
    try:
        settings = Settings.from_env(args.env_file, output_dir=args.output_dir, quiet=args.quiet)
        run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
        request = AgentRequest(
            run_id=run_id,
            task_id="workflow",
            technologies=args.technologies,
            domain=args.domain,
            as_of_date=args.as_of_date,
            limits=settings.limits,
        )
        state = run_workflow(settings, request, demo=args.demo)
    except Exception as exc:
        print(f"실행 준비 실패: {redact(str(exc), settings.secrets if 'settings' in locals() else [])}")
        return 1
    print(f"상태: {state['status']}")
    for label, path in state.get("final_artifacts", {}).items():
        print(f"{label}: {path}")
    return 1 if state["status"] == "failed" else (0 if args.demo or state["status"] == "completed" else 2)


if __name__ == "__main__":
    raise SystemExit(cli())
