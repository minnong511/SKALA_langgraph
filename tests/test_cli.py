"""Offline end-to-end checks for artifact publishing and honest CLI status."""

import json
from pathlib import Path

from pydantic import SecretStr
from pypdf import PdfReader

from main import cli, run_workflow
from src.config import Settings
from src.demo import demo_agents
from src.schemas import AgentContext, AgentRequest


def settings_for(tmp_path, **overrides):
    return Settings(output_dir=tmp_path, quiet=True, **overrides)


def test_demo_cli_produces_inspectable_draft_files(tmp_path, capsys):
    status = cli(
        [
            "--demo",
            "--quiet",
            "--run-id",
            "cli-demo",
            "--output-dir",
            str(tmp_path),
            "--env-file",
            str(tmp_path / "missing.env"),
        ]
    )
    assert status == 0
    directory = tmp_path / "cli-demo"
    persisted = json.loads((directory / "result.json").read_text())
    assert persisted["status"] == "needs_review"
    assert "예시 데이터" in (directory / "draft-report.md").read_text()
    assert len(PdfReader(directory / "draft-report.pdf").pages) >= 1
    assert not (directory / "report.md").exists()
    assert not (directory / "report.pdf").exists()
    assert len(list((directory / "tasks").rglob("attempt-*.json"))) >= 8
    output = capsys.readouterr().out
    assert "needs_review" in output
    assert str(directory / "draft-report.md") in output
    rows = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    assert rows[0]["event"] == "run_start"
    assert rows[-1]["event"] == "run_end"
    assert rows[-1]["details"]["status"] == "needs_review"


def test_secrets_are_redacted_from_failed_attempts_reports_and_logs(tmp_path):
    secret = "private-provider-key-12345"
    settings = settings_for(tmp_path, llm_api_key=SecretStr(secret))
    implementations = demo_agents()
    research = implementations["technical"]
    report = implementations["report"]

    def failing_technical(request, context):
        if request.attempt == 1:
            raise RuntimeError(f"Provider rejected {secret}")
        return research(request, context)

    def sensitive_report(request, context):
        result = report(request, context)
        result.report_markdown = result.report_markdown.replace(
            "# 1. 분석 배경", f"진단: {secret}\n\n# 1. 분석 배경"
        )
        result.data["credentials"] = {"api_key": secret}
        return result

    implementations.update(technical=failing_technical, report=sensitive_report)
    state = run_workflow(
        settings, AgentRequest(run_id="private", task_id="workflow"), demo=True, agents=implementations
    )
    assert state["technical_result"].attempt == 2
    assert state["report_result"].report_markdown
    for path in (tmp_path / "private").rglob("*"):
        if path.suffix in {".json", ".jsonl", ".md"}:
            assert secret not in path.read_text(), str(path)
        elif path.suffix == ".pdf":
            assert secret not in "\n".join(page.extract_text() for page in PdfReader(path).pages)
    failed = json.loads((tmp_path / "private/tasks/technical/attempt-1.json").read_text())
    assert failed["status"] == "failed"
    assert "[REDACTED]" in failed["errors"][0]


def test_pdf_failure_preserves_markdown_and_never_claims_pdf_saved(tmp_path):
    settings = settings_for(tmp_path, pdf_font_path=tmp_path / "missing-font.ttf", max_report_revisions=0)
    request = AgentRequest(run_id="font-failure", task_id="workflow", limits=settings.limits)
    state = run_workflow(settings, request, demo=True)
    assert state["status"] == "needs_review"
    assert Path(state["final_artifacts"]["markdown"]).is_file()
    assert "pdf" not in state["final_artifacts"]
    assert not list((tmp_path / "font-failure").glob("*.pdf"))
    rows = [json.loads(line) for line in (tmp_path / "font-failure/events.jsonl").read_text().splitlines()]
    assert not any(row["event"] == "file_saved" and row["message"] == "PDF 저장" for row in rows)
    assert any("PDF 저장 실패" in issue for issue in state["review"]["issues"])


def test_duplicate_cli_run_does_not_replace_any_output(tmp_path, capsys):
    args = [
        "--demo",
        "--quiet",
        "--run-id",
        "duplicate",
        "--output-dir",
        str(tmp_path),
        "--env-file",
        str(tmp_path / "missing.env"),
    ]
    assert cli(args) == 0
    directory = tmp_path / "duplicate"
    before = {
        str(path.relative_to(directory)): path.read_bytes() for path in directory.rglob("*") if path.is_file()
    }
    assert cli(args) == 1
    after = {
        str(path.relative_to(directory)): path.read_bytes() for path in directory.rglob("*") if path.is_file()
    }
    assert before == after


def test_injected_live_dependencies_still_use_runtime_and_budget(tmp_path):
    # Keep deterministic agents while exercising the non-demo workflow boundary.
    implementations = demo_agents()
    real_supervisor = implementations["supervisor"]

    def fixture_supervisor(request, context):
        context.demo = True
        return real_supervisor(request, context)

    implementations["supervisor"] = fixture_supervisor
    settings = settings_for(tmp_path, max_total_calls=2)
    state = run_workflow(
        settings,
        AgentRequest(run_id="live-injected", task_id="workflow", limits=settings.limits),
        demo=False,
        agents=implementations,
        context=AgentContext(),
    )
    assert state["status"] == "needs_review"
    assert sum(state["call_counts"].values()) == 2
    assert Path(state["final_artifacts"]["result"]).is_file()
    assert (tmp_path / "live-injected/tasks/technical/attempt-1.json").is_file()


def test_successful_live_review_publishes_final_names_and_complete_manifest(tmp_path):
    class StructuredReply:
        def __init__(self, schema):
            self.schema = schema

        def invoke(self, messages):
            if self.schema.__name__ == "PlanOutput":
                return self.schema(objective="실행 경로 검증용 계획")
            return self.schema(passed=True, issues=[])

    class FixtureLLM:
        def with_structured_output(self, schema):
            return StructuredReply(schema)

    # Only worker/model responses are fixtures: scheduling, review, budget, and export are real.
    settings = settings_for(tmp_path)
    state = run_workflow(
        settings,
        AgentRequest(run_id="review-passed", task_id="workflow"),
        demo=False,
        agents=demo_agents(),
        context=AgentContext(llm=FixtureLLM()),
    )
    assert state["status"] == "completed"
    assert state["review"]["passed"] is True
    assert state["call_counts"]["llm"] == 2
    directory = tmp_path / "review-passed"
    assert (directory / "report.pdf").is_file()
    assert (directory / "report.md").is_file()
    assert not (directory / "draft-report.pdf").exists()
    assert not (directory / "draft-report.md").exists()
    persisted = json.loads((directory / "result.json").read_text())
    assert persisted["final_artifacts"] == state["final_artifacts"]
    assert all(Path(path).is_file() for path in persisted["final_artifacts"].values())
