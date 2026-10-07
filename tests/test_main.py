"""Exercise the CLI path and the compact summary contract without network."""

import json
import sys

import pytest

from kv_cache_agent import main as cli


def test_mock_cli_saves_report_and_small_summary(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "OUTPUTS_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["kv-cache-agent", "--mock", "--no-pdf"])
    cli.main()
    summary = json.loads(next((tmp_path / "logs").glob("*.json")).read_text())
    assert summary["status"] == "completed"
    assert summary["termination_reason"] == "quality_pass"
    assert summary["worker_count"] == 5
    assert len(list((tmp_path / "reports").glob("*.md"))) == 1
    assert "workflow_result" not in summary
    assert summary["evaluation"]["overall_pass"] is True
    events = [
        json.loads(line)
        for line in next((tmp_path / "logs").glob("*.jsonl")).read_text().splitlines()
    ]
    assert all(event["trace_id"] == summary["trace_id"] for event in events)


def test_mock_cli_returns_nonzero_on_bound_and_preserves_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "OUTPUTS_DIR", tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["kv-cache-agent", "--mock", "--no-pdf", "--max-steps", "1"]
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    summary = json.loads(next((tmp_path / "logs").glob("*.json")).read_text())
    assert summary["status"] == "failed"
    assert summary["termination_reason"] == "max_steps"
    assert summary["report_path"] is None


def test_cli_pdf_failure_is_not_success(monkeypatch, tmp_path):
    from kv_cache_agent.tools.pdf_writer import ReportPageLimitError

    monkeypatch.setattr(cli, "OUTPUTS_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["kv-cache-agent", "--mock"])
    def fail(*args, **kwargs):
        raise ReportPageLimitError("too long")
    monkeypatch.setattr(cli, "write_pdf", fail)
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    summary = json.loads(next((tmp_path / "logs").glob("*.json")).read_text())
    assert summary["pdf_path"] is None
    assert summary["pdf_status"] == "failed:ReportPageLimitError"
    assert summary["status"] == "failed"
    assert summary["workflow_status"] == "completed"
    assert summary["termination_reason"] == "pdf_export_failed"
