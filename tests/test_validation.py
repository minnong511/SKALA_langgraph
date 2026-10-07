"""Intermediate artifacts must reflect actual checkpoints and fail honestly."""

import json

import pytest

from kv_cache_agent import validation


def test_validation_captures_real_loops_and_recovery(tmp_path):
    pytest.importorskip("langgraph.checkpoint.sqlite")
    data, html = validation.run_validation(tmp_path)
    cases = {case["id"]: case for case in data["cases"]}
    assert data["passed"] is True
    assert len(cases) == 12
    assert len(cases["baseline"]["tasks"]) != len(cases["expanded"]["tasks"])
    coverage = cases["coverage"]
    evaluations = [
        row["detail"]
        for row in coverage["snapshots"]
        if row["node"] == "quality_evaluator"
    ]
    assert [evaluation["overall_pass"] for evaluation in evaluations] == [False, True]
    assert coverage["control"]["planning_round"] == 2
    assert cases["worker_excluded"]["control"]["failed_tasks"]
    assert cases["mandatory_failure"]["control"]["status"] == "failed"
    assert (
        cases["max_revisions"]["control"]["termination_reason"]
        == "max_report_revisions"
    )
    assert (
        cases["resume"]["recovery"]["trace_before"]
        == cases["resume"]["control"]["trace_id"]
    )
    assert cases["resume"]["recovery"]["namespaces"] == [""]
    assert html.exists()
    assert (tmp_path / "baseline.report.md").read_text().startswith("# SUMMARY")
    assert json.loads((tmp_path / "results.json").read_text())["passed"] is True
    assert "__VALIDATION_DATA__" not in html.read_text()


def test_report_script_tag_is_embedded_as_data(tmp_path):
    path = validation.write_results(
        {"report": "</script><script>alert(1)</script>"}, tmp_path
    )
    assert "</script><script>alert(1)</script>" not in path.read_text()
    assert "\\u003c/script>" in path.read_text()


def test_live_failure_does_not_expose_provider_error_message(monkeypatch):
    from kv_cache_agent import llm

    class AuthenticationError(Exception):
        status_code = 401
        code = "invalid_api_key"

    def fail():
        raise AuthenticationError("SECRET-API-KEY and provider response")

    monkeypatch.setattr(llm, "get_llm", fail)
    result = validation._live_status(True)
    assert result["status"] == "blocked"
    assert result["checks"][0]["error_code"] == "invalid_api_key"
    assert "SECRET" not in json.dumps(result)


def test_nonterminal_pending_writes_cannot_be_exported_as_finished(
    tmp_path, monkeypatch
):
    """After a process interruption, next can be empty with uncommitted writes."""
    pytest.importorskip("langgraph.checkpoint.sqlite")
    from types import SimpleNamespace
    from unittest.mock import Mock

    workflow = Mock()
    workflow.get_state.return_value = SimpleNamespace(
        values={"control": {"status": "researching"}, "payload": {}}, next=()
    )
    monkeypatch.setattr(validation, "build_workflow", lambda **kwargs: workflow)
    with pytest.raises(ValueError, match="still running"):
        validation.export_live_results(
            tmp_path / "pending.sqlite", "pending-writes", tmp_path / "review", {}
        )


def test_export_checkpoint_preserves_actual_report_and_trace(tmp_path):
    sqlite = pytest.importorskip("langgraph.checkpoint.sqlite")
    from kv_cache_agent.graph.workflow import build_workflow
    from kv_cache_agent.mock_run import mock_services

    db = tmp_path / "export.sqlite"
    config = {"configurable": {"thread_id": "export-test"}}
    with (
        mock_services(),
        validation.collect_events() as events,
        sqlite.SqliteSaver.from_conn_string(str(db)) as saver,
    ):
        result = build_workflow(checkpointer=saver).invoke(
            {"payload": {"user_query": "두 기술 평가"}}, config
        )
    event_path = tmp_path / "events.jsonl"
    event_path.write_text("\n".join(json.dumps(event) for event in events))
    data, html = validation.export_live_results(
        db,
        "export-test",
        tmp_path / "export",
        {
            "events_path": str(event_path),
            "openai": {"status": "passed", "current_run": True, "reason": "Fixture"},
        },
    )
    assert data["mode"] == "live"
    assert data["passed"] is True
    case = data["cases"][0]
    assert case["control"]["trace_id"] == result["control"]["trace_id"]
    assert case["report"] == result["payload"]["report"]
    assert case["evidence_cards"] == result["payload"]["evidence_cards"]
    assert html.exists()


def test_cli_cannot_label_mock_log_as_live(tmp_path, monkeypatch, capsys):
    summary = tmp_path / "mock.json"
    summary.write_text(json.dumps({"mock": True, "thread_id": "mock-thread"}))
    monkeypatch.setattr(
        "sys.argv",
        [
            "validation",
            "--live-run-log",
            str(summary),
            "--checkpoint-db",
            str(tmp_path / "unused.sqlite"),
        ],
    )
    with pytest.raises(SystemExit) as stopped:
        validation.main()
    assert stopped.value.code == 2
    assert "--mock 없음" in capsys.readouterr().err
    assert not (tmp_path / "unused.sqlite").exists()


def test_pending_checkpoint_is_not_presented_as_final_pass(tmp_path):
    sqlite = pytest.importorskip("langgraph.checkpoint.sqlite")
    from kv_cache_agent.graph.workflow import build_workflow
    from kv_cache_agent.mock_run import mock_services

    db = tmp_path / "pending.sqlite"
    with (
        mock_services(),
        validation.collect_events() as events,
        sqlite.SqliteSaver.from_conn_string(str(db)) as saver,
    ):
        build_workflow(checkpointer=saver, interrupt_before=["research_worker"]).invoke(
            {"payload": {"user_query": "두 기술 평가"}},
            {"configurable": {"thread_id": "pending"}},
        )
    event_path = tmp_path / "events.jsonl"
    event_path.write_text("\n".join(json.dumps(event) for event in events))
    services = {"events_path": str(event_path), "openai": {"status": "not_run"}}
    with pytest.raises(ValueError, match="still running"):
        validation.export_live_results(db, "pending", tmp_path / "view", services)
    data, _ = validation.export_live_results(
        db, "pending", tmp_path / "view", services, allow_running=True
    )
    assert data["passed"] is False
    assert data["cases"][0]["status"] == "running"
    assert "중간 결과" in data["scope"]
