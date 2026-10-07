from unittest.mock import Mock

from kv_cache_agent import observability


def test_each_fan_out_decision_is_preserved_as_trace_event(monkeypatch):
    run = Mock()
    monkeypatch.setattr(observability, "get_current_run_tree", lambda: run)
    for task_id in ("task-1", "task-2"):
        observability.record_event("trace-1", "assign_workers", "send", task_id=task_id)
    events = [call.args[0] for call in run.add_event.call_args_list]
    assert [e["kwargs"]["task_id"] for e in events] == ["task-1", "task-2"]
    assert all(e["kwargs"]["trace_id"] == "trace-1" for e in events)


def test_flush_failure_does_not_change_workflow_outcome(monkeypatch):
    import langsmith.run_trees
    import langsmith.utils

    client = Mock()
    client.flush.side_effect = RuntimeError("SECRET credentials")
    monkeypatch.setattr(langsmith.utils, "tracing_is_enabled", lambda: True)
    monkeypatch.setattr(langsmith.run_trees, "get_cached_client", lambda: client)
    observability.flush_traces()
    client.flush.assert_called_once_with(timeout=10)
