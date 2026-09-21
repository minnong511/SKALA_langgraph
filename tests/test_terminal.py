import io
import json

import pytest
from rich.console import Console

from src.common.events import EventLogger
from src.common.runtime import BudgetExceeded, CallBudget
from src.common.terminal import TerminalDashboard
from src.schemas import AgentContext


def test_live_display_receives_only_redacted_events_and_keeps_json_log(tmp_path):
    output = io.StringIO()
    console = Console(file=output, force_terminal=True, width=100)
    with TerminalDashboard(console=console) as display:
        logger = EventLogger(tmp_path, "run", secrets=["private-key"], display=display)
        logger.emit("agent_start", "[red]private-key", agent="market")
        display.live.refresh()
        logger.emit("agent_return", "done", agent="market", status="completed")
        logger.emit("run_end", "review", status="needs_review", unresolved=["demo"])
    assert "private-key" not in output.getvalue()
    assert "[red]" in output.getvalue()  # Untrusted text is not interpreted as markup.
    rows = [json.loads(line) for line in logger.path.read_text().splitlines()]
    assert len(rows) == 3
    assert display.rows["market"][0] == "completed"
    assert display.status == "검토 필요"


def test_progress_resets_on_new_round_and_agent_retry(tmp_path):
    display = TerminalDashboard(console=Console(file=io.StringIO(), force_terminal=True))
    logger = EventLogger(tmp_path, "run", display=display)
    logger.emit("parallel_progress", "done", current=3, total=3)
    logger.emit("parallel_start", "retry", current=0, total=1, round=2)
    task = display.progress.tasks[display.bars["parallel"]]
    assert task.completed == 0 and task.total == 1
    logger.emit("agent_return", "done", agent="verification", status="completed")
    logger.emit("verification_progress", "done", current=8, total=8)
    logger.emit("agent_start", "retry", agent="verification", attempt=2)
    assert display.rows["verification"][:2] == ["running", 2]
    assert display.progress.tasks[display.bars["verification"]].completed == 0


def test_live_stops_on_interrupt_and_non_terminal_falls_back():
    console = Console(file=io.StringIO(), force_terminal=True)
    with pytest.raises(KeyboardInterrupt), TerminalDashboard(console=console) as display:
        raise KeyboardInterrupt
    assert display.status == "중단"
    assert not display.live.is_started
    assert not TerminalDashboard(console=Console(file=io.StringIO())).enabled


def test_budget_display_tracks_actual_limit_and_does_not_consume_calls():
    output = io.StringIO()
    console = Console(file=output, width=120)
    display = TerminalDashboard(console=console)
    display.budget = CallBudget(2, 60)
    display.budget.consume("llm")
    display.budget.consume("source_read")
    with pytest.raises(BudgetExceeded):
        display.budget.consume("search")
    console.print(display.render())
    assert "호출: 2/2 (100%)" in output.getvalue()
    assert "새 호출 차단" in output.getvalue()
    assert display.budget.snapshot()["counts"] == {"llm": 1, "source_read": 1}


def test_failed_call_records_duration_and_clears_active_display(tmp_path, monkeypatch):
    clock = iter([10.0, 12.5])
    monkeypatch.setattr("src.schemas.monotonic", lambda: next(clock))
    display = TerminalDashboard(console=Console(file=io.StringIO(), force_terminal=True))
    logger = EventLogger(tmp_path, "run", display=display)
    context = AgentContext(events=logger, agent="market", task_id="market", attempt=2)
    with pytest.raises(ValueError):
        with context.timed_call("search", "FakeSearch"):
            assert len(display.active_calls) == 1
            raise ValueError("private error body")
    rows = [json.loads(line) for line in logger.path.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["call_start", "call_end"]
    assert rows[0]["details"]["call_id"] == rows[1]["details"]["call_id"]
    assert rows[1]["details"]["elapsed_seconds"] == 2.5
    assert rows[1]["details"]["status"] == "failed"
    assert rows[1]["attempt"] == 2
    assert "private error body" not in logger.path.read_text()
    assert not display.active_calls
    assert display.call_times["search"] == (1, 2.5, 2.5, 2.5)
