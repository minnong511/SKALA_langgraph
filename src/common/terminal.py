"""Live terminal view fed exclusively by already-redacted events."""

from time import monotonic

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

NAMES = {
    "supervisor": "Supervisor",
    "technical": "기술 조사",
    "market": "시장",
    "stakeholder": "이해관계자",
    "domain": "도메인",
    "verification": "근거 검증",
    "synthesis": "평가 종합",
    "report": "보고서",
}
STATUSES = {
    "waiting": "대기",
    "running": "실행 중",
    "completed": "완료",
    "failed": "실패",
    "partial": "부분 완료",
    "needs_review": "검토 필요",
}


class TerminalDashboard:
    def __init__(self, *, enabled=True, console=None):
        self.console = console or Console()
        self.enabled = enabled and self.console.is_terminal
        self.rows = {name: ["waiting", 0, ""] for name in NAMES}
        self.status = "준비 중"
        self.started = monotonic()
        self.message = "설정, 모델, 인덱스 준비"
        self.notice = ""
        self.progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            TextColumn("{task.completed:.0f}/{task.total:.0f}"),
            auto_refresh=False,
        )
        self.bars = {}
        self.live = None
        self.spinner = Spinner("dots", text="현재 작업 처리 중")
        self.budget = None
        self.final_budget = None
        self.active_calls = {}
        self.call_times = {}

    def render(self):
        table = Table(expand=True, box=None)
        for title in ("단계", "상태", "호출 회차", "최근 작업"):
            table.add_column(title, overflow="fold")
        for name, (status, attempt, message) in self.rows.items():
            active = [value for value in self.active_calls.copy().values() if value[0] == name]
            if active:
                message = ", ".join(f"{kind} {monotonic() - start:.1f}초" for _, kind, start in active)
            table.add_row(
                Text(NAMES[name]),
                Text(STATUSES.get(status, status)),
                str(attempt or "-"),
                Text(message[:140]),
            )
        parts = [Text(f"상태: {self.status} | 경과: {monotonic() - self.started:.1f}초"), table]
        if self.budget is not None:
            budget = self.final_budget or self.budget.snapshot()
            used, limit = budget["used"], budget["limit"]
            ratio = used / limit if limit else 0
            style = "red" if ratio >= 1 else ("yellow" if ratio >= 0.8 else "cyan")
            counts = budget["counts"]
            parts.insert(
                1,
                Text(f"호출: {used}/{limit} ({ratio:.0%}) | 남은 호출: {budget['remaining']}회", style=style),
            )
            parts.insert(
                2,
                Text(
                    f"에이전트 {counts.get('agent', 0)} | LLM {counts.get('llm', 0)} | "
                    f"검색 {counts.get('search', 0)} | 원문 읽기 {counts.get('source_read', 0)}"
                ),
            )
            parts.insert(
                3,
                Text(
                    f"조사 중단 기준: {limit - budget['reserved_calls']}회 | "
                    f"후속 단계 보존: {budget['reserved_calls']}회"
                ),
            )
            parts.insert(
                3,
                Text(
                    f"작업 시간 한도: {budget['elapsed']:.1f}/{budget['seconds_limit']:.0f}초 | "
                    f"남은 시간: {budget['seconds_remaining']:.1f}초"
                ),
            )
            if not budget["remaining"] or not budget["seconds_remaining"]:
                parts.insert(4, Text("실행 한도 도달, 새 호출 차단", style="red"))
        if self.status in ("준비 중", "실행 중"):
            parts.append(self.spinner)
        if self.bars:
            parts.append(self.progress)
        for kind, values in self.call_times.copy().items():
            count, latest, total, maximum = values
            parts.append(
                Text(
                    f"{kind}: 종료 {count}회 | 최근 {latest:.2f}초 | 평균 {total / count:.2f}초 | "
                    f"누적 {total:.2f}초 | 최대 {maximum:.2f}초"
                )
            )
        if self.call_times:
            parts.append(Text("호출 시간: 실패 포함, 병렬 호출 누적 시간은 실제 경과와 다름", style="dim"))
        parts.append(Text(self.message[:240]))
        if self.notice:
            parts.append(Text(self.notice[:300], style="yellow"))
        parts.append(Text("진행 막대: 해당 회차 처리 건수, 전체 예상 시간 비율 아님", style="dim"))
        return Panel(Group(*parts), title="SKALA 실행 상태", border_style="cyan")

    def __enter__(self):
        if self.enabled:
            self.live = Live(console=self.console, get_renderable=self.render, refresh_per_second=4)
            self.live.start(refresh=True)
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.live:
            if exc_type:
                self.status = "중단" if issubclass(exc_type, KeyboardInterrupt) else "실패"
                self.live.update(self.render(), refresh=True)
            self.live.stop()

    def update(self, entry):
        event, agent, details = entry["event"], entry["agent"], entry["details"]
        self.message = str(entry["message"])
        if event == "call_start":
            self.active_calls[details["call_id"]] = (agent, details["kind"], monotonic())
        elif event == "call_end":
            self.active_calls.pop(details["call_id"], None)
            duration = details["elapsed_seconds"]
            count, _, total, maximum = self.call_times.get(details["kind"], (0, 0, 0, 0))
            self.call_times[details["kind"]] = (count + 1, duration, total + duration, max(maximum, duration))
            self.message = f"{details['kind']} {details['operation']} {duration:.2f}초 ({details['status']})"
        if event == "run_start":
            self.status = "실행 중"
        if agent in self.rows:
            row = self.rows[agent]
            row[1:] = [entry["attempt"], self.message]
            if event == "agent_start":
                row[0] = "running"
                if agent == "verification" and "verification" in self.bars:
                    self.progress.update(self.bars["verification"], completed=0)
            elif event == "agent_return":
                row[0] = details.get("status", "completed")
        if event == "task_assigned" and details.get("target_agent") in self.rows:
            self.rows[details["target_agent"]][0] = "waiting"
        if event in ("parallel_start", "parallel_progress", "verification_progress"):
            key = "verification" if event == "verification_progress" else "parallel"
            label = "근거 검증" if key == "verification" else "병렬 평가 처리"
            total, current = details.get("total", 0), details.get("current", 0)
            if total > 0:
                if key not in self.bars:
                    self.bars[key] = self.progress.add_task(label, total=total)
                self.progress.update(self.bars[key], total=total, completed=current)
        if event in ("error", "search_error", "limit_reached"):
            self.notice = self.message + " " + str(details.get("error", details.get("reason", "")))
        if event == "run_end":
            if self.budget is not None:
                self.final_budget = self.budget.snapshot()
            self.status = STATUSES.get(details.get("status"), str(details.get("status")))
            self.notice = "; ".join(details.get("unresolved", [])) or self.notice
        if self.live:
            self.live.update(self.render(), refresh=event == "run_end")
