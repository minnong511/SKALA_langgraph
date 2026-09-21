"""One execution wrapper handles agent lifecycle events and immutable attempts."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from threading import Event, Lock, Thread
from time import monotonic
from typing import Callable

from src.schemas import AgentContext, AgentRequest, AgentResult


class BudgetExceeded(RuntimeError):
    pass


class CallBudget:
    def __init__(self, max_calls: int, max_seconds: float, reserved_calls: int = 0):
        self.max_calls = max_calls
        self.max_seconds = max_seconds
        self.started = monotonic()
        self._lock = Lock()
        self.counts: Counter = Counter()
        self.reserved_calls = min(reserved_calls, max_calls // 4)

    def check(self):
        if monotonic() - self.started >= self.max_seconds:
            raise BudgetExceeded("전체 실행 시간 한도에 도달했습니다.")
        if sum(self.counts.values()) >= self.max_calls:
            raise BudgetExceeded("전체 호출 한도에 도달했습니다.")

    def consume(self, kind: str, *, research=False):
        with self._lock:
            self.check()
            if research and sum(self.counts.values()) >= self.max_calls - self.reserved_calls:
                raise BudgetExceeded("후속 검증 및 보고서를 위한 호출 예산 보존: 조사 호출 중단")
            self.counts[kind] += 1

    @property
    def research_exhausted(self):
        with self._lock:
            return sum(self.counts.values()) >= self.max_calls - self.reserved_calls

    def snapshot(self):
        """Return an atomic view for monitoring without consuming budget."""
        with self._lock:
            used = sum(self.counts.values())
            elapsed = max(0.0, monotonic() - self.started)
            return {
                "used": used,
                "limit": self.max_calls,
                "remaining": max(0, self.max_calls - used),
                "counts": dict(self.counts),
                "reserved_calls": self.reserved_calls,
                "elapsed": elapsed,
                "seconds_limit": self.max_seconds,
                "seconds_remaining": max(0.0, self.max_seconds - elapsed),
            }

    @property
    def exhausted(self):
        try:
            self.check()
            return False
        except BudgetExceeded:
            return True


class AgentRuntime:
    def __init__(self, context: AgentContext, artifacts, *, heartbeat_seconds=15):
        self.context = context
        self.artifacts = artifacts
        self.heartbeat_seconds = heartbeat_seconds
        self._lock = Lock()
        self._round = 0
        self._expected: set[str] = set()
        self._completed: set[str] = set()

    def start_round(self, agents: list[str], round_number: int):
        with self._lock:
            self._round, self._expected, self._completed = round_number, set(agents), set()
        self.context.events.emit(
            "parallel_start",
            f"관점별 평가 0/{len(agents)} 완료",
            current=0,
            total=len(agents),
            round=round_number,
        )

    def execute(
        self, name: str, agent: Callable, request: AgentRequest, results: dict[str, AgentResult]
    ) -> AgentResult:
        context = replace(
            self.context, results=results, agent=name, task_id=request.task_id, attempt=request.attempt
        )
        started = monotonic()
        stop = Event()
        context.emit(request, "agent_start", f"{name} 시작")

        def heartbeat():
            while not stop.wait(self.heartbeat_seconds):
                context.emit(
                    request, "progress", f"{name} 실행 중", elapsed_seconds=round(monotonic() - started, 3)
                )

        thread = Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            if context.budget:
                context.budget.consume(
                    "agent", research=name in ("technical", "market", "stakeholder", "domain")
                )
            value = agent(request, context)
            result = value if isinstance(value, AgentResult) else AgentResult.model_validate(value)
            if (result.agent, result.task_id, result.attempt) != (name, request.task_id, request.attempt):
                raise ValueError("에이전트 결과의 작업 식별자 또는 시도 번호가 일치하지 않습니다.")
        except Exception as exc:
            result = AgentResult(
                task_id=request.task_id,
                agent=name,
                attempt=request.attempt,
                status="failed",
                summary=f"{name} 실행 실패",
                gaps=[f"{name} 결과를 확인하지 못했습니다."],
                errors=[f"{type(exc).__name__}: {exc}"],
            )
            context.emit(request, "error", result.summary, errors=result.errors)
        finally:
            stop.set()
            thread.join(timeout=1)
        try:
            path = self.artifacts.save_result(result)
            context.emit(request, "file_saved", "시도별 결과 저장", path=str(path))
        except Exception as exc:
            result = result.model_copy(
                update={"status": "failed", "errors": [*result.errors, f"결과 저장 실패: {exc}"]}
            )
            context.emit(request, "error", "시도별 결과 저장 실패", errors=result.errors)
        context.emit(
            request,
            "agent_return",
            result.summary,
            status=result.status,
            evidence_count=len(result.evidence_cards),
            elapsed_seconds=round(monotonic() - started, 3),
        )
        with self._lock:
            if name in self._expected and name not in self._completed:
                self._completed.add(name)
                context.emit(
                    request,
                    "parallel_progress",
                    f"관점별 평가 {len(self._completed)}/{len(self._expected)} 완료",
                    current=len(self._completed),
                    total=len(self._expected),
                    round=self._round,
                )
        return result
