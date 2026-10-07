"""Reproducible intermediate checks and a local, source-backed result viewer."""

import argparse
import json
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from collections import Counter
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver

from kv_cache_agent.agents.quality_evaluator import (
    check_report_rules,
    quality_evaluator_agent,
)
from kv_cache_agent.agents.report_writer import report_writer_agent
from kv_cache_agent.agents.research_worker import _research_once
from kv_cache_agent.config import OPENAI_MODEL, OUTPUTS_DIR, ROOT_DIR, WorkflowLimits
from kv_cache_agent.graph.workflow import build_workflow
from kv_cache_agent.mock_run import mock_services
from kv_cache_agent.schemas.tasks import REQUIRED_PERSPECTIVES

PLAN = [
    {
        "stage": "환경과 모델",
        "inspect": "브랜치, 실제 모델 설정, 테스트, API 인증",
        "accept": "feature 브랜치, gpt-6-luna, 테스트 통과. 실제 API는 별도 인증 성공 필요",
    },
    {
        "stage": "계획",
        "inspect": "ResearchPlan, Task 수, 관점, 검색어, 출처 유형",
        "accept": "필수 네 관점 포함, Task ID 고유, 다른 입력에서 Task 수 변화",
    },
    {
        "stage": "Worker와 Reducer",
        "inspect": "Send 수, 완료 결과, 재시도, Evidence ID",
        "accept": "Task별 결과 한 개, 병합 손실 없음, 실패 결과의 근거 제외",
    },
    {
        "stage": "근거 검증",
        "inspect": "verified/partial 카드, 원문 URL, 페이지, 제외 이유",
        "accept": "인용 가능한 근거와 제외 근거 구분. 실제 자료의 의미 일치는 사람이 표본 확인",
    },
    {
        "stage": "종합",
        "inspect": "관점별 통합, 일치점과 상충점, 사실과 추론",
        "accept": "Worker 문자열 단순 연결 없이 근거 ID 유지. 승자 선정과 조건 다른 수치의 직접 비교 제한",
    },
    {
        "stage": "보고서",
        "inspect": "SUMMARY, 네 관점, 한계, REFERENCE, 인용 연결",
        "accept": "본문 번호 → Evidence ID → 원문 URL 연결, 필수 절 존재",
    },
    {
        "stage": "품질과 수정",
        "inspect": "네 품질 플래그, 실패 이유, 실제 다음 노드",
        "accept": "근거/관점 부족은 추가 조사, 편향은 재종합, 중립성 문제는 재작성",
    },
    {
        "stage": "실패와 종료",
        "inspect": "일시 실패, 영구 실패, 관점 전체 실패, 계속 FAIL",
        "accept": "Task별 retry 상한, Worker 제외, max_steps/max_revision에서 명시적 종료",
    },
    {
        "stage": "복구",
        "inspect": "Worker 실행 전 중단, SQLite 닫기/재열기, trace_id",
        "accept": "같은 Task와 trace_id로 재개, 결과 중복 없음, 부모 namespace만 저장",
    },
    {
        "stage": "실제 조사 승인 기준",
        "inspect": "유효 키로 연결 확인 → 실제 PDF/웹 조사 → 인용 표본 확인",
        "accept": "실제 API와 RAG 실행 및 사람이 검토한 출처까지 확인한 뒤 실제 연구 결과로 판단",
    },
]

NODE_LABELS = {
    "initialize": "입력 초기화",
    "orchestrator": "계획 생성",
    "research_worker": "Worker 병렬 실행",
    "reduce_results": "Reducer 병합",
    "verifier": "근거 검증",
    "synthesis": "종합",
    "report_writer": "보고서 작성",
    "quality_evaluator": "품질 평가",
    "finalize": "종료",
}


class EventCollector(logging.Handler):
    def __init__(self):
        super().__init__()
        self.events = []

    def emit(self, record):
        self.events.append(json.loads(record.getMessage()))


@contextmanager
def collect_events():
    logger = logging.getLogger("kv_cache_agent.events")
    handler, old_level = EventCollector(), logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        yield handler.events
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
        handler.close()


def _snapshots(workflow, config):
    """Read actual parent checkpoints; avoid copying full State into artifacts."""
    rows, previous_next = [], ()
    for snapshot in reversed(list(workflow.get_state_history(config))):
        nodes = list(dict.fromkeys(previous_next))
        previous_next = snapshot.next
        if not nodes or nodes[0] not in NODE_LABELS:
            continue
        payload, control = snapshot.values["payload"], snapshot.values["control"]
        node = nodes[0]
        if node == "orchestrator":
            detail = payload.get("research_plan", {})
        elif node == "research_worker":
            detail = [
                {**r, "evidence_cards": [c["evidence_id"] for c in r["evidence_cards"]]}
                for r in payload.get("worker_results", [])
            ]
        elif node == "verifier":
            detail = {
                "verification": payload.get("verification"),
                "usable_evidence_cards": payload.get("usable_evidence_cards", []),
            }
        elif node in {"synthesis", "report_writer", "quality_evaluator"}:
            key = {
                "synthesis": "synthesis",
                "report_writer": "report",
                "quality_evaluator": "evaluation",
            }[node]
            detail = payload.get(key)
        elif node == "initialize":
            detail = {
                key: payload.get(key)
                for key in ("user_query", "selected_technologies", "target_domain")
            }
        else:
            detail = {
                "task_status": control.get("task_status"),
                "retry_count": control.get("retry_count"),
                "failed_tasks": control.get("failed_tasks"),
                "limitations": payload.get("limitations", []),
                "termination_reason": control.get("termination_reason"),
            }
        rows.append(
            {
                "node": node,
                "label": NODE_LABELS[node],
                "checkpoint_step": snapshot.metadata["step"],
                "next": list(snapshot.next),
                "step_count": control["step_count"],
                "revision_count": control["revision_count"],
                "planning_round": control["planning_round"],
                "status": control["status"],
                "decision": control.get("decision"),
                "decision_reason": control.get("decision_reason", ""),
                "task_count": len(payload.get("tasks", [])),
                "result_count": len(payload.get("worker_results", [])),
                "evidence_count": len(payload.get("evidence_cards", [])),
                "usable_count": len(payload.get("usable_evidence_cards", [])),
                "detail": detail,
            }
        )
    return rows


def _check(label, condition, observed):
    return {"label": label, "passed": bool(condition), "observed": observed}


def _run_case(case_id, title, purpose, output_dir):
    counts = Counter()
    options, limits = {}, WorkflowLimits()
    query = (
        "공급망과 위험도 평가해줘" if case_id == "expanded" else "두 기술을 평가해줘"
    )

    def worker(task, request):
        counts[task.task_id] += 1
        if case_id == "mandatory_failure" and task.perspective == "technical_maturity":
            raise RuntimeError("injected mandatory perspective failure")
        if task.task_id.startswith("r0-0") and (
            case_id == "worker_excluded"
            or (case_id == "retry" and task.retry_count == 0)
        ):
            raise RuntimeError("injected worker failure")
        return _research_once(task, request)

    def writer(state):
        counts["writer"] += 1
        update = report_writer_agent(state)
        if counts["writer"] == 1 and case_id in {"coverage", "groundedness"}:
            old, new = (
                ("4.2 시장성", "4.2 기타")
                if case_id == "coverage"
                else ("# REFERENCE", "# 자료")
            )
            update["payload"]["report"] = update["payload"]["report"].replace(old, new)
        return update

    def evaluator(state):
        counts["judge"] += 1
        update = quality_evaluator_agent(state)
        if case_id in {"max_steps", "max_revisions"}:
            update["payload"]["evaluation"].update(
                groundedness=False,
                groundedness_reason="검증용 FAIL 주입: 근거 부족 판정을 계속 반환",
                overall_pass=False,
            )
        elif counts["judge"] == 1 and case_id in {"bias", "neutrality"}:
            flag = "bias_control" if case_id == "bias" else "neutrality"
            update["payload"]["evaluation"].update({flag: False, "overall_pass": False})
            reason = "bias_reason" if case_id == "bias" else "neutrality_reason"
            update["payload"]["evaluation"][reason] = "검증용 FAIL 주입: " + (
                "편향 재종합 경로 확인"
                if case_id == "bias"
                else "중립성 재작성 경로 확인"
            )
        return update

    if case_id in {"retry", "worker_excluded", "mandatory_failure"}:
        options["worker_service"] = worker
    if case_id in {"coverage", "groundedness"}:
        options["report_node"] = writer
    if case_id in {"bias", "neutrality", "max_steps", "max_revisions"}:
        options["evaluator_node"] = evaluator
    if case_id == "max_steps":
        limits = WorkflowLimits(max_steps=8, max_report_revisions=100)
    elif case_id == "max_revisions":
        limits = WorkflowLimits(max_steps=100, max_report_revisions=2)
    elif case_id == "mandatory_failure":
        limits = WorkflowLimits(max_report_revisions=0)

    config = {"configurable": {"thread_id": str(uuid4())}}
    input_state = {"payload": {"user_query": query}}
    recovery = None
    with mock_services(), collect_events() as events:
        if case_id == "resume":
            try:
                from langgraph.checkpoint.sqlite import SqliteSaver
            except ImportError:
                return {
                    "id": case_id,
                    "title": title,
                    "purpose": purpose,
                    "status": "skipped",
                    "error_type": "SQLite extra missing",
                    "snapshots": [],
                    "events": [],
                    "checks": [],
                }
            db = output_dir / "recovery.sqlite"
            with SqliteSaver.from_conn_string(str(db)) as saver:
                workflow = build_workflow(
                    checkpointer=saver, interrupt_before=["research_worker"]
                )
                paused = workflow.invoke(input_state, config)
                pending = list(workflow.get_state(config).next)
            with SqliteSaver.from_conn_string(str(db)) as saver:
                workflow = build_workflow(checkpointer=saver)
                result = workflow.invoke(None, config)
                snapshots = _snapshots(workflow, config)
            with sqlite3.connect(db) as connection:
                namespaces = [
                    row[0]
                    for row in connection.execute(
                        "SELECT DISTINCT checkpoint_ns FROM checkpoints"
                    )
                ]
            recovery = {
                "pending_before_close": pending,
                "trace_before": paused["control"]["trace_id"],
                "trace_after": result["control"]["trace_id"],
                "task_ids_before": [t["task_id"] for t in paused["payload"]["tasks"]],
                "namespaces": namespaces,
            }
        else:
            workflow = build_workflow(
                limits=limits, checkpointer=InMemorySaver(), **options
            )
            result = workflow.invoke(input_state, config)
            snapshots = _snapshots(workflow, config)

    payload, control = result["payload"], result["control"]
    results = payload.get("worker_results", [])
    all_tasks = {
        task["task_id"]: task
        for row in snapshots
        if row["node"] == "orchestrator"
        for task in (row["detail"] or {}).get("tasks", [])
    }
    cards = payload.get("evidence_cards", [])
    expected_cards = {c["evidence_id"] for r in results for c in r["evidence_cards"]}
    fanouts = [e for e in events if e["node"] == "dynamic_fan_out"]
    checks = [
        _check(
            "Task별 결과와 Send 수 일치",
            len(fanouts) == len(results) == len(all_tasks),
            {"tasks": len(all_tasks), "send": len(fanouts), "results": len(results)},
        ),
        _check(
            "Reducer 근거 손실과 중복 없음",
            expected_cards == {c["evidence_id"] for c in cards}
            and len(cards) == len(expected_cards),
            {"worker_card_ids": len(expected_cards), "merged_cards": len(cards)},
        ),
        _check(
            "Trace 상관 관계 유지",
            all(e["trace_id"] == control["trace_id"] for e in events),
            control["trace_id"],
        ),
        _check(
            "종료 상한 준수",
            control["step_count"] <= limits.max_steps
            and control["revision_count"] <= limits.max_report_revisions,
            {"steps": control["step_count"], "revisions": control["revision_count"]},
        ),
    ]
    nodes = Counter(row["node"] for row in snapshots)
    routes = [e["decision"] for e in events if e["node"] == "quality_routing"]
    if case_id in {"max_steps", "max_revisions", "mandatory_failure"}:
        reason = "max_steps" if case_id == "max_steps" else "max_report_revisions"
        checks.append(
            _check(
                "예상된 제한 종료",
                control["status"] in {"best_effort", "failed"}
                and control["termination_reason"] == reason,
                {"status": control["status"], "reason": control["termination_reason"]},
            )
        )
    else:
        checks.append(
            _check(
                "품질 PASS 후 정상 종료",
                control["status"] == "completed"
                and payload.get("evaluation", {}).get("overall_pass"),
                control["status"],
            )
        )
    if case_id in {"baseline", "expanded"}:
        checks.append(
            _check(
                "첫 계획의 필수 네 관점",
                set(REQUIRED_PERSPECTIVES)
                <= {t["perspective"] for t in all_tasks.values()},
                sorted({t["perspective"] for t in all_tasks.values()}),
            )
        )
    if case_id == "retry":
        checks.append(
            _check(
                "동일 Task 재시도 후 회복",
                any(
                    r["retry_count"] == 1 and r["status"] == "success" for r in results
                ),
                dict(counts),
            )
        )
    if case_id == "worker_excluded":
        failed = [r for r in results if r["status"] == "failed"]
        checks.append(
            _check(
                "3회 시도 후 근거 제외",
                len(failed) == 1
                and failed[0]["retry_count"] == 2
                and not failed[0]["evidence_cards"]
                and bool(failed[0]["limitations"]),
                dict(counts),
            )
        )
    if case_id == "mandatory_failure":
        checks.append(
            _check(
                "관점 전체 실패는 PASS 금지",
                control["status"] == "failed"
                and "technical_maturity"
                in payload["evaluation"]["missing_perspectives"],
                payload["evaluation"]["missing_perspectives"],
            )
        )
    if case_id in {"coverage", "groundedness"}:
        checks.append(
            _check(
                "추가 계획, Worker, 재평가 실행",
                nodes["orchestrator"] == 2
                and nodes["quality_evaluator"] == 2
                and "additional_research" in routes
                and len(results) > 5,
                {"nodes": dict(nodes), "routes": routes},
            )
        )
    if case_id in {"bias", "neutrality"}:
        route = "resynthesis" if case_id == "bias" else "rewrite"
        checks.append(
            _check(
                "추가 Worker 없이 지정 수정 경로",
                len(results) == 5
                and nodes["orchestrator"] == 1
                and nodes["report_writer"] == 2
                and nodes["synthesis"] == (2 if case_id == "bias" else 1)
                and route in routes,
                {"nodes": dict(nodes), "routes": routes},
            )
        )
    if recovery:
        checks.append(
            _check(
                "SQLite 재열기 후 같은 실행 복구",
                recovery["trace_before"] == recovery["trace_after"]
                and set(recovery["task_ids_before"]) == {r["task_id"] for r in results}
                and recovery["namespaces"] == [""],
                recovery,
            )
        )
    report_name = f"{case_id}.report.md"
    if payload.get("report"):
        (output_dir / report_name).write_text(payload["report"], encoding="utf-8")
    return {
        "id": case_id,
        "title": title,
        "purpose": purpose,
        "query": query,
        "status": "passed" if all(c["passed"] for c in checks) else "failed",
        "control": control,
        "checks": checks,
        "snapshots": snapshots,
        "events": events,
        "tasks": list(all_tasks.values()),
        "worker_results": results,
        "evidence_cards": cards,
        "usable_evidence_cards": payload.get("usable_evidence_cards", []),
        "report": payload.get("report", ""),
        "report_file": report_name if payload.get("report") else None,
        "evaluation": payload.get("evaluation"),
        "rules": check_report_rules(payload),
        "recovery": recovery,
    }


SCENARIOS = [
    ("baseline", "기본 조사", "정상 계획, 병렬 실행, 검증, 보고서와 품질 PASS"),
    ("expanded", "공급망 추가", "입력에 따라 Task 수와 Worker 수가 달라지는지 확인"),
    ("retry", "일시 실패와 회복", "첫 Worker 시도를 실패시켜 동일 Task 재시도 확인"),
    (
        "worker_excluded",
        "Worker 제외",
        "한 Worker를 계속 실패시켜 제외 후 나머지 진행 확인",
    ),
    ("coverage", "관점 부족 보완", "첫 보고서의 시장성 제목을 제거해 추가 조사 유도"),
    ("groundedness", "근거 부족 보완", "첫 보고서의 REFERENCE를 제거해 추가 조사 유도"),
    ("bias", "편향 재종합", "첫 Judge의 편향 판정을 FAIL로 주입해 재종합 경로 확인"),
    (
        "neutrality",
        "중립성 재작성",
        "첫 Judge의 중립성 판정을 FAIL로 주입해 재작성 경로 확인",
    ),
    ("max_steps", "Step 상한", "Judge가 계속 FAIL일 때 8 steps 안에서 종료 확인"),
    (
        "max_revisions",
        "Revision 상한",
        "Judge가 계속 FAIL일 때 2 revisions 후 종료 확인",
    ),
    (
        "mandatory_failure",
        "필수 관점 실패",
        "기술 관점의 모든 Worker 실패를 품질 PASS로 오인하지 않는지 확인",
    ),
    ("resume", "SQLite 복구", "Worker 실행 전 중단하고 DB를 다시 열어 같은 실행 재개"),
]


def _live_status(run_live):
    if not run_live:
        path = OUTPUTS_DIR / "logs" / "gpt6_luna_smoke.json"
        if path.exists():
            old = json.loads(path.read_text(encoding="utf-8"))
            failed = next(
                (c for c in old.get("checks", []) if not c.get("passed")), None
            )
            return {
                "status": "blocked" if failed else "not_run",
                "current_run": False,
                "reason": "이전 호출 기록이며 이번 검증에서 API를 다시 호출하지 않았습니다.",
                "previous_check": failed,
                "source": str(path),
                "recorded_at": datetime.fromtimestamp(
                    path.stat().st_mtime, UTC
                ).isoformat(),
            }
        return {
            "status": "not_run",
            "current_run": False,
            "reason": "실제 API 연결은 --live-smoke로 별도 확인합니다.",
        }
    from kv_cache_agent.llm import get_llm
    from kv_cache_agent.schemas.evaluation import QualityEvaluation
    from kv_cache_agent.schemas.tasks import ResearchPlan

    checks = []
    prompts = [
        (None, "Reply exactly OK."),
        (
            ResearchPlan,
            "Return a short four-task research plan for TurboQuant and CXL-based KV Cache covering technical_maturity, market, stakeholder, domain_application. Set retry_count 0. Do not perform research.",
        ),
        (
            QualityEvaluation,
            "Evaluate an empty report with no evidence: no groundedness, no coverage, no source diversity, neutrality true, overall_pass false, recommended_action additional_research. Return brief reasons.",
        ),
    ]
    with patch.dict(
        "os.environ", {"LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false"}
    ):
        for schema, prompt in prompts:
            name = schema.__name__ if schema else "text"
            try:
                model = get_llm()
                model.request_timeout, model.max_retries = 20, 0
                if schema:
                    parsed = model.with_structured_output(schema).invoke(prompt)
                    schema.model_validate(parsed)
                else:
                    model.invoke(prompt)
                checks.append({"name": name, "passed": True})
            except Exception as error:  # noqa: BLE001 - diagnostic boundary
                code = str(getattr(error, "code", ""))
                checks.append(
                    {
                        "name": name,
                        "passed": False,
                        "error_type": type(error).__name__,
                        "status_code": getattr(error, "status_code", None),
                        "error_code": code
                        if re.fullmatch(r"[a-zA-Z0-9_.-]{1,80}", code)
                        else None,
                    }
                )
                break
    return {
        "status": "passed"
        if len(checks) == 3 and all(c["passed"] for c in checks)
        else "blocked",
        "current_run": True,
        "checks": checks,
        "reason": "연결과 Schema 계약만 확인합니다. 실제 자료 조사와 보고서 정확성은 별도 검증입니다.",
    }


def _tool_checks(output_dir):
    commands = [
        (
            "pytest",
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                f"--junitxml={output_dir / 'tests.xml'}",
            ],
        ),
        ("ruff", [sys.executable, "-m", "ruff", "check", "src", "tests"]),
        ("git_diff", ["git", "diff", "--check"]),
    ]
    rows = []
    for name, command in commands:
        result = subprocess.run(
            command,
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
            env={
                **os.environ,
                "LANGSMITH_TRACING": "false",
                "LANGCHAIN_TRACING_V2": "false",
            },
        )
        path = output_dir / f"{name}.log"
        path.write_text(result.stdout + result.stderr, encoding="utf-8")
        rows.append(
            {
                "name": name,
                "status": "passed" if result.returncode == 0 else "failed",
                "exit_code": result.returncode,
                "log_file": path.name,
                "summary": next(
                    (
                        line
                        for line in reversed(result.stdout.splitlines())
                        if line.strip()
                    ),
                    "",
                ),
            }
        )
    return rows


def write_results(data, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # A closing script tag from any report/query must remain data, never markup.
    serialized = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    template = (
        Path(__file__).with_name("validation_view.html").read_text(encoding="utf-8")
    )
    html = template.replace("__VALIDATION_DATA__", serialized)
    path = output_dir / "index.html"
    path.write_text(html, encoding="utf-8")
    return path


def export_live_results(
    db_path, thread_id, output_dir, live_checks, *, allow_running=False
):
    """Inspect saved live checkpoints without replaying external calls."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {"configurable": {"thread_id": thread_id}}
    with SqliteSaver.from_conn_string(str(db_path)) as saver:
        workflow = build_workflow(checkpointer=saver)
        saved = workflow.get_state(config)
        pending = bool(saved.next) or saved.values.get("control", {}).get(
            "status"
        ) not in {"completed", "best_effort", "failed"}
        if not saved.values or (pending and not allow_running):
            raise ValueError("live thread missing or still running")
        snapshots = _snapshots(workflow, config)
    payload, control = saved.values["payload"], saved.values["control"]
    tasks = {
        task["task_id"]: task
        for row in snapshots
        if row["node"] == "orchestrator"
        for task in (row["detail"] or {}).get("tasks", [])
    }
    rows = payload.get("worker_results", [])
    cards = payload.get("evidence_cards", [])
    rules = check_report_rules(payload)
    event_path = Path(live_checks["events_path"])
    events = [
        json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()
    ]
    fanouts = [event for event in events if event["node"] == "dynamic_fan_out"]
    checks = [
        _check(
            "계획된 Task 결과 수 일치",
            set(tasks) == {r["task_id"] for r in rows},
            {"tasks": len(tasks), "worker_results": len(rows)},
        ),
        _check("Task별 Dynamic Send 수 일치", len(fanouts) == len(tasks), len(fanouts)),
        _check(
            "Reducer 근거 손실과 중복 없음",
            {c["evidence_id"] for r in rows for c in r["evidence_cards"]}
            == {c["evidence_id"] for c in cards}
            and len(cards) == len({c["evidence_id"] for c in cards}),
            len(cards),
        ),
        _check(
            "필수 네 관점의 사용 가능한 근거",
            not rules["missing_perspectives"],
            rules["missing_perspectives"],
        ),
        _check("보고서 섹션과 인용 규칙", not rules["failures"], rules["failures"]),
        _check(
            "실제 품질 평가 PASS",
            bool(payload.get("evaluation", {}).get("overall_pass")),
            payload.get("evaluation"),
        ),
        _check(
            "완료 상태와 종료 이유",
            control["status"] == "completed",
            {"status": control["status"], "reason": control["termination_reason"]},
        ),
    ]
    report = payload.get("report", "")
    report_file = "live.report.md"
    (output_dir / report_file).write_text(report, encoding="utf-8")
    pdf_file = None
    pdf_path = live_checks.get("run_summary", {}).get("pdf_path")
    if pdf_path and Path(pdf_path).is_file():
        pdf_file = "live.report.pdf"
        shutil.copy2(pdf_path, output_dir / pdf_file)
    case = {
        "id": "live",
        "title": "실제 API 전체 실행",
        "purpose": "OpenAI, Tavily, BGE-M3/FAISS, 원문 검증을 실제 실행한 결과",
        "status": "running"
        if pending
        else ("passed" if all(c["passed"] for c in checks) else "failed"),
        "query": payload["user_query"],
        "control": control,
        "checks": checks,
        "tasks": list(tasks.values()),
        "worker_results": rows,
        "evidence_cards": cards,
        "usable_evidence_cards": payload.get("usable_evidence_cards", []),
        "evaluation": payload.get("evaluation"),
        "rules": rules,
        "snapshots": snapshots,
        "report": report,
        "report_file": report_file,
        "pdf_file": pdf_file,
        "events": events,
        "recovery": live_checks.get("recovery"),
    }
    data = {
        "generated_at": datetime.now(UTC).isoformat(),
        "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=ROOT_DIR, text=True
        ).strip(),
        "model": OPENAI_MODEL,
        "mode": "live",
        "plan": PLAN,
        "cases": [case],
        "checks": [],
        "tools": [],
        "live": live_checks["openai"],
        "service_checks": live_checks,
        "passed": case["status"] == "passed",
        "scope": (
            "실행 중인 체크포인트의 중간 결과이며 최종 판정이 아닙니다. "
            if pending
            else ""
        )
        + "Mock을 사용하지 않고 실제 OpenAI, Tavily, BGE-M3/FAISS와 원문 검증을 실행했습니다. 품질 PASS는 규칙과 LLM Judge의 자동 평가 결과이며, 기술 전문가의 검토를 대신하지 않습니다.",
    }
    return data, write_results(data, output_dir)


def run_validation(output_dir, *, live_smoke=False, with_tests=False):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT_DIR, text=True
    ).strip()
    cases = []
    for case_id, title, purpose in SCENARIOS:
        try:
            case = _run_case(case_id, title, purpose, output_dir)
        except Exception as error:  # noqa: BLE001 - keep other diagnostic cases
            case = {
                "id": case_id,
                "title": title,
                "purpose": purpose,
                "status": "failed",
                "error_type": type(error).__name__,
                "snapshots": [],
                "events": [],
                "checks": [],
            }
        cases.append(case)
    baseline, expanded = cases[:2]
    varying = len(baseline.get("tasks", [])) != len(expanded.get("tasks", []))
    checks = [
        _check(
            "입력에 따른 동적 Task 수 변화",
            varying and baseline["status"] == expanded["status"] == "passed",
            {
                "basic": len(baseline.get("tasks", [])),
                "expanded": len(expanded.get("tasks", [])),
            },
        )
    ]
    data = {
        "generated_at": datetime.now(UTC).isoformat(),
        "branch": branch,
        "model": OPENAI_MODEL,
        "mode": "mock",
        "plan": PLAN,
        "cases": cases,
        "checks": checks,
        "tools": _tool_checks(output_dir) if with_tests else [],
        "live": _live_status(live_smoke),
        "scope": "LangGraph 노드와 체크포인트를 실제 실행했습니다. LLM, Tavily, 논문 검색과 원문 조회는 합성 Mock입니다. 의도적인 실패 주입은 분기 동작을 확인하며, LLM의 편향 탐지 정확성을 증명하지 않습니다.",
    }
    data["passed"] = (
        all(case["status"] == "passed" for case in cases)
        and all(c["passed"] for c in checks)
        and all(t["status"] == "passed" for t in data["tools"])
    )
    path = write_results(data, output_dir)
    return data, path


def main():
    parser = argparse.ArgumentParser(description="중간 검증 계획과 실제 Mock 결과 뷰어")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR
        / "validation"
        / datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f"),
    )
    parser.add_argument(
        "--with-tests", action="store_true", help="전체 pytest, Ruff, diff 검사도 기록"
    )
    parser.add_argument(
        "--live-smoke",
        action="store_true",
        help="현재 키로 소규모 실제 API 연결/Schema 확인",
    )
    parser.add_argument(
        "--live-run-log", type=Path, help="완료된 실제 CLI 요약 JSON을 뷰어로 내보내기"
    )
    parser.add_argument(
        "--checkpoint-db", type=Path, help="실제 실행에 사용한 SQLite DB"
    )
    args = parser.parse_args()
    if args.live_run_log:
        if not args.checkpoint_db:
            parser.error("--live-run-log에는 --checkpoint-db가 필요합니다.")
        summary = json.loads(args.live_run_log.read_text(encoding="utf-8"))
        if summary.get("mock") is not False:
            parser.error("실제 실행(--mock 없음)의 요약만 내보낼 수 있습니다.")
        data, path = export_live_results(
            args.checkpoint_db,
            summary["thread_id"],
            args.output_dir,
            {
                "events_path": summary["events_path"],
                "run_summary": summary,
                "openai": _live_status(True)
                if args.live_smoke
                else {
                    "status": "not_run",
                    "current_run": False,
                    "reason": "저장된 실제 실행을 표시합니다. 추가 API 연결 검사는 호출하지 않았습니다.",
                },
            },
        )
        if args.with_tests:
            data["tools"] = _tool_checks(args.output_dir)
            data["passed"] &= all(t["status"] == "passed" for t in data["tools"])
            path = write_results(data, args.output_dir)
        print(f"실제 Workflow: {data['cases'][0]['control']['status']}")
    else:
        data, path = run_validation(
            args.output_dir, live_smoke=args.live_smoke, with_tests=args.with_tests
        )
        print(
            f"Mock 시나리오: {sum(c['status'] == 'passed' for c in data['cases'])}/{len(data['cases'])} 통과"
        )
    print(f"실제 API 연결: {data['live']['status']}")
    print(f"결과 화면: {path}")
    print(f"원본 결과: {path.with_name('results.json')}")
    if not data["passed"] or (args.live_smoke and data["live"]["status"] != "passed"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
