"""Structured runtime planner for initial and targeted supplemental research."""

import json
from pathlib import Path

import yaml
from langchain_core.messages import HumanMessage, SystemMessage

from kv_cache_agent.llm import get_llm
from kv_cache_agent.observability import record_event
from kv_cache_agent.schemas.tasks import (
    LEGACY_PERSPECTIVES,
    REQUIRED_PERSPECTIVES,
    ResearchPlan,
    SubTask,
)

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "orchestrator.yaml"


def orchestrator_agent(state):
    payload, control = state["payload"], state["control"]
    evaluation = payload.get("evaluation") or {}
    if hasattr(evaluation, "model_dump"):
        evaluation = evaluation.model_dump()
    context = {
        "user_query": payload["user_query"],
        "selected_technologies": payload["selected_technologies"],
        "target_domain": payload["target_domain"],
        "planning_round": control["planning_round"],
        "quality_feedback": evaluation,
        "previous_results": [
            {
                key: result.get(key, []) if key == "search_attempts" else result[key]
                for key in (
                    "task_id",
                    "perspective",
                    "technology",
                    "status",
                    "limitations",
                    "search_attempts",
                )
            }
            for result in payload.get("worker_results", [])
        ],
        "evidence_topics": [
            {
                key: card.get(key)
                for key in (
                    "evidence_id",
                    "technology",
                    "perspective",
                    "claim",
                    "source_id",
                    "baseline",
                    "conditions",
                )
            }
            for card in payload.get("usable_evidence_cards", [])
        ],
        "max_tasks_per_plan": control["limits"]["max_tasks_per_plan"],
    }
    prompt = yaml.safe_load(PROMPT_PATH.read_text(encoding="utf-8"))
    response = (
        get_llm()
        .with_structured_output(ResearchPlan)
        .invoke(
            [
                SystemMessage(content=prompt["system_prompt"]),
                HumanMessage(content=json.dumps(context, ensure_ascii=False)),
            ]
        )
    )
    plan = ResearchPlan.model_validate(response)
    tasks = list(plan.tasks)
    required = (
        set(REQUIRED_PERSPECTIVES)
        if control["planning_round"] == 0
        else set(evaluation.get("missing_perspectives", []))
    )
    # Also detect an entire mandatory perspective failing, regardless of judge.
    if control["planning_round"]:
        required.update(
            perspective
            for perspective in REQUIRED_PERSPECTIVES
            if not any(
                c.get("perspective") == LEGACY_PERSPECTIVES[perspective]
                for c in payload.get("usable_evidence_cards", [])
            )
        )
    present = {task.perspective for task in tasks}
    for perspective in REQUIRED_PERSPECTIVES:
        if perspective in required - present:
            tasks.append(
                SubTask(
                    task_id=f"coverage_{perspective}",
                    perspective=perspective,
                    technology=payload["selected_technologies"],
                    objective=f"누락 관점 {perspective}의 직접 근거와 반대 근거 보완",
                    query=(
                        f"{' '.join(payload['selected_technologies'])} "
                        f"{payload['target_domain']} {perspective} evidence limitations"
                    ),
                    preferred_source="paper"
                    if perspective == "technical_maturity"
                    else "web",
                    priority=1,
                )
            )
    if len(tasks) > control["limits"]["max_tasks_per_plan"]:
        raise ValueError("plan exceeds MAX_TASKS_PER_PLAN")
    scoped_tasks = []
    for index, task in enumerate(sorted(tasks, key=lambda task: task.priority)):
        if task.search_queries:
            task = task.model_copy(update={"query": task.search_queries[0]})
        technologies = (
            [task.technology] if isinstance(task.technology, str) else task.technology
        )
        if not set(technologies) <= set(payload["selected_technologies"]):
            raise ValueError("task technology outside selected scope")
        scoped_tasks.append(
            task.model_copy(
                update={
                    "task_id": f"r{control['planning_round']}-{index}-{task.task_id[:70]}",
                    "retry_count": 0,
                }
            )
        )
    plan = ResearchPlan(tasks=scoped_tasks, planning_reason=plan.planning_reason)
    statuses = {
        **control.get("task_status", {}),
        **{task.task_id: "pending" for task in scoped_tasks},
    }
    record_event(
        control["trace_id"],
        "orchestrator",
        "plan",
        plan.planning_reason,
        task_count=len(scoped_tasks),
    )
    return {
        "payload": {
            "research_plan": plan.model_dump(),
            "tasks": [task.model_dump() for task in scoped_tasks],
        },
        "control": {
            **control,
            "status": "researching",
            "task_status": statuses,
            "planning_round": control["planning_round"] + 1,
            "decision": "fan_out",
            "decision_reason": plan.planning_reason,
        },
    }
