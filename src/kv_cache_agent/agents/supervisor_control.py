"""Semantic planning/routing bounded by executable coverage and budget policy."""

import json
from pathlib import Path

import yaml
from langchain_core.messages import HumanMessage, SystemMessage

from kv_cache_agent.graph.progress import (
    build_coverage,
    current_claims,
    eligible_targets,
    pending_claim_ids,
    plan_cells,
    ready_to_finalize,
    validate_research_plan,
)
from kv_cache_agent.llm import get_llm
from kv_cache_agent.observability.logger import emit
from kv_cache_agent.schemas.base import stable_id
from kv_cache_agent.schemas.report import ReportPlan
from kv_cache_agent.schemas.research import ResearchTask, SupervisorDecision
from kv_cache_agent.schemas.supervision import ReportOutline, RouteProposal

PROMPTS = Path(__file__).resolve().parents[1] / "prompts"


def _system_prompt(name):
    with (PROMPTS / f"{name}.yaml").open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)["system_prompt"]


class SupervisorController:
    def __init__(self, *, model=None, planner=None, router=None, no_progress_limit=2):
        if no_progress_limit < 1:
            raise ValueError("No-progress limit must be positive")
        self.model = model
        self.planner = planner
        self.router = router
        self.no_progress_limit = no_progress_limit
        self.plan_prompt = _system_prompt("supervisor_plan")
        self.route_prompt = _system_prompt("supervisor_route")

    def plan(self, user_query, *, budget, run_id, config=None):
        budget.reserve(model_calls=1)
        if self.planner:
            outline = self.planner(user_query, config=config)
        else:
            outline = (
                (self.model or get_llm())
                .with_structured_output(ReportOutline)
                .invoke(
                    [
                        SystemMessage(content=self.plan_prompt),
                        HumanMessage(content=user_query),
                    ],
                    config=config,
                )
            )
        outline = ReportOutline.model_validate(outline)
        plan = ReportPlan(
            plan_id=stable_id("plan", (run_id, user_query)),
            user_query=user_query,
            sections=outline.sections,
        )
        plan = validate_research_plan(plan)
        emit(
            "report_plan_created",
            details={
                "plan_id": plan.plan_id,
                "version": plan.version,
                "sections": [
                    {
                        "section_id": s.section_id,
                        "title": s.title,
                        "owner": s.owner,
                        "criteria": s.criteria,
                    }
                    for s in plan.sections
                ],
            },
        )
        return plan

    def routing_context(self, state, budget):
        return {
            "report_plan": state["report_plan"].model_dump(mode="json"),
            "round_id": state.get("round_id", 0),
            "coverage": [c.model_dump(mode="json") for c in build_coverage(state)],
            "eligible_targets": eligible_targets(state),
            "current_drafts": [
                d.model_dump(mode="json")
                for d in state.get("drafts_by_section", {}).values()
            ],
            "evidence": [
                e.model_dump(mode="json")
                for e in state.get("evidence_by_id", {}).values()
                if e.evidence_id
                in {
                    key
                    for c in current_claims(state).values()
                    for key in c.evidence_ids
                }
            ],
            "recent_results": [
                {
                    "task_id": r.task_id,
                    "agent": r.agent,
                    "status": r.status,
                    "missing_items": r.missing_items,
                    "errors": r.errors,
                }
                for r in state.get("task_results", {}).values()
                if r.round_id == state.get("round_id", 0)
            ],
            "budget_used": budget.snapshot(),
            "budget_available": budget.available(),
            "round_limit": budget.limits.research_rounds,
            "no_progress_rounds": state.get("no_progress_rounds", 0),
            "unresolved_questions": state.get("semantic_gaps", {}),
            "reviewable_targets": [
                {
                    "section_id": key,
                    "technology": technology,
                    "criterion": criterion,
                    "questions": [
                        {
                            "question_id": stable_id(
                                "question", (state["report_plan"].plan_id, key, q)
                            ),
                            "text": q,
                        }
                        for s in state["report_plan"].sections
                        if s.section_id == key
                        for q in s.questions
                    ],
                }
                for key, technology, criterion in plan_cells(state["report_plan"])
            ],
        }

    def decide(self, state, *, budget, config=None):
        if state.get("fatal_errors"):
            return SupervisorDecision(
                action="stop", reason="contract_or_execution_failure"
            )
        pending = pending_claim_ids(state)
        if pending:
            if budget.available()["model_calls"] < 1:
                return SupervisorDecision(
                    action="stop", reason="verification_budget_exhausted"
                )
            return SupervisorDecision(
                action="verify",
                reason="새로 생성되거나 변경된 실제 문장을 검증합니다",
                claim_ids=pending,
            )
        complete = ready_to_finalize(state)
        if not complete and state.get("round_id", 0) >= budget.limits.research_rounds:
            return SupervisorDecision(action="stop", reason="research_round_limit")
        if (
            not complete
            and state.get("no_progress_rounds", 0) >= self.no_progress_limit
        ):
            return SupervisorDecision(action="stop", reason="no_progress_limit")
        # Keep capacity for a worker and intermediate verification before routing.
        if budget.available()["model_calls"] < (1 if complete else 3):
            return SupervisorDecision(
                action="stop",
                reason="completion_review_budget_exhausted"
                if complete
                else "research_model_budget_exhausted",
            )
        context = self.routing_context(state, budget)
        context["completion_review"] = complete
        if not context["eligible_targets"] and not complete:
            return SupervisorDecision(action="stop", reason="no_eligible_targets")
        budget.reserve(model_calls=1)
        if self.router:
            proposal = self.router(context, config=config)
        else:
            proposal = (
                (self.model or get_llm())
                .with_structured_output(RouteProposal)
                .invoke(
                    [
                        SystemMessage(content=self.route_prompt),
                        HumanMessage(content=json.dumps(context, ensure_ascii=False)),
                    ],
                    config=config,
                )
            )
        proposal = RouteProposal.model_validate(proposal)
        decision = self.validate_proposal(
            proposal, state, allow_semantic_review=complete
        )
        unresolved = {
            "/".join(cell): task.unanswered_questions
            for task in decision.tasks
            if task.unanswered_questions
            for cell in (
                (s, t, c)
                for s in task.section_ids
                for t in task.technologies
                for c in task.criteria
            )
        }
        if decision.tasks and state.get("round_id", 0) >= budget.limits.research_rounds:
            return SupervisorDecision(
                action="stop",
                reason="research_round_limit",
                unresolved_questions=unresolved,
            )
        if (
            decision.tasks
            and state.get("no_progress_rounds", 0) >= self.no_progress_limit
        ):
            return SupervisorDecision(
                action="stop",
                reason="no_progress_limit",
                unresolved_questions=unresolved,
            )
        return decision

    def validate_proposal(self, proposal, state, *, allow_semantic_review=False):
        if proposal.action == "finalize":
            if not ready_to_finalize(state):
                raise ValueError(
                    "Supervisor tried to finalize incomplete or unverified coverage"
                )
            return SupervisorDecision(action="finalize", reason=proposal.reason)
        if proposal.action == "stop":
            return SupervisorDecision(action="stop", reason=proposal.reason)
        if len(proposal.assignments) > 4:
            raise ValueError("Too many assignments in one dispatch")
        targets = {
            (t["section_id"], t["technology"], t["criterion"]): t
            for t in eligible_targets(state)
        }
        tasks = []
        assigned = set()
        plan = state["report_plan"]
        specs = {s.section_id: s for s in plan.sections}
        drafts = state.get("drafts_by_section", {})
        for index, assignment in enumerate(proposal.assignments):
            if assignment.unanswered_question_ids:
                if len(assignment.section_ids) != 1:
                    raise ValueError("Question-ID follow-up must target one section")
                section_id = assignment.section_ids[0]
                if section_id not in specs:
                    raise ValueError("Unknown semantic follow-up section")
                question_map = {
                    stable_id(
                        "question", (plan.plan_id, section_id, question)
                    ): question
                    for question in specs[section_id].questions
                }
                if not set(assignment.unanswered_question_ids) <= question_map.keys():
                    raise ValueError("Unknown unanswered question ID")
                assignment = assignment.model_copy(
                    update={
                        "unanswered_questions": tuple(
                            question_map[key]
                            for key in assignment.unanswered_question_ids
                        )
                    }
                )
            if (
                assignment.unanswered_questions or assignment.unanswered_question_ids
            ) and assignment.trigger != "semantic_gap":
                raise ValueError(
                    "Unanswered questions require a semantic-gap assignment"
                )
            cells = {
                (section, technology, criterion)
                for section in assignment.section_ids
                for technology in assignment.technologies
                for criterion in assignment.criteria
            }
            if assignment.trigger == "semantic_gap":
                if not assignment.unanswered_questions or not assignment.feedback:
                    raise ValueError(
                        "Semantic follow-up requires concrete unanswered questions and feedback"
                    )
                for cell in cells:
                    section = specs.get(cell[0])
                    if section is None or not set(
                        assignment.unanswered_questions
                    ) <= set(section.questions):
                        raise ValueError(
                            "Semantic follow-up introduces questions outside the report plan"
                        )
                    known_gap = state.get("semantic_gaps", {}).get("/".join(cell), ())
                    if not allow_semantic_review and not known_gap:
                        raise ValueError(
                            "Fulfilled cells require a recorded semantic gap or completion review"
                        )
                    if (
                        cell not in targets
                        and allow_semantic_review
                        and cell in plan_cells(plan)
                    ):
                        targets[cell] = {
                            "agent": section.owner,
                            "coverage_status": "verified",
                            "verification_feedback": [],
                            "reported_gaps": [],
                            "unanswered_questions": assignment.unanswered_questions,
                        }
            if not cells or not cells <= targets.keys() or cells & assigned:
                raise ValueError(
                    "Dispatch has fulfilled, unready, out-of-scope or overlapping targets"
                )
            if any(targets[cell]["agent"] != assignment.agent for cell in cells):
                raise ValueError("Dispatch violates section ownership")
            for cell in cells:
                target = targets[cell]
                if target["coverage_status"] in {"verified", "inferred"}:
                    if target["verification_feedback"]:
                        if assignment.trigger != "verification_feedback":
                            raise ValueError(
                                "Revisiting coverage requires verification feedback"
                            )
                    elif assignment.trigger == "semantic_gap":
                        pass  # Validated against the original plan questions above.
                    elif (
                        assignment.trigger != "reported_gap"
                        or not target["reported_gaps"]
                    ):
                        raise ValueError(
                            "Revisiting fulfilled coverage requires a current reported gap"
                        )
                if assignment.action == "revise":
                    feedback = target["verification_feedback"]
                    has_sources = any(
                        c.evidence_ids
                        for c in current_claims(state).values()
                        if (c.section_id, c.technology, c.criterion) == cell
                    )
                    if not has_sources or any(
                        f["requested_action"] == "research" for f in feedback
                    ):
                        raise ValueError(
                            "Missing evidence cannot be solved by a draft-only revision"
                        )
            assigned.update(cells)
            existing_ids = tuple(
                dict.fromkeys(
                    key
                    for claim in current_claims(state).values()
                    if (claim.section_id, claim.technology, claim.criterion) in cells
                    for key in claim.evidence_ids
                )
            )
            round_id = state.get("round_id", 0) + 1
            web_refs = any(
                ref.location.startswith("http") for ref in state.get("source_refs", ())
            ) or any(
                ref.location.startswith("http")
                for key in existing_ids
                for ref in state["evidence_by_id"][key].source_refs
            )
            task = ResearchTask(
                task_id=stable_id(
                    "task",
                    (
                        state.get("workflow_run_id"),
                        plan.plan_id,
                        round_id,
                        index,
                        assignment.model_dump(mode="json"),
                    ),
                ),
                plan_id=plan.plan_id,
                plan_version=plan.version,
                round_id=round_id,
                agent=assignment.agent,
                section_ids=assignment.section_ids,
                technologies=assignment.technologies,
                criteria=assignment.criteria,
                objective=assignment.objective,
                questions=assignment.questions,
                existing_evidence_ids=existing_ids,
                feedback=tuple(
                    dict.fromkeys(
                        (
                            *assignment.feedback,
                            *(
                                f["reason"]
                                for cell in cells
                                for f in targets[cell]["verification_feedback"]
                            ),
                            *(
                                item
                                for cell in cells
                                for item in targets[cell]["reported_gaps"]
                            ),
                        )
                    )
                ),
                action=assignment.action,
                unanswered_questions=assignment.unanswered_questions,
                max_search_calls=0
                if assignment.action == "revise"
                else assignment.max_search_calls,
                max_model_calls=1,
                max_extract_calls=0
                if assignment.agent == "technical" and not web_refs
                else 1,
                max_extract_urls=6,
                draft_versions={
                    key: drafts[key].version + 1 if key in drafts else 1
                    for key in assignment.section_ids
                },
            )
            task.validate_against(plan)
            tasks.append(task)
        action = "revise" if all(t.action == "revise" for t in tasks) else "research"
        return SupervisorDecision(
            action=action, reason=proposal.reason, tasks=tuple(tasks)
        )
