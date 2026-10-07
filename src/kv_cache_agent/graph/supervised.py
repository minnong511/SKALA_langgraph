"""Dynamic research loop with isolated Send inputs and one coordinator per round."""

from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from kv_cache_agent.agents.supervisor_control import SupervisorController
from kv_cache_agent.agents.task_worker import get_worker
from kv_cache_agent.graph.progress import (
    WORKERS,
    build_coverage,
    claim_is_usable,
    current_claims,
    pending_claim_ids,
    plan_cells,
    progress_signature,
    ready_to_finalize,
    reported_gaps,
    safe_drafts,
    validate_research_plan,
    verification_input,
)
from kv_cache_agent.graph.routing import (
    dispatch_workers,
    route_after_collection,
    route_supervisor_decision,
)
from kv_cache_agent.graph.state import SupervisedState
from kv_cache_agent.observability.logger import CURRENT_CONTEXT, emit
from kv_cache_agent.observability.nodes import add_logged_node
from kv_cache_agent.observability.tracing import TracingSettings
from kv_cache_agent.schemas.base import fingerprint
from kv_cache_agent.schemas.report import SectionDraft
from kv_cache_agent.schemas.research import (
    BudgetLedger,
    BudgetLimits,
    SupervisorDecision,
    TaskResult,
    merge_versioned,
)
from kv_cache_agent.schemas.supervision import WorkerDelivery, WorkflowInput
from kv_cache_agent.schemas.worker import WorkerInput
from kv_cache_agent.verification.pipeline import (
    POLICY_VERSION,
    VerificationPipeline,
    VerificationResult,
)
from kv_cache_agent.verification.sources import SourceLoader, reference_key


def _default_verifier(budget):
    return VerificationPipeline(loader=SourceLoader(budget=budget).load, budget=budget)


def _task_cells(task, plan):
    return set(plan_cells(plan)) & {
        (key, technology, criterion)
        for key in task.section_ids
        for technology in task.technologies
        for criterion in task.criteria
    }


def _technical_inputs(state, task):
    valid = {
        key
        for key, claim in current_claims(state).items()
        if claim.perspective == "technical"
        and claim.technology in task.technologies
        and claim_is_usable(claim, state)
    }
    results = []
    for original in state["task_results"].values():
        if original.agent != "technical":
            continue
        drafts = tuple(
            d.model_copy(
                update={
                    "claims": tuple(c for c in d.claims if c.claim_id in valid),
                    "status": "draft",
                }
            )
            for d in original.drafts
            if any(c.claim_id in valid for c in d.claims)
        )
        ids = {
            key
            for draft in drafts
            for claim in draft.claims
            for key in claim.evidence_ids
        }
        if drafts:
            results.append(
                TaskResult.model_validate(
                    original.model_copy(
                        update={
                            "drafts": drafts,
                            "evidence": tuple(
                                state["evidence_by_id"][key] for key in ids
                            ),
                        }
                    )
                )
            )
    return tuple(results)


def _validate_worker_result(result, request):
    result = TaskResult.model_validate(result)
    task = request.task
    if result.round_id < task.round_id or result.plan_version < task.plan_version:
        return result, "stale"
    if (result.task_id, result.agent, result.plan_version, result.round_id) != (
        task.task_id,
        task.agent,
        task.plan_version,
        task.round_id,
    ):
        raise ValueError("Worker returned a result for a different assignment")
    cells = set(request.cells())
    if len({d.section_id for d in result.drafts}) != len(result.drafts):
        raise ValueError("Duplicate result sections")
    evidence = {e.evidence_id: e for e in result.evidence}
    if len(evidence) != len(result.evidence):
        raise ValueError("Duplicate result evidence IDs")
    allowed_ids = set(evidence) | set(task.existing_evidence_ids)
    for draft in result.drafts:
        if (
            draft.section_id not in task.section_ids
            or draft.version != task.draft_versions[draft.section_id]
        ):
            raise ValueError(
                "Worker returned an unassigned or outdated section version"
            )
        for claim in draft.claims:
            if (claim.section_id, claim.technology, claim.criterion) not in cells:
                raise ValueError("Worker returned an out-of-scope assertion")
            if (
                claim.perspective != task.agent
                or not claim.evidence_ids
                or not set(claim.evidence_ids) <= allowed_ids
            ):
                raise ValueError(
                    "Worker assertion has invalid perspective or evidence links"
                )
    for card in result.evidence:
        if card.perspective != task.agent or not any(
            (key, card.technology, card.criterion) in cells for key in task.section_ids
        ):
            raise ValueError("Worker returned out-of-scope evidence")
    return result, "none"


def _failed_result(task, message):
    return TaskResult(
        task_id=task.task_id,
        agent=task.agent,
        plan_version=task.plan_version,
        round_id=task.round_id,
        status="failed",
        errors=(message,),
        missing_items=tuple(
            f"{s}/{t}/{c}: {message}"
            for s in task.section_ids
            for t in task.technologies
            for c in task.criteria
        ),
    )


class SupervisedWorkflow:
    """Configuration can be reused; state, budgets and checkpointer are per invocation."""

    def __init__(
        self,
        *,
        supervisor=None,
        workers=None,
        verifier_factory=None,
        finalizer=None,
        budget_limits=None,
        parallel=True,
    ):
        self.supervisor = supervisor or SupervisorController()
        self.workers = dict(workers or {})
        if set(self.workers) - set(WORKERS):
            raise ValueError("Unknown worker dependency")
        self.verifier_factory = verifier_factory or _default_verifier
        self.finalizer = finalizer
        self.budget_limits = budget_limits or BudgetLimits()
        self.parallel = parallel

    def invoke(self, inputs, config: RunnableConfig | None = None):
        inputs = WorkflowInput.model_validate(inputs)
        context = CURRENT_CONTEXT.get()
        run_id = str(uuid4())
        budget = BudgetLedger(self.budget_limits)
        allocations = {}
        latest = {}
        state = {
            "user_query": inputs.user_query,
            "workflow_run_id": run_id,
            "source_refs": inputs.source_refs,
            "task_results": {},
            "tasks_by_id": {},
            "deliveries": {},
            "evidence_by_id": {},
            "drafts_by_section": {},
            "verification_decisions": {},
            "verification_inputs": {},
            "semantic_gaps": {},
            "coverage": [],
            "active_task_ids": [],
            "active_requests": {},
            "processed_task_ids": [],
            "decisions": [],
            "round_id": 0,
            "no_progress_rounds": 0,
            "round_pending_progress": False,
            "progress_baseline": {},
            "errors": [],
            "fatal_errors": [],
            "status": "planning",
        }
        if inputs.report_plan is not None:
            state["report_plan"] = inputs.report_plan
        try:
            settings = (
                context.session.tracing if context else TracingSettings.from_env()
            )
            settings.validate()
            verifier = self.verifier_factory(budget)
        except Exception as error:  # noqa: BLE001 - fail before planning or external research
            emit(
                "workflow_preflight_failed",
                level="ERROR",
                status="failed",
                message=str(error),
            )
            return {
                **state,
                "status": "failed",
                "termination_reason": "configuration_failed",
                "fatal_errors": [f"{type(error).__name__}: {error}"],
                "budget_usage": budget.snapshot(),
            }

        def remember(updates, current):
            # Used only to preserve the last committed parent state on terminal errors.
            latest.clear()
            latest.update({**current, **updates})
            return updates

        def plan_report(current, config: RunnableConfig):
            try:
                plan = current.get("report_plan")
                if plan is None:
                    plan = self.supervisor.plan(
                        current["user_query"],
                        budget=budget,
                        run_id=run_id,
                        config=config,
                    )
                else:
                    plan = validate_research_plan(plan)
                    if plan.user_query != current["user_query"]:
                        raise ValueError(
                            "Provided plan belongs to a different user query"
                        )
                return remember({"report_plan": plan}, current)
            except Exception as error:  # noqa: BLE001 - return explicit terminal failure
                return remember(
                    {
                        "fatal_errors": [f"Planning: {type(error).__name__}: {error}"],
                        "status": "failed",
                        "termination_reason": "planning_failed",
                    },
                    current,
                )

        def review_progress(current):
            coverage = build_coverage(current)
            updates = {"coverage": coverage, "budget_usage": budget.snapshot()}
            gap_cells = reported_gaps(current)
            drafts = {}
            for key, draft in current["drafts_by_section"].items():
                complete = all(
                    c.status in {"verified", "inferred"}
                    for c in coverage
                    if c.section_id == key
                )
                complete = complete and not any(cell[0] == key for cell in gap_cells)
                complete = complete and not any(
                    cell.startswith(key + "/")
                    for cell in current.get("semantic_gaps", {})
                )
                approved = (
                    complete
                    and bool(draft.claims)
                    and all(claim_is_usable(c, current) for c in draft.claims)
                )
                drafts[key] = draft.model_copy(
                    update={
                        "status": "approved"
                        if approved
                        else "needs_revision"
                        if draft.claims
                        else "unavailable"
                    }
                )
            updates["drafts_by_section"] = drafts
            if current["round_pending_progress"] and not pending_claim_ids(current):
                signature = progress_signature(current)
                previous = current.get("progress_baseline", {})
                improved = any(
                    signature[key] - previous.get(key, set()) for key in signature
                )
                count = 0 if improved else current["no_progress_rounds"] + 1
                updates.update(
                    {
                        "no_progress_rounds": count,
                        "round_pending_progress": False,
                        "progress_baseline": signature,
                    }
                )
                emit(
                    "research_progress",
                    details={
                        "improved": improved,
                        "no_progress_rounds": count,
                        "round_id": current["round_id"],
                        "coverage": [c.model_dump(mode="json") for c in coverage],
                    },
                )
            return remember(updates, current)

        def supervisor_decide(current, config: RunnableConfig):
            try:
                if current.get("remaining_steps", 100) <= 3:
                    decision = SupervisorDecision(
                        action="stop", reason="recursion_limit"
                    )
                else:
                    decision = self.supervisor.decide(
                        current, budget=budget, config=config
                    )
                updates = {
                    "decision": decision,
                    "decisions": [*current["decisions"], decision],
                    "budget_usage": budget.snapshot(),
                }
                semantic_gaps = dict(current.get("semantic_gaps", {}))
                semantic_gaps.update(decision.unresolved_questions)
                for task in decision.tasks:
                    if task.unanswered_questions:
                        for cell in _task_cells(task, current["report_plan"]):
                            semantic_gaps["/".join(cell)] = task.unanswered_questions
                updates["semantic_gaps"] = (
                    {} if decision.action == "finalize" else semantic_gaps
                )
            except Exception as error:  # noqa: BLE001 - no unsafe fallback dispatch
                decision = SupervisorDecision(
                    action="stop", reason="supervisor_decision_failed"
                )
                updates = {
                    "decision": decision,
                    "decisions": [*current["decisions"], decision],
                    "fatal_errors": [
                        *current["fatal_errors"],
                        f"Routing: {type(error).__name__}: {error}",
                    ],
                    "status": "failed",
                }
            emit(
                "supervisor_decision",
                action=decision.action,
                reason=decision.reason,
                details={
                    "tasks": [t.model_dump(mode="json") for t in decision.tasks],
                    "claim_ids": decision.claim_ids,
                    "budget": budget.snapshot(),
                },
            )
            return remember(updates, current)

        def prepare_dispatch(current):
            try:
                available = budget.available()
                admitted = []
                for task in current["decision"].tasks:
                    if task.task_id in current["tasks_by_id"]:
                        raise ValueError("Task ID was already dispatched")
                    counts = {
                        "model_calls": 1,
                        "search_calls": min(
                            task.max_search_calls, available["search_calls"]
                        )
                        if task.agent != "technical"
                        else 0,
                        "extract_calls": min(
                            task.max_extract_calls, available["extract_calls"]
                        ),
                        "extract_urls": min(
                            task.max_extract_urls, available["extract_urls"]
                        ),
                    }
                    web_research = (
                        task.agent != "technical"
                        and task.action == "research"
                        and task.max_search_calls > 0
                    )
                    if available["model_calls"] < 2 or (
                        web_research
                        and any(
                            counts[key] < 1
                            for key in ("search_calls", "extract_calls", "extract_urls")
                        )
                    ):
                        emit(
                            "dispatch_deferred",
                            level="WARNING",
                            details={
                                "task_id": task.task_id,
                                "reason": "budget_capacity",
                            },
                        )
                        continue
                    if task.max_extract_calls == 0:
                        counts["extract_urls"] = 0
                    task = task.model_copy(
                        update={
                            "max_search_calls": counts["search_calls"]
                            if task.agent != "technical"
                            else task.max_search_calls,
                            "max_extract_calls": counts["extract_calls"],
                            "max_extract_urls": counts["extract_urls"],
                        }
                    )
                    for key, amount in counts.items():
                        available[key] -= amount
                    admitted.append((task, counts))
                if not admitted:
                    return remember(
                        {
                            "decision": SupervisorDecision(
                                action="stop", reason="dispatch_budget_exhausted"
                            )
                        },
                        current,
                    )
                requests = {}
                for task, _ in admitted:
                    requests[task.task_id] = WorkerInput(
                        task=task,
                        plan=current["report_plan"],
                        source_refs=current["source_refs"]
                        if task.agent == "technical"
                        else (),
                        existing_evidence=tuple(
                            current["evidence_by_id"][key]
                            for key in task.existing_evidence_ids
                        ),
                        existing_drafts=tuple(
                            current["drafts_by_section"][key]
                            for key in task.section_ids
                            if key in current["drafts_by_section"]
                        ),
                        technical_results=_technical_inputs(current, task)
                        if task.agent == "cloud_domain"
                        else (),
                    )
                allocations.update(
                    budget.allocate_many(
                        {t.task_id: caps for t, caps in admitted},
                        retain={"model_calls": 1},
                    )
                )
                tasks = {
                    **current["tasks_by_id"],
                    **{t.task_id: t for t, _ in admitted},
                }
                updates = {
                    "tasks_by_id": tasks,
                    "active_task_ids": list(requests),
                    "active_requests": requests,
                    "round_id": admitted[0][0].round_id,
                    "round_pending_progress": True,
                    "progress_baseline": progress_signature(current),
                    "status": "researching",
                }
                emit(
                    "dispatch_reserved",
                    details={
                        "task_ids": list(requests),
                        "held": budget.held_snapshot(),
                        "remaining": budget.available(),
                        "mode": "parallel" if self.parallel else "sequential",
                    },
                )
                return remember(updates, current)
            except Exception as error:  # noqa: BLE001 - preserve state on allocation/contract failure
                return remember(
                    {
                        "decision": SupervisorDecision(
                            action="stop", reason="dispatch_failed"
                        ),
                        "fatal_errors": [
                            *current["fatal_errors"],
                            f"Dispatch: {type(error).__name__}: {error}",
                        ],
                    },
                    current,
                )

        def dispatch(current):
            return {}

        def worker(packet, config: RunnableConfig):
            request = packet["request"]
            task = request.task
            reserved = allocations[task.task_id]
            issue_kind = "none"
            message = ""
            try:
                implementation = self.workers.get(task.agent) or get_worker(task.agent)
                try:
                    method = (
                        implementation
                        if callable(implementation)
                        else implementation.run
                    )
                    raw = method(request, budget=reserved, config=config)
                except Exception as error:  # noqa: BLE001 - one worker must not lose sibling results
                    raw = _failed_result(
                        task, f"Worker execution: {type(error).__name__}: {error}"
                    )
                try:
                    result, issue_kind = _validate_worker_result(raw, request)
                    if issue_kind == "stale":
                        message = "Delayed older task result discarded"
                        result = _failed_result(task, message)
                        emit(
                            "stale_result_discarded",
                            level="WARNING",
                            details={"task_id": task.task_id},
                        )
                except Exception as error:  # noqa: BLE001 - classify invalid protocol explicitly
                    issue_kind = "contract"
                    message = f"Worker contract: {type(error).__name__}: {error}"
                    result = _failed_result(task, message)
                delivery = WorkerDelivery(
                    task_id=task.task_id,
                    plan_version=task.plan_version,
                    round_id=task.round_id,
                    result=result.model_copy(
                        update={"usage": {**result.usage, **reserved.snapshot()}}
                    ),
                    issue_kind=issue_kind,
                    message=message,
                )
                return {"deliveries": {task.task_id: delivery}}
            finally:
                reserved.close()

        def collect_results(current):
            drafts = dict(current["drafts_by_section"])
            evidence = dict(current["evidence_by_id"])
            results = dict(current["task_results"])
            processed = list(current["processed_task_ids"])
            fatal = list(current["fatal_errors"])
            errors = list(current["errors"])
            for task_id in current["active_task_ids"]:
                if task_id in processed or task_id not in current["deliveries"]:
                    continue
                delivery = current["deliveries"][task_id]
                task = current["tasks_by_id"][task_id]
                result = delivery.result
                processed.append(task_id)
                results[task_id] = result
                errors.extend(result.errors)
                if delivery.issue_kind == "contract":
                    fatal.append(delivery.message)
                    continue
                evidence = merge_versioned(
                    evidence, {card.evidence_id: card for card in result.evidence}
                )
                targets = _task_cells(task, current["report_plan"])
                for section in task.section_ids:
                    previous = drafts.get(section)
                    incoming = next(
                        (d for d in result.drafts if d.section_id == section), None
                    )
                    if incoming is None:
                        continue
                    keep = (
                        tuple(
                            c
                            for c in previous.claims
                            if (c.section_id, c.technology, c.criterion) not in targets
                        )
                        if previous
                        else ()
                    )
                    prefixes = tuple(f"{s}/{t}/{c}:" for s, t, c in targets)
                    limits = (
                        tuple(
                            item
                            for item in previous.limitations
                            if not item.startswith(prefixes)
                        )
                        if previous
                        else ()
                    )
                    drafts[section] = SectionDraft(
                        section_id=section,
                        plan_version=task.plan_version,
                        version=task.draft_versions[section],
                        owner=task.agent,
                        claims=(*keep, *incoming.claims),
                        limitations=tuple(
                            dict.fromkeys((*limits, *incoming.limitations))
                        ),
                        status="draft" if keep or incoming.claims else "unavailable",
                    )
            updates = {
                "task_results": results,
                "processed_task_ids": processed,
                "drafts_by_section": drafts,
                "evidence_by_id": evidence,
                "fatal_errors": fatal,
                "errors": list(dict.fromkeys(errors)),
                "budget_usage": budget.snapshot(),
            }
            emit(
                "round_results_collected",
                details={
                    "selected": current["active_task_ids"],
                    "completed": [
                        key for key in current["active_task_ids"] if key in processed
                    ],
                    "outstanding": [
                        key
                        for key in current["active_task_ids"]
                        if key not in processed
                    ],
                    "round_id": current["round_id"],
                },
            )
            return remember(updates, current)

        def verify_claims(current, config: RunnableConfig):
            try:
                claims = current_claims(current)
                ids = current["decision"].claim_ids
                selected = [claims[key] for key in ids]
                evidence_ids = {key for c in selected for key in c.evidence_ids}
                result = VerificationResult.model_validate(
                    verifier.verify(
                        selected,
                        [current["evidence_by_id"][key] for key in evidence_ids],
                        config=config,
                    )
                )
                if any(
                    error.startswith("Claim comparison failed:")
                    and "BudgetExceeded" not in error
                    for error in result.errors
                ):
                    raise ValueError(
                        f"Verification execution/contract failed: {'; '.join(result.errors)}"
                    )
                if {d.claim_id for d in result.decisions} != set(ids) or len(
                    result.decisions
                ) != len(ids):
                    raise ValueError(
                        "Verification omitted or duplicated current claims"
                    )
                decisions = dict(current["verification_decisions"])
                inputs = dict(current["verification_inputs"])
                for decision in result.decisions:
                    claim = claims[decision.claim_id]
                    if (
                        decision.claim_version != claim.version
                        or decision.claim_hash != fingerprint(claim)
                        or decision.policy_version != POLICY_VERSION
                    ):
                        raise ValueError("Verification returned a stale claim decision")
                    expected_sources = {
                        reference_key(ref)
                        for key in claim.evidence_ids
                        for ref in current["evidence_by_id"][key].source_refs
                    }
                    if set(decision.source_versions) != expected_sources:
                        raise ValueError(
                            "Verification returned decisions for different original sources"
                        )
                    if decision.status == "verified" and (
                        decision.support_level != "full"
                        or decision.criterion_assessment != "relevant"
                        or decision.claim_type_assessment != "correct"
                    ):
                        raise ValueError(
                            "Verified decision lacks full source/scope support"
                        )
                    decisions[claim.claim_id] = decision
                    inputs[claim.claim_id] = verification_input(
                        claim, current["evidence_by_id"]
                    )
                return remember(
                    {
                        "verification_decisions": decisions,
                        "verification_inputs": inputs,
                        "errors": [*current["errors"], *result.errors],
                        "status": "verifying",
                        "budget_usage": budget.snapshot(),
                    },
                    current,
                )
            except Exception as error:  # noqa: BLE001 - terminal verification protocol errors
                return remember(
                    {
                        "fatal_errors": [
                            *current["fatal_errors"],
                            f"Verification: {type(error).__name__}: {error}",
                        ]
                    },
                    current,
                )

        def finish(current, config: RunnableConfig):
            failed = bool(current["fatal_errors"]) or "report_plan" not in current
            ready = (
                not failed
                and ready_to_finalize(current)
                and current.get("decision")
                and current["decision"].action == "finalize"
            )
            status = (
                "failed"
                if failed
                else "ready_for_finalization"
                if ready
                else "provisional"
            )
            reason = current.get("termination_reason") or (
                current["decision"].reason
                if current.get("decision")
                else "execution_failed"
            )
            payload = {}
            artifact = {}
            if not failed:
                payload = {
                    "report_plan": current["report_plan"],
                    "drafts": safe_drafts(current),
                    "coverage": tuple(build_coverage(current)),
                    "status": status,
                    "termination_reason": reason,
                    "unresolved_questions": current.get("semantic_gaps", {}),
                    "missing_items": tuple(
                        c
                        for c in build_coverage(current)
                        if c.status not in {"verified", "inferred"}
                    ),
                }
                if self.finalizer:
                    try:
                        artifact = self.finalizer(payload, budget=budget, config=config)
                        if not isinstance(artifact, dict):
                            raise TypeError("Finalization hook must return a mapping")
                    except Exception as error:  # noqa: BLE001 - never label finalization failure ready
                        status = "failed"
                        reason = "finalization_failed"
                        current = {
                            **current,
                            "fatal_errors": [
                                *current["fatal_errors"],
                                f"Finalization: {type(error).__name__}: {error}",
                            ],
                        }
            emit(
                "research_loop_finished",
                status=status,
                reason=reason,
                details={
                    "round_id": current["round_id"],
                    "budget_usage": budget.snapshot(),
                    "report_generation_pending": self.finalizer is None,
                },
            )
            return remember(
                {
                    "status": status,
                    "termination_reason": reason,
                    "finalization_input": payload,
                    "finalization_result": artifact,
                    "fatal_errors": current["fatal_errors"],
                    "budget_usage": budget.snapshot(),
                    **(
                        {
                            "drafts_by_section": {
                                d.section_id: d for d in payload["drafts"]
                            }
                        }
                        if ready
                        else {}
                    ),
                },
                current,
            )

        graph = StateGraph(SupervisedState)
        for name, function in (
            ("plan_report", plan_report),
            ("review_progress", review_progress),
            ("supervisor_decide", supervisor_decide),
            ("prepare_dispatch", prepare_dispatch),
            ("dispatch", dispatch),
            ("worker", worker),
            ("collect_results", collect_results),
            ("verify_claims", verify_claims),
            ("finish", finish),
        ):
            add_logged_node(graph, name, function, path=f"supervised.{name}")
        graph.add_edge(START, "plan_report")
        graph.add_conditional_edges(
            "plan_report",
            lambda s: "finish" if s.get("fatal_errors") else "review_progress",
        )
        graph.add_edge("review_progress", "supervisor_decide")
        graph.add_conditional_edges("supervisor_decide", route_supervisor_decision)
        graph.add_conditional_edges(
            "prepare_dispatch",
            lambda s: (
                "dispatch"
                if s["decision"].action in {"research", "revise"}
                else "finish"
            ),
        )
        graph.add_conditional_edges(
            "dispatch", lambda s: dispatch_workers(s, parallel=self.parallel)
        )
        graph.add_edge("worker", "collect_results")
        graph.add_conditional_edges("collect_results", route_after_collection)
        graph.add_edge("verify_claims", "review_progress")
        graph.add_edge("finish", END)
        allowed = (
            [
                ("kv_cache_agent.schemas.report", name)
                for name in ("ReportPlan", "SectionSpec", "SectionDraft", "DraftClaim")
            ]
            + [
                ("kv_cache_agent.schemas.research", name)
                for name in (
                    "ResearchTask",
                    "TaskResult",
                    "SupervisorDecision",
                    "CoverageItem",
                )
            ]
            + [
                ("kv_cache_agent.schemas.evidence", name)
                for name in (
                    "EvidenceCard",
                    "SourceRef",
                    "ClaimAttribution",
                    "VerificationDecision",
                )
            ]
            + [
                ("kv_cache_agent.schemas.worker", "WorkerInput"),
                ("kv_cache_agent.schemas.supervision", "WorkerDelivery"),
            ]
        )
        runnable = graph.compile(
            checkpointer=InMemorySaver(
                serde=JsonPlusSerializer(allowed_msgpack_modules=allowed)
            ),
            name="supervised_research",
        )
        actual_config = {**(config or {})}
        actual_config["configurable"] = {
            **actual_config.get("configurable", {}),
            "thread_id": run_id,
        }
        actual_config.setdefault(
            "recursion_limit", 20 + self.budget_limits.research_rounds * 18
        )
        actual_config["max_concurrency"] = min(
            actual_config.get("max_concurrency", 4), 4
        )
        try:
            return runnable.invoke(state, config=actual_config)
        except Exception as error:  # noqa: BLE001 - retain partial parent state and failed outcome
            current = runnable.get_state(actual_config).values or latest or state
            reason = (
                "recursion_limit"
                if isinstance(error, GraphRecursionError)
                else "graph_execution_failed"
            )
            emit(
                "research_loop_error",
                level="ERROR",
                status="failed",
                message=type(error).__name__,
            )
            return {
                **current,
                "status": "failed",
                "termination_reason": reason,
                "fatal_errors": [
                    *current.get("fatal_errors", []),
                    f"{type(error).__name__}: {error}",
                ],
                "budget_usage": budget.snapshot(),
            }
        finally:
            for reservation in allocations.values():
                reservation.close()
