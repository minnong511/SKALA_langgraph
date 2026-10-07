from unittest.mock import Mock

import pytest

from kv_cache_agent.agents.supervisor_control import SupervisorController
from kv_cache_agent.graph.progress import (
    build_coverage,
    eligible_targets,
    pending_claim_ids,
    validate_research_plan,
)
from kv_cache_agent.graph.workflow import build_supervised_workflow
from kv_cache_agent.schemas.research import BudgetLedger, BudgetLimits
from kv_cache_agent.schemas.supervision import Assignment, ReportOutline, RouteProposal
from tests.supervised_fixtures import QUERY, FixtureResearch, controller, plan, raw_plan


def completed_state():
    research = FixtureResearch()
    return build_supervised_workflow(
        supervisor=controller(),
        workers=research.workers,
        verifier_factory=research.verifier,
    ).invoke({"user_query": QUERY})


def test_plan_role_prompts_are_used_and_titles_objectives_come_from_model():
    model = Mock()
    model.with_structured_output.return_value.invoke.return_value = ReportOutline(
        sections=raw_plan().sections
    )
    service = SupervisorController(model=model)
    budget = BudgetLedger()
    result = service.plan(
        QUERY, budget=budget, run_id="r", config={"tags": ["plan-test"]}
    )
    messages = model.with_structured_output.return_value.invoke.call_args.args[0]
    assert "목차" in messages[0].content and messages[1].content == QUERY
    assert result.sections[0].title == raw_plan().sections[0].title
    assert budget.snapshot()["model_calls"] == 1
    assert result.sections[1].direct_evidence_criteria == ("adoption",)
    assert "technical" in result.sections[1].dependencies


@pytest.mark.parametrize(
    "defect",
    ("wrong_owner", "missing_technology", "empty_criteria", "final_dependency"),
)
def test_plan_cannot_fake_required_worker_coverage(defect):
    source = raw_plan()
    sections = list(source.sections)
    if defect == "wrong_owner":
        sections[0] = sections[0].model_copy(update={"owner": "supervisor"})
    if defect == "missing_technology":
        sections[1] = sections[1].model_copy(update={"technologies": ("TurboQuant",)})
    if defect == "empty_criteria":
        sections[2] = sections[2].model_copy(update={"criteria": ()})
    if defect == "final_dependency":
        sections[0] = sections[0].model_copy(update={"dependencies": ("conclusion",)})
    with pytest.raises(ValueError):
        validate_research_plan(source.model_copy(update={"sections": tuple(sections)}))


def test_new_claim_is_verified_before_routing_or_round_limit_stop():
    state = completed_state()
    state["verification_decisions"] = {}
    router = Mock()
    service = SupervisorController(router=router)
    decision = service.decide(
        state, budget=BudgetLedger(BudgetLimits(research_rounds=1))
    )
    assert decision.action == "verify" and len(decision.claim_ids) == 8
    router.assert_not_called()


def test_changed_evidence_or_claim_invalidates_previous_verification():
    state = completed_state()
    card = next(iter(state["evidence_by_id"].values()))
    state["evidence_by_id"] = {
        **state["evidence_by_id"],
        card.evidence_id: card.model_copy(
            update={"version": 2, "caveat": "Changed conditions"}
        ),
    }
    pending = pending_claim_ids(state)
    assert len(pending) == 1
    coverage = build_coverage(state)
    assert sum(c.status == "unsupported" for c in coverage) == 1


def test_completed_target_cannot_be_dispatched_again_without_current_feedback():
    state = completed_state()
    proposal = RouteProposal(
        action="research",
        reason="repeat",
        assignments=(
            Assignment(
                agent="market",
                section_ids=("market",),
                technologies=("CXL-based",),
                criteria=("adoption",),
                objective="Repeat fulfilled work",
                questions=("Search again",),
            ),
        ),
    )
    with pytest.raises(ValueError, match="fulfilled"):
        controller().validate_proposal(proposal, state)


def test_unsupported_cell_routes_only_to_its_owner_with_dynamic_instructions():
    state = completed_state()
    claim = next(
        c
        for c in state["drafts_by_section"]["market"].claims
        if c.technology == "CXL-based"
    )
    state["verification_decisions"][claim.claim_id] = state["verification_decisions"][
        claim.claim_id
    ].model_copy(
        update={
            "status": "unsupported",
            "support_level": "none",
            "rationale": "Need a real deployment source",
        }
    )
    assert len(eligible_targets(state)) == 1
    model = Mock()
    model.with_structured_output.return_value.invoke.return_value = RouteProposal(
        action="research",
        reason="Only CXL adoption needs new evidence",
        assignments=(
            Assignment(
                agent="market",
                section_ids=("market",),
                technologies=("CXL-based",),
                criteria=("adoption",),
                objective="Find a named production deployment with its original documentation",
                questions=("Which operator has deployed CXL KV cache in production?",),
                trigger="verification_feedback",
            ),
        ),
    )
    service = SupervisorController(model=model)
    decision = service.decide(state, budget=BudgetLedger())
    task = decision.tasks[0]
    assert task.technologies == ("CXL-based",) and task.agent == "market"
    assert "named production deployment" in task.objective and task.feedback
    messages = model.with_structured_output.return_value.invoke.call_args.args[0]
    assert (
        "원문" in messages[0].content
        and "Need a real deployment source" in messages[1].content
    )
    assert task.draft_versions["market"] == 2


def test_scope_ownership_overlap_and_dependency_violations_are_rejected():
    state = {
        "report_plan": plan(),
        "drafts_by_section": {},
        "evidence_by_id": {},
        "verification_decisions": {},
        "verification_inputs": {},
        "round_id": 0,
        "task_results": {},
        "tasks_by_id": {},
    }
    service = controller()
    base = Assignment(
        agent="technical",
        section_ids=("technical",),
        technologies=("TurboQuant",),
        criteria=("mechanism",),
        objective="Find mechanism",
        questions=("What is the mechanism?",),
    )
    with pytest.raises(ValueError, match="overlapping"):
        service.validate_proposal(
            RouteProposal(action="research", reason="x", assignments=(base, base)),
            state,
        )
    with pytest.raises(ValueError, match="ownership"):
        service.validate_proposal(
            RouteProposal(
                action="research",
                reason="x",
                assignments=(base.model_copy(update={"agent": "market"}),),
            ),
            state,
        )
    with pytest.raises(ValueError, match="unready"):
        assignment = base.model_copy(
            update={
                "agent": "market",
                "section_ids": ("market",),
                "criteria": ("adoption",),
            }
        )
        service.validate_proposal(
            RouteProposal(action="research", reason="x", assignments=(assignment,)),
            state,
        )


def test_plan_and_coverage_cannot_be_replaced_by_model_finalize_assertion():
    state = {
        "report_plan": plan(),
        "drafts_by_section": {},
        "evidence_by_id": {},
        "verification_decisions": {},
        "verification_inputs": {},
        "round_id": 0,
        "task_results": {},
        "tasks_by_id": {},
    }
    with pytest.raises(ValueError, match="incomplete"):
        controller().validate_proposal(
            RouteProposal(action="finalize", reason="Trust me"), state
        )


def test_market_direct_criterion_does_not_accept_technical_or_inference_claim():
    state = completed_state()
    draft = state["drafts_by_section"]["market"]
    claim = draft.claims[0]
    card = state["evidence_by_id"][claim.evidence_ids[0]]
    state["evidence_by_id"][card.evidence_id] = card.model_copy(
        update={"perspective": "technical"}
    )
    assert not any(
        c.status == "verified"
        for c in build_coverage(state)
        if c.section_id == "market" and c.technology == claim.technology
    )


def test_semantic_revisit_requires_questions_from_original_plan():
    state = completed_state()
    assignment = Assignment(
        agent="market",
        section_ids=("market",),
        technologies=("CXL-based",),
        criteria=("adoption",),
        objective="Fill missing answer",
        questions=("New source",),
        trigger="semantic_gap",
        unanswered_questions=("Invented unrelated question",),
        feedback=("Insufficient",),
    )
    proposal = RouteProposal(
        action="research", reason="Semantic gap", assignments=(assignment,)
    )
    with pytest.raises(ValueError, match="outside the report plan"):
        controller().validate_proposal(proposal, state, allow_semantic_review=True)
    assignment = assignment.model_copy(
        update={"unanswered_questions": ("Find original support.",)}
    )
    with pytest.raises(ValueError, match="recorded semantic gap"):
        controller().validate_proposal(
            proposal.model_copy(update={"assignments": (assignment,)}), state
        )


def test_unrecognized_question_id_cannot_reopen_approved_coverage():
    state = completed_state()
    assignment = Assignment(
        agent="market",
        section_ids=("market",),
        technologies=("CXL-based",),
        criteria=("adoption",),
        objective="Fix",
        questions=("Fix",),
        trigger="semantic_gap",
        unanswered_question_ids=("invented-question-id",),
        feedback=("Not enough",),
    )
    with pytest.raises(ValueError, match="Unknown unanswered question ID"):
        controller().validate_proposal(
            RouteProposal(action="research", reason="x", assignments=(assignment,)),
            state,
            allow_semantic_review=True,
        )
