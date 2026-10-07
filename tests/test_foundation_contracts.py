from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import ValidationError

from kv_cache_agent.schemas.evidence import SourceRef, SourceSnapshot
from kv_cache_agent.schemas.report import (
    DraftClaim,
    ReportPlan,
    SectionDraft,
    SectionSpec,
)
from kv_cache_agent.schemas.research import (
    BudgetExceeded,
    BudgetLedger,
    BudgetLimits,
    ResearchTask,
    merge_versioned,
)


def plan():
    return ReportPlan(
        plan_id="p",
        user_query="Compare KV cache",
        sections=tuple(
            SectionSpec(
                section_id=f"s-{p}",
                title=p,
                order=i,
                owner=p,
                objective="Evaluate both technologies",
                perspective=p,
            )
            for i, p in enumerate(
                ("technical", "market", "stakeholder", "cloud_domain")
            )
        ),
    )


def test_report_plan_cannot_drop_required_views_or_accept_dependency_cycles():
    valid = plan()
    with pytest.raises(ValidationError, match="Missing evaluation sections"):
        ReportPlan.model_validate(
            {**valid.model_dump(), "sections": valid.sections[:-1]}
        )
    with pytest.raises(ValidationError, match="Cyclic"):
        sections = [
            s.model_copy(update={"dependencies": (s.section_id,)})
            for s in valid.sections
        ]
        ReportPlan.model_validate({**valid.model_dump(), "sections": sections})


def test_dynamic_task_is_bound_to_section_owner_and_plan_version():
    task = ResearchTask(
        task_id="m1",
        plan_id="p",
        plan_version=1,
        round_id=1,
        agent="market",
        section_ids=("s-market",),
        technologies=("CXL-based",),
        criteria=("adoption",),
        objective="Find direct deployment evidence",
    )
    task.validate_against(plan())
    with pytest.raises(ValueError, match="unowned"):
        task.model_copy(update={"section_ids": ("s-technical",)}).validate_against(
            plan()
        )
    with pytest.raises(ValueError, match="Stale"):
        task.model_copy(update={"plan_version": 2}).validate_against(plan())


def test_actual_draft_fact_requires_evidence_and_rejects_blank_text():
    fields = {
        "claim_id": "c",
        "section_id": "s-market",
        "technology": "CXL-based",
        "perspective": "market",
        "criterion": "adoption",
        "text": "It was deployed.",
        "claim_type": "fact",
    }
    with pytest.raises(ValidationError, match="require evidence"):
        DraftClaim(**fields)
    with pytest.raises(ValidationError):
        DraftClaim(**{**fields, "text": " "}, evidence_ids=("e",))


def test_source_contract_rejects_bad_pages_and_tampered_body():
    with pytest.raises(ValidationError):
        SourceRef(source_id="x", location="data/papers/a.pdf", pages=(0,))
    with pytest.raises(ValidationError):
        SourceSnapshot(
            reference=SourceRef(source_id="x", location="https://example.com"),
            acquisition="tavily_extract",
            status="ok",
            content="body",
            content_hash="fake",
        )


def test_version_merge_does_not_overwrite_new_plan_with_old_draft():
    current = SectionDraft(section_id="s", owner="market", plan_version=2, version=1)
    old = SectionDraft(section_id="s", owner="market", plan_version=1, version=9)
    assert merge_versioned({"s": current}, {"s": old})["s"] == current
    with pytest.raises(ValueError, match="Conflicting"):
        merge_versioned(
            {"s": current}, {"s": current.model_copy(update={"status": "unavailable"})}
        )


def test_parallel_reservations_are_atomic_and_preserve_finish_calls():
    budget = BudgetLedger(BudgetLimits(model_calls=6, finish_reserve=2))

    def reserve(_):
        try:
            budget.reserve(model_calls=1)
            return True
        except BudgetExceeded:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(reserve, range(12))) == 4
    budget.reserve(model_calls=2, finishing=True)
    with pytest.raises(BudgetExceeded):
        budget.reserve(model_calls=1, finishing=True)
    before = budget.snapshot()
    with pytest.raises(BudgetExceeded):
        budget.reserve(search_calls=1, extract_urls=100)
    assert budget.snapshot() == before


def test_worker_cannot_self_approve_a_report_section():
    from kv_cache_agent.schemas.research import TaskResult

    with pytest.raises(ValidationError, match="cannot approve"):
        TaskResult(
            task_id="m",
            plan_version=1,
            round_id=1,
            agent="market",
            status="ok",
            drafts=(
                SectionDraft(
                    section_id="s", plan_version=1, owner="market", status="approved"
                ),
            ),
        )
