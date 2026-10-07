"""Coverage comes from current verified assertions, never worker success flags."""

from kv_cache_agent.schemas.base import fingerprint
from kv_cache_agent.schemas.report import ReportPlan, SectionDraft
from kv_cache_agent.schemas.research import CoverageItem
from kv_cache_agent.verification.pipeline import POLICY_VERSION

WORKERS = ("technical", "market", "stakeholder", "cloud_domain")
TECHNOLOGIES = ("TurboQuant", "CXL-based")
DIRECT_CRITERIA = {
    "market_size",
    "market_scale",
    "adoption",
    "commercialization",
    "public_position",
    "public_statement",
    "deployment",
}


def validate_research_plan(plan: ReportPlan) -> ReportPlan:
    """Normalize explicit policy requirements; preserve model-authored titles and scope."""
    plan = ReportPlan.model_validate(plan)
    sections = []
    for section in plan.sections:
        if section.owner in WORKERS:
            if (
                section.perspective != section.owner
                or not section.criteria
                or not section.questions
            ):
                raise ValueError(
                    "Worker sections require matching perspective, questions and criteria"
                )
            technologies = section.technologies or plan.technologies
            if not set(technologies) <= set(TECHNOLOGIES):
                raise ValueError(
                    "Worker sections must evaluate the two technologies separately"
                )
            if len(set(technologies)) != len(technologies):
                raise ValueError("Duplicate section technologies")
            direct = set(section.direct_evidence_criteria) | (
                set(section.criteria) & DIRECT_CRITERIA
            )
            if section.owner in {"market", "stakeholder"}:
                direct.add(section.criteria[0])
            section = section.model_copy(
                update={
                    "technologies": tuple(technologies),
                    "direct_evidence_criteria": tuple(
                        c for c in section.criteria if c in direct
                    ),
                }
            )
        sections.append(section)
    for worker in WORKERS:
        covered = {t for s in sections if s.owner == worker for t in s.technologies}
        if not set(TECHNOLOGIES) <= covered:
            raise ValueError(f"Both technologies need actual {worker} worker sections")
    if not any(s.owner == "supervisor" for s in sections):
        raise ValueError("Report outline needs Supervisor-owned final sections")
    specs = {s.section_id: s for s in sections}
    technical = {
        technology: next(
            s.section_id
            for s in sections
            if s.owner == "technical" and technology in s.technologies
        )
        for technology in TECHNOLOGIES
    }
    normalized = []
    for section in sections:
        if section.owner in WORKERS:
            if any(specs[key].owner not in WORKERS for key in section.dependencies):
                raise ValueError(
                    "Research cannot depend on unwritten final report sections"
                )
            if section.owner != "technical":
                deps = (
                    *section.dependencies,
                    *(technical[t] for t in section.technologies),
                )
                section = section.model_copy(
                    update={"dependencies": tuple(dict.fromkeys(deps))}
                )
        normalized.append(section)
    return ReportPlan.model_validate({**plan.model_dump(), "sections": normalized})


def plan_cells(plan):
    return tuple(
        (s.section_id, technology, criterion)
        for s in sorted(plan.sections, key=lambda s: s.order)
        if s.owner in WORKERS
        for technology in s.technologies
        for criterion in s.criteria
    )


def current_claims(state):
    claims = {}
    for draft in state.get("drafts_by_section", {}).values():
        for claim in draft.claims:
            if claim.claim_id in claims and claims[claim.claim_id] != claim:
                raise ValueError("Conflicting current claim ID")
            claims[claim.claim_id] = claim
    return claims


def verification_input(claim, evidence):
    return fingerprint(
        {
            "claim": claim.model_dump(mode="json"),
            "evidence": {
                key: evidence[key].model_dump(mode="json") if key in evidence else None
                for key in claim.evidence_ids
            },
            "policy": POLICY_VERSION,
        }
    )


def decision_is_current(claim, state):
    decision = state.get("verification_decisions", {}).get(claim.claim_id)
    return bool(
        decision
        and decision.claim_version == claim.version
        and decision.claim_hash == fingerprint(claim)
        and decision.policy_version == POLICY_VERSION
        and state.get("verification_inputs", {}).get(claim.claim_id)
        == verification_input(claim, state.get("evidence_by_id", {}))
    )


def pending_claim_ids(state):
    return tuple(
        key
        for key, claim in current_claims(state).items()
        if not decision_is_current(claim, state)
    )


def claim_is_usable(claim, state):
    if not decision_is_current(claim, state):
        return False
    decision = state["verification_decisions"][claim.claim_id]
    if (
        decision.status != "verified"
        or decision.support_level != "full"
        or decision.criterion_assessment != "relevant"
        or decision.claim_type_assessment != "correct"
    ):
        return False
    cards = [state["evidence_by_id"].get(key) for key in claim.evidence_ids]
    if not cards or any(card is None or not card.source_refs for card in cards):
        return False
    if claim.claim_type == "fact":
        return all(card.perspective == claim.perspective for card in cards)
    return claim.claim_type == "inference" and all(
        card.caveat.strip() for card in cards
    )


def build_coverage(state):
    plan = state["report_plan"]
    specs = {s.section_id: s for s in plan.sections}
    claims = current_claims(state)
    coverage = []
    for section_id, technology, criterion in plan_cells(plan):
        matching = [
            c
            for c in claims.values()
            if (c.section_id, c.technology, c.criterion)
            == (section_id, technology, criterion)
        ]
        supported = [c for c in matching if claim_is_usable(c, state)]
        facts = [c for c in supported if c.claim_type == "fact"]
        direct = criterion in specs[section_id].direct_evidence_criteria
        accepted = facts if direct else (facts or supported)
        status = (
            "verified"
            if facts
            else "inferred"
            if accepted
            else "unsupported"
            if matching
            else "missing"
        )
        coverage.append(
            CoverageItem(
                section_id=section_id,
                technology=technology,
                perspective=specs[section_id].perspective,
                criterion=criterion,
                status=status,
                claim_ids=tuple(c.claim_id for c in accepted),
                reason="Verified direct facts"
                if facts
                else "Explicit supported inference"
                if accepted
                else "This criterion requires verified direct facts"
                if direct
                else "No current supported assertion",
            )
        )
    return coverage


def dependencies_ready(section, technology, plan, coverage):
    specs = {s.section_id: s for s in plan.sections}
    cells = {(c.section_id, c.technology, c.criterion): c for c in coverage}
    for key in section.dependencies:
        dependency = specs[key]
        if technology not in dependency.technologies:
            continue
        if any(
            cells[(key, technology, criterion)].status not in {"verified", "inferred"}
            for criterion in dependency.criteria
        ):
            return False
    return True


def reported_gaps(state):
    """Only latest target-specific feedback can reopen a fulfilled cell."""
    latest = {}
    tasks = state.get("tasks_by_id", {})
    for task in sorted(tasks.values(), key=lambda t: t.round_id):
        result = state.get("task_results", {}).get(task.task_id)
        if result is None:
            continue
        for section_id in task.section_ids:
            for technology in task.technologies:
                for criterion in task.criteria:
                    prefix = f"{section_id}/{technology}/{criterion}:"
                    latest[(section_id, technology, criterion)] = tuple(
                        item for item in result.missing_items if item.startswith(prefix)
                    )
                    if (
                        result.status == "failed"
                        and not latest[(section_id, technology, criterion)]
                    ):
                        latest[(section_id, technology, criterion)] = (
                            f"{prefix} Worker failed: {'; '.join(result.errors)}",
                        )
    return {cell: items for cell, items in latest.items() if items}


def revision_feedback(state):
    result = {}
    for claim in current_claims(state).values():
        if decision_is_current(claim, state) and not claim_is_usable(claim, state):
            decision = state["verification_decisions"][claim.claim_id]
            cell = (claim.section_id, claim.technology, claim.criterion)
            result.setdefault(cell, []).append(
                {
                    "claim_id": claim.claim_id,
                    "status": decision.status,
                    "reason": decision.rationale,
                    "issues": decision.issues,
                    "requested_action": "research"
                    if decision.support_level == "none"
                    or decision.criterion_assessment == "irrelevant"
                    or "Original source unavailable" in decision.issues
                    else "revise",
                }
            )
    return result


def eligible_targets(state):
    plan = state["report_plan"]
    specs = {s.section_id: s for s in plan.sections}
    feedback = revision_feedback(state)
    gaps = reported_gaps(state)
    targets = []
    for item in build_coverage(state):
        cell = (item.section_id, item.technology, item.criterion)
        unanswered = state.get("semantic_gaps", {}).get("/".join(cell), ())
        if (
            item.status in {"verified", "inferred"}
            and cell not in feedback
            and cell not in gaps
            and not unanswered
        ):
            continue
        if not dependencies_ready(
            specs[item.section_id], item.technology, plan, build_coverage(state)
        ):
            continue
        targets.append(
            {
                "section_id": item.section_id,
                "agent": specs[item.section_id].owner,
                "technology": item.technology,
                "criterion": item.criterion,
                "coverage_status": item.status,
                "verification_feedback": feedback.get(cell, []),
                "reported_gaps": gaps.get(cell, ()),
                "unanswered_questions": unanswered,
            }
        )
    return targets


def ready_to_finalize(state):
    return bool(
        build_coverage(state)
        and all(c.status in {"verified", "inferred"} for c in build_coverage(state))
        and not pending_claim_ids(state)
        and not revision_feedback(state)
        and not reported_gaps(state)
    )


def safe_drafts(state):
    coverage = build_coverage(state)
    safe = []
    specs = {s.section_id: s for s in state["report_plan"].sections}
    for draft in state.get("drafts_by_section", {}).values():
        claims = tuple(c for c in draft.claims if claim_is_usable(c, state))
        gaps = [
            c
            for c in coverage
            if c.section_id == draft.section_id
            and c.status not in {"verified", "inferred"}
        ]
        all_good = (
            not gaps
            and len(claims) == len(draft.claims)
            and bool(claims)
            and not any(
                key.startswith(draft.section_id + "/")
                for key in state.get("semantic_gaps", {})
            )
        )
        safe.append(
            SectionDraft(
                section_id=draft.section_id,
                plan_version=draft.plan_version,
                version=draft.version,
                owner=specs[draft.section_id].owner,
                claims=claims,
                limitations=tuple(
                    dict.fromkeys(
                        (
                            *draft.limitations,
                            *(
                                f"{c.technology}/{c.criterion}: {c.reason}"
                                for c in gaps
                            ),
                        )
                    )
                ),
                status="approved"
                if all_good
                else "needs_revision"
                if claims
                else "unavailable",
            )
        )
    return tuple(sorted(safe, key=lambda d: specs[d.section_id].order))


def progress_signature(state):
    approved = set()
    sources = set()
    for claim in current_claims(state).values():
        if claim_is_usable(claim, state):
            approved.add(
                fingerprint(
                    {
                        "cell": (claim.section_id, claim.technology, claim.criterion),
                        "text": claim.text,
                        "claim_type": claim.claim_type,
                    }
                )
            )
        if decision_is_current(claim, state):
            decision = state["verification_decisions"][claim.claim_id]
            if (
                decision.support_level != "none"
                and "Original source unavailable" not in decision.issues
            ):
                sources.update(decision.source_versions.values())
    cells = {
        (c.section_id, c.technology, c.criterion, c.status)
        for c in build_coverage(state)
        if c.status in {"verified", "inferred"}
    }
    return {"approved": approved, "sources": sources, "cells": cells}
