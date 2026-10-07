"""Synthetic originals and assignments; no project data or external API calls."""

from threading import Lock

from kv_cache_agent.agents.supervisor_control import SupervisorController
from kv_cache_agent.graph.progress import WORKERS, validate_research_plan
from kv_cache_agent.schemas.base import fingerprint, stable_id
from kv_cache_agent.schemas.evidence import EvidenceCard, SourceRef, SourceSnapshot
from kv_cache_agent.schemas.report import (
    DraftClaim,
    ReportPlan,
    SectionDraft,
    SectionSpec,
)
from kv_cache_agent.schemas.research import TaskResult
from kv_cache_agent.schemas.supervision import Assignment, ReportOutline, RouteProposal
from kv_cache_agent.verification.pipeline import (
    ClaimJudgement,
    JudgementBatch,
    VerificationPipeline,
)

CRITERIA = {
    "technical": "mechanism",
    "market": "adoption",
    "stakeholder": "public_position",
    "cloud_domain": "latency",
}
QUERY = "클라우드 KV cache 기술을 두 기술과 네 관점에서 비교한다."


def raw_plan():
    return ReportPlan(
        plan_id="test-plan",
        user_query=QUERY,
        sections=(
            *(
                SectionSpec(
                    section_id=role,
                    title=role,
                    order=i,
                    owner=role,
                    perspective=role,
                    objective="Evaluate requested criterion",
                    technologies=("TurboQuant", "CXL-based"),
                    criteria=(CRITERIA[role],),
                    questions=("Find original support.",),
                )
                for i, role in enumerate(WORKERS)
            ),
            SectionSpec(
                section_id="conclusion",
                title="결론",
                order=4,
                owner="supervisor",
                objective="Compare approved results",
                dependencies=WORKERS,
            ),
        ),
    )


def plan():
    return validate_research_plan(raw_plan())


def choose_missing(context, config=None):
    if context.get("completion_review") and not context["eligible_targets"]:
        return RouteProposal(
            action="finalize", reason="All planned questions have supported answers"
        )
    groups = {}
    for target in context["eligible_targets"]:
        cell = (target["agent"], target["section_id"], target["criterion"])
        groups.setdefault(cell, []).append(target)
    assignments = []
    for (role, section, criterion), targets in groups.items():
        if len(assignments) >= 4:
            break
        feedback = [f for t in targets for f in t["verification_feedback"]]
        action = (
            "revise"
            if feedback and all(f["requested_action"] == "revise" for f in feedback)
            else "research"
        )
        trigger = (
            "verification_feedback"
            if feedback
            else "reported_gap"
            if any(t["reported_gaps"] for t in targets)
            else "missing"
        )
        assignments.append(
            Assignment(
                agent=role,
                section_ids=(section,),
                technologies=tuple(t["technology"] for t in targets),
                criteria=(criterion,),
                objective=f"Fill only {criterion}; preserve evidence scope.",
                questions=(f"What original source supports {criterion}?",),
                action=action,
                trigger=trigger,
            )
        )
    return RouteProposal(
        action="research",
        reason="Fill eligible gaps only",
        assignments=tuple(assignments),
    )


def controller(router=choose_missing):
    return SupervisorController(
        planner=lambda query, config=None: ReportOutline(sections=raw_plan().sections),
        router=router,
    )


class FixtureResearch:
    def __init__(self, behavior=None):
        self.behavior = behavior
        self.calls = []
        self.sources = {}
        self.lock = Lock()
        self.workers = {role: self.for_role(role) for role in WORKERS}

    def for_role(self, role):
        def worker(request, *, budget, config=None):
            with self.lock:
                self.calls.append(request)
            budget.reserve(model_calls=1)
            if role != "technical" and request.task.max_search_calls:
                budget.reserve(search_calls=1, extract_calls=1, extract_urls=1)
            cards = []
            claims = []
            gaps = []
            for section, technology, criterion in request.cells():
                spec = {
                    "text": f"{technology} has documented {criterion} evidence for {role}.",
                    "body": f"{technology} has documented {criterion} evidence for {role}.",
                    "claim_type": "fact",
                    "skip": False,
                    "caveat": "",
                }
                if self.behavior:
                    spec.update(
                        self.behavior(request, (section, technology, criterion), spec)
                        or {}
                    )
                if spec["skip"]:
                    gaps.append(
                        f"{section}/{technology}/{criterion}: No original evidence found"
                    )
                    continue
                url = (
                    f"https://docs.nvidia.com/synthetic/{role}/{technology}/{criterion}"
                )
                ref = SourceRef(
                    source_id=stable_id("source", (url, spec["body"])),
                    location=url,
                    source_type="official",
                    version=fingerprint(spec["body"]),
                )
                with self.lock:
                    self.sources[url] = SourceSnapshot(
                        reference=ref,
                        content=spec["body"],
                        status="ok",
                        acquisition="tavily_extract",
                    )
                evidence_id = stable_id(
                    "evidence",
                    (
                        role,
                        technology,
                        criterion,
                        spec["text"],
                        spec["body"],
                        spec["claim_type"],
                    ),
                )
                cards.append(
                    EvidenceCard(
                        evidence_id=evidence_id,
                        technology=technology,
                        perspective=role,
                        criterion=criterion,
                        claim=spec["text"],
                        evidence_text=spec["body"],
                        source_refs=(ref,),
                        claim_type=spec["claim_type"],
                        caveat=spec["caveat"],
                    )
                )
                claims.append(
                    DraftClaim(
                        claim_id=stable_id("claim", (section, evidence_id)),
                        version=request.task.draft_versions[section],
                        section_id=section,
                        technology=technology,
                        perspective=role,
                        criterion=criterion,
                        text=spec["text"],
                        claim_type=spec["claim_type"],
                        evidence_ids=(evidence_id,),
                    )
                )
            drafts = tuple(
                SectionDraft(
                    section_id=key,
                    owner=role,
                    plan_version=request.plan.version,
                    version=request.task.draft_versions[key],
                    claims=tuple(c for c in claims if c.section_id == key),
                    limitations=tuple(gaps),
                    status="draft" if claims else "unavailable",
                )
                for key in request.task.section_ids
            )
            return TaskResult(
                task_id=request.task.task_id,
                agent=role,
                plan_version=request.plan.version,
                round_id=request.task.round_id,
                status="insufficient_evidence" if gaps else "ok",
                drafts=drafts,
                evidence=tuple(cards),
                missing_items=tuple(gaps),
                usage=budget.snapshot(),
            )

        return worker

    def loader(self, ref):
        return self.sources[ref.location].model_copy(update={"reference": ref})

    def judge(self, claims, evidence, snapshots):
        decisions = []
        for claim in claims:
            card = evidence[claim.evidence_ids[0]]
            partial = "always" in claim.text
            decisions.append(
                ClaimJudgement(
                    claim_id=claim.claim_id,
                    support_level="partial" if partial else "full",
                    matched_text=card.evidence_text,
                    rationale="Restrict the universal claim to its stated conditions"
                    if partial
                    else "Exact synthetic original",
                    claim_type_assessment="should_be_inference"
                    if partial
                    else "correct",
                    criterion_assessment="relevant",
                )
            )
        return JudgementBatch(decisions=decisions)

    def verifier(self, budget):
        return VerificationPipeline(loader=self.loader, judge=self.judge, budget=budget)
