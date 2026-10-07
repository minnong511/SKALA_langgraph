"""Task isolation, versioned merges, coverage, and per-run budgets."""

from threading import Lock
from typing import Any, Literal

from pydantic import Field, model_validator

from kv_cache_agent.schemas.base import Contract
from kv_cache_agent.schemas.evidence import EvidenceCard, Perspective, Technology
from kv_cache_agent.schemas.report import DraftClaim, ReportPlan, SectionDraft, Worker


class ResearchTask(Contract):
    task_id: str = Field(min_length=1)
    plan_id: str = Field(min_length=1)
    plan_version: int = Field(ge=1)
    round_id: int = Field(ge=1)
    agent: Worker
    section_ids: tuple[str, ...] = Field(min_length=1)
    technologies: tuple[Technology, ...] = Field(min_length=1)
    criteria: tuple[str, ...] = Field(min_length=1)
    objective: str = Field(min_length=1)
    questions: tuple[str, ...] = ()
    existing_evidence_ids: tuple[str, ...] = ()
    feedback: tuple[str, ...] = ()
    unanswered_questions: tuple[str, ...] = ()
    action: Literal["research", "revise"] = "research"
    max_search_calls: int = Field(default=3, ge=0)
    max_model_calls: int = Field(default=1, ge=0)
    max_extract_calls: int = Field(default=2, ge=0)
    max_extract_urls: int = Field(default=6, ge=0)
    draft_versions: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def unique_targets(self):
        if len(set(self.section_ids)) != len(self.section_ids):
            raise ValueError("Duplicate task sections")
        if not set(self.draft_versions) <= set(self.section_ids) or any(
            v < 1 for v in self.draft_versions.values()
        ):
            raise ValueError("Invalid requested draft versions")
        return self

    def validate_against(self, plan: ReportPlan) -> None:
        if (self.plan_id, self.plan_version) != (plan.plan_id, plan.version):
            raise ValueError("Stale task plan")
        sections = {s.section_id: s for s in plan.sections}
        for key in self.section_ids:
            if key not in sections or sections[key].owner != self.agent:
                raise ValueError("Task targets an unknown or unowned section")


class TaskResult(Contract):
    task_id: str
    plan_version: int = Field(ge=1)
    round_id: int = Field(ge=1)
    version: int = Field(default=1, ge=1)
    agent: Worker
    status: Literal["ok", "needs_retry", "insufficient_evidence", "failed"]
    drafts: tuple[SectionDraft, ...] = ()
    evidence: tuple[EvidenceCard, ...] = ()
    missing_items: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    usage: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_result(self):
        if any(d.status == "approved" for d in self.drafts):
            raise ValueError("A worker cannot approve its own draft")
        if any(
            d.owner != self.agent or d.plan_version != self.plan_version
            for d in self.drafts
        ):
            raise ValueError("Result contains an unowned or stale draft")
        if self.status == "failed" and not self.errors:
            raise ValueError("Failed task must explain its failure")
        return self


class SupervisorDecision(Contract):
    action: Literal["research", "verify", "revise", "finalize", "stop"]
    reason: str = Field(min_length=1)
    tasks: tuple[ResearchTask, ...] = ()
    claim_ids: tuple[str, ...] = ()
    unresolved_questions: dict[str, tuple[str, ...]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_action(self):
        if self.action in {"research", "revise"} and not self.tasks:
            raise ValueError("Dispatch requires tasks")
        if self.action not in {"research", "revise"} and self.tasks:
            raise ValueError("Tasks are only valid for dispatch actions")
        if self.action == "verify" and not self.claim_ids:
            raise ValueError("Verification requires claims")
        ids = [t.task_id for t in self.tasks]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate dispatch task IDs")
        return self


class CoverageItem(Contract):
    section_id: str
    technology: Technology
    perspective: Perspective
    criterion: str
    status: Literal["verified", "inferred", "missing", "unsupported"]
    claim_ids: tuple[str, ...] = ()
    reason: str = ""

    @model_validator(mode="after")
    def valid_coverage(self):
        if self.status in {"verified", "inferred"} and not self.claim_ids:
            raise ValueError("Coverage requires traceable claims")
        return self


class BudgetLimits(Contract):
    research_rounds: int = Field(default=3, ge=1)
    search_calls: int = Field(default=15, ge=0)
    extract_calls: int = Field(default=6, ge=0)
    extract_urls: int = Field(default=20, ge=0)
    model_calls: int = Field(default=20, ge=1)
    finish_reserve: int = Field(default=4, ge=0)

    @model_validator(mode="after")
    def valid_reserve(self):
        if self.finish_reserve > self.model_calls:
            raise ValueError("Finish reserve exceeds model budget")
        return self


class BudgetExceeded(RuntimeError):
    pass


class BudgetLedger:
    """Reserve before dispatch; attempts consume budget even when they fail."""

    def __init__(self, limits: BudgetLimits | None = None):
        self.limits = limits or BudgetLimits()
        self.used = {
            key: 0
            for key in ("search_calls", "extract_calls", "extract_urls", "model_calls")
        }
        self._lock = Lock()
        self._held = dict.fromkeys(self.used, 0)

    def _capacity(self, key, finishing=False):
        cap = getattr(self.limits, key)
        if key == "model_calls" and not finishing:
            cap -= self.limits.finish_reserve
        return cap - self.used[key] - self._held[key]

    def reserve(self, *, finishing: bool = False, **amounts: int) -> None:
        with self._lock:
            for key, amount in amounts.items():
                if key not in self.used or not isinstance(amount, int) or amount < 0:
                    raise ValueError("Invalid budget reservation")
                if amount > self._capacity(key, finishing):
                    raise BudgetExceeded(f"Budget exhausted: {key}")
            for key, amount in amounts.items():
                self.used[key] += amount

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self.used)

    def available(self, *, finishing: bool = False) -> dict[str, int]:
        with self._lock:
            return {key: self._capacity(key, finishing) for key in self.used}

    def held_snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._held)

    def allocate_many(
        self,
        allocations: dict[str, dict[str, int]],
        *,
        retain: dict[str, int] | None = None,
    ) -> dict[str, "BudgetReservation"]:
        """Hold a complete dispatch atomically; count only attempted calls as used."""
        with self._lock:
            totals = dict.fromkeys(self.used, 0)
            for amounts in (*allocations.values(), retain or {}):
                for key, amount in amounts.items():
                    if key not in totals or not isinstance(amount, int) or amount < 0:
                        raise ValueError("Invalid budget allocation")
                    totals[key] += amount
            for key, amount in totals.items():
                if amount > self._capacity(key):
                    raise BudgetExceeded(f"Dispatch budget exhausted: {key}")
            reservations = {}
            for task_id, amounts in allocations.items():
                caps = {key: amounts.get(key, 0) for key in self.used}
                for key, cap in caps.items():
                    self._held[key] += cap
                reservations[task_id] = BudgetReservation(self, caps)
            return reservations


class BudgetReservation:
    """Unused capacity returns at completion; failed attempts remain charged."""

    def __init__(self, ledger: BudgetLedger, caps: dict[str, int]):
        self.ledger = ledger
        self.limits = ledger.limits
        self.remaining = dict(caps)
        self.used = dict.fromkeys(caps, 0)
        self.closed = False

    def reserve(self, *, finishing: bool = False, **amounts):
        with self.ledger._lock:
            if self.closed or finishing:
                raise BudgetExceeded(
                    "Worker reservation is closed or used for finalization"
                )
            for key, amount in amounts.items():
                if key not in self.used or not isinstance(amount, int) or amount < 0:
                    raise ValueError("Invalid reservation use")
                if amount > self.remaining[key]:
                    raise BudgetExceeded(f"Task allocation exhausted: {key}")
            for key, amount in amounts.items():
                self.remaining[key] -= amount
                self.ledger._held[key] -= amount
                self.ledger.used[key] += amount
                self.used[key] += amount

    def snapshot(self):
        with self.ledger._lock:
            return dict(self.used)

    def close(self):
        with self.ledger._lock:
            if not self.closed:
                for key, amount in self.remaining.items():
                    self.ledger._held[key] -= amount
                self.remaining = dict.fromkeys(self.remaining, 0)
                self.closed = True


def merge_versioned(
    existing: dict[str, Any], incoming: dict[str, Any]
) -> dict[str, Any]:
    """Immutable ID maps: ignore stale versions; reject conflicting same versions."""
    merged = dict(existing)
    for key, item in incoming.items():
        previous = merged.get(key)
        if previous is None:
            merged[key] = item
            continue
        version = (
            getattr(item, "plan_version", 0),
            getattr(item, "round_id", 0),
            item.version,
        )
        old_version = (
            getattr(previous, "plan_version", 0),
            getattr(previous, "round_id", 0),
            previous.version,
        )
        if version < old_version:
            continue
        if version == old_version and item != previous:
            raise ValueError(f"Conflicting record version: {key}")
        merged[key] = item
    return merged


def legacy_claim(card: dict, *, section_id: str = "legacy") -> DraftClaim:
    return DraftClaim(
        claim_id=str(card["evidence_id"]),
        section_id=section_id,
        technology=card.get("technology", "general"),
        perspective=card["perspective"],
        criterion=card.get("criterion", "legacy"),
        text=card["claim"],
        claim_type=card["claim_type"],
        evidence_ids=(str(card["evidence_id"]),),
    )
