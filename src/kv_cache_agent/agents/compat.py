"""Temporary legacy views; never checkpoint agent-local documents."""

from kv_cache_agent.evidence import claim_fingerprint
from kv_cache_agent.schemas.tasks import LEGACY_PERSPECTIVES


def grounded_worker_results(state, cards):
    """Use verified card claims instead of raw worker findings in generation prompts."""
    rows = []
    for result in state.get("dynamic_worker_results", []):
        selected = [
            cards[card["evidence_id"]]
            for card in result.get("evidence_cards", [])
            if card["evidence_id"] in cards
        ]
        excluded = len(result.get("evidence_cards", [])) - len(selected)
        rows.append(
            {
                **result,
                "findings": [card["claim"] for card in selected],
                "evidence_cards": selected,
                "limitations": [
                    *result.get("limitations", []),
                    *([f"원문 검증에서 제외한 근거 {excluded}개"] if excluded else []),
                ],
            }
        )
    return rows


def legacy_view(state):
    payload = state["payload"]
    results = payload.get("worker_results", [])
    view = {
        "control": state.get("control", {}),
        "user_query": payload.get("user_query", ""),
        "research_plan": payload.get("research_plan", {}),
        "evidence_cards": payload.get("evidence_cards", []),
        "verified_evidence_cards": payload.get("verified_evidence_cards", []),
        "usable_evidence_cards": payload.get("usable_evidence_cards", []),
        "verification_result": payload.get("verification", {}),
        "synthesis_result": payload.get("synthesis", {}),
        "final_report": payload.get("report", ""),
        "dynamic_worker_results": results,
        "quality_feedback": payload.get("evaluation") or {},
    }
    for perspective, legacy in LEGACY_PERSPECTIVES.items():
        selected = [r for r in results if r["perspective"] == perspective]
        usable = [r for r in selected if r["status"] != "failed"]
        view[f"{legacy}_result"] = {
            "status": "ok" if usable else "insufficient_evidence",
            "summary": "\n".join(f for r in usable for f in r["findings"]),
            "evidence_ids": [
                c["evidence_id"] for r in usable for c in r["evidence_cards"]
            ],
            "limitations": [s for r in selected for s in r["limitations"]],
            "errors": [s for r in selected for s in r["errors"]],
            "payload": {},
        }
    return view


def compact_verification(result):
    # Keep all input IDs for audit, but do not count duplicate facts as new support.
    usable = []
    seen = set()
    for card in sorted(
        result["usable_evidence_cards"],
        key=lambda c: c.get("verification_status") != "verified",
    ):
        fingerprint = claim_fingerprint(card)
        if fingerprint not in seen:
            seen.add(fingerprint)
            usable.append(card)
    verification = dict(result["verification_result"])
    verification["payload"] = {
        key: verification.get("payload", {}).get(key)
        for key in ("balance_result", "uncertain_evidence_ids", "retry_count")
    }
    verification["errors"] = (
        ["verification_error"] if verification.get("errors") else []
    )
    return {
        "verification": verification,
        "verified_evidence_cards": result["verified_evidence_cards"],
        "usable_evidence_cards": usable,
    }
