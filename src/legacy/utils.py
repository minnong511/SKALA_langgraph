"""Shared normalization helpers for the validation and synthesis nodes."""

from __future__ import annotations

from typing import Any, Mapping


EVIDENCE_KEYS = (
    "technical_evidence_cards",
    "market_evidence_cards",
    "stakeholder_evidence_cards",
    "domain_evidence_cards",
)

RESULT_KEYS = (
    "technical_result",
    "market_result",
    "stakeholder_result",
    "domain_result",
)


def collect_evidence_cards(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return unified evidence cards, with safe fallbacks before the join node exists."""

    cards = state.get("evidence_cards") or []
    if cards:
        return [dict(card) for card in cards if isinstance(card, Mapping)]

    merged: list[dict[str, Any]] = []
    for key in EVIDENCE_KEYS:
        value = state.get(key) or []
        merged.extend(dict(card) for card in value if isinstance(card, Mapping))

    if merged:
        return merged

    # Early integration fallback: some agents may still embed cards in *_result.
    for key in RESULT_KEYS:
        result = state.get(key) or {}
        if isinstance(result, Mapping):
            value = result.get("evidence_cards") or []
            merged.extend(dict(card) for card in value if isinstance(card, Mapping))
    return merged


def normalize_evidence_cards(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign stable IDs and normalize common aliases without mutating the input."""

    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(cards, start=1):
        card = dict(raw)
        claim_id = str(
            card.get("claim_id")
            or card.get("evidence_id")
            or f"claim-{index:04d}"
        )
        card["claim_id"] = claim_id
        card.setdefault("evidence_id", claim_id)

        if "claim" not in card:
            card["claim"] = card.get("statement", "")
        if "evidence_text" not in card:
            card["evidence_text"] = card.get("quote", card.get("content", ""))
        if "source_agent" not in card:
            card["source_agent"] = _agent_from_perspective(card.get("perspective"))

        # Keep prompts bounded while retaining enough surrounding evidence.
        if isinstance(card.get("evidence_text"), str):
            card["evidence_text"] = card["evidence_text"][:4000]
        normalized.append(card)
    return normalized


def compact_agent_results(state: Mapping[str, Any]) -> dict[str, Any]:
    """Remove duplicated evidence payloads before sending results to the synthesis LLM."""

    compact: dict[str, Any] = {}
    for key in RESULT_KEYS:
        result = state.get(key) or {}
        if not isinstance(result, Mapping):
            compact[key] = result
            continue
        compact[key] = {
            field: value
            for field, value in result.items()
            if field != "evidence_cards"
        }
    return compact


def _agent_from_perspective(value: Any) -> str:
    perspective = str(value or "").lower()
    if perspective in {"technical", "trl"}:
        return "technical"
    if perspective in {"market", "stakeholder", "domain"}:
        return perspective
    return "supervisor"
