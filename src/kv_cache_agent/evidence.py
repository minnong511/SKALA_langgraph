"""Source identity and claim deduplication, independent of task-scoped IDs."""

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


def source_identity(card):
    url = str(card.get("source_url", "")).strip()
    # This repository's PDFs are the same publications as the cited arXiv pages.
    publications = {
        "turboquant.pdf": "2504.19874",
        "cxl_based_kv_cache.pdf": "2606.12556",
    }
    if not urlsplit(url).scheme or url.startswith("file:"):
        paper = publications.get(Path(url).name)
        if paper:
            return f"arxiv:{paper}"
    match = re.search(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})(?:v\d+)?", url)
    if match:
        return f"arxiv:{match[1]}"
    parts = urlsplit(url)
    if parts.scheme in {"http", "https"}:
        query = [
            (k, v)
            for k, v in parse_qsl(parts.query)
            if not k.lower().startswith("utm_")
        ]
        return urlunsplit(
            (
                "https",
                parts.netloc.lower().removeprefix("www."),
                parts.path.rstrip("/"),
                urlencode(sorted(query)),
                "",
            )
        )
    return url


def claim_fingerprint(card):
    claim = (
        re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(card.get("claim", ""))))
        .strip()
        .casefold()
    )
    key = "|".join(
        [
            source_identity(card),
            str(card.get("technology", "")),
            str(card.get("perspective", "")),
            claim,
        ]
    )
    if card.get("metric"):
        key = "|".join(
            [
                source_identity(card),
                str(card.get("technology", "")),
                str(card.get("perspective", "")),
                *[str(card.get(k)) for k in ("metric", "value", "unit", "baseline")],
                json.dumps(card.get("conditions", {}), sort_keys=True),
            ]
        )
    return hashlib.sha256(key.encode()).hexdigest()[:24]


def comparison_context(cards):
    """Pairwise comparability only; never label differing numbers as conflicts."""
    measured = [c for c in cards if c.get("metric")]
    rows = []
    for index, left in enumerate(measured):
        for right in measured[index + 1 :]:
            if left["evidence_id"] == right["evidence_id"]:
                continue
            same_metric = left.get("metric") == right.get("metric") and left.get(
                "unit"
            ) == right.get("unit")
            a, b = left.get("conditions", {}), right.get("conditions", {})
            reasons = []
            if not same_metric:
                reasons.append("different_metric_or_unit")
            if not left.get("baseline") or not right.get("baseline"):
                reasons.append("unknown_baseline")
            elif left["baseline"] != right["baseline"]:
                reasons.append("different_baseline")
            if any(
                not a.get(k) or not b.get(k) for k in ("model", "hardware", "workload")
            ):
                reasons.append("unknown_experimental_conditions")
            elif a != b:
                reasons.append("different_experimental_conditions")
            rows.append(
                {
                    "evidence_ids": [left["evidence_id"], right["evidence_id"]],
                    "directly_comparable": not reasons,
                    "reasons": reasons,
                }
            )
            if len(rows) == 200:
                return rows
    return rows
