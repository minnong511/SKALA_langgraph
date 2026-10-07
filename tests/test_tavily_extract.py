from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pytest

from kv_cache_agent.schemas.research import BudgetExceeded, BudgetLedger, BudgetLimits
from kv_cache_agent.tools.tavily_extract import SourceCache, TavilyExtractor


def test_extract_handles_out_of_order_partial_and_omitted_results():
    client = Mock()
    client.extract.return_value = {
        "results": [{"url": "https://example.org/b", "raw_content": "B body"}],
        "failed_results": [{"url": "https://example.org/a", "error": "HTTP 403"}],
        "usage": {"credits": 0},
        "request_id": "request-1",
    }
    result = TavilyExtractor(client=client).extract(
        [
            "https://example.org/a",
            "https://example.org/b",
            "https://example.org/c",
        ]
    )
    assert [s.status for s in result.snapshots] == ["blocked", "ok", "error"]
    assert result.snapshots[1].content == "B body"
    assert result.credits == 0
    assert "query" not in client.extract.call_args.kwargs
    assert client.extract.call_args.kwargs["format"] == "markdown"


def test_cache_is_single_flight_under_parallel_access_and_expires(tmp_path):
    client = Mock()
    client.extract.return_value = {
        "results": [{"url": "https://example.org/a", "raw_content": "A body"}]
    }
    now = datetime(2026, 10, 7, tzinfo=UTC)
    clock = [now]
    service = TavilyExtractor(
        client=client,
        cache=SourceCache(tmp_path, ttl_seconds=60),
        clock=lambda: clock[0],
    )
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(lambda _: service.extract(["https://example.org/a"]), range(4))
        )
    assert client.extract.call_count == 1
    assert sum(r.cache_hits for r in results) == 3
    first = results[0].snapshots[0]
    clock[0] += timedelta(seconds=61)
    client.extract.return_value = {
        "results": [{"url": "https://example.org/a", "raw_content": "Changed body"}]
    }
    latest = service.extract(["https://example.org/a"]).snapshots[0]
    assert client.extract.call_count == 2
    assert latest.content_hash != first.content_hash


def test_all_failed_response_is_not_success_and_failed_attempts_consume_budget():
    client = Mock()
    client.extract.side_effect = TimeoutError("provider timeout")
    ledger = BudgetLedger(BudgetLimits(extract_calls=1, extract_urls=1))
    extractor = TavilyExtractor(client=client, budget=ledger)
    result = extractor.extract(["https://example.org/a"])
    assert result.snapshots[0].status == "error"
    assert ledger.snapshot()["extract_calls"] == 1
    with pytest.raises(BudgetExceeded):
        extractor.extract(["https://example.org/a"])
    assert client.extract.call_count == 1


def test_invalid_url_never_reaches_provider():
    client = Mock()
    with pytest.raises(ValueError):
        TavilyExtractor(client=client).extract(["ftp://example.org/a"])
    client.extract.assert_not_called()


def test_production_source_fetch_uses_extract_instead_of_direct_http(monkeypatch):
    from kv_cache_agent.schemas.evidence import SourceRef, SourceSnapshot
    from kv_cache_agent.tools import source_fetcher, tavily_extract

    snapshot = SourceSnapshot(
        reference=SourceRef(source_id="s", location="https://example.org/a"),
        acquisition="tavily_extract",
        status="ok",
        content="actual body",
    )
    monkeypatch.setattr(tavily_extract, "web_fetch", lambda _: snapshot)
    monkeypatch.setattr(
        source_fetcher.httpx,
        "Client",
        lambda **_: pytest.fail("Direct HTTP is not the default"),
    )
    result = source_fetcher.fetch_source("https://example.org/a")
    assert result["content"] == "actual body"
    assert result["retrieval_method"] == "tavily_extract"
    assert result["status_code"] == 0  # unknown origin status, not provider HTTP 200


def test_versioned_character_span_reads_cached_body_and_rejects_stale_version(tmp_path):
    from kv_cache_agent.schemas.evidence import SourceRef
    from kv_cache_agent.verification.sources import SourceLoader

    url = "https://docs.nvidia.com/page"
    client = Mock()
    client.extract.return_value = {
        "results": [
            {
                "url": url,
                "raw_content": "Large preamble. Important evidence. Large appendix.",
            }
        ]
    }
    service = TavilyExtractor(client=client, cache=SourceCache(tmp_path))
    snapshot = service.extract([url]).snapshots[0]
    ref = SourceRef(
        source_id="s",
        location=url,
        version=snapshot.content_hash,
        character_range=(16, 35),
    )
    scoped = SourceLoader(extractor=service).load(ref)
    assert scoped.content == "Important evidence."
    with pytest.raises(ValueError, match="Source changed"):
        SourceLoader(extractor=service).load(
            ref.model_copy(update={"version": "stale"})
        )

    assert client.extract.call_count == 1
