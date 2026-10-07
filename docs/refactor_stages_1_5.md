# Refactoring foundation: stages 1–5

Baseline: `ee4f860`, 88 tests passed. This increment preserves the legacy orchestration and worker interfaces; the dynamic five-agent workflow is scheduled for stages 6–13.

## Implemented

1. Regression cases preserve the known page-cache defect and identify deferred cloud locator, report validation, and CLI status defects with strict `xfail` markers.
2. Validated contracts describe report plans, section ownership, tasks/results, actual draft claims, multiple source references, version-bound decisions, coverage, and atomic budgets. `FoundationState` is additive to legacy state. Workers cannot self-approve sections.
3. Every existing graph node uses shared registration for start/end/error/heartbeat events. The CLI creates a run session and writes flushed JSONL and a run summary. Context includes task, round, section, technology, criterion, nested invocation and trace IDs. LangSmith uses native graph/model tracing and explicit SDK/tool spans; credentials and large snapshots are sanitized. Tracing errors preserve local events.
4. Production web fetch uses Tavily Extract; injected HTTP clients remain a test/compatibility adapter. Search summary and `raw_content` remain distinct. Extraction has per-URL outcomes, bounded batches/requests, TTL cache, body hashes, single-flight access, and usage accounting. Provider HTTP success is not treated as origin HTTP success.
5. Common verification uses four nodes: metadata, original-source fetch, actual-claim judgement, finalization. The legacy comparison entrypoint delegates to it. PDF page lists/ranges and multiple sources remain separate. Unknown hosts and search excerpts cannot support verified facts. Numeric checks use fetched source content, quotes must exist in the original, and decisions bind to claim/source/policy versions. Author-generated excerpts are omitted from the judge's evidence context.

## Interface examples

```python
from pathlib import Path
from kv_cache_agent.observability.logger import RunSession
from kv_cache_agent.verification.pipeline import VerificationPipeline

with RunSession(Path("outputs/logs"), metadata={"section_ids": ["section_4_2"]}) as run:
    result = VerificationPipeline().verify(claims, evidence)
    run.finish("completed" if result.status == "ok" else "provisional")
```

LangSmith settings are in `.env.example`. Offline tests must set `LANGSMITH_TRACING=false`. Enabled tracing validates key/project settings before tools run. `langsmith` 0.14.0 was already locked transitively and is now declared directly; `uv lock --check --offline` validates the unchanged dependency versions.

Large original bodies are cached intact. Model context over 80,000 source characters is rejected explicitly. Select a `SourceRef.character_range` tied to the full-body `version` hash to submit a bounded passage with surrounding conditions. Local PDF snapshots also track the document digest. This avoids silent truncation and stale span reuse.

## Live checks

Using the configured account, one Tavily search and one Extract batch for three public URLs succeeded. The batch included both URLs previously blocked by direct HTTP: Fortune Business Insights and SEC EDGAR, plus Tavily documentation. Returned bodies were approximately 36k, 1.29m, and 13k characters; source acquisition alone does not establish factual correctness.

A first live judgement paraphrased its purported quotation; deterministic quotation validation rejected it. A positive control using an actual source statement then passed the default LLM comparison. LangSmith traces and nested graph/tool/model calls were retrieved from the server. Raw API keys and complete source documents are not included in this record.

## Remaining migration

- Stage 8: legacy cloud worker still replaces a requested locator with the last technical card for the URL.
- Stage 11: legacy report writer still permits uncited numerical assertions in provisional mode.
- Stage 12: the legacy aggregate CLI JSON and exit status still need the completed/provisional/failed migration. New per-run event/summary logs already expose the foundation outcome separately.

These three cases are strict expected failures, not silently omitted checks. Remove their markers when their planned implementations land. The old eight-agent graph remains active until the worker and supervisor migration is implemented.
