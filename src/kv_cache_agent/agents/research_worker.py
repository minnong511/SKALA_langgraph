"""One generic task node, reusing the existing research and citation helpers."""

from pathlib import Path

import yaml

from kv_cache_agent.agents import cloud_domain, market, stakeholder, technical
from kv_cache_agent.evidence import claim_fingerprint, source_identity
from kv_cache_agent.observability import node_span, record_event
from kv_cache_agent.schemas.outputs import WorkerResult
from kv_cache_agent.schemas.tasks import LEGACY_PERSPECTIVES, SubTask
from kv_cache_agent.tools.paper_retriever import retrieve_paper_chunks
from kv_cache_agent.tools.tavily_search import search_web

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "worker.yaml"


def _extract_web_cards(local, web):
    """Reuse source-bound structured extraction when heuristic coverage is absent."""
    chunks = [
        {
            "chunk_id": f"web-{index}",
            "content": item.get("raw_content") or item.get("content") or "",
            "metadata": {
                "source_title": item.get("title", ""),
                "source_url": item["url"],
                "source_locator": "web page",
                "published_date": item.get("published_date", ""),
                "retrieval_method": "tavily",
                "source_type": market._source_type(
                    None, item.get("title", ""), item["url"]
                ),
            },
        }
        for index, item in enumerate(web)
    ]
    extraction = technical._extract_findings(local, chunks)
    cards, skipped = technical._build_evidence_cards(extraction, chunks)
    limitations = list(extraction.limitations)
    if skipped:
        limitations.append("웹 원문에 연결되지 않은 구조화 주장은 제외함")
    return cards, limitations


def _legacy_web_cards(task, web):
    records = []
    module = stakeholder if task.perspective == "stakeholder" else market
    for item in web:
        record = module._normalize_result(item, task.query)
        if record is not None:
            record["technology"] = market._technology_from_text(
                f"{record['title']} {record['content']}"
            )
            records.append(record)
    if task.perspective == "stakeholder":
        return stakeholder._build_stakeholder_evidence({"source_records": records})[
            "evidence_cards"
        ]
    return market._build_evidence_cards(records)[0]


def _research_once(task: SubTask, request: dict) -> WorkerResult:
    """One source strategy per attempt, with bounded documents kept outside State."""
    from kv_cache_agent.tools.source_fetcher import fetch_source

    chunks, web, cards, limitations, errors = [], [], [], [], []
    if task.preferred_source in {"paper", "hybrid"}:
        try:
            chunks = retrieve_paper_chunks([task.query], top_k=6)
        except Exception as error:  # noqa: BLE001
            errors.append(f"paper:{type(error).__name__}")
    if task.preferred_source in {"web", "hybrid"}:
        try:
            kwargs = {
                "max_results": 5,
                "include_raw_content": True,
                "search_depth": "advanced",
            }
            if request.get("restrict_domains") and task.preferred_domains:
                kwargs["include_domains"] = task.preferred_domains
            web = search_web(task.query, **kwargs)
            for item in web:
                content = item.get("raw_content")
                if not content:
                    fetched = fetch_source(item["url"], max_chars=12000, timeout=10)
                    if fetched.get("fetch_status") == "ok":
                        content = fetched.get("content")
                    else:
                        limitations.append(
                            "일부 원문 접근 실패, 검색 요약은 잠정 근거로만 사용"
                        )
                item["content"] = str(content or item.get("content", ""))[:12000]
                item["raw_content"] = item["content"]
        except Exception as error:  # noqa: BLE001
            errors.append(f"web:{type(error).__name__}")

    instruction = yaml.safe_load(PROMPT_PATH.read_text(encoding="utf-8"))[
        "system_prompt"
    ]
    local = {
        "user_query": f"{request['user_query']}\n도메인: {request['target_domain']}\nTask: {task.model_dump_json()}",
        "task_instruction": instruction,
    }
    try:
        if task.perspective == "domain_application" and (chunks or web):
            extraction = cloud_domain._extract_cloud_assessment(local, [], web, chunks)
            cards, skipped = cloud_domain._build_evidence_cards(
                extraction, [], web, chunks
            )
            limitations.extend(extraction.limitations)
        else:
            skipped = []
            if chunks:
                extraction = technical._extract_findings(local, chunks)
                paper_cards, skipped = technical._build_evidence_cards(
                    extraction, chunks
                )
                cards.extend(paper_cards)
                limitations.extend(extraction.limitations)
            if web:
                web_cards, web_limitations = _extract_web_cards(local, web)
                if not web_cards and task.perspective in {"market", "stakeholder"}:
                    web_cards = _legacy_web_cards(task, web)
                cards.extend(web_cards)
                limitations.extend(web_limitations)
        if skipped:
            limitations.append("입력 출처에 연결되지 않은 주장은 제외함")
    except Exception as error:  # noqa: BLE001
        errors.append(f"extraction:{type(error).__name__}")
        if web and task.perspective in {"market", "stakeholder"}:
            cards.extend(_legacy_web_cards(task, web))

    sources = {
        source_identity(
            {
                "source_url": c["metadata"].get("source_url")
                or c["metadata"].get("source_path", "")
            }
        )
        for c in chunks
    }
    sources.update(source_identity({"source_url": item["url"]}) for item in web)
    if not cards:
        limitations.append("배정된 목표의 출처 연결 근거를 확보하지 못함")
    failure = (
        "none"
        if cards
        else "extraction"
        if any(e.startswith("extraction:") for e in errors)
        else "transport"
        if errors and not (chunks or web)
        else "insufficient_evidence"
    )
    return WorkerResult(
        task_id=task.task_id,
        perspective=task.perspective,
        technology=task.technology,
        status="partial"
        if cards and (limitations or errors)
        else "success"
        if cards
        else "failed",
        findings=[c["claim"] for c in cards],
        evidence_cards=cards,
        limitations=limitations,
        errors=errors,
        failure_kind=failure,
        search_attempts=[
            {
                "query": task.query,
                "strategy": "preferred_domains"
                if request.get("restrict_domains") and task.preferred_domains
                else "open_search",
                "source_ids": sorted(sources)[:12],
                "source_count": len(sources),
                "paper_chunks": len(chunks),
                "web_results": len(web),
                "card_count": len(cards),
            }
        ],
    )


def research_worker(request, researcher=None):
    """Bounded same-task retry; parallel tasks only write payload reducer fields."""
    task = SubTask.model_validate(request["task"])
    max_retries = request["limits"]["max_worker_retries"]
    service = researcher or _research_once
    errors = []
    history = []
    queries = task.search_queries or [task.query[:180]]
    if len(queries) < 3:
        technologies = (
            task.technology
            if isinstance(task.technology, str)
            else " ".join(task.technology)
        )
        queries = list(
            dict.fromkeys(
                [
                    *queries,
                    f"{technologies} {task.perspective} official documentation support limitations"[
                        :180
                    ],
                    f"{technologies} {task.objective} independent evidence"[:180],
                ]
            )
        )[:3]
    query_index = 0
    known = set(request.get("known_evidence_keys", []))
    known_sources = set(request.get("known_source_ids", []))
    result = None
    for attempt in range(task.retry_count, max_retries + 1):
        active = task.model_copy(
            update={"retry_count": attempt, "query": queries[query_index]}
        )
        with node_span(
            request["trace_id"],
            "worker_attempt",
            task_id=task.task_id,
            perspective=task.perspective,
            retry_count=attempt,
        ):
            try:
                result = WorkerResult.model_validate(
                    service(active, {**request, "restrict_domains": query_index == 0})
                )
                if result.task_id != task.task_id:
                    raise ValueError("worker returned a different task ID")
                if result.status != "failed" and not result.evidence_cards:
                    result = result.model_copy(
                        update={
                            "status": "failed",
                            "failure_kind": "insufficient_evidence",
                        }
                    )
                cards = []
                seen = set(known)
                for original in result.evidence_cards:
                    card = {
                        **original,
                        "perspective": LEGACY_PERSPECTIVES[task.perspective],
                        "source_id": source_identity(original),
                    }
                    fingerprint = claim_fingerprint(card)
                    if fingerprint not in seen:
                        seen.add(fingerprint)
                        cards.append(card)
                duplicate_count = len(result.evidence_cards) - len(cards)
                if result.status != "failed" and not cards:
                    result = result.model_copy(
                        update={
                            "status": "failed",
                            "failure_kind": "insufficient_evidence",
                            "limitations": [
                                *result.limitations,
                                "검색 결과가 기존 근거와 중복됨",
                            ],
                        }
                    )
                result = result.model_copy(update={"evidence_cards": cards})
            except Exception as error:  # noqa: BLE001 - worker failure isolation
                result = WorkerResult(
                    task_id=task.task_id,
                    perspective=task.perspective,
                    technology=task.technology,
                    status="failed",
                    errors=[type(error).__name__],
                    failure_kind="transport",
                )
                duplicate_count = 0
            errors.extend(result.errors)
            rows = result.search_attempts or [{"query": active.query, "source_ids": []}]
            for row in rows:
                row = {
                    **row,
                    "status": result.status,
                    "failure_kind": result.failure_kind,
                    "retry_count": attempt,
                    "duplicate_cards": duplicate_count,
                    "new_source_count": len(
                        set(row.get("source_ids", [])) - known_sources
                    ),
                }
                history.append(row)
                record_event(
                    request["trace_id"],
                    "worker_search",
                    result.status,
                    result.failure_kind,
                    task_id=task.task_id,
                    perspective=task.perspective,
                    **row,
                )
            record_event(
                request["trace_id"],
                "research_worker",
                "retry"
                if result.status == "failed" and attempt < max_retries
                else result.status,
                result.failure_kind if result.status == "failed" else "task complete",
                task_id=task.task_id,
                perspective=task.perspective,
                retry_count=attempt,
            )
        if result.status != "failed":
            break
        if result.failure_kind == "insufficient_evidence":
            query_index = min(query_index + 1, len(queries) - 1)
    if result is None:
        result = WorkerResult(
            task_id=task.task_id,
            perspective=task.perspective,
            technology=task.technology,
            status="failed",
        )
        attempt = max_retries

    cards = []
    if result.status != "failed":
        for index, original in enumerate(
            result.evidence_cards[: request["limits"]["max_cards_per_task"]],
            1,
        ):
            cards.append(
                {
                    **original,
                    "task_id": task.task_id,
                    # IDs are unique across split tasks and supplemental rounds.
                    "evidence_id": f"{task.task_id}:e{index}",
                    "perspective": LEGACY_PERSPECTIVES[task.perspective],
                    "evidence_text": str(original.get("evidence_text", ""))[:2000],
                    "claim": str(original.get("claim", ""))[:1000],
                    "caveat": str(original.get("caveat", ""))[:1000],
                }
            )
    limitations = list(result.limitations)
    if result.status == "failed":
        limitations.append(f"{task.task_id}: 최대 재시도 초과로 조사 결과 제외")
    result = result.model_copy(
        update={
            "perspective": task.perspective,
            "technology": task.technology,
            "evidence_cards": cards,
            "findings": [c["claim"] for c in cards],
            "retry_count": attempt,
            "search_attempts": history[: max_retries + 1],
            "limitations": [s[:1000] for s in limitations[:12]],
            "errors": list(dict.fromkeys(errors))[: max_retries + 1],
        }
    )
    return {
        "payload": {
            "worker_results": [result.model_dump()],
            "evidence_cards": cards,
        }
    }
