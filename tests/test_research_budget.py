from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier, Lock

import pytest

from src.agents import technical
from src.agents.base import ResearchOutput, repair_source_links, sanitize_research, source_excerpts
from src.common.runtime import BudgetExceeded, CallBudget
from src.schemas import AgentContext, ExecutionLimits
from tests.test_agents import FakeLLM, Reader, Search, card, request, research_output, source


def test_concurrent_research_reads_share_result_and_verification_reopens():
    class CountingReader:
        calls = 0
        lock = Lock()

        def read(self, document):
            with self.lock:
                self.calls += 1
            return document

    reader = CountingReader()
    context = AgentContext(source_reader=reader, budget=CallBudget(20, 60))
    barrier = Barrier(3)

    def run(agent):
        barrier.wait()
        return replace(context, agent=agent).read(source())

    with ThreadPoolExecutor(3) as pool:
        results = list(pool.map(run, ["technical", "market", "domain"]))
    assert reader.calls == 1
    results[0].content = "mutated"
    assert results[1].content == source().content
    replace(context, agent="verification").read(source())
    assert reader.calls == 2
    assert context.budget.counts["source_read"] == 2


def test_failed_source_not_repeated_during_research_but_rechecked_in_verification():
    reader = Reader({"s1": RuntimeError("unavailable")})
    context = AgentContext(source_reader=reader, agent="market")
    for agent in ["market", "domain", "verification"]:
        with pytest.raises(RuntimeError):
            replace(context, agent=agent).read(source())
    assert reader.calls == ["s1", "s1"]


def test_reserved_calls_remain_available_to_downstream():
    budget = CallBudget(8, 60, reserved_calls=2)
    for _ in range(6):
        budget.consume("search", research=True)
    with pytest.raises(BudgetExceeded, match="예산 보존"):
        budget.consume("llm", research=True)
    assert budget.research_exhausted and not budget.exhausted
    budget.consume("source_read")
    budget.consume("llm")
    assert budget.exhausted


def test_known_excerpt_repairs_wrong_source_and_url_without_inventing_evidence():
    original = source()
    excerpt = source_excerpts(original)[0]
    output = ResearchOutput.model_validate(
        research_output(
            cards=[
                card(
                    source_id="wrong",
                    source_url="https://wrong.example/",
                    source_excerpt_id=excerpt["excerpt_id"],
                )
            ]
        )
    )
    repaired, changes = repair_source_links(output, {"s1": original})
    cards, findings, gaps = sanitize_research(repaired, {"s1": original}, "technical", set())
    assert changes and len(cards) == len(findings) == 1 and not gaps
    assert cards[0].source_id == "s1" and cards[0].source_url == original.url
    output.evidence_cards[0].source_excerpt_id = "invented"
    repaired, changes = repair_source_links(output, {"s1": original})
    assert not changes
    assert not sanitize_research(repaired, {"s1": original}, "technical", set())[0]


def test_no_new_sources_keeps_valid_analysis_without_repeated_llm_analysis():
    llm = FakeLLM(
        SearchQueryPlan=[{"queries": ["q1"]}, {"queries": ["q2"]}],
        ResearchOutput=[research_output(missing=["Official deployment not confirmed"])],
    )
    reader = Reader()
    result = technical.run(
        request().model_copy(update={"limits": ExecutionLimits()}),
        AgentContext(llm=llm, retriever=Search([source()]), source_reader=reader),
    )
    assert result.status == "partial" and result.evidence_cards
    assert len(llm.calls) == 3 and reader.calls == ["s1"]
    assert "새 원문 없음" in result.data["search_stop_reason"]
    assert not result.follow_up_requests


def test_research_caps_queries_and_successful_original_reads():
    docs = [
        source().model_copy(update={"source_id": f"s{i}", "url": f"https://example.org/{i}"})
        for i in range(1, 6)
    ]
    llm = FakeLLM(SearchQueryPlan=[{"queries": ["q1", "q2", "q3", "q4"]}], ResearchOutput=[research_output()])
    search, reader = Search(docs), Reader()
    limits = ExecutionLimits(max_search_retries=0, research_source_limit=3)
    technical.run(
        request().model_copy(update={"limits": limits}),
        AgentContext(llm=llm, retriever=search, source_reader=reader),
    )
    assert search.calls == ["q1"]
    assert len(reader.calls) == 3
