"""Run every production agent through LangGraph using deterministic local tools."""

from __future__ import annotations

import json
import socket
from collections import Counter
from pathlib import Path
from threading import Lock

from pypdf import PdfReader

from main import run_workflow
from src.config import Settings
from src.schemas import REPORT_HEADINGS, AgentContext, AgentRequest, SourceDocument

CRITERIA = {"latency": "TTFT and TPOT under stated conditions", "maturity": "public prototype evidence"}
PERSPECTIVES = {"technical": "trl", "market": "market", "stakeholder": "stakeholder", "domain": "domain"}
TECHNOLOGIES = ("TurboQuant", "ITME")


def evidence_id(agent, technology):
    return f"{agent}-{technology.lower()}"


def fixture_source():
    return SourceDocument(
        source_id="fixture-primary-paper",
        title="Local integration fixture: primary source",
        author="Fixture Author",
        url="https://example.org/integration-primary-source",
        source_type="paper",
        published_at="2025-01-01",
        accessed_at="2026-09-22",
        page_or_section="page 1",
        content="\n".join(
            f"{technology} has a research prototype under the fixture workload."
            for technology in TECHNOLOGIES
        ),
    )


class MemoryTools:
    def __init__(self):
        self._lock = Lock()
        self.search_calls = []
        self.read_calls = []

    def search(self, query, limit=5):
        with self._lock:
            self.search_calls.append((query, limit))
        return [fixture_source()]

    def read(self, source):
        with self._lock:
            self.read_calls.append(source.source_id)
        assert source.source_id == fixture_source().source_id
        return fixture_source().model_copy(update={"metadata": {"original_read": True}})


class PipelineLLM:
    """Structured outputs are keyed by their production schemas, never call order."""

    def __init__(self):
        self._lock = Lock()
        self.calls = []

    def with_structured_output(self, schema):
        owner = self

        class Bound:
            def invoke(self, messages):
                payload = json.loads(messages[-1][1])
                with owner._lock:
                    owner.calls.append((schema.__name__, payload))
                return owner.respond(schema.__name__, payload)

        return Bound()

    def respond(self, schema, payload):
        if schema == "PlanOutput":
            assert payload["context"]["evaluation_criteria"] == CRITERIA
            return {
                "objective": "Compare documented applicability conditions",
                **{f"{agent}_questions": [f"Check {agent} evidence"] for agent in PERSPECTIVES},
            }
        if schema in {
            "SearchQueryPlan",
            "ResearchOutput",
            "VerificationOutput",
            "SynthesisOutput",
            "ReportOutput",
        }:
            assert payload["evaluation_criteria"] == CRITERIA
        if schema == "SearchQueryPlan":
            return {"queries": ["Read local primary source"], "rationale": "Use inspectable evidence"}
        if schema == "ResearchOutput":
            agent = payload["evidence_id_prefix"].rstrip("-")
            if agent != "technical":
                assert len(payload["technical_result"]["evidence_cards"]) == 2
            return {
                "summary": f"{agent} evidence extracted",
                "evidence_cards": [
                    dict(
                        evidence_id=evidence_id(agent, technology),
                        technology=technology,
                        perspective=PERSPECTIVES[agent],
                        claim=f"{technology} prototype is documented.",
                        source_id="fixture-primary-paper",
                        source_title="Model supplied title must be overwritten",
                        source_url=fixture_source().url,
                        evidence_text=f"{technology} has a research prototype under the fixture workload.",
                        statement_type="author_claim",
                        conditions="fixture workload only",
                    )
                    for technology in TECHNOLOGIES
                ],
                "findings": [
                    dict(
                        finding_id=f"{agent}-finding-{technology.lower()}",
                        technology=technology,
                        perspective=PERSPECTIVES[agent],
                        statement=f"{technology} remains conditional on the evaluated workload.",
                        evidence_ids=[evidence_id(agent, technology)],
                        confidence="medium",
                    )
                    for technology in TECHNOLOGIES
                ],
                "limitations": ["Synthetic fixture used only for offline implementation tests"],
            }
        if schema == "VerificationOutput":
            assert len(payload["reopened_sources"]) == 1
            assert payload["reopened_sources"][0]["metadata"]["original_read"]
            assert not any(payload["deterministic_prechecks"].values())
            return {
                "summary": "Every fixture claim matches its reopened original",
                "verification": [
                    {
                        "evidence_id": item["evidence_id"],
                        "status": "verified",
                        "reason": "Quote and conditions agree with the reopened fixture source",
                    }
                    for item in payload["evidence_cards"]
                ],
            }
        if schema == "SynthesisOutput":
            assert len(payload["usable_evidence"]) == 8
            return {
                "comparison_matrix": [
                    dict(
                        perspective=perspective,
                        turboquant="Documented prototype, conditional applicability",
                        itme="Documented prototype, conditional applicability",
                        conflict_or_condition="The fixture workload defines the scope",
                        evidence_ids=[evidence_id(agent, technology) for technology in TECHNOLOGIES],
                        confidence="medium",
                    )
                    for agent, perspective in PERSPECTIVES.items()
                ],
                "neutral_conclusion": "Interpret the two documented prototypes within their workload conditions.",
                "conclusion_evidence_ids": [
                    evidence_id("technical", technology) for technology in TECHNOLOGIES
                ],
                "neutrality_check": "No winner or aggregate ranking",
                "evidence_limitations": ["Fixture workload only"],
            }
        if schema == "ReportOutput":
            assert len(payload["evidence_cards"]) == 8
            pairs = {
                agent: " ".join(f"[{evidence_id(agent, technology)}]" for technology in TECHNOLOGIES)
                for agent in PERSPECTIVES
            }
            contents = [
                f"각 기술의 적용 조건을 근거에 따라 검토했습니다. {pairs['technical']}",
                f"클라우드 서빙의 적용 조건을 검토합니다. {pairs['domain']}",
                f"두 기술의 공개된 근거를 같은 기준으로 검토했습니다. {pairs['technical']}",
                f"원문에 기술된 프로토타입의 범위를 확인했습니다. {pairs['technical']}",
                f"시장, 이해관계자, 도메인 관점을 함께 검토했습니다. {pairs['market']} {pairs['stakeholder']} {pairs['domain']}",
                f"조건에 따라 평가가 달라질 수 있습니다. {pairs['domain']}",
                f"기록된 실험 조건 범위에서만 해석합니다. {pairs['technical']}",
                "Model-generated bibliography must be replaced by canonical cited sources.",
            ]
            return {
                "markdown": "\n\n".join(
                    f"## {heading}\n\n{content}" for heading, content in zip(REPORT_HEADINGS, contents)
                )
            }
        if schema == "QualityOutput":
            assert len(payload["evidence"]) == 8
            assert "Model-generated bibliography" not in payload["report"]
            return {"passed": True, "issues": []}
        raise AssertionError(f"Unexpected production output schema: {schema}")


def test_all_actual_agents_complete_live_pipeline_and_publish_real_files(tmp_path, monkeypatch):
    def block_network(*args, **kwargs):
        raise AssertionError("This integration test must not access the network")

    monkeypatch.setattr(socket.socket, "connect", block_network)
    memory = MemoryTools()
    llm = PipelineLLM()
    context = AgentContext(llm=llm, retriever=memory, web_search=memory, source_reader=memory)
    settings = Settings(
        output_dir=tmp_path,
        quiet=True,
        max_search_retries=0,
        max_research_retries=0,
        max_synthesis_retries=0,
        max_report_revisions=0,
    )
    request = AgentRequest(
        run_id="full-production-pipeline",
        task_id="workflow",
        as_of_date="2026-09-22",
        limits=settings.limits,
        context={"evaluation_criteria": CRITERIA},
    )

    # Omitting agents invokes all eight production modules, including supervisor.
    state = run_workflow(settings, request, demo=False, context=context)

    assert state["status"] == "completed", state.get("review", state.get("run_error"))
    assert not context.demo
    assert state["unresolved"] == []
    assert state["review"]["passed"] is True
    assert set(state["attempts"]) == {"supervisor", *PERSPECTIVES, "verification", "synthesis", "report"}
    assert all(state["attempts"][name] == 1 for name in state["attempts"] if name != "supervisor")
    for name in (*PERSPECTIVES, "verification", "synthesis", "report"):
        assert state[f"{name}_result"].status == "completed"
    assert Counter(name for name, _ in llm.calls) == {
        "PlanOutput": 1,
        "SearchQueryPlan": 4,
        "ResearchOutput": 4,
        "VerificationOutput": 1,
        "SynthesisOutput": 1,
        "ReportOutput": 1,
        "QualityOutput": 1,
    }
    assert len(memory.read_calls) == 5  # Four research assignments and independent verification.
    assert len(memory.search_calls) == 5  # Domain uses both retrieval and web search.
    expected_ids = {evidence_id(agent, technology) for agent in PERSPECTIVES for technology in TECHNOLOGIES}
    assert set(state["report_result"].used_evidence_ids) == expected_ids
    for name in PERSPECTIVES:
        assert all(
            card.source_title == fixture_source().title for card in state[f"{name}_result"].evidence_cards
        )
    directory = tmp_path / request.run_id
    assert Path(state["final_artifacts"]["markdown"]) == directory / "report.md"
    assert Path(state["final_artifacts"]["pdf"]) == directory / "report.pdf"
    markdown = (directory / "report.md").read_text()
    references = markdown.split("## REFERENCE", 1)[1]
    assert all(f"[{item}]" in references for item in expected_ids)
    assert references.count(fixture_source().url) == 8
    assert "Model-generated bibliography" not in markdown
    pdf_text = "\n".join(page.extract_text() for page in PdfReader(directory / "report.pdf").pages)
    assert "관점별 평가" in pdf_text
    assert "REFERENCE" in pdf_text
    assert state["pdf_layout"]["summary_within_half_page"] is True
    persisted = json.loads((directory / "result.json").read_text())
    assert persisted["status"] == "completed"
    assert not (directory / "draft-report.md").exists()
    assert not (directory / "draft-report.pdf").exists()
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    progress = [row["details"] for row in events if row["event"] == "verification_progress"]
    assert [row["current"] for row in progress] == list(range(1, 9))
    assert all(row["total"] == 8 and row["verdict"] == "verified" for row in progress)
    completed = [row for row in events if row["event"] == "verification_completed"]
    assert len(completed) == 1
    assert completed[0]["details"] == {"total": 8, "verified": 8, "uncertain": 0, "rejected": 0}
