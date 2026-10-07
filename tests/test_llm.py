"""Exercise Luna's Responses transport and structured result parsing offline."""

import json

import httpx
import pytest
from langchain_openai import ChatOpenAI

from kv_cache_agent import llm
from kv_cache_agent.schemas.evaluation import QualityEvaluation
from kv_cache_agent.schemas.tasks import ResearchPlan


@pytest.mark.parametrize(
    ("schema", "output"),
    [
        (
            ResearchPlan,
            {
                "tasks": [
                    {
                        "task_id": "technical_sw",
                        "perspective": "technical_maturity",
                        "technology": "TurboQuant",
                        "objective": "Check software maturity",
                        "query": "TurboQuant experimental conditions",
                        "preferred_source": "paper",
                        "priority": 1,
                        "retry_count": 0,
                    }
                ],
                "planning_reason": "Focused structured output fixture",
            },
        ),
        (
            QualityEvaluation,
            {
                "groundedness": False,
                "neutrality": True,
                "bias_control": False,
                "perspective_coverage": False,
                "groundedness_reason": "Evidence absent",
                "neutrality_reason": "No recommendation",
                "bias_reason": "Sources absent",
                "coverage_reason": "Missing market",
                "missing_perspectives": ["market"],
                "missing_evidence_topics": ["commercial adoption"],
                "overall_pass": False,
                "recommended_action": "additional_research",
                "rule_failures": [],
            },
        ),
    ],
)
def test_luna_structured_output_uses_responses(monkeypatch, schema, output):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "resp_fixture",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": "gpt-6-luna",
                "output": [
                    {
                        "id": "msg_fixture",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps(output),
                                "annotations": [],
                            }
                        ],
                    }
                ],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(llm, "OPENAI_MODEL", "gpt-6-luna")
        monkeypatch.setattr(llm, "OPENAI_API_KEY", "test-key")
        monkeypatch.setattr(
            llm,
            "ChatOpenAI",
            lambda **kwargs: ChatOpenAI(**kwargs, http_client=client, max_retries=0),
        )
        result = llm.get_llm().with_structured_output(schema).invoke("Fixture request")

    assert result == schema.model_validate(output)
    assert len(requests) == 1
    request = requests[0]
    payload = json.loads(request.content)
    assert request.url.path == "/v1/responses"
    assert payload["model"] == "gpt-6-luna"
    assert payload["reasoning"]["effort"] == "none"
    assert payload["text"]["format"]["type"] == "json_schema"
    assert request.extensions["timeout"]["read"] == llm.OPENAI_REQUEST_TIMEOUT


def test_timeout_is_bounded_and_configurable(monkeypatch):
    monkeypatch.setattr(llm, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(llm, "OPENAI_REQUEST_TIMEOUT", 45)
    assert llm.get_llm().request_timeout == 45
    monkeypatch.setattr(llm, "OPENAI_REQUEST_TIMEOUT", 0)
    with pytest.raises(ValueError, match="must be positive"):
        llm.get_llm()
