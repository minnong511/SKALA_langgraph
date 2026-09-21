"""Exercise the actual OpenAI/LangChain adapter with an in-process HTTP transport."""

from __future__ import annotations

import json

import httpx
import pytest
from langchain_openai import ChatOpenAI
from openai import RateLimitError
from pydantic import ValidationError

from src.agents.base import ResearchOutput
from src.config import Settings, create_llm
from src.schemas import AgentContext, EvidenceCard

MODEL = "user-selected-model-exact-id"
SYNTHETIC_KEY = "sk-test-only-adapter-fixture-not-a-real-secret"


def _completion(arguments: dict) -> dict:
    return {
        "id": "chatcmpl-local-fixture",
        "object": "chat.completion",
        "created": 1,
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_local_fixture",
                            "type": "function",
                            "function": {"name": "ResearchOutput", "arguments": json.dumps(arguments)},
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _settings() -> Settings:
    return Settings(
        llm_model=MODEL,
        llm_api_key=SYNTHETIC_KEY,
        tavily_api_key="tvly-synthetic-fixture",
        llm_timeout_seconds=7,
    )


def test_luna_routes_function_tools_through_responses_api():
    settings = _settings().model_copy(update={"llm_model": "gpt-5.6-luna"})
    llm = create_llm(settings)
    runnable = llm.with_structured_output(ResearchOutput, method="function_calling", strict=False)
    assert llm.use_responses_api is True
    assert runnable is not None


def _configured_llm(monkeypatch, client: httpx.Client) -> ChatOpenAI:
    """Keep production construction options, replacing only the HTTP transport."""

    def injected_transport(**kwargs):
        return ChatOpenAI(**kwargs, http_client=client, base_url="https://openai.test/v1")

    monkeypatch.setattr("langchain_openai.ChatOpenAI", injected_transport)
    return create_llm(_settings())


def test_real_adapter_parses_research_tool_output_and_preserves_open_dicts(monkeypatch):
    requests: list[httpx.Request] = []
    expected = {
        "summary": "검증 가능한 근거입니다.",
        "evidence_cards": [
            {
                "evidence_id": "technical-1",
                "technology": "TurboQuant",
                "perspective": "technical",
                "claim": "Memory use falls under stated conditions.",
                "source_id": "PDF-fixture",
                "evidence_text": "Memory use falls under stated conditions.",
                "missing_metadata": {"published_at": "원문 발행일 없음", "author": "원문 저자 없음"},
            }
        ],
        "structured_output": [{"key": "trl", "value": "추가 검토 필요"}],
    }

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json=_completion(expected))

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        llm = _configured_llm(monkeypatch, client)
        context = AgentContext(llm=llm, structured_output_method="function_calling")
        result = context.ask(
            ResearchOutput,
            "원문에서 근거를 추출한다.",
            {"query": "TurboQuant", "config": _settings().public_config()},
        )
        assert llm.max_retries == llm.root_client.max_retries == 0
    assert isinstance(result, ResearchOutput)
    assert isinstance(result.evidence_cards[0], EvidenceCard)
    assert result.evidence_cards[0].missing_metadata == expected["evidence_cards"][0]["missing_metadata"]
    assert result.structured_output[0].key == "trl"
    assert len(requests) == 1
    sent = requests[0]
    body = json.loads(sent.content)
    assert sent.url.path == "/v1/chat/completions"
    assert body["model"] == MODEL
    assert body["tool_choice"] == {"type": "function", "function": {"name": "ResearchOutput"}}
    function = body["tools"][0]["function"]
    assert function["name"] == "ResearchOutput"
    assert function["strict"] is False
    metadata_schema = function["parameters"]["properties"]["evidence_cards"]["items"]["properties"][
        "missing_metadata"
    ]
    assert metadata_schema["additionalProperties"] == {"type": "string"}
    assert sent.extensions["timeout"]["read"] == 7
    # Authentication belongs in headers, never in model prompts or serialized config.
    assert sent.headers["authorization"] == f"Bearer {SYNTHETIC_KEY}"
    assert SYNTHETIC_KEY not in sent.content.decode()
    assert "tvly-synthetic-fixture" not in sent.content.decode()


def test_real_adapter_still_validates_malformed_non_strict_model_output(monkeypatch):
    malformed = {
        "summary": "invalid output",
        "evidence_cards": [
            {
                "evidence_id": "bad",
                "technology": "TurboQuant",
                "perspective": "invalid-perspective",
                "source_id": "PDF-fixture",
                "claim": "claim",
                "evidence_text": "quote",
            }
        ],
    }
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=_completion(malformed)))
    ) as client:
        llm = _configured_llm(monkeypatch, client)
        context = AgentContext(llm=llm, structured_output_method="function_calling")
        with pytest.raises(ValidationError, match="perspective"):
            context.ask(ResearchOutput, "extract", {"query": "fixture"})


def test_real_sdk_rate_limit_is_not_retried_outside_application_budget(monkeypatch):
    requests: list[httpx.Request] = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            429,
            json={
                "error": {
                    "message": "Synthetic rate limit",
                    "type": "rate_limit_error",
                    "code": "rate_limit_exceeded",
                }
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        llm = _configured_llm(monkeypatch, client)
        context = AgentContext(llm=llm, structured_output_method="function_calling")
        with pytest.raises(RateLimitError):
            context.ask(ResearchOutput, "extract", {"query": "fixture"})
    assert len(requests) == 1
