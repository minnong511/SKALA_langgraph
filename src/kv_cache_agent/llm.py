"""Shared OpenAI loader for planning, research, writing, and evaluation."""

from langchain_openai import ChatOpenAI

from kv_cache_agent.config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_REQUEST_TIMEOUT


def get_llm() -> ChatOpenAI:
    """Return a deterministic chat model for structured research outputs."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set")
    if OPENAI_REQUEST_TIMEOUT <= 0:
        raise ValueError("OPENAI_REQUEST_TIMEOUT must be positive")

    # Luna supports reasoning with structured/tool outputs through Responses.
    # Preserve the previous non-reasoning workload and temperature setting.
    luna_options = (
        {"use_responses_api": True, "reasoning": {"effort": "none"}}
        if OPENAI_MODEL == "gpt-6-luna"
        else {}
    )
    return ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=0,
        timeout=OPENAI_REQUEST_TIMEOUT,
        **luna_options,
    )
