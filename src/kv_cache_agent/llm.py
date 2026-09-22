"""Shared GPT-4o-mini loader for Supervisor and worker agents."""

from langchain_openai import ChatOpenAI

from kv_cache_agent.config import OPENAI_API_KEY, OPENAI_MODEL


def get_llm() -> ChatOpenAI:
    """Return a deterministic chat model for structured research outputs."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set")

    return ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=0,
    )
