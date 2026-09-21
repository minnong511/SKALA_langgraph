"""Market agent, preserving the original project research scope."""

from ..schemas import AgentContext, AgentRequest, AgentResult
from .base import run_research

DEFAULT_QUERIES = [
    "TurboQuant KV cache commercial adoption serving framework support",
    "KV cache quantization cloud inference cost adoption ecosystem",
    "ITME CXL-Hybrid memory product adoption cloud LLM inference",
    "CXL memory server ecosystem cloud AI infrastructure adoption",
    "long context LLM inference memory demand cloud market",
]


def run(request: AgentRequest, context: AgentContext) -> AgentResult:
    return run_research(
        request,
        context,
        agent="market",
        default_queries=DEFAULT_QUERIES,
        use_retriever=context.market_rag,
        use_web=True,
    )
