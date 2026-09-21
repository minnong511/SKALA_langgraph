"""Domain agent, preserving the original project research scope."""

from ..schemas import AgentContext, AgentRequest, AgentResult
from .base import run_research

DEFAULT_QUERIES = [
    "TurboQuant cloud LLM serving memory throughput latency quality conditions",
    "ITME cloud LLM serving TTFT throughput memory capacity latency conditions",
    "KV cache quantization multi tenant long context serving operational cost",
    "CXL tiered memory multi tenant long context LLM serving operational complexity",
    "TurboQuant ITME deployment requirements compatibility serving engine",
]


def run(request: AgentRequest, context: AgentContext) -> AgentResult:
    return run_research(
        request, context, agent="domain", default_queries=DEFAULT_QUERIES, use_retriever=True, use_web=True
    )
