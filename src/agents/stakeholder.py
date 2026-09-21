"""Stakeholder agent, preserving the original project research scope."""

from ..schemas import AgentContext, AgentRequest, AgentResult
from .base import run_research

DEFAULT_QUERIES = [
    "cloud provider developer views KV cache quantization accuracy deployment barriers",
    "LLM serving developer KV cache compression integration maintenance concerns",
    "cloud operator CXL memory expansion cost latency operational complexity",
    "GPU CPU memory vendors CXL strategy AI inference memory",
    "investment analyst view inference optimization memory expansion AI infrastructure",
]


def run(request: AgentRequest, context: AgentContext) -> AgentResult:
    return run_research(
        request,
        context,
        agent="stakeholder",
        default_queries=DEFAULT_QUERIES,
        use_retriever=context.stakeholder_rag,
        use_web=True,
    )
