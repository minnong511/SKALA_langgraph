"""Technical agent, preserving the original project research scope."""

from ..schemas import AgentContext, AgentRequest, AgentResult
from .base import run_research

DEFAULT_QUERIES = [
    "TurboQuant KV cache quantization mechanism distortion guarantee experiments limitations",
    "TurboQuant LongBench needle in a haystack bit width baseline experimental conditions",
    "ITME CXL-Hybrid memory architecture KV cache prefetch prototype limitations",
    "ITME throughput baseline ShareGPT FPGA prototype evaluation conditions",
    "TurboQuant ITME TRL evidence code prototype production deployment",
]


def run(request: AgentRequest, context: AgentContext) -> AgentResult:
    return run_research(
        request,
        context,
        agent="technical",
        default_queries=DEFAULT_QUERIES,
        use_retriever=True,
        use_web=False,
    )
