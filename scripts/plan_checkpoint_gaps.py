"""Plan a focused follow-up from saved quality feedback, without running Workers."""

import argparse
import json
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from langsmith import Client, trace

from kv_cache_agent.agents.orchestrator import orchestrator_agent
from kv_cache_agent.graph.workflow import build_workflow
from kv_cache_agent.observability import node_span


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-db", type=Path, required=True)
    parser.add_argument("--thread-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-tasks", type=int, default=6)
    args = parser.parse_args()
    with SqliteSaver.from_conn_string(str(args.checkpoint_db)) as cp:
        state = deepcopy(
            build_workflow(checkpointer=cp)
            .get_state({"configurable": {"thread_id": args.thread_id}})
            .values
        )
    source_trace_id = state["control"]["trace_id"]
    trace_id = str(uuid4())
    state["control"]["trace_id"] = trace_id
    state["control"]["limits"]["max_tasks_per_plan"] = args.max_tasks
    client = Client(timeout_ms=10_000)
    with trace(
        "quality-gap-follow-up-planning",
        client=client,
        inputs={},
        metadata={
            "trace_id": trace_id,
            "source_trace_id": source_trace_id,
            "purpose": "planning only, no Workers or report generation",
        },
    ) as root:
        with node_span(trace_id, "orchestrator"):
            update = orchestrator_agent(state)
        plan = update["payload"]["research_plan"]
        root.end(outputs={"task_count": len(plan["tasks"])})
    client.flush(timeout=10)
    result = {
        "trace_id": trace_id,
        "source_thread": args.thread_id,
        "workers_executed": False,
        "report_reevaluated": False,
        "run_url": client.get_run_url(run=root),
        "research_plan": plan,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "follow-up-plan.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
