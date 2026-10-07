"""Upload a small diagnostic trace and read it back, without exposing secrets."""

import argparse
import json
import logging
import time
from pathlib import Path
from uuid import uuid4

from langsmith import Client, trace
from langsmith.utils import tracing_is_enabled

from kv_cache_agent.config import OUTPUTS_DIR
from kv_cache_agent.observability import record_event


def check_langsmith():
    import os

    trace_id = str(uuid4())
    project = os.getenv("LANGSMITH_PROJECT", "default")
    result = {
        "trace_id": trace_id,
        "project": project,
        "status": "failed",
        "tracing_enabled": tracing_is_enabled(),
        "key_configured": bool(os.getenv("LANGSMITH_API_KEY")),
        "purpose": "connection_check, not a research workflow",
    }
    if not result["tracing_enabled"] or not result["key_configured"]:
        result["error_type"] = "TracingConfigurationMissing"
        return result
    logging.getLogger("langsmith").setLevel(logging.ERROR)
    try:
        client = Client(timeout_ms=10_000)
        with trace(
            "kv-cache-langsmith-connection-check",
            client=client,
            project_name=project,
            inputs={},
            metadata={"trace_id": trace_id, "purpose": "connection_check"},
        ) as root:
            with trace(
                "event-upload-check",
                client=client,
                inputs={},
                metadata={"trace_id": trace_id},
            ) as child:
                for task_id in ("diagnostic-1", "diagnostic-2"):
                    record_event(
                        trace_id, "tracing_check", "connection_probe", task_id=task_id
                    )
            root.end(outputs={"status": "uploaded"})
        client.flush(timeout=10)
        result["run_id"] = str(root.id)
        saved = None
        for attempt in range(3):
            try:
                saved = client.read_run(root.id, load_child_runs=True)
                if saved.end_time is not None and saved.child_runs:
                    break
            except Exception:
                if attempt == 2:
                    raise
            if attempt < 2:
                time.sleep(2)
        saved_child = client.read_run(child.id)
        events = saved_child.events or []
        result.update(
            status="ok" if saved and saved.end_time and len(events) >= 2 else "failed",
            run_url=client.get_run_url(run=saved, project_name=project),
            child_names=[c.name for c in saved.child_runs or []],
            event_task_ids=[
                e.get("kwargs", {}).get("task_id")
                for e in events
                if e.get("name") == "connection_probe"
            ],
            completed=bool(saved.end_time),
        )
    except Exception as error:  # noqa: BLE001 - provider bodies can include sensitive material
        result["error_type"] = type(error).__name__
    return result


def main():
    parser = argparse.ArgumentParser(description="LangSmith 업로드/조회 연결 확인")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR / "validation" / "langsmith-check",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = check_langsmith()
    path = args.output_dir / "result.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"연결 확인 결과: {path}")
    if result["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
