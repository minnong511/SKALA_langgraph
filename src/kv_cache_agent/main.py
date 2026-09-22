from kv_cache_agent.graph.workflow import build_workflow


def main() -> None:
    workflow = build_workflow()
    print("Workflow initialized:", workflow)


if __name__ == "__main__":
    main()
