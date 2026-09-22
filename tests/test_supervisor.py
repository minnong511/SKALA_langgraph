from kv_cache_agent.agents.supervisor import supervisor_agent


def test_supervisor_loads_yaml_plan_and_starts_with_technical_research():
    result = supervisor_agent({"user_query": "KV Cache 기술 비교"})

    plan = result["research_plan"]
    control = result["control"]

    assert plan["technologies"] == ["TurboQuant", "CXL-based"]
    assert set(plan["perspectives"]) == {
        "technical",
        "market",
        "stakeholder",
        "cloud_domain",
    }
    assert all(
        plan["search_questions"][perspective]
        for perspective in plan["perspectives"]
    )
    assert control["status"] == "researching"
    assert control["next_agent"] == "technical"
    assert control["retry_count"] == {}
