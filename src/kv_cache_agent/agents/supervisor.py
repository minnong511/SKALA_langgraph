"""Supervisor 에이전트의 연구 계획과 실행 시작점을 관리하는 모듈."""

from pathlib import Path
from typing import Any, TypedDict

import yaml

from kv_cache_agent.graph.state import GlobalState
from kv_cache_agent.schemas.outputs import ResearchPlan

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "supervisor.yaml"

EXPECTED_TECHNOLOGIES = {"TurboQuant", "CXL-based"}
EXPECTED_PERSPECTIVES = {
    "technical",
    "market",
    "stakeholder",
    "cloud_domain",
}


class SupervisorConfig(TypedDict, total=False):
    """Supervisor YAML에서 사용하는 설정 형식."""

    agent_name: str
    version: str
    mission: str
    technologies: list[str]
    perspectives: list[str]
    execution_order: dict[str, list[str]]
    search_questions: dict[str, list[str]]
    policies: dict[str, bool]


def _load_supervisor_config() -> SupervisorConfig:
    """YAML 설정을 읽고 Supervisor 실행에 필요한 항목을 검증한다."""
    with PROMPT_PATH.open(encoding="utf-8") as prompt_file:
        config = yaml.safe_load(prompt_file) or {}

    if not isinstance(config, dict):
        raise TypeError("supervisor.yaml의 최상위 값은 mapping이어야 합니다.")

    technologies = config.get("technologies")
    perspectives = config.get("perspectives")
    search_questions = config.get("search_questions")

    if not isinstance(technologies, list) or not all(
        isinstance(item, str) and item.strip() for item in technologies
    ):
        raise ValueError(
            "supervisor.yaml의 technologies는 비어 있지 않은 문자열 목록이어야 합니다."
        )

    if set(technologies) != EXPECTED_TECHNOLOGIES:
        raise ValueError(
            "technologies에는 TurboQuant와 CXL-based만 정의해야 합니다."
        )

    if not isinstance(perspectives, list) or not all(
        isinstance(item, str) and item.strip() for item in perspectives
    ):
        raise ValueError("supervisor.yaml의 perspectives는 문자열 목록이어야 합니다.")

    if set(perspectives) != EXPECTED_PERSPECTIVES:
        raise ValueError(
            "perspectives에는 technical, market, stakeholder, cloud_domain이 "
            "모두 포함되어야 합니다."
        )

    if not isinstance(search_questions, dict):
        raise TypeError("supervisor.yaml에 search_questions가 필요합니다.")

    missing_question_groups = EXPECTED_PERSPECTIVES - set(search_questions)
    if missing_question_groups:
        raise ValueError(
            "다음 관점의 search_questions가 없습니다: "
            f"{sorted(missing_question_groups)}"
        )

    for perspective in EXPECTED_PERSPECTIVES:
        questions = search_questions[perspective]
        if not isinstance(questions, list) or not all(
            isinstance(question, str) and question.strip() for question in questions
        ):
            raise ValueError(
                f"{perspective}의 search_questions는 문자열 목록이어야 합니다."
            )

    return SupervisorConfig(
        agent_name=str(config.get("agent_name", "supervisor")),
        version=str(config.get("version", "1.0")),
        mission=str(config.get("mission", "")),
        technologies=technologies,
        perspectives=perspectives,
        execution_order=config.get("execution_order", {}),
        search_questions=search_questions,
        policies=config.get("policies", {}),
    )


def supervisor_agent(state: GlobalState) -> dict[str, Any]:
    """YAML 기반 연구 계획을 만들고 기술 조사를 첫 실행 대상으로 지정한다."""
    config = _load_supervisor_config()
    plan: ResearchPlan = {
        "technologies": config["technologies"],
        "perspectives": config["perspectives"],
        "search_questions": config["search_questions"],
    }

    return {
        "research_plan": plan,
        "control": {
            "status": "researching",
            "next_agent": "technical",
            "retry_count": {},
            "errors": [],
        },
    }
