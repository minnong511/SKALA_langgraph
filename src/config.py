"""Validated environment configuration; unresolved model names stay explicit."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values
from pydantic import BaseModel, Field, SecretStr

from src.schemas import ExecutionLimits


class Settings(BaseModel):
    llm_provider: str = "openai"
    llm_model: str = ""
    llm_api_key: SecretStr = SecretStr("")
    tavily_api_key: SecretStr = SecretStr("")
    embedding_model: str = "BAAI/bge-m3"
    index_path: Path = Path("data/index")
    raw_dir: Path = Path("data/raw")
    output_dir: Path = Path("outputs")
    pdf_font_path: Path | None = None
    max_search_retries: int = Field(default=2, ge=0, le=10)
    max_research_retries: int = Field(default=2, ge=0, le=10)
    max_report_revisions: int = Field(default=2, ge=0, le=10)
    max_synthesis_retries: int = Field(default=2, ge=0, le=10)
    max_total_calls: int = Field(default=200, ge=1)
    max_run_seconds: float = Field(default=1800, gt=0)
    search_timeout_seconds: float = Field(default=30, gt=0, le=300)
    llm_timeout_seconds: float = Field(default=120, gt=0, le=600)
    heartbeat_seconds: float = Field(default=15, gt=0)
    retrieval_top_k: int = Field(default=5, ge=1, le=20)
    market_rag: bool = False
    stakeholder_rag: bool = False
    quiet: bool = False

    @classmethod
    def from_env(cls, env_file: str | Path = ".env", **overrides):
        env = {**dotenv_values(env_file), **os.environ}
        values = {
            name: env[name.upper()] for name in cls.model_fields if env.get(name.upper()) not in (None, "")
        }
        values.update({key: value for key, value in overrides.items() if value is not None})
        return cls.model_validate(values)

    @property
    def limits(self) -> ExecutionLimits:
        fields = ExecutionLimits.model_fields
        values = {name: getattr(self, name) for name in fields if hasattr(self, name)}
        return ExecutionLimits(**values, top_k=self.retrieval_top_k)

    @property
    def secrets(self) -> list[str]:
        return [
            value.get_secret_value()
            for value in (self.llm_api_key, self.tavily_api_key)
            if value.get_secret_value()
        ]

    def public_config(self) -> dict:
        return self.model_dump(mode="json", exclude={"llm_api_key", "tavily_api_key"})

    def validate_live(self) -> None:
        if self.llm_provider.lower() != "openai":
            raise ValueError("현재 구현한 LLM_PROVIDER는 openai입니다.")
        missing = [
            name
            for name, value in [
                ("LLM_MODEL", self.llm_model),
                ("LLM_API_KEY", self.llm_api_key.get_secret_value()),
                ("TAVILY_API_KEY", self.tavily_api_key.get_secret_value()),
            ]
            if not value
        ]
        if missing:
            raise ValueError("실제 실행에 필요한 설정: " + ", ".join(missing))


def create_llm(settings: Settings):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key.get_secret_value(),
        timeout=settings.llm_timeout_seconds,
        max_retries=0,
    )
