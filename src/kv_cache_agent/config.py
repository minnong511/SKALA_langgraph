import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# config.py 기준 두 단계 위가 프로젝트 루트다.
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
PAPERS_DIR = DATA_DIR / "papers"
VECTOR_DB_DIR = DATA_DIR / "vector_db"
OUTPUTS_DIR = ROOT_DIR / "outputs"

load_dotenv(ROOT_DIR / ".env")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-6-luna")
OPENAI_REQUEST_TIMEOUT = float(os.getenv("OPENAI_REQUEST_TIMEOUT", "120"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
HF_TOKEN = os.getenv("HF_TOKEN", "")


@dataclass(frozen=True)
class ReportBudget:
    """References count toward the PDF limit; shortening has a separate bound."""

    max_pdf_pages: int = 10
    max_length_rewrites: int = 2
    max_analysis_rewrites: int = 2
    target_body_chars: int = 15000

    def __post_init__(self):
        if not 1 <= self.max_pdf_pages <= 10:
            raise ValueError("report page limit must be between 1 and 10")
        if (
            self.max_length_rewrites < 0
            or self.max_analysis_rewrites < 0
            or self.target_body_chars < 500
        ):
            raise ValueError("invalid report shortening budget")

    @classmethod
    def from_env(cls):
        return cls(
            max_pdf_pages=int(os.getenv("MAX_REPORT_PAGES", "10")),
            max_length_rewrites=int(os.getenv("MAX_LENGTH_REWRITES", "2")),
            max_analysis_rewrites=int(os.getenv("MAX_ANALYSIS_REWRITES", "2")),
            target_body_chars=int(os.getenv("REPORT_BODY_CHARS", "15000")),
        )


@dataclass(frozen=True)
class WorkflowLimits:
    max_steps: int = 20
    max_report_revisions: int = 3
    max_worker_retries: int = 2
    max_tasks_per_plan: int = 24
    max_cards_per_task: int = 24

    def __post_init__(self):
        if (
            self.max_steps < 1
            or self.max_cards_per_task < 1
            or self.max_tasks_per_plan < 4
        ):
            raise ValueError("steps/cards must be positive; task cap must be >= 4")
        if min(self.max_report_revisions, self.max_worker_retries) < 0:
            raise ValueError("retry and revision limits must be nonnegative")

    @classmethod
    def from_env(cls):
        return cls(
            max_steps=int(os.getenv("MAX_STEPS", "20")),
            max_report_revisions=int(os.getenv("MAX_REPORT_REVISIONS", "3")),
            max_worker_retries=int(os.getenv("MAX_WORKER_RETRIES", "2")),
            max_tasks_per_plan=int(os.getenv("MAX_TASKS_PER_PLAN", "24")),
            max_cards_per_task=int(os.getenv("MAX_CARDS_PER_TASK", "24")),
        )
