import os
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
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Plain values only; tracing clients are created after preflight, never at import.
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").lower() in {"true", "1"}
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT") or None
TAVILY_EXTRACT_DEPTH = os.getenv("TAVILY_EXTRACT_DEPTH", "advanced")
TAVILY_EXTRACT_TIMEOUT = float(os.getenv("TAVILY_EXTRACT_TIMEOUT", "30"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
