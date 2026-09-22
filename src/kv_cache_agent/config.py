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
