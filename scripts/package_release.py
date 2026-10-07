"""실제 키/가상환경/Git 이력을 제외한 실행용 ZIP을 만든다.

실키가 들어 있는 .env는 읽지도 않는다. 배포 템플릿을 메모리에서 별도로
생성해 넣는다. --report로 명시한 결과물만 포함하고 기존 outputs 로그는 제외한다.
"""

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
ENV_TEMPLATE = """OPENAI_API_KEY=
TAVILY_API_KEY=
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=BAAI/bge-m3
HF_TOKEN=
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=kv-cache-supervisor
"""


def package(destination: Path, report: Path | None = None) -> None:
    """기존 ZIP을 덮어쓰지 않고 프로젝트 폴더 하나로 압축한다."""
    if destination.exists():
        raise FileExistsError(f"기존 파일을 덮어쓰지 않습니다: {destination}")
    excluded = {
        ".venv",
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        "outputs",
    }
    with ZipFile(destination, "x", compression=ZIP_DEFLATED) as archive:
        for path in sorted(ROOT.rglob("*")):
            relative = path.relative_to(ROOT)
            if not path.is_file() or any(part in excluded for part in relative.parts):
                continue
            if path.name.startswith(".env") or path.suffix in {".pyc", ".zip"}:
                continue
            if any(
                part.startswith(".") and part not in {".gitignore", ".python-version"}
                for part in relative.parts
            ):
                continue
            archive.write(path, f"SKALA_langgraph/{relative.as_posix()}")
        for name in (".env", ".env.example"):
            archive.writestr(f"SKALA_langgraph/{name}", ENV_TEMPLATE)
        if report:
            for path in (report, report.with_suffix(".md")):
                if path.is_file():
                    archive.write(path, f"SKALA_langgraph/outputs/reports/{path.name}")
    print(f"ZIP 생성: {destination} / 실제 .env는 제외, 배포 키는 모두 빈 값")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    parser.add_argument("--report", type=Path)
    arguments = parser.parse_args()
    package(arguments.destination, arguments.report)
