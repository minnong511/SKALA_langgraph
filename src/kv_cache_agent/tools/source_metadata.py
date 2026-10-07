"""논문 서지 정보와 배포 가능한 상대 경로를 일관되게 관리한다.

날짜·저자는 추측하지 않는다. 아래 두 항목은 동봉된 PDF 첫 페이지의
arXiv 버전/날짜/저자를 직접 확인한 값이다. 웹 자료는 수집 결과만 사용한다.
"""

from copy import deepcopy
from pathlib import Path
from urllib.parse import urlparse

from kv_cache_agent.config import ROOT_DIR

PAPERS = {
    "turboquant.pdf": {
        "source_title": "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate",
        "authors": "Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni",
        "published_date": "2025-04-28",
        "publisher": "arXiv:2504.19874v1",
        "canonical_url": "https://arxiv.org/abs/2504.19874v1",
    },
    "cxl_based_kv_cache.pdf": {
        "source_title": "ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories",
        "authors": "Hakbeom Jang, Younghoon Min, Sunwoong Kim, Taeyoung Ahn, Hanyee Kim, Youngpyo Joo, Hoshik Kim, Jongryool Kim",
        "published_date": "2026-06-16",
        "publisher": "arXiv:2606.12556v2",
        "canonical_url": "https://arxiv.org/abs/2606.12556v2",
    },
}


def normalize_source(record: dict) -> dict:
    """옛 컴퓨터 경로는 알려진 동봉 논문에 한해서만 상대 경로로 교체한다.

    임의 파일명으로 로컬 파일을 열지 않고, 등록된 논문 및 논문형 자료에만
    적용한다. 카드 ID·주장·평가 결과는 변경하지 않는다.
    """
    output = deepcopy(record)
    source = str(record.get("source_url") or record.get("source_path") or "")
    if urlparse(source).scheme in {"http", "https"}:
        return output
    name = Path(source).name
    if name not in PAPERS or not (ROOT_DIR / "data" / "papers" / name).is_file():
        return output
    relative = f"data/papers/{name}"
    output.update(PAPERS[name])
    output["source_url"] = relative
    output["source_path"] = relative
    if "title" in output:
        output["title"] = output["source_title"]
    return output
