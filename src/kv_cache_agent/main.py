import argparse
from datetime import UTC, datetime

from kv_cache_agent.config import OUTPUTS_DIR
from kv_cache_agent.graph.workflow import build_workflow

DEFAULT_QUERY = (
    "클라우드 LLM 서빙에서 TurboQuant와 CXL-based KV Cache 최적화 기술을 "
    "기술, 시장, 이해관계자 관점에서 비교 평가해줘."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help="평가 질문. 생략하면 기본 KV Cache 비교 질문을 사용합니다.",
    )
    args = parser.parse_args()
    query = args.query.strip() or DEFAULT_QUERY

    result = build_workflow().invoke({"user_query": query})

    output_dir = OUTPUTS_DIR / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"kv_cache_evaluation_{timestamp}.md"
    report_path.write_text(result["final_report"], encoding="utf-8")

    print(f"질문: {query}")
    print(f"보고서 저장 완료: {report_path}")
    print("검증 상태:", result.get("verification_result", {}).get("status"))
    print("종합 상태:", result.get("synthesis_result", {}).get("status"))
    print("보고서 길이:", len(result.get("final_report", "")))


if __name__ == "__main__":
    main()
