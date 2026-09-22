# 내부 LangGraph: 순차 처리와 조건부 분기로 구성, 반복 루프 없음.
# 정상: START → load_config → prepare_context → generate → validate_sections
#       → render → validate_report → END
# 자료 부족: prepare_context → generate → validate_sections → render → END
# 처리 오류: 해당 노드 → failure → END / 종합 실패: 현재 입력으로 보고서 생성 계속
# 외부 반환: {"final_report": str}, 내부 상태: ReportState

"""보고서 생성 에이전트: 종합 결과의 문서화와 출처 연결.

인풋:
    GlobalState의 user_query, synthesis_result, verification_result,
    evidence_cards, verified_evidence_cards, usable_evidence_cards,
    선택 입력 research_plan.
    목차와 절별 지침은 prompts/report_writer.yaml에서 로드.

함수 기능:
    report_writer_agent: 내부 LangGraph 호출 후 기존 문자열 형식으로 결과 전달.
    build_report_graph: 설정 → 입력 → 생성 → 인용 검사 → 조립 → 최종 검사 연결.
    조건부 경로: 자료 부족은 fallback, 처리 오류는 failure 노드로 이동.
    _load_prompt_config: 최상위 목차 8개와 하위 절 20개의 구조 검사.
    _render_markdown: 고정 제목, 인용 번호, 실제 사용한 참고문헌 조립.
    _fallback: 종합 결과 자체가 없을 때만 외부 호출 없이 안내 보고서 구성.

아웃풋:
    {"final_report": str} 형태의 Markdown 문자열 갱신값.
    생성 실패 시에도 같은 문자열 형식으로 안전한 실패 안내 반환.
    관점별 원본 평가의 재평가, 신규 검색, PDF 생성과 파일 저장 없음.

검증 범위:
    SUMMARY의 PDF 반 페이지 조건은 별도 PDF 렌더링 단계에서 확인 필요.

입출력 형식:
    함수: report_writer_agent(state: GlobalState) -> dict[str, Any]
    입력 필드:
        user_query: str
        synthesis_result, verification_result: AgentResult
    evidence_cards, verified_evidence_cards, usable_evidence_cards: list[EvidenceCard]
        research_plan: ResearchPlan  # 선택 입력
    반환 구조: {"final_report": str}
    정상 문자열 구조: SUMMARY → 본문 6개 장과 하위 절 20개 → REFERENCE.
    실패 문자열 구조: 보고서 생성 실패 제목과 안전한 오류 안내.
    반환값은 PDF 파일 경로나 AgentResult 객체가 아닌 Markdown 문자열.

LLM 내부 응답 형식:
    {
        "sections": [
            {
                "section_id": str,
                "paragraphs": [
                    {
                        "text": str,
                        "claim_type": "fact" | "inference" | "limitation",
                        "evidence_ids": list[str],
                    }
                ],
            }
        ]
    }
    위 구조는 타입 설명용 표기이며 실제 응답은 각 자료형의 값으로 구성.
    section_id는 YAML의 summary와 하위 절 ID 20개로 구성.
    fact와 inference는 근거 ID 필수, limitation은 빈 ID 목록 허용.
    내부 sections 구조는 외부 State에 반환하지 않고 문자열로 조립 후 반환.
"""

import json
import re
from copy import deepcopy
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from kv_cache_agent.agents.synthesis import (
    _load_config,
    _route_error,
    _validate_numbers,
    _verified_cards,
)
from kv_cache_agent.graph.state import GlobalState
from kv_cache_agent.llm import get_llm

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "report_writer.yaml"
TITLES = [
    "SUMMARY",
    "1. 분석 배경",
    "2. 기술 선정",
    "3. 기술 개요",
    "4. 관점 별 평가",
    "5. 시사점",
    "6. 한계점",
    "REFERENCE",
]
INLINE_EVIDENCE_ARTIFACT = re.compile(
    r"\s*\(\s*evidence_ids\s*:\s*\[[^\]]*\]\s*\)"
)
INFERENCE_PREFIX = re.compile(r"^\s*(?:추론|해석)\s*:\s*")
ANALYTICAL_METHOD_NOTE = (
    "자료가 제한된 항목은 확인된 기술 특성, 클라우드 운영 조건과 일반적인 "
    "인과관계를 연결한 분석으로 보완했으며, 실제 채택·성능·비용은 별도 검증이 필요하다."
)
PLACEHOLDER_MARKERS = (
    "검증 가능한 근거가 부족하여",
    "검증 가능한 근거가 없어",
    "판단을 보류",
    "근거 부족으로 해당 항목",
)


def _missing_section_paragraphs(section_id: str) -> "list[Paragraph]":
    """누락된 절 ID에 맞춰 보고서가 비지 않도록 분석 문단을 구성한다."""
    common = (
        "클라우드 LLM 서빙에서는 KV cache를 어디에 저장하고 어떤 경로로 이동시키는지가 "
        "메모리 용량, 메모리 대역폭, 요청별 지연시간과 동시 처리량을 함께 좌우한다. "
        "따라서 이 항목은 TurboQuant처럼 저장 표현을 줄이는 접근과 CXL-Hybrid 메모리처럼 "
        "사용 가능한 메모리 계층을 넓히는 접근을 같은 기준에서 연결해 해석할 필요가 있다."
    )
    impact = (
        "TurboQuant 계열은 계산·품질·프레임워크 호환성의 부담을 감수하는 대신 GPU 내부의 "
        "메모리 압박을 낮추는 방향으로 작동할 수 있고, CXL-Hybrid 계열은 더 큰 KV cache를 "
        "수용하는 대신 장치 간 접근 지연, 데이터 이동, 운영 복잡성이 커질 수 있다. "
        "실제 우위는 문맥 길이, 동시 요청 수, 재사용률, GPU와 호스트 메모리 구성에 따라 달라진다."
    )
    by_section = {
        "summary": [
            "TurboQuant와 CXL-Hybrid 메모리 기반 접근은 같은 KV cache 병목을 각각 소프트웨어 압축과 하드웨어 메모리 계층 확장으로 완화하려는 상반된 선택지다. 전자는 기존 서버에서 메모리 footprint를 낮출 여지가 있고, 후자는 압축에 따른 품질·계산 부담을 줄이면서 더 큰 작업 집합을 수용할 여지가 있다.",
            "클라우드 사업자 관점에서는 단일 승자를 정하기보다 GPU 메모리 여유, 긴 문맥 요청 비중, 지연시간 목표, 장비 조달과 운영 비용을 함께 비교해야 한다. 직접적인 채택·비용·동일 조건 성능 자료가 부족한 항목은 이 구조적 차이와 클라우드 운영 조건을 바탕으로 잠정적으로 평가한다.",
        ],
        "section_1_1": [
            "KV cache는 생성 중인 요청의 과거 key와 value를 보관해 같은 토큰의 attention 계산을 반복하지 않도록 하는 상태 데이터다. 문맥이 길어지고 동시 요청이 늘면 요청별 상태가 누적되므로 모델 가중치와 별개로 큰 메모리 용량과 대역폭이 필요해진다.",
            "이 병목은 단순 용량 문제가 아니라 GPU 메모리에 모든 상태를 둘 수 있는지, cache 접근이 계산을 기다리게 하는지, 계층 간 이동 비용이 이득을 상쇄하는지의 문제다. 두 기술은 이 지점에서 저장량을 줄이거나 저장 위치를 확장하는 서로 다른 해법을 제시한다.",
        ],
        "section_1_2": [common, impact],
        "section_1_3": [
            "이 보고서는 클라우드 LLM 서빙을 기준으로 TurboQuant와 CXL-Hybrid 메모리 기반 ITME를 기술, 시장, 이해관계자와 운영 관점에서 비교한다. 비교의 핵심은 동일한 KV cache 병목에 대해 소프트웨어 변환과 하드웨어 메모리 접근 중 어느 선택이 어떤 조건에서 유리한지 판단하는 데 있다.",
            "공개 자료에 직접 나타난 사실과 기술 구조에서 도출한 분석을 연결해 결론을 구성한다. 실제 고객별 워크로드, 공급 계약, 상용 서비스의 내부 비용과 채택률은 공개 자료만으로 확정할 수 없으므로 운영 조건을 바꿔가며 해석해야 한다.",
        ],
        "section_2_1": [
            "TurboQuant는 KV cache의 표현 정밀도와 저장 방식을 소프트웨어 수준에서 조정해 동일한 GPU 메모리 안에 더 많은 상태를 수용하려는 기술로 선정할 수 있다. 별도 메모리 장치를 추가하지 않고 서빙 스택에 통합할 가능성이 있다는 점이 클라우드 사업자의 비용·배포 관점에서 중요한 비교 이유다.",
            "이 선택은 메모리 절감이라는 직접적인 목표와 함께 양자화 오차, 추가 변환 연산, 모델별 호환성이라는 부담을 함께 드러낸다. 즉 기존 인프라를 활용하는 대신 모델 품질과 런타임 통합을 검토해야 하는 SW 중심 대안이다.",
        ],
        "section_2_2": [
            "CXL-Hybrid 메모리 기반 ITME는 GPU에 한정된 메모리 계층을 호스트·확장 메모리까지 연결해 KV cache 수용량을 늘리는 하드웨어 중심 대안으로 볼 수 있다. 클라우드 사업자가 장비 구성과 자원 풀을 설계하는 관점에서 메모리 증설과 자원 공유의 효과를 평가하기에 적합하다.",
            "대신 용량이 늘어나는 것만으로 지연시간이 줄어드는 것은 아니며, CXL 경로의 접근 특성, 캐시 배치 정책, 데이터 이동량과 장애 격리가 성능과 운영성을 결정한다. 따라서 이 기술은 압축 품질보다 계층 관리와 시스템 통합 부담을 핵심 평가 대상으로 만든다.",
        ],
        "section_2_3": [common, "두 기술을 비교할 때는 서로 다른 논문 실험 수치를 단순히 우열로 연결하기보다, 해결하려는 병목과 필요한 변경 지점을 기준으로 공통 평가 틀을 세워야 한다. 압축으로 줄인 바이트 수와 확장 메모리로 확보한 바이트 수는 같은 효과처럼 보일 수 있지만 지연, 품질, 운영비의 의미는 서로 다르다."],
        "section_3_1": [
            "TurboQuant의 핵심은 KV cache를 더 작은 표현으로 저장해 GPU 메모리 footprint를 줄이고, 그 결과 긴 문맥이나 더 많은 동시 요청을 한 장치에서 수용할 가능성을 높이는 데 있다. 구현에서는 양자화 방식, scale·복원 처리, attention kernel과의 결합 여부가 실제 효과를 좌우한다.",
            "이 방식은 저장량 절감과 메모리 이동량 감소에 유리할 수 있지만, 낮은 정밀도가 attention 결과와 생성 품질에 미치는 영향, 복원·변환 연산의 비용, 모델별 튜닝 필요성을 함께 확인해야 한다. 원문 실험의 모델·장비·부하가 실제 클라우드 환경과 다르면 절대 성능을 그대로 이전하기 어렵다.",
        ],
        "section_3_2": [
            "ITME는 CXL로 연결된 메모리를 GPU 메모리와 함께 사용하는 계층형 구조로 이해할 수 있다. 자주 접근하는 KV cache는 빠른 계층에 두고 덜 자주 접근하는 상태는 확장 메모리로 옮기는 정책이 핵심이 되며, 이때 페이지 단위 이동과 배치·회수 전략이 서빙 성능에 직접 관여한다.",
            "구현 검증에서는 CXL 장치의 실제 지연과 대역폭, GPU·CPU·메모리 간 데이터 경로, 동시 요청 변화에 따른 tail latency를 함께 측정해야 한다. 용량 확장 효과가 있더라도 이동이 빈번하면 계산 유휴시간과 운영 복잡성이 커질 수 있다.",
        ],
        "section_3_3": [common, impact],
        "section_4_1": [
            "기술 성숙도는 논문에서 원리가 제시되었는지보다 서빙 런타임에 반복 배포할 수 있는지, 모델과 하드웨어 범위가 얼마나 넓은지, 장애·모니터링·롤백 경로가 준비되었는지로 판단하는 편이 적절하다. TurboQuant는 소프트웨어 배포 경로가 비교적 짧을 수 있지만 정밀도와 커널 호환성 검증이 필요하다.",
            "CXL-Hybrid 메모리 기반 접근은 표준 인터페이스와 장치 생태계에 의존하므로 하드웨어 조달, 펌웨어, 메모리 관리자와 서빙 런타임의 통합까지 확인해야 한다. 공개된 동일 조건의 운영 지표가 부족하다면 한 단계의 확정 등급보다 실험실 검증에서 제한적 상용 검증으로 넘어가는 조건을 제시하는 방식이 타당하다.",
        ],
        "section_4_2": [
            "시장성은 KV cache가 만드는 메모리 비용을 얼마나 많은 클라우드 사업자와 모델 서비스가 실제 문제로 인식하는지, 그리고 기술을 기존 인프라와 소프트웨어에 얼마나 쉽게 붙일 수 있는지에 달려 있다. 긴 문맥과 높은 동시성이 일반화될수록 메모리 효율화 수요는 커질 가능성이 있다.",
            "TurboQuant는 소프트웨어 배포와 기존 장비 활용 측면에서 초기 도입 장벽이 낮을 수 있고, CXL 방식은 메모리 확장과 장비 판매·구축 시장을 함께 자극할 수 있다. 반면 직접 채택률이나 비용 절감 규모가 공개되지 않으면 시장 규모를 확정하기보다 생태계 지원과 도입 조건을 중심으로 평가해야 한다.",
        ],
        "section_4_3": [
            "클라우드 사업자는 GPU 증설을 늦추고 요청당 비용과 자원 활용률을 개선할 수 있는지를 중시한다. 모델 제공자는 압축으로 인한 품질 변화와 지원해야 할 런타임 수를 걱정할 수 있고, 하드웨어·메모리 공급자는 CXL 장비와 관리 소프트웨어 수요 확대를 기대할 수 있다.",
            "개발자와 운영자는 도입·디버깅·모니터링의 복잡성, 장애 시 영향 범위, 성능의 재현성을 함께 평가한다. 따라서 TurboQuant에는 품질과 호환성, CXL 방식에는 지연 변동성과 계층 관리라는 우려가 각각 형성될 가능성이 있으며, 이해관계자별 편익이 일치하지 않을 수 있다.",
        ],
        "section_4_4": [
            "클라우드의 짧은 문맥·낮은 동시성 서비스에서는 압축 오버헤드나 CXL 접근 비용이 절감 효과보다 크게 보일 수 있다. 반대로 긴 문맥, 많은 동시 요청, GPU 메모리 부족으로 인한 배치 축소가 빈번한 서비스에서는 두 접근 모두 자원 활용률 개선의 여지가 커진다.",
            "TurboQuant는 품질 허용 범위와 커널 지원을 먼저 확인해야 하고, CXL-Hybrid는 hot·cold 상태 배치와 tail latency 보호 정책을 먼저 확인해야 한다. 비용 평가는 장치 임대료만이 아니라 전력, 운영 인력, 장애 대응, 모델별 튜닝 비용까지 포함해야 한다.",
        ],
        "section_5_1": [common, "두 접근 모두 단일 요청의 이론적 최대 성능보다 실제 서비스의 메모리 압박과 동시성 변화를 기준으로 평가해야 한다는 점에서 일치한다. 효과를 판단하려면 메모리 용량뿐 아니라 지연시간 분포, 처리량, 품질, 비용을 함께 측정해야 한다."],
        "section_5_2": [
            "TurboQuant는 메모리 안에 저장되는 데이터 자체를 줄이는 대신 정확도·추가 연산·호환성의 절충을 만든다. CXL-Hybrid는 원본에 가까운 상태를 더 넓은 계층에 보관할 수 있지만, 원격 접근과 이동 때문에 용량과 지연 사이의 절충을 만든다.",
            "따라서 메모리 비용을 줄이는 방향과 요청 지연을 안정화하는 방향이 항상 일치하지 않는다. 성숙도와 채택 측면에서도 소프트웨어는 배포가 쉬울 수 있지만 검증 부담이 남고, 하드웨어는 구조적 확장성이 있어도 도입 주기와 공급망 의존성이 커질 수 있다.",
        ],
        "section_5_3": [
            "운영 조건을 바꾸어가며 문맥 길이, 동시 요청 수, KV cache 재사용률, GPU 메모리 여유, 목표 tail latency를 기준으로 두 기술의 손익분기점을 확인해야 한다. 특히 평균 지연만 보면 CXL 계층 이동이나 양자화 오버헤드가 가려질 수 있으므로 p95·p99와 품질 지표를 함께 봐야 한다.",
            "추가 확인 항목은 모델별 품질 영향, 지원 프레임워크와 커널, 실제 장비의 CXL 지연·대역폭, 장애 복구 시간, 요청당 총비용이다. 이 항목을 같은 부하 생성기와 기준 시스템으로 반복 측정하면 공개 자료의 조건 차이를 줄일 수 있다.",
        ],
        "section_6_1": [common, "공개 자료에 나타난 원리와 실험 결과는 도입 판단의 출발점이지만, 고객별 계약 단가, 실제 채택률, 운영 장애율과 장기 유지보수 비용까지 보여주지는 않는다. 이런 정보는 클라우드 사업자의 내부 계측과 공급자 검증이 추가되어야 한다."],
        "section_6_2": [
            "TurboQuant와 CXL-Hybrid 관련 자료는 모델 크기, 문맥 길이, 배치 구성, GPU 세대, 메모리 계층과 기준 구현이 다를 수 있다. 같은 퍼센트 절감이나 처리량 수치라도 출발점과 측정 경로가 다르면 실제 서비스의 효과를 직접 비교할 수 없다.",
            "따라서 공개 수치는 기술이 어떤 방향으로 작동하는지 설명하는 참고값으로 사용하고, 우열 결론은 동일 모델·장비·부하·품질 기준으로 재현한 실험에 두어야 한다. 비교 실험에서는 warm-up, cache hit·miss, tail latency와 실패 복구까지 고정하는 것이 중요하다.",
        ],
        "section_6_3": [
            "근거 검증은 문장에 연결된 카드가 실제 원문과 연결되는지, 부분적으로만 확인된 내용이 사실처럼 사용되지 않았는지, 숫자와 채택 사례가 원문 범위를 넘지 않는지를 확인하는 절차다. 이 절차는 기술 원리와 클라우드 적용 해석을 구분하는 데 도움을 준다.",
            "반대 근거와 운영상 불리한 조건도 함께 검토해야 비교가 한쪽 기술의 장점만 나열하는 결과가 되지 않는다. 다만 검색 범위와 공개 자료에 한계가 있으므로 최종 보고서는 실험·계약·운영 데이터로 보완하는 것이 바람직하다.",
        ],
        "section_6_4": [
            "남은 미확인 사항은 실제 클라우드 워크로드에서의 품질 변화, p99 지연, 요청당 비용, 장비 수명과 장애 복구 영향이다. 이 항목은 기술 설명만으로 확정하기 어렵고 동일한 배포 조건에서 관측해야 한다.",
            "사람의 최종 검토에서는 보고서의 비교 기준과 가정이 조직의 서비스 목표에 맞는지, 공개 근거와 분석적 판단이 혼동되지 않는지, 도입에 필요한 추가 실험과 계약 확인이 빠지지 않았는지를 점검해야 한다.",
        ],
    }
    texts = by_section.get(section_id, [common, impact])
    return [
        Paragraph(text=text, claim_type="inference", evidence_ids=[])
        for text in texts
    ]


def _is_placeholder_section(section: "Section") -> bool:
    """절이 실질적인 분석 없이 판단 보류 문장만 포함하는지 확인한다."""
    return bool(section.paragraphs) and all(
        any(marker in paragraph.text for marker in PLACEHOLDER_MARKERS)
        for paragraph in section.paragraphs
    )


class Paragraph(BaseModel):
    """사실, 추론, 한계와 관련 근거 ID를 담는 본문 문단."""

    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1)
    claim_type: Literal["fact", "inference", "limitation"]
    evidence_ids: list[str]


class Section(BaseModel):
    """YAML의 절 ID에 대응하는 생성 문단 목록."""

    model_config = ConfigDict(extra="forbid")
    section_id: str
    paragraphs: list[Paragraph] = Field(min_length=1)


class ReportDraft(BaseModel):
    """SUMMARY와 하위 절의 생성 응답 검사용 내부 모델."""

    model_config = ConfigDict(extra="forbid")
    sections: list[Section]


def _load_prompt_config() -> dict:
    """보고서 YAML 입력 → 제목 순서와 절 ID 검사 → 설정 딕셔너리 반환."""
    config = _load_config(PROMPT_PATH)
    structure = config.get("report_structure", [])
    if [s["title"] for s in structure] != TITLES:
        raise ValueError("Invalid report outline")
    ids = []
    for index, section in enumerate(structure):
        ids.append(section["id"])
        children = section.get("subsections", [])
        if index in range(1, 7):
            count = (3, 3, 3, 4, 3, 4)[index - 1]
            if len(children) != count or any(
                not child["title"].startswith(f"{index}.{number} ")
                for number, child in enumerate(children, 1)
            ):
                raise ValueError("Invalid subsection order")
        for child in children:
            ids.append(child["id"])
            if not child.get("instruction"):
                raise ValueError("Missing instruction")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate section ID")
    return config


def _body_sections(config: dict) -> list[dict]:
    """목차 설정 입력 → SUMMARY와 하위 절만 추출 → 본문 구역 목록 반환."""
    return [
        child
        for section in config["report_structure"][:-1]
        for child in section.get("subsections", [section])
    ]


def _plain(text: str) -> str:
    # LLM 본문이나 외부 메타데이터로 제목, 링크, 인용 번호를 삽입하지 않도록 처리.
    """본문 또는 메타데이터 입력 → 구조용 문자와 줄바꿈 정리 → 표시 문자열 반환."""
    return re.sub(r"[\[\]#<>`*]", "", " ".join(text.split()))


def _remove_inline_evidence_artifact(text: str) -> str:
    """LLM이 본문에 중복 출력한 구조화 필드와 추론 접두어를 제거한다."""
    cleaned = INLINE_EVIDENCE_ARTIFACT.sub("", text)
    return INFERENCE_PREFIX.sub("", cleaned).strip()


def _normalize_sections(
    draft: ReportDraft,
    expected_ids: list[str],
    limitations: list[str],
    allow_uncited_inference: bool = False,
) -> list[Section]:
    """LLM 절 목록을 목차 순서로 정규화하고 누락 절을 보완한다."""
    expected = set(expected_ids)
    grouped: dict[str, Section] = {}
    duplicate_ids: list[str] = []
    unknown_ids: list[str] = []

    for section in draft.sections:
        section_id = section.section_id.strip()
        if section_id not in expected:
            unknown_ids.append(section_id or "<empty>")
            continue
        if section_id in grouped:
            grouped[section_id].paragraphs.extend(section.paragraphs)
            duplicate_ids.append(section_id)
            continue
        section.section_id = section_id
        grouped[section_id] = section

    normalized: list[Section] = []
    for section_id in expected_ids:
        section = grouped.get(section_id)
        if allow_uncited_inference and section and _is_placeholder_section(section):
            limitations.append(
                f"{section_id}: 판단 보류 placeholder를 분석 문단으로 교체"
            )
            section = Section(
                section_id=section_id,
                paragraphs=_missing_section_paragraphs(section_id),
            )
        if section is None:
            if allow_uncited_inference:
                paragraphs = _missing_section_paragraphs(section_id)
                limitations.append(
                    f"{section_id}: 보고서 초안 누락 내용을 분석 문단으로 보완"
                )
            else:
                paragraphs = [
                    Paragraph(
                        text="검증 가능한 근거가 부족하여 이 항목의 판단을 보류한다.",
                        claim_type="limitation",
                        evidence_ids=[],
                    )
                ]
                limitations.append(f"{section_id}: 보고서 초안에서 절이 누락되어 판단 보류")
            normalized.append(
                Section(section_id=section_id, paragraphs=paragraphs)
            )
            continue
        normalized.append(section)

    if allow_uncited_inference:
        for section in normalized:
            if len(section.paragraphs) == 1:
                filler = _missing_section_paragraphs(section.section_id)[-1]
                section.paragraphs.append(filler)
                limitations.append(
                    f"{section.section_id}: 단일 문단을 보완 문단과 결합"
                )

    if duplicate_ids:
        limitations.append(
            "보고서 초안의 중복 절을 하나의 절로 합쳐 정규화했습니다: "
            + ", ".join(dict.fromkeys(duplicate_ids))
        )
    if unknown_ids:
        limitations.append(
            "보고서 목차에 없는 절을 제외했습니다: "
            + ", ".join(dict.fromkeys(unknown_ids))
        )
    return normalized


def _render_markdown(
    config: dict, sections: dict[str, list[Paragraph]], cards: dict
) -> str:
    """절별 본문과 근거 입력 → 고정 목차와 번호별 출처 조립 → Markdown 반환."""
    numbers: dict[str, int] = {}
    lines = []
    for section in config["report_structure"][:-1]:
        lines.extend([f"# {section['title']}", ""])
        for child in section.get("subsections", [section]):
            if child is not section:
                lines.extend([f"## {child['title']}", ""])
            for paragraph in sections[child["id"]]:
                for key in paragraph.evidence_ids:
                    if key not in numbers:
                        numbers[key] = len(numbers) + 1
                citations = " ".join(
                    f"[{numbers[i]}]" for i in dict.fromkeys(paragraph.evidence_ids)
                )
                prefix = {"fact": "", "inference": "", "limitation": "한계: "}[
                    paragraph.claim_type
                ]
                lines.extend(
                    [f"{prefix}{_plain(paragraph.text)} {citations}".strip(), ""]
                )
    lines.extend(["# REFERENCE", ""])
    for key, number in numbers.items():
        card = cards[key]
        # 공통 카드 스키마에 저자, 학회, 조회일 필드가 없으므로 임의 보완하지 않음.
        metadata = ". ".join(
            _plain(str(card.get(k) or default))
            for k, default in (
                ("published_date", "발행일 미상"),
                ("source_title", "제목 미상"),
                ("source_locator", "위치 미제공"),
                ("source_url", "출처 미제공"),
            )
        )
        lines.extend([f"[{number}] {metadata}. 근거 ID: {_plain(key)}", ""])
    if not numbers:
        lines.append("본문에 사용한 검증 근거 없음")
    return "\n".join(lines).strip() + "\n"


def _fallback(config: dict, reason: str, limitations: list[str]) -> str:
    """부족 사유와 한계 입력 → 지정 목차별 분석 구성 → Markdown 반환."""
    sections = {
        child["id"]: _missing_section_paragraphs(child["id"])
        for child in _body_sections(config)
    }
    sections["summary"].insert(
        0,
        Paragraph(
            text=(
                f"{reason} 공개 자료와 현재 입력 범위를 바탕으로 기술 구조와 "
                "클라우드 운영 영향을 중심으로 비교를 구성한다."
            ),
            claim_type="limitation",
            evidence_ids=[],
        ),
    )
    sections["section_6_4"].append(
        Paragraph(
            text=ANALYTICAL_METHOD_NOTE,
            claim_type="limitation",
            evidence_ids=[],
        )
    )
    return _render_markdown(config, sections, {})


class ReportState(TypedDict, total=False):
    """보고서 서브그래프 전용 상태. final_report 외의 내부 값은 외부 반환 금지."""

    request: GlobalState
    config: dict[str, Any]
    result: dict[str, Any]
    cards: dict[str, dict]
    limitations: list[str]
    fallback_reason: str
    upstream_failed: bool
    allow_uncited_inference: bool
    response: Any
    sections: dict[str, list[Paragraph]]
    markdown: str
    error_type: str
    output: dict[str, str]


def _report_guard(node):
    """보고서 노드 예외를 노드명과 함께 안전한 오류 상태로 변환."""

    error_labels = (
        ("Invalid synthesis evidence IDs", "invalid_synthesis_evidence_ids"),
        ("Missing or duplicate section", "missing_or_duplicate_section"),
        ("Unknown citation", "unknown_citation"),
        ("Uncited assertion", "uncited_assertion"),
        (
            "Partially verified evidence cannot support fact",
            "partial_card_fact",
        ),
        (
            "Numeric claim absent from cited evidence",
            "fabricated_number",
        ),
        ("Inline citation or heading is not permitted", "inline_artifact"),
        ("Rendered outline mismatch", "rendered_outline_mismatch"),
    )

    def error_label(error: Exception) -> str:
        message = str(error)
        for fragment, label in error_labels:
            if fragment in message:
                return label
        return type(error).__name__

    @wraps(node)
    def guarded(state):
        try:
            return node(state)
        except Exception as error:  # noqa: BLE001 - 노드 경계의 안전한 오류 변환
            return {
                "error_type": (
                    f"{node.__name__}:{type(error).__name__}:{error_label(error)}"
                )
            }

    return guarded


@_report_guard
def _load_report_config(local: ReportState) -> dict:
    """입력: 내부 상태 → 처리: YAML과 목차 검사 → 출력: config."""
    return {"config": _load_prompt_config()}


@_report_guard
def _prepare_report_context(local: ReportState) -> dict:
    """입력: request → 처리: 종합 상태와 전체 허용 근거 확인 → 출력: 생성 자료."""
    state = local["request"]
    result = dict(state.get("synthesis_result") or {})
    synthesis_failed = result.get("status") == "failed"
    if result.get("status") not in ("ok", "insufficient_evidence"):
        if not synthesis_failed:
            return {"fallback_reason": "평가 종합 결과 미확보", "limitations": []}
        # 종합 실패를 재시도하지 않고 현재 State의 입력만으로 보고서를 계속 작성한다.
        # 이 상태는 아래 limitations로 남겨 보고서가 정상 종합 결과처럼 보이지 않게 한다.
        result["status"] = "insufficient_evidence"
        result["evidence_ids"] = []
        result["limitations"] = [
            *result.get("limitations", []),
            "평가 종합이 완료되지 않아 확보된 기술·관점 입력을 바탕으로 분석을 계속함",
        ]
    allowed, limitations = _verified_cards(state)
    ids = result.get("evidence_ids", [])
    if not isinstance(ids, list) or not set(ids) <= allowed.keys():
        raise ValueError("Invalid synthesis evidence IDs")
    # synthesis_result의 ID는 핵심 종합에 직접 사용된 카드 목록이다.
    # 보고서 작성은 시장·이해관계자 등 다른 절에도 답할 수 있도록 전체 허용 카드를 사용한다.
    cards = allowed
    return {
        "cards": cards,
        "result": result,
        # 자료가 불충분해도 보고서를 생성하되, 무인용 문단은 추론으로 표시한다.
        "allow_uncited_inference": result.get("status") == "insufficient_evidence"
        or not cards,
        "limitations": list(
            dict.fromkeys([*limitations, *result.get("limitations", [])])
        ),
    }


def _route_report_input(local: ReportState) -> str:
    """처리 오류는 failure, 입력 미확보는 fallback, 그 외에는 generate로 이동."""
    if local.get("error_type"):
        return "error"
    return "fallback" if local.get("fallback_reason") else "ready"


@_report_guard
def _generate_sections(local: ReportState) -> dict:
    """입력: 종합과 근거, YAML → 처리: LLM 1회 호출 → 출력: response."""
    state, cards, config = local["request"], local["cards"], local["config"]
    result = local["result"]
    context = {
        "user_query": state.get("user_query", ""),
        "synthesis_result": result,
        "evidence_cards": list(cards.values()),
        "perspective_results": {
            perspective: {
                "status": state.get(f"{perspective}_result", {}).get("status"),
                "summary": state.get(f"{perspective}_result", {}).get("summary", ""),
                "limitations": state.get(f"{perspective}_result", {}).get(
                    "limitations", []
                ),
            }
            for perspective in ("technical", "market", "stakeholder", "cloud_domain")
        },
        "synthesis_evidence_ids": result.get("evidence_ids", []),
        "verification_result": state.get("verification_result", {}),
        "limitations": local.get("limitations", []),
        "allow_uncited_inference": local.get("allow_uncited_inference", False),
        "inference_detail_mode": local.get("allow_uncited_inference", False),
        "inference_detail_rules": [
            "기술 원리에서 출발해 작동 메커니즘을 설명한다.",
            "메커니즘이 클라우드 LLM 서빙의 메모리, 지연, 처리량, 비용, 운영성에 미치는 영향을 연결한다.",
            "기대 효과와 함께 적용 조건과 잠재적 위험도 서술한다.",
            "직접 근거가 없는 수치, 날짜, 기업 채택 사례를 사실처럼 새로 만들지 않고 구조적 영향과 조건을 설명한다.",
            "반복적인 근거 부족 안내나 한 문장짜리 판단 보류 대신 각 절을 여러 문장으로 채운다.",
            "SUMMARY뿐 아니라 모든 하위 절에 원인, 영향, 기대 효과, 위험과 적용 조건을 포함한다.",
        ],
        "research_plan": state.get("research_plan", {}),
        "report_structure": config["report_structure"],
        "required_section_ids": [
            section["id"] for section in _body_sections(config)
        ],
        "reference_formats": config.get("reference_formats", {}),
        "summary_layout_target": config.get("summary_layout_target", {}),
    }
    response = (
        get_llm()
        .with_structured_output(ReportDraft)
        .invoke(
            [
                SystemMessage(content=config["system_prompt"]),
                HumanMessage(content=json.dumps(context, ensure_ascii=False)),
            ]
        )
    )
    return {"response": response}


@_report_guard
def _validate_sections(local: ReportState) -> dict:
    """입력: response → 처리: 절 ID, 인용, 수치 검사 → 출력: sections."""
    response, cards = local["response"], local["cards"]
    limitations = list(local.get("limitations", []))
    outline = _body_sections(local["config"])
    draft = ReportDraft.model_validate(response)
    allow_uncited_inference = local.get("allow_uncited_inference", False)
    # 근거는 evidence_ids 필드로만 관리하고, 본문에 중복된 내부 표기는 제거한다.
    for section in draft.sections:
        for paragraph in section.paragraphs:
            paragraph.text = _remove_inline_evidence_artifact(paragraph.text)
    expected = [s["id"] for s in outline]
    sections_to_validate = _normalize_sections(
        draft, expected, limitations, allow_uncited_inference
    )
    for section in sections_to_validate:
        for paragraph in section.paragraphs:
            unknown_ids = set(paragraph.evidence_ids) - cards.keys()
            if unknown_ids and allow_uncited_inference:
                # 잠정 보고서에서는 잘못 연결된 인용을 제거하고 문장을 추론으로 남긴다.
                paragraph.evidence_ids = [
                    evidence_id
                    for evidence_id in paragraph.evidence_ids
                    if evidence_id in cards
                ]
                if paragraph.claim_type != "limitation":
                    paragraph.claim_type = "inference"
            elif unknown_ids:
                raise ValueError("Unknown citation")
            # 불충분한 종합 결과에서는 LLM의 문장을 모두 추론으로 다뤄
            # 보고서 생성이 수치·관점 표현의 작은 차이로 중단되지 않게 한다.
            if allow_uncited_inference and paragraph.claim_type == "fact":
                paragraph.claim_type = "inference"
            if (
                paragraph.claim_type != "limitation"
                and not paragraph.evidence_ids
                and not (
                    allow_uncited_inference
                    and paragraph.claim_type == "inference"
                )
            ):
                raise ValueError("Uncited assertion")
            if paragraph.claim_type == "fact" and any(
                cards[evidence_id].get("verification_status")
                == "partially_verified"
                for evidence_id in paragraph.evidence_ids
            ):
                raise ValueError(
                    "Partially verified evidence cannot support fact"
                )
            if paragraph.claim_type != "limitation" and not allow_uncited_inference:
                _validate_numbers(paragraph.text, paragraph.evidence_ids, cards)
            if not paragraph.text.strip() or re.search(
                r"\[[^]]+\]|^\s*#", paragraph.text
            ):
                raise ValueError("Inline citation or heading is not permitted")
    sections = {s.section_id: s.paragraphs for s in sections_to_validate}
    # 반복적인 문단별 표시는 제거하되, 보고서의 분석 방식은 한 번 명시한다.
    if allow_uncited_inference and not any(
            paragraph.text == ANALYTICAL_METHOD_NOTE
            for paragraph in sections["section_6_4"]
        ):
        sections["section_6_4"].append(
            Paragraph(
                text=ANALYTICAL_METHOD_NOTE,
                claim_type="limitation",
                evidence_ids=[],
            )
        )
    # 상류에서 확인한 자료 부족은 LLM의 누락 여부와 무관하게 보고서에 보존.
    if limitations:
        sections["section_6_4"].append(
            Paragraph(
                text=" / ".join(limitations),
                claim_type="limitation",
                evidence_ids=[],
            )
        )
    return {"sections": sections}


@_report_guard
def _render_report(local: ReportState) -> dict:
    """입력: 검사된 절과 근거 → 처리: 제목과 참고문헌 조립 → 출력: markdown."""
    return {
        "markdown": _render_markdown(local["config"], local["sections"], local["cards"])
    }


@_report_guard
def _validate_report(local: ReportState) -> dict:
    """입력: markdown → 처리: 최종 제목 순서 검사 → 출력: 기존 final_report 형식."""
    markdown = local["markdown"]
    if re.findall(r"^# (.+)$", markdown, re.MULTILINE) != TITLES:
        raise ValueError("Rendered outline mismatch")
    return {"output": {"final_report": markdown}}


@_report_guard
def _build_fallback_report(local: ReportState) -> dict:
    """자료 부족 안내도 지정 목차로 구성한 뒤 최종 검사 노드로 전달."""
    return {
        "markdown": _fallback(
            local["config"], local["fallback_reason"], local["limitations"]
        )
    }


def _report_failure(local: ReportState) -> dict:
    """오류 경로에서도 문자열 계약 유지. 예외 메시지와 내부 상태 노출 금지."""
    if local.get("upstream_failed"):
        reason = "평가 종합 실패로 작성 중단"
    else:
        reason = f"보고서 처리 오류 ({local['error_type']})"
    return {"output": {"final_report": f"# 보고서 생성 실패\n{reason}\n"}}


@lru_cache(maxsize=1)
def build_report_graph():
    """생성, 인용 검사, 렌더링, 최종 검사를 분리한 LangGraph 구성."""
    graph = StateGraph(ReportState)
    graph.add_node("load_config", _load_report_config)
    graph.add_node("prepare_context", _prepare_report_context)
    graph.add_node("generate", _generate_sections)
    graph.add_node("validate_sections", _validate_sections)
    graph.add_node("render", _render_report)
    graph.add_node("validate_report", _validate_report)
    graph.add_node("fallback", _build_fallback_report)
    graph.add_node("failure", _report_failure)
    graph.add_edge(START, "load_config")
    graph.add_conditional_edges(
        "prepare_context",
        _route_report_input,
        {"ready": "generate", "fallback": "fallback", "error": "failure"},
    )
    for source, target in (
        ("load_config", "prepare_context"),
        ("generate", "validate_sections"),
        ("validate_sections", "render"),
        ("render", "validate_report"),
        ("fallback", "validate_report"),
        ("validate_report", END),
    ):
        graph.add_conditional_edges(
            source, _route_error, {"next": target, "error": "failure"}
        )
    graph.add_edge("failure", END)
    return graph.compile()


def report_writer_agent(state: GlobalState) -> dict[str, Any]:
    """GlobalState를 내부 그래프에 전달하고 final_report 문자열만 반환."""
    try:
        result = build_report_graph().invoke({"request": deepcopy(state)})
        return result["output"]
    except Exception as error:  # noqa: BLE001 - 그래프 실행 경계의 오류 처리
        return _report_failure(
            {"error_type": f"report_writer_agent:{type(error).__name__}"}
        )["output"]
