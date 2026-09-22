# KV Cache SW vs HW 개발 순서

## 1. 프로젝트 목표

TurboQuant와 CXL-based KV Cache 기술을 대상으로 기술성, 시장성, 이해관계자 반응, 클라우드 도메인 적합성을 평가하는 Supervisor 기반 Agentic RAG를 구현한다.

## 2. 담당 에이전트

| 번호 | 에이전트                      | 담당자 |
| ---: | ----------------------------- | ------ |
|    1 | Supervisor 에이전트           | 재혁   |
|    2 | 기술 조사 에이전트            | 재혁   |
|    3 | 시장 평가 에이전트            | 시은   |
|    4 | 이해관계자 평가 에이전트      | 시은   |
|    5 | 클라우드 도메인 평가 에이전트 | 연수   |
|    6 | 근거 검증 에이전트            | 연수   |
|    7 | 평가 종합 에이전트            | 민형   |
|    8 | 보고서 생성 에이전트          | 민형   |

## 3. 전체 실행 흐름

```text
Supervisor
    ↓
기술 조사
    ↓
시장 평가 ─┐
이해관계자 ─┼─ 병렬 실행
클라우드 평가 ┘
    ↓
근거 검증
    ↓
평가 종합
    ↓
보고서 생성
```

기술 조사는 두 기술에 대한 공통 기준을 제공하므로 먼저 실행한다. 시장, 이해관계자, 클라우드 평가는 기술 조사 결과를 받은 뒤 병렬로 실행한다. 세 평가가 끝나면 근거를 검증하고, 검증이 끝난 결과만 종합과 보고서 생성에 사용한다.

## 4. 개발 순서

### 0단계. 공통 규격 확정

담당: 재혁 주도, 전원 합의

다른 에이전트 개발 전에 다음 규격을 먼저 확정한다.

- `GlobalState` 필드
- `EvidenceCard` 필수 필드
- `AgentResult` 출력 형식
- 에이전트별 입력·출력 범위
- YAML 프롬프트 파일명과 필드 규칙
- 성공·실패 상태값

주요 파일:

- `src/kv_cache_agent/graph/state.py`
- `src/kv_cache_agent/schemas/outputs.py`
- `src/kv_cache_agent/graph/workflow.py`
- `src/kv_cache_agent/prompts/*.yaml`

이 단계가 끝나면 State 구조를 임의로 변경하지 않는다. 변경이 필요하면 모든 담당자에게 먼저 공유한다.

### 1단계. 재혁 개발: Supervisor와 기술 조사

#### 1번 Supervisor 에이전트

먼저 전체 그래프의 뼈대를 작성한다.

구현 범위:

- 사용자 질문 수신
- 연구 계획 생성
- 기술 조사 우선 실행
- 시장·이해관계자·클라우드 평가 병렬 실행
- 세 평가 완료 여부 확인
- 근거 검증 실행
- 평가 종합 실행
- 보고서 생성 실행
- 에이전트 실패 시 오류 상태 기록

처음에는 각 에이전트를 실제로 완성하지 않아도 된다. 각 에이전트를 임시 노드로 연결하여 전체 그래프가 실행되는지 먼저 확인한다.

#### 2번 기술 조사 에이전트

논문 2건을 FAISS 기반 RAG로 조사한다.

```text
PDF 읽기
  ↓
문서 분할
  ↓
BGE-M3 임베딩
  ↓
FAISS 검색
  ↓
기술 근거 추출
  ↓
EvidenceCard 반환
```

조사 대상:

- TurboQuant 작동 방식
- CXL-based KV Cache 작동 방식
- 해결하려는 KV Cache 병목
- 장점과 한계
- 필요한 하드웨어·소프트웨어 조건
- 성능 또는 메모리 절감 근거

기술 조사 결과는 다른 에이전트가 사용할 수 있도록 샘플 JSON으로도 저장한다.

```text
tests/fixtures/sample_technical_result.json
```

### 2단계. 시은 개발: 시장과 이해관계자 평가

시은은 기술 조사 에이전트가 완성될 때까지 기다리지 않고 샘플 기술 결과를 사용해 개발한다.

#### 3번 시장 평가 에이전트

Tavily로 다음 내용을 조사한다.

- 관련 시장 규모와 성장 전망
- 클라우드 사업자의 채택 가능성
- 상용화 또는 채택 사례
- 관련 기업과 경쟁 기술
- 비용 절감 가능성
- 도입 장벽

#### 4번 이해관계자 평가 에이전트

다음 이해관계자별로 평가한다.

- 클라우드 사업자
- AI 모델 개발사
- 하드웨어 제조사
- 클라우드 고객
- 오픈소스 개발자
- 연구자와 투자자

각 이해관계자의 기대 효과, 우려 사항, 도입 조건, 예상 반응을 근거 카드로 반환한다.

### 3단계. 연수 개발: 클라우드 도메인과 근거 검증

#### 5번 클라우드 도메인 평가 에이전트

평가 도메인은 클라우드 하나로 고정한다.

평가 항목:

- 클라우드 LLM 추론 서비스 적합성
- GPU 메모리 비용 절감 가능성
- 긴 컨텍스트 처리 적합성
- 멀티테넌시 환경 적합성
- 지연시간과 처리량 영향
- 클라우드 사업자 관점의 운영 난이도
- 고객이 체감할 수 있는 효과

#### 6번 근거 검증 에이전트

시장·이해관계자·클라우드 평가 에이전트가 만든 근거 카드를 검증한다.

검증 항목:

- 주장을 출처가 실제로 뒷받침하는가?
- 논문 내용과 웹 자료가 혼동되지 않았는가?
- 최신 자료인지 확인했는가?
- 사실과 추론이 구분되어 있는가?
- 출처가 중복되거나 신뢰성이 낮지 않은가?
- TurboQuant와 CXL-based를 공정하게 비교했는가?

권장 검증 상태:

```text
verified
partially_verified
unsupported
needs_more_evidence
```

### 4단계. 민형 개발: 평가 종합과 보고서 생성

#### 7번 평가 종합 에이전트

검증을 통과한 결과를 관점별로 비교한다.

| 비교 항목         | TurboQuant | CXL-based |
| ----------------- | ---------- | --------- |
| 기술성            |            |           |
| 시장성            |            |           |
| 클라우드 적합성   |            |           |
| 이해관계자 수용성 |            |           |
| 주요 위험         |            |           |
| 도입 조건         |            |           |

단순 점수 합산보다 다음 내용을 중심으로 종합한다.

- 관점별 평가가 일치하는 부분
- 관점별 평가가 충돌하는 부분
- 특정 이해관계자에게만 유리한 부분
- 추가 검증이 필요한 부분
- 어떤 조건에서 어느 기술이 적합한지

#### 8번 보고서 생성 에이전트

새로운 검색을 수행하지 않고, 검증된 평가 종합 결과만 사용한다.

보고서 목차:

1. 조사 목적
2. KV Cache 병목 개요
3. TurboQuant 기술 소개
4. CXL-based 기술 소개
5. 기술 관점 비교
6. 시장 관점 비교
7. 이해관계자 관점 비교
8. 클라우드 도메인 평가
9. 근거 검증 결과
10. 종합 평가
11. 한계와 추가 조사 과제
12. 결론

### 5단계. 전체 통합

모든 에이전트가 완료되면 재혁이 `workflow.py`에서 전체 그래프를 통합한다.

통합 확인 순서:

1. 기술 조사 결과가 State에 저장되는지 확인
2. 시장·이해관계자·클라우드 평가가 병렬 실행되는지 확인
3. 세 평가 결과가 근거 검증으로 전달되는지 확인
4. 검증 결과가 평가 종합으로 전달되는지 확인
5. 평가 종합 결과가 보고서 생성으로 전달되는지 확인
6. 최종 보고서에 출처가 포함되는지 확인

## 5. 공통 State 사용 규칙

에이전트 간 데이터는 반드시 `GlobalState`를 통해 전달한다. 전역 변수, 직접 객체 참조, 비공식 임시 파일로 결과를 공유하지 않는다.

각 에이전트는 자신이 담당한 State 필드만 수정한다.

| 에이전트        | 주요 수정 필드                              |
| --------------- | ------------------------------------------- |
| Supervisor      | `control`, `research_plan`              |
| 기술 조사       | `technical_result`, `evidence_cards`    |
| 시장 평가       | `market_result`, `evidence_cards`       |
| 이해관계자 평가 | `stakeholder_result`, `evidence_cards`  |
| 클라우드 평가   | `cloud_domain_result`, `evidence_cards` |
| 근거 검증       | `verification_result`                     |
| 평가 종합       | `synthesis_result`                        |
| 보고서 생성     | `final_report`                            |

## 6. 확정 데이터 구조

공통 데이터 구조는 `src/kv_cache_agent/schemas/outputs.py`에서 관리한다. 에이전트별로 별도의 결과 형식을 만들지 않고 아래 형식을 공통으로 사용한다.

### EvidenceCard

`EvidenceCard`는 출처가 있는 하나의 주장과 근거를 표현한다.

```python
class EvidenceCard(TypedDict, total=False):
    evidence_id: str
    technology: Literal["TurboQuant", "CXL-based", "both", "general"]
    perspective: Literal[
        "technical", "market", "stakeholder", "cloud_domain"
    ]
    claim: str
    evidence_text: str
    source_title: str
    source_url: str
    source_type: Literal["paper", "official", "news", "blog", "report"]
    source_locator: str
    retrieval_method: Literal["faiss", "tavily", "direct"]
    published_date: str
    claim_type: Literal["fact", "inference"]
    confidence: float
    caveat: str
    verification_status: Literal[
        "unverified",
        "verified",
        "partially_verified",
        "unsupported",
    ]
```

`technology`에 `both`와 `general`을 허용한다. 두 기술을 비교하는 주장이나 특정 기술에 한정되지 않는 클라우드·시장 동향도 표현해야 하기 때문이다.

`source_locator`에는 논문 페이지·절 번호 또는 웹 문서 내 관련 위치를 기록한다. `retrieval_method`에는 해당 근거를 가져온 방식을 기록한다.

### AgentResult

모든 에이전트는 동일한 `AgentResult` 형식으로 결과를 반환한다.

```python
class AgentResult(TypedDict, total=False):
    agent_name: str
    status: Literal[
        "ok",
        "needs_retry",
        "insufficient_evidence",
        "failed",
    ]
    summary: str
    evidence_ids: list[str]
    limitations: list[str]
    errors: list[str]
    payload: dict[str, Any]
```

`payload`에는 에이전트별 추가 결과를 저장한다. 예를 들어 시장 평가 에이전트는 `market_findings`, `adoption_barriers`, `competitors`를 사용할 수 있다. 다른 에이전트가 사용하는 payload 키는 문서나 코드에 명시한다.

`limitations`는 조사의 범위나 자료의 한계를 기록하고, `errors`는 API 실패·파싱 실패·검색 실패 등 실행 오류를 기록한다.

### GlobalState 저장 범위

`GlobalState`에는 정제된 결과만 저장한다. FAISS의 원문 청크나 Tavily의 전체 검색 결과는 각 에이전트의 LocalState에서 처리하고, 최종적으로 필요한 근거만 `EvidenceCard`로 변환하여 공유한다.

## 7. 근거 카드 규칙

모든 외부 주장은 근거 카드로 관리한다.

```python
{
    "evidence_id": "market-001",
    "technology": "TurboQuant",
    "perspective": "market",
    "claim": "주장 내용",
    "evidence_text": "주장을 뒷받침하는 원문 또는 요약",
    "source_title": "출처 제목",
    "source_url": "출처 URL",
    "source_type": "report",
    "source_locator": "p. 12 또는 관련 절",
    "retrieval_method": "tavily",
    "published_date": "발행일 또는 확인일",
    "claim_type": "fact",
    "confidence": 0.85,
    "caveat": "자료의 한계",
    "verification_status": "unverified"
}
```

출처가 없는 시장 규모, 채택 현황, 성능 수치는 보고서에 포함하지 않는다. 사실과 에이전트의 추론을 구분해서 기록한다.

## 8. RAG 사용 규칙

- 기술 조사 에이전트: 논문 2건을 저장한 FAISS 사용
- 시장·이해관계자·클라우드 에이전트: Tavily 웹 검색 사용
- 근거 검증 에이전트: 기존 근거 카드와 원 출처를 비교
- 평가 종합 에이전트: 검증 완료 결과 사용
- 보고서 생성 에이전트: 검증 완료 결과와 종합 결과 사용

평가 종합과 보고서 생성 단계에서는 원칙적으로 새로운 검색을 수행하지 않는다.

## 9. 프롬프트 파일 규칙

프롬프트는 `prompts/`의 YAML 파일에서 관리한다.

```text
prompts/
  supervisor.yaml
  technical.yaml
  market.yaml
  stakeholder.yaml
  cloud_domain.yaml
  verifier.yaml
  synthesis.yaml
  report_writer.yaml
```

각 YAML에는 다음 항목을 포함하는 것을 권장한다.

```yaml
agent_name: technical_research
version: "1.0"
role: |
  에이전트의 역할
goals:
  - 목표 1
  - 목표 2
system_prompt: |
  시스템 프롬프트
input_fields:
  - user_query
output_fields:
  - summary
  - evidence_cards
constraints:
  require_citation: true
```

## 10. 개발 규칙

### 에이전트 호출 규칙

에이전트가 다른 에이전트를 직접 호출하지 않는다. 에이전트 실행 순서와 분기는 Supervisor와 LangGraph가 담당한다.

### 검색 제한 규칙

- Tavily 검색어: 관점별 최대 3개
- 검색 결과: 검색어당 상위 3~5개
- 근거 검증 재시도: 최대 1회
- Supervisor 오류 재실행: 최대 1회

무한 반복과 무제한 검색을 금지한다.

### 테스트 규칙

일반 테스트에서는 외부 API를 호출하지 않는다.

- OpenAI API: Mock
- Tavily API: Mock
- BGE-M3: Mock 또는 고정 임베딩
- FAISS 검색: 테스트용 문서 사용

권장 테스트 파일:

```text
tests/
  test_state.py
  test_supervisor.py
  test_technical.py
  test_web_agents.py
  test_verifier.py
  fixtures/
    sample_technical_result.json
```

### 보안 규칙

- `.env`를 커밋하지 않는다.
- API 키를 Python 코드나 YAML에 직접 작성하지 않는다.
- 실제 보고서와 대용량 벡터 DB는 커밋하지 않는다.
- `.env.example`만 공유한다.

## 11. 파일 충돌 방지 규칙

담당 파일은 다음과 같이 나눈다.

| 담당자 | 담당 파일                                                                                                            |
| ------ | -------------------------------------------------------------------------------------------------------------------- |
| 재혁   | `agents/supervisor.py`, `agents/technical.py`, `graph/state.py`, `graph/workflow.py`, `schemas/outputs.py` |
| 시은   | `agents/market.py`, `agents/stakeholder.py`, 관련 YAML과 테스트                                                  |
| 연수   | `agents/cloud_domain.py`, `agents/verifier.py`, 관련 YAML과 테스트                                               |
| 민형   | `agents/synthesis.py`, `agents/report_writer.py`, 관련 YAML과 테스트                                             |

다음 공통 파일은 재혁이 관리한다.

- `graph/state.py`
- `graph/workflow.py`
- `schemas/outputs.py`
- `pyproject.toml`
- `uv.lock`

공통 파일을 수정해야 할 때는 먼저 팀에 알리고, 수정 후 전체 테스트를 실행한다.

## 12. 한 브랜치 개발 규칙

하루 개발이라면 한 브랜치에서 개발해도 된다. 단, 다음 규칙을 지킨다.

1. 각자 담당 파일만 수정한다.
2. 공통 파일은 재혁만 수정한다.
3. 작업 단위가 끝날 때마다 작은 커밋을 만든다.
4. 커밋 전에 `uv run pytest -q`를 실행한다.
5. 다른 사람의 변경사항을 덮어쓰지 않는다.
6. 통합 시에는 재혁이 최종적으로 `workflow.py`를 수정한다.

권장 커밋 단위:

```text
feat: implement technical research agent
feat: implement market evaluation agent
feat: implement evidence verifier
feat: connect supervisor workflow
test: add agent unit tests
```

## 13. 완료 기준

각 에이전트는 다음 조건을 만족해야 완료로 본다.

- 지정된 YAML 프롬프트를 읽는다.
- 입력 State를 받는다.
- 담당 State 필드만 수정한다.
- `AgentResult` 형식으로 결과를 반환한다.
- 근거가 있는 주장을 `EvidenceCard`로 반환한다.
- 오류 발생 시 예외를 삼키지 않고 `errors`에 기록한다.
- 외부 API가 실패해도 전체 그래프가 중단되지 않도록 상태를 반환한다.
- 단위 테스트가 최소 1개 이상 존재한다.

## 14. 팀원별 작업 체크리스트

### 재혁 — Supervisor와 기술 조사

담당 범위:

- agents/supervisor.py
- agents/technical.py
- graph/state.py
- graph/workflow.py
- schemas/outputs.py
- rag/
- tools/paper_retriever.py

작업 목록:

- [X] GlobalState, AgentResult, EvidenceCard 구조 확정
- [X] Supervisor 전체 실행 그래프 작성
- [X] 논문 PDF chunk 분할 구현
- [X] BGE-M3 임베딩 연결
- [X] FAISS 인덱스 생성·저장·로드·검색 구현
- [X] TurboQuant와 CXL-based 논문 FAISS 인덱스 생성
- [X] 기술 조사 에이전트의 검색 결과 연결
- [X] tests/fixtures/sample_technical_result.json 작성
- [ ] 기술 조사 에이전트의 GPT-4o-mini 구조화 출력 확인
- [ ] 기술 조사 결과의 EvidenceCard 출처 연결 확인
- [ ] Supervisor가 모든 에이전트 결과를 올바르게 연결하는지 확인
- [ ] 전체 통합 테스트 실행
- [ ] 최종 샘플 보고서 생성

완료 기준:

- [ ] Supervisor 그래프가 START부터 END까지 실행된다.
- [ ] 기술 조사 결과에 TurboQuant와 CXL-based 근거가 모두 포함된다.
- [ ] 모든 기술 근거에 논문 제목과 페이지가 포함된다.
- [ ] pytest와 ruff 검사가 통과한다.

### 시은 — 시장 평가와 이해관계자 평가

담당 범위:

- agents/market.py
- agents/stakeholder.py
- prompts/market.yaml
- prompts/stakeholder.yaml
- tests/test_market.py
- tests/test_stakeholder.py

시장 평가 작업:

- [ ] sample_technical_result.json 또는 technical_result를 입력으로 사용
- [ ] TurboQuant와 CXL-based별 Tavily 질문 작성
- [ ] 시장 규모와 성장 전망 조사
- [ ] 클라우드 사업자 채택 가능성 조사
- [ ] 상용화·채택 사례 조사
- [ ] 관련 기업과 경쟁 기술 조사
- [ ] 비용 절감 가능성과 도입 장벽 조사
- [ ] 검색 결과를 EvidenceCard로 변환
- [ ] market_result에 요약·한계·오류 저장
- [ ] 출처 URL과 발행일 저장
- [ ] Tavily Mock 테스트 작성

이해관계자 평가 작업:

- [ ] 클라우드 사업자 관점 조사
- [ ] AI 모델 개발사 관점 조사
- [ ] 하드웨어 제조사 관점 조사
- [ ] 클라우드 고객 관점 조사
- [ ] 오픈소스 개발자·연구자 관점 조사
- [ ] 기대 효과와 우려 사항 분리
- [ ] 이해관계자별 근거를 EvidenceCard로 변환
- [ ] stakeholder_result에 요약·한계·오류 저장
- [ ] Tavily Mock 테스트 작성

완료 기준:

- [ ] 시장 평가와 이해관계자 평가가 각각 독립적으로 실행된다.
- [ ] 검색 결과에 출처 URL이 포함된다.
- [ ] 사실과 추론이 claim_type으로 구분된다.
- [ ] 근거 없는 시장 규모나 채택 사례를 생성하지 않는다.

### 연수 — 클라우드 도메인 평가와 근거 검증

담당 범위:

- agents/cloud_domain.py
- agents/verifier.py
- prompts/cloud_domain.yaml
- prompts/verifier.yaml
- tests/test_cloud_domain.py
- tests/test_verifier.py

클라우드 도메인 평가 작업:

- [ ] 평가 범위를 클라우드 LLM 서빙으로 제한
- [ ] TurboQuant의 클라우드 적용 적합성 평가
- [ ] CXL-based의 클라우드 적용 적합성 평가
- [ ] GPU 메모리 비용 영향 평가
- [ ] 긴 컨텍스트 처리 적합성 평가
- [ ] 멀티테넌시 환경 적합성 평가
- [ ] 지연시간·처리량·SLO 영향 평가
- [ ] 클라우드 사업자의 운영 난이도 평가
- [ ] 고객이 체감할 수 있는 효과 평가
- [ ] cloud_domain_result와 근거 카드 반환

근거 검증 작업:

- [ ] 기술·시장·이해관계자·클라우드 근거 카드 수집
- [ ] 주장과 출처 내용의 일치 여부 확인
- [ ] 출처 URL 또는 논문 페이지 존재 여부 확인
- [ ] 사실과 추론의 구분 확인
- [ ] 출처 발행일과 최신성 확인
- [ ] 근거가 부족한 카드를 unsupported로 표시
- [ ] 일부만 뒷받침되는 카드를 partially_verified로 표시
- [ ] 검증 결과를 verification_result에 저장
- [ ] 검증 실패와 재시도 조건 구현
- [ ] 검증 테스트 작성

완료 기준:

- [ ] 클라우드 도메인 외의 불필요한 평가를 하지 않는다.
- [ ] 검증되지 않은 근거가 종합 단계로 그대로 전달되지 않는다.
- [ ] 모든 근거 카드에 검증 상태가 기록된다.

### 민형 — 평가 종합과 보고서 생성

담당 범위:

- agents/synthesis.py
- agents/report_writer.py
- prompts/synthesis.yaml
- prompts/report_writer.yaml
- tests/test_synthesis.py
- tests/test_report_writer.py

평가 종합 작업:

- [ ] 검증 완료된 결과만 입력으로 사용
- [ ] TurboQuant와 CXL-based 비교표 작성
- [ ] 기술성·시장성·이해관계자 수용성 비교
- [ ] 클라우드 적합성 비교
- [ ] 장점·한계·위험·도입 조건 정리
- [ ] 관점별 일치점과 충돌점 분석
- [ ] 특정 조건에서 적합한 기술 구분
- [ ] 단순 점수 합산 대신 근거 중심으로 종합
- [ ] synthesis_result 반환

보고서 생성 작업:

- [ ] 새로운 웹 검색을 수행하지 않도록 제한
- [ ] 검증 완료 근거와 종합 결과만 사용
- [ ] 정해진 보고서 목차 적용
- [ ] 기술·시장·이해관계자·클라우드 관점 포함
- [ ] 본문 주장에 출처 연결
- [ ] 사실·추론·한계 구분
- [ ] 최종 결론에 조건부 판단 포함
- [ ] final_report 반환
- [ ] 샘플 보고서 생성 테스트 작성

완료 기준:

- [ ] 두 기술이 같은 평가 기준으로 비교된다.
- [ ] 보고서에 검증되지 않은 주장이 포함되지 않는다.
- [ ] 보고서가 정해진 12개 목차를 따른다.
- [ ] 최종 결과가 final_report에 저장된다.

## 16. 최종 통합 체크리스트

담당: 전원 확인, 재혁 통합

- [ ] 모든 에이전트가 동일한 GlobalState를 사용한다.
- [ ] 모든 에이전트가 AgentResult 형식으로 결과를 반환한다.
- [ ] 모든 외부 주장이 EvidenceCard로 관리된다.
- [ ] 기술 조사가 가장 먼저 실행된다.
- [ ] 시장·이해관계자·클라우드 평가가 병렬 실행된다.
- [ ] 근거 검증이 세 평가 이후에 실행된다.
- [ ] 평가 종합이 검증 결과만 사용한다.
- [ ] 보고서 생성이 새로운 검색을 수행하지 않는다.
- [ ] OpenAI·Tavily 호출이 테스트에서 Mock 처리된다.
- [ ] .env와 API 키가 커밋되지 않는다.
- [ ] pytest 검사가 통과한다.
- [ ] ruff 검사가 통과한다.
- [ ] 샘플 보고서가 생성된다.
