# 공통 인터페이스

원본 명세: [Notion](https://app.notion.com/p/3e209004d9f8806ebbfbd59733bb5293)

모든 에이전트의 외부 인터페이스는 `run(request: AgentRequest, context: AgentContext) -> AgentResult`입니다.
정의는 `src/schemas.py`, 그래프 State는 `src/state.py`에 있습니다. Python 호출부에는 Pydantic 객체를 사용하고
파일에는 `model_dump(mode="json")`에 해당하는 구조를 저장합니다.

## 입력

| 필드 | 형식/의미 |
|---|---|
| run_id, task_id | 경로로 사용할 수 있는 영문/숫자/밑줄/하이픈 식별자 |
| technologies | 기술명 문자열 목록, 기본 TurboQuant/ITME |
| domain | 적용 도메인 |
| as_of_date | ISO 날짜, 조사 기준일 |
| questions | 이번 작업의 질문 |
| context | 작업에 필요한 계획/기존 데이터 |
| feedback | 이전 검증/보고서 검토의 보완 요청 |
| limits | 검색/재조사/종합/보고서 추가 재시도, 총 호출, 시간, top_k |
| attempt | 최초 1부터 시작하는 시도 번호 |

`AgentContext`에는 LLM, 검색 도구, 원문 리더, 이전 에이전트 결과 스냅샷, 로그, 실행 예산을 주입합니다.
`ask`, `search`, `read`를 통해 총 호출 한도를 적용합니다. 이 객체는 State나 파일에 저장하지 않습니다.

## 출력

공통 필드: `task_id`, `agent`, `attempt`, `status`, `summary`, `findings`, `evidence_cards`,
`gaps`, `follow_up_requests`, `errors`.
상태는 `completed`, `partial`, `failed`이며 작업 완료와 근거 검증 통과는 별개입니다.
추가 필드: 검색 원문의 `sources`, 주장별 `verification`, `report_markdown`, `used_evidence_ids`, 역할별 `data`.
예외도 공통 실행 래퍼가 오류 결과로 변환하여 슈퍼바이저에게 반환합니다.

근거 카드에는 `evidence_id`, 기술/관점, 주장, 원문 인용, 출처 ID/제목/저자/URL/파일,
페이지/절, 발행일/조회일, 조건/한계, 사실/저자 주장/분석 추론을 기록합니다.
누락 메타데이터는 빈 값과 `missing_metadata` 사유를 사용합니다.
추론은 `supporting_evidence_ids`로 근거를 연결합니다. 알 수 없는 출처나 원문에 없는 인용은 제외합니다.
검증 판정은 `verified`, `uncertain`, `rejected`이며 판정 누락을 통과로 간주하지 않습니다.

## State 소유권

- 시작 시: request, config
- 슈퍼바이저: plan, pending_tasks, phase, attempts, retry_counts, feedback, review_history, review, status
- 에이전트: technical_result, market_result, stakeholder_result, domain_result, verification_result, synthesis_result, report_result 각각 하나
- 실행부: final_artifacts, 최종 내보내기 상태

병렬 노드는 자기 결과 필드만 갱신합니다. 슈퍼바이저는 다음 super-step에서 합쳐진 결과를 받습니다.
공통 목록을 여러 노드가 동시에 덮어쓰지 않습니다. 새로운 기술 조사로 의존 평가를 무효화할 때는 해당 결과를 None으로 초기화합니다.
`retry_counts`는 총 시도 횟수가 아니라 최초 실행 이후의 추가 실행 횟수입니다.

## 기존 코드 반영

`src/legacy/models.py`의 구조화된 조사/검증/종합 설계, `research_base.py`의 도구 주입 및 조사 계획,
각 에이전트의 역할 프롬프트를 새 공통 계약으로 옮겼습니다.
이전 `validation_result`는 `verification_result`, `claim_id`는 `evidence_id`,
외부 누락 모듈 `state`는 `src.state`로 통합했습니다. 원본 파일은 변경하지 않았습니다.
