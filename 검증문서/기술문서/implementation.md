# 구현 및 검증 현황

기준: [Notion 명세](https://app.notion.com/p/3e209004d9f8806ebbfbd59733bb5293).
원본 `src/legacy/`의 15개 Python 파일은 해시 비교로 내용 보존을 확인했습니다.
기존 역할별 프롬프트와 조사·검증·종합 설계를 공통 계약에 맞게 이식했습니다.

| 명세 범위 | 구현 및 확인 |
|---|---|
| 8개 에이전트 공통 호출 | `run(request, context) -> AgentResult`, 실패도 구조화 반환 |
| 공유 State/스키마 | Pydantic 요청·근거·판정·결과, 독립적인 병렬 결과 필드 |
| 슈퍼바이저 제어 | 전 단계 복귀, 관점별 병렬 실행, 대상별 재조사, 기술 변경 시 후속 평가 무효화 |
| 종료 한도 | 추가 재시도와 최초 실행 구분, 전체 호출/시간 한도, 검토 필요/실패 상태 보존 |
| Tavily | 실제 REST 어댑터, 제한 시간, 출처 및 중복 제거, 오류 처리 |
| 원문 RAG | BGE-M3 임베딩, PDF 페이지 단위 분할, 사전 FAISS 구축/로드, JSON 메타데이터 |
| 출처 검증 | 원문 재읽기, 인용문/출처/위치 확인, 발행일 기준일 확인, 추론 연결 검증 |
| 종합 | 네 관점과 조건/일치/상충, 미검증 근거 제외, 불확실성 명시 |
| 보고서 | 지정 목차, 본문 인용 기반 참고문헌 재구성, 내용 및 인용 검토 |
| PDF | 한글 폰트 포함, 실제 SUMMARY 높이 반 페이지 검증 및 수정 요청 |
| 로그 | UTC 이벤트, 작업/시도 식별자, 실제 소요 시간, 병렬 진행 수, 재조사 사유 |
| 저장 | 실행별 로그, 시도별 JSON, 원자적 저장, 기존 결과 덮어쓰기 거부, 비밀정보 마스킹 |
| 설치/협업 | `.env.example`, `pyproject.toml`, `uv.lock`, 설치/원문/실행 문서, 산출물 Git 제외 |

## 검증 방법

```bash
python -m pytest -q
ruff check main.py src scripts tests
ruff format --check main.py src scripts tests
python main.py --demo
```

`tests/test_full_pipeline.py`는 실제 8개 에이전트, LangGraph, 슈퍼바이저, 최종 저장을 모두 사용합니다.
LLM과 검색 응답만 고정된 테스트 값으로 대체하고 외부 네트워크 연결을 차단해 검사합니다.
`tests/test_openai_adapter.py`는 실제 ChatOpenAI/OpenAI SDK에 가짜 HTTP 전송 계층을 주입합니다.
모델 ID/timeout/재시도 0 설정과 구조화 출력 파싱을 확인합니다. 실 API 키는 사용하지 않습니다.
도구 테스트는 실제 FAISS 인덱스와 PDF를 사용하며 임베딩 계산만 작은 테스트 구현으로 대체합니다.
PDF는 한글 텍스트 추출, 내장 폰트, 긴 SUMMARY 초과 및 파일 충돌을 확인합니다.

구조화 LLM 응답은 function calling을 사용하고 Pydantic으로 반환값을 검증합니다.
동적 메타데이터 사전이 있으므로 서버의 strict JSON Schema 모드는 강제하지 않습니다.
참고: [OpenAI 함수 호출](https://developers.openai.com/api/docs/guides/function-calling),
[LangGraph 상태 및 병렬 단계](https://docs.langchain.com/oss/python/langgraph/graph-api),
[Tavily 검색](https://docs.tavily.com/documentation/api-reference/endpoint/search),
[SentenceTransformer](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html),
[FAISS](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes).

## 실제 실행 전에 필요한 입력

- 사용 가능한 정확한 `LLM_MODEL`과 OpenAI/Tavily API 키.
- 제공된 `data/raw/CXL.pdf`, `data/raw/Turboquant.pdf`로 만든 인덱스(`RAW_DIR=data/raw` 기본 설정).
- 실제 계정에서의 모델 접근, Tavily 응답, BGE-M3 모델 다운로드와 품질은 아직 확인하지 않았습니다.
- `--demo` 산출물은 합성 예시이며 기술별 주장이나 최종 제출 근거로 사용할 자료가 아닙니다.

기술별 연구 결과의 정확성과 의미 판단은 입력 원문 및 모델 응답에 의존합니다.
팀의 미확정 선택(정확한 모델 ID, Python/패키지 버전 합의, 시장/이해관계자의 RAG 범위)은 README에 명시했습니다.
중단 지점 재개와 웹 수집 자료의 자동 벡터 인덱싱은 명세의 확정 범위에 포함하지 않아 구현하지 않았습니다.

최종 검증: Python 3.13 환경에서 테스트 69개 통과, Ruff 검사/서식 검사 통과, uv.lock 검증 통과.
