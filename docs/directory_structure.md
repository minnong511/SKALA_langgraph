# 구현 디렉토리

```text
main.py                     # 설정, 의존성 생성, 그래프 실행, 최종 저장
pyproject.toml              # 설치/의존성/테스트 설정
.env.example                # 키 없는 환경변수 예시
src/
  schemas.py                # 요청/결과/근거/검증/실행 문맥
  state.py                  # LangGraph 공유 상태
  config.py                 # 환경변수, 기본 한도, OpenAI 생성
  graph.py                  # 슈퍼바이저 중심 그래프
  demo.py                   # 합성 예시 에이전트
  agents/
    base.py                 # 기존 조사 엔진을 공통 계약으로 이식
    supervisor.py           # 계획, 배정, 재조사, 최종 검토
    technical.py            # 기술 성숙도/원문 조사
    market.py               # 시장성
    stakeholder.py          # 이해관계자
    domain.py               # 도메인 적용
    verification.py         # 원문 재확인 및 주장 검증
    synthesis.py            # 네 관점 종합
    report.py               # 목차/본문/실제 인용 목록
  tools/
    web_search.py           # Tavily
    retriever.py            # BGE-M3/FAISS
    source_reader.py        # 원문 PDF/웹 읽기
  common/
    runtime.py              # 호출 예산, 에이전트 공통 래퍼
    events.py               # 직렬 로그
    artifacts.py            # 실행/작업/시도별 저장
  exporters/pdf.py          # 한글 PDF/요약 높이 검증
  legacy/                   # 최초 제공된 15개 Python 파일 원본
scripts/build_index.py      # 원문 인덱스 사전 생성
prompts/                    # 8개 역할 프롬프트
data/raw/                   # 직접 확보한 원문 (Git 제외)
data/index/                 # 생성 인덱스 (Git 제외)
outputs/                    # 실행 산출물 (Git 제외)
tests/                      # 도구, 근거 무결성, 병렬/재시도, 저장, PDF
docs/                       # 명세, 흐름, 구현 상태
```

Python 패키지에는 `__init__.py`가 있습니다. 원본 `src/legacy/`는 실행 경로에서 가져오지 않으며
기존 상대 import와 미완성 State에 의존하지 않습니다. 새 모듈은 `src.*` 공통 계약을 사용합니다.
