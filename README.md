# SKALA: 근거 기반 기술 다관점 평가

TurboQuant와 CXL-Hybrid 메모리 기반 ITME를 클라우드 LLM 서빙에서 기술 성숙도, 시장성,
이해관계자, 도메인 적용 관점으로 조사합니다. 단일 점수나 승패 대신 조건별 차이와 불확실성을 설명합니다.
[Notion 구현 명세](https://app.notion.com/p/3e209004d9f8806ebbfbd59733bb5293)를 기준으로 구현했습니다.
기존 구현은 `src/legacy/`에 원문 그대로 보존했습니다.

## 빠른 시작: 키 없는 통합 예시

Python 3.11–3.13을 사용합니다. 개발 검증 환경은 Python 3.13입니다. `uv.lock`에 재현 가능한 의존성 버전을 기록했습니다.
uv 사용 시 `uv sync --locked --extra dev`로 준비할 수 있습니다(실제 RAG는 `--extra rag` 추가).

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python main.py --demo
python -m pytest
```

`--demo`는 합성 결과로 그래프, 병렬 실행, 검토, 로그 및 Markdown/PDF 저장을 확인합니다.
실제 검색이나 LLM을 호출하지 않으며, 결과는 항상 `needs_review`로 표시합니다.
실제 에이전트의 근거 검증과 실패 처리는 별도 fake-LLM 테스트로 검증합니다.
PDF에는 한글 TrueType 폰트가 필요합니다. 시스템의 NanumGothic 등을 자동으로 찾으며,
찾지 못하면 `PDF_FONT_PATH`에 폰트 파일의 절대 경로를 지정합니다.
Ubuntu에서는 `fonts-nanum` 패키지로 준비할 수 있습니다. 폰트가 없으면 PDF 저장 실패와 검토 필요 상태를 남깁니다.

## 실제 자료로 실행

```bash
python -m pip install -e '.[rag,dev]'
cp .env.example .env
# .env에 LLM_MODEL, LLM_API_KEY, TAVILY_API_KEY를 입력
# 확보한 TurboQuant / ITME 원문 PDF를 data/raw/에 배치
python -m scripts.build_index --raw-dir data/raw --index-path data/index
python main.py
```

LLM 제공자는 OpenAI입니다. 명세에 적힌 `GPT-5 luna` 문자열은 API 모델 식별자로 임의 변환하지 않았습니다.
사용할 정확한 모델 ID를 확인해 `LLM_MODEL`에 입력해야 합니다. 실제 API 인증과 해당 모델 접근 권한은 실제 실행에서 확인됩니다.
출처: [OpenAI 모델 목록](https://developers.openai.com/api/docs/models),
[ChatOpenAI 연결 및 구조화 출력](https://docs.langchain.com/oss/python/integrations/chat/openai).

원문 PDF와 API 키는 저장소에 포함하지 않습니다. 원문은 팀이 선정한 공식 배포처에서 직접 확보하여
`data/raw/` 아래에 놓습니다. 파일명은 자유이며 PDF 페이지와 파일 지문을 인덱스에 저장합니다.
스캔본은 텍스트 추출/OCR을 먼저 수행해야 합니다. PDF 생성일을 발행일로 추정하지 않습니다.
현재 `docs/CXL.pdf`, `docs/Turboquant.pdf`가 제공되어 있으며 생성된 인덱스는 아직 없습니다.
이 파일들을 현재 위치에서 사용하려면 다음 명령을 실행하고 `.env`에 `RAW_DIR=docs`를 설정합니다.
원본 PDF는 이 구현 작업에서 변경하지 않았습니다.

```bash
python -m scripts.build_index --raw-dir docs --index-path data/index
# .env: RAW_DIR=docs
python main.py
```
BGE-M3 모델은 첫 인덱스 생성 때 내려받으며 저장 공간과 메모리가 필요합니다.
인덱스는 준비 단계에서 한 번 만들고 에이전트 호출마다 재생성하지 않습니다.
PDF 변경 시 `python -m scripts.build_index --force`로 재생성합니다.

실제 호출은 요금이 발생할 수 있으므로 `.env`의 호출/시간/재시도 한도를 프로젝트에 맞게 설정합니다.
내장 SDK 재시도는 끄고 애플리케이션의 횟수 한도로 관리합니다.
`MAX_TOTAL_CALLS`는 에이전트·LLM·검색·원문 읽기 호출을 합한 보수적 한도입니다.
전체 시간 한도는 새 호출 시작 전 확인하며, 진행 중인 네트워크 요청은 요청별 timeout으로 종료됩니다.

## 결과와 상태

```text
outputs/{run_id}/
├── events.jsonl
├── tasks/{task_id}/attempt-{attempt}.json
├── result.json
├── report.md / draft-report.md
└── report.pdf / draft-report.pdf
```

- `completed`: 에이전트/근거/보고서 검토와 PDF SUMMARY 반 페이지 조건을 통과한 결과입니다. 최종 제출 전 사람이 확인합니다.
- `needs_review`: 불확실성, 한도 도달, 검토 실패 또는 예시 실행입니다. 초안 파일명으로 저장합니다.
- `failed`: 실행이나 저장 실패입니다. 가능한 로그와 상태를 남깁니다.
- CLI 종료 코드: 완료/정상 예시 0, 실패 1, 실제 실행의 검토 필요 2.

실행 ID가 같으면 이전 기록을 덮어쓰지 않고 실패합니다. 기본 ID는 시각과 UUID로 생성합니다.
시도 번호는 최초 실행 1이며 추가 재시도 2회라면 총 3번까지 실행합니다.
생성 인덱스, 실행 결과, 원문 자료는 Git에서 제외됩니다.

## 역할 및 파일

| 담당 | 에이전트 | 구현 |
|---|---|---|
| 시은 | 슈퍼바이저, 기술 조사 | `src/agents/supervisor.py`, `technical.py` |
| 민형 | 시장, 이해관계자 | `src/agents/market.py`, `stakeholder.py` |
| 연수 | 도메인, 근거 검증 | `src/agents/domain.py`, `verification.py` |
| 재혁 | 평가 종합, 보고서 | `src/agents/synthesis.py`, `report.py` |

공통 실행은 `main.py`, 그래프 연결은 `src/graph.py`, 입출력은 `src/schemas.py`, 공유 State는 `src/state.py`입니다.
검색은 `src/tools/`, 로그/시도별 저장은 `src/common/`, PDF는 `src/exporters/`, 프롬프트는 `prompts/`에 있습니다.

## 확정 전 설정과 범위

명세의 미확정 항목은 숨기지 않고 설정으로 분리했습니다. Python 범위와 의존성 범위, 추가 재시도 2회,
총 호출 200회/전체 1800초는 구현 기본값이며 팀 합의 후 변경할 수 있습니다.
시장·이해관계자는 기본적으로 Tavily 웹 검색과 원문 읽기를 수행합니다.
`MARKET_RAG` / `STAKEHOLDER_RAG=true`이면 이미 준비된 PDF 인덱스도 검색합니다.
웹 본문을 자동으로 벡터화하는 기능과 중단 지점 재개는 구현 범위에 포함하지 않습니다.

상세 문서: [구조](docs/directory_structure.md), [인터페이스](docs/interfaces.md),
[흐름과 종료 조건](docs/workflow.md), [구현·검증 현황](docs/implementation.md).
