# KV Cache SW vs HW

Supervisor-based Agentic RAG project for comparing:

- SW: TurboQuant-based KV Cache compression
- HW: CXL-based KV Cache storage / memory expansion

The report evaluates both technologies from technical, market, stakeholder, and cloud-domain perspectives.

## Fixed technology stack

- Paper RAG vector store: FAISS
- Paper embedding model: `BAAI/bge-m3`
- Supervisor and worker LLM: `gpt-4o-mini`
- External web search: Tavily

The paper pipeline stores only the two selected papers in a local FAISS index. The
web-research agents use Tavily for current sources, and `gpt-4o-mini` converts both
retrieved paper chunks and web results into consistent evidence cards.

## Setup

```bash
uv sync
cp .env.example .env
```

`BAAI/bge-m3`는 공개 모델이므로 Hugging Face 토큰 없이도 다운로드할 수 있지만,
다운로드 제한을 줄이려면 `.env`의 `HF_TOKEN`에 Hugging Face User Access Token을
설정하는 것을 권장한다. 토큰은 코드나 YAML에 직접 작성하지 않는다.

## Build the paper Vector DB

Place the two source papers in `data/papers/`, then run:

```bash
uv run python -m kv_cache_agent.rag.ingest_papers
```

## Run

```bash
uv run python -m kv_cache_agent.main
```

## Team ownership

- Supervisor + verifier: `agents/supervisor.py`, `agents/verifier.py`, `graph/`
- Technical + market: `agents/technical.py`, `agents/market.py`
- Stakeholder + cloud domain: `agents/stakeholder.py`, `agents/cloud_domain.py`
- Synthesis + report writer: `agents/synthesis.py`, `agents/report_writer.py`
