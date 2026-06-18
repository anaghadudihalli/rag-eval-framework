# RAG Evaluation & Observability Framework

An automated pipeline for measuring and monitoring RAG system quality. Built to mirror the OpenSearch + LangChain production stack used at Electronic Arts, this framework closes the gap of detecting retrieval quality regressions before users notice them.

## What It Measures

| Metric | Method |
|--------|--------|
| **Retrieval Quality** | Precision@K and Recall@K against golden doc IDs |
| **Semantic Similarity** | Cosine similarity via `all-MiniLM-L6-v2` (sentence-transformers) |
| **Hallucination Rate** | LLM-as-judge (GPT-3.5-turbo) with structured JSON output |
| **Confidence Routing** | OpenAI logprobs → mean token confidence → human escalation |

## Architecture

```
Golden Dataset (30 queries)
        │
        ▼
  RAG Pipeline (LangChain + OpenSearch/ChromaDB + GPT-3.5-turbo)
        │
        ├──► Retrieval Evaluator  → Precision@K, Recall@K
        ├──► Similarity Evaluator → Cosine sim (sentence-transformers)
        ├──► Hallucination Judge  → GPT-3.5-turbo LLM-as-judge
        └──► Confidence Router    → logprobs → human escalation
                │
                ▼
         EvalReport (Pydantic v2)
                │
        ┌───────┴────────┐
        ▼                ▼
  Rich Console     JSON Report File
  (terminal)       (reports/<run_id>.json)
                         │
                         ▼
                  AlertEngine → CI exit code
```

## Vector Store

**Active: OpenSearch** (primary — matches EA production pattern)

OpenSearch 2.11 with k-NN plugin is used for vector similarity search. The system automatically falls back to **ChromaDB** if:
- OpenSearch is unreachable at startup
- `VECTOR_STORE_BACKEND=chroma` is set in `.env`

Both backends implement the same `VectorStoreBackend` interface — all metric code is backend-agnostic.

## Quickstart

### Prerequisites
- Python 3.11+
- Docker + Docker Compose (for OpenSearch)
- OpenAI API key

### 1. Clone & Install

```bash
git clone <repo>
cd rag-eval-framework
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Start OpenSearch

```bash
make docker-up
# Wait for health check: http://localhost:9200/_cluster/health
```

### 4. Ingest Knowledge Base

```bash
make ingest
```

### 5. Run Evaluation

```bash
make eval
```

A rich terminal report will display. The JSON report is saved to `reports/`.

### 6. Run Tests

```bash
make test           # unit tests only (no Docker required)
make test-all       # unit + integration
```

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make docker-up` | Start OpenSearch container |
| `make docker-down` | Stop and remove containers |
| `make ingest` | Load documents into vector store |
| `make eval` | Run full evaluation pipeline |
| `make report` | Print latest report from `reports/` |
| `make test` | Run unit tests |
| `make test-all` | Run all tests including integration |
| `make lint` | Run ruff + black check |
| `make format` | Auto-format with black |

## Thresholds & Alerts

Defaults in `rag_eval/models/thresholds.py`:

| Metric | Default Threshold |
|--------|------------------|
| Precision@K | ≥ 0.60 |
| Recall@K | ≥ 0.60 |
| Semantic Similarity | ≥ 0.75 |
| Hallucination Rate | ≤ 0.15 |
| Routing Accuracy | ≥ 0.90 |

Override via environment variables or by editing `thresholds.py`.

## CI/CD

GitHub Actions runs weekly (Monday 9am UTC) and on every push to `main`. If any metric falls below threshold, the workflow fails — preventing silent regression.

```
.github/workflows/eval.yml
```

To run manually: **Actions → RAG Eval Pipeline → Run workflow**

## Project Structure

```
rag-eval-framework/
├── rag_eval/           # Core package
│   ├── config.py       # Settings (pydantic BaseSettings)
│   ├── models/         # Pydantic v2 data models
│   ├── store/          # Vector store abstraction (OpenSearch + Chroma)
│   ├── ingest/         # Document loading and ingestion
│   ├── metrics/        # Retrieval, similarity, hallucination, routing
│   ├── pipeline/       # RAG chain + eval orchestration
│   └── reporting/      # Rich console, JSON, alert engine
├── data/
│   ├── golden_dataset.json   # 30-sample ground-truth eval set
│   └── documents/            # Knowledge base (10 JSON files)
├── tests/
│   ├── unit/           # Fast tests, no external services
│   └── integration/    # Require Docker or ChromaDB
├── docker/
│   └── docker-compose.yml
├── .github/workflows/eval.yml
└── reports/            # Generated at runtime (gitignored)
```

## Design Decisions & Deviations

1. **Confidence scoring via logprobs**: The spec mentions confidence-score routing but not the signal source. We use OpenAI's `logprobs=True` to compute `exp(mean(token_logprobs))` as the answer confidence. Falls back to cosine similarity score if logprobs are unavailable.

2. **Golden dataset domain**: Software engineering / developer tools FAQ (Git, Docker, Python packaging, CI/CD). Self-contained domain with verifiable ground-truth answers.

3. **`langchain-opensearch` package**: LangChain 0.3+ moved OpenSearch integration to `langchain-opensearch` (separate from `langchain-community`). This is the recommended import path.

4. **No RAGAS/LangSmith**: Metrics are implemented from first principles for full observability and zero external evaluation service dependencies.

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | ✅ | — | OpenAI API key |
| `OPENSEARCH_URL` | ❌ | `http://localhost:9200` | OpenSearch endpoint |
| `OPENSEARCH_INDEX` | ❌ | `rag-eval-docs` | Index name |
| `VECTOR_STORE_BACKEND` | ❌ | `opensearch` | `opensearch` or `chroma` |
| `EMBEDDING_MODEL` | ❌ | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `JUDGE_MODEL` | ❌ | `gpt-3.5-turbo` | OpenAI model for LLM-as-judge |
| `TOP_K` | ❌ | `5` | Retrieval top-K |
| `CONFIDENCE_THRESHOLD` | ❌ | `0.7` | Below this → route to human |
