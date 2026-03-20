# Memory Agent (LangGraph + Redis + Chroma)

This project implements a 4-node memory/retrieval agent using **LangGraph**:

1. **Intent**: rewrite/classify the user query
2. **Retrieval**: semantic search in **ChromaDB** (papers/benchmark haystack chunks)
3. **Inference**: call an LLM to answer using the retrieved context (and in-session chat history)
4. **Memory store**: persist the turn to **Redis** (conversation history), so later turns can use it

It also includes:
- A **one-time embedding/prebuild step** that writes benchmark haystack chunks into Chroma collections
- A **single-query test script** that runs exactly one query against a prebuilt Chroma collection
- A **full benchmark runner** (optionally using the prebuilt collections)

---

## Architecture Overview

The agent is a LangGraph `StateGraph` with state passed between nodes:
- `AgentState` (see `agents/state.py`)

### Nodes (in execution order)

1. `agents/intent_agent.py`
   - Produces:
     - `rewritten_query`
     - `query_type` (`factual|comparative|exploratory`)
     - `keywords`
2. `agents/retrieval_agent.py`
   - Embeds the query (two-pass semantic retrieval using both `query` and `rewritten_query`)
   - Queries the configured Chroma collection
   - Produces:
     - `retrieved_chunks`
     - `retrieval_context` (formatted text snippets)
3. `agents/inference_agent.py`
   - Calls the configured LLM (vLLM-compatible or Azure OpenAI)
   - Produces:
     - `answer`
4. `agents/memory_store_agent.py`
   - Appends the turn to Redis conversation history as a Redis `LIST`
   - Produces:
     - updated `conversation_history`
     - `stored_memory_id`

LangGraph wiring:
- `graph/pipeline.py`

---

## Repository Layout

Key folders/files:
- `agents/`
  - `state.py` — `AgentState` contract + defaults
  - `intent_agent.py` — Node 1
  - `retrieval_agent.py` — Node 2
  - `inference_agent.py` — Node 3
  - `memory_store_agent.py` — Node 4
- `graph/pipeline.py` — builds the 4-node LangGraph pipeline (Redis checkpointer)
- `memory/`
  - `embedder.py` — lazy-loaded sentence-transformers embeddings
  - `chroma_client.py` — Chroma HTTP client helpers
  - `redis_history.py` — Redis LIST append helpers
- `benchmark/`
  - `loader.py` — loads `ai-hyz/MemoryAgentBench` and extracts haystack chunks
  - `build_chroma.py` — one-time prebuild: embed + upsert haystack chunks into Chroma
  - `runner.py` — benchmark evaluation runner (supports prebuilt mode)
- `scripts/`
  - `single_prebuilt_query_test.py` — runs exactly one prebuilt-query test (with optional node traces)
  - `quick_benchmark_rag_test.py` — legacy debug smoke test (not required for normal usage)
  - `inspect_benchmark_split.py` — checks how many non-empty examples exist in a split
- `utils/`
  - `llm_client.py` — creates the shared LLM client (vLLM vs Azure)
  - `console_trace.py` — optional terminal tracing for each node
- `main.py`
  - CLI entry: `--mode repl` or `--mode benchmark`
- `docker-compose.yml`
  - Redis Stack (RedisJSON required by LangGraph checkpointing)
  - ChromaDB

---

## Setup

### 1) Python dependencies

From `memory-agent/`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Start infrastructure (Redis + Chroma)

From the project root (where `docker-compose.yml` lives):

```bash
docker compose up -d
```

### 3) Environment variables

Copy the template:

```bash
cp .env.example .env
```

Important:
- Do **not** commit `.env` (it is in `.gitignore`).
- Update LLM + provider variables (`LLM_PROVIDER`, vLLM/Azure fields).
- Update connection variables (`REDIS_URL`, `CHROMA_HOST`, `CHROMA_PORT`, etc.).

---

## One-time: Build Chroma embeddings (prebuilt collections)

The prebuild script:
- Loads benchmark examples
- Extracts haystack chunks
- Embeds + upserts them into **per-example** Chroma collections

Collection naming format:
- `${CHROMA_COLLECTION_PAPERS}__${BENCHMARK_SPLIT}__${example_id}`

Build the first 5 non-empty examples (recommended for quick iteration):

```bash
python benchmark/build_chroma.py --split Accurate_Retrieval --max-examples 5 --embed-batch-size 128
```

Notes:
- `--max-examples` counts **non-empty haystack examples** (skips empty ones).
- This step is the expensive part; you typically do it once per dataset/split.

---

## Run a single query (prebuilt) with node-by-node logs

This script runs exactly one retrieval+inference call against a prebuilt Chroma collection.

Random prebuilt example + random question:

```bash
TOP_K_CHUNKS=10 \
python scripts/single_prebuilt_query_test.py --split Accurate_Retrieval --trace
```

Pick a specific example/question:

```bash
TOP_K_CHUNKS=20 \
python scripts/single_prebuilt_query_test.py --split Accurate_Retrieval --example-id 17 --question-index 40 --trace
```

What you get:
- The printed `thread_id` (useful for LangSmith filtering)
- Node steps (Intent / Retrieval / Inference / Redis write) when `--trace` is used
- `retrieval_context` preview + timings

To enable traces from any script, you can also set:
- `AGENT_CONSOLE_TRACE=1`

---

## Full benchmark (evaluation)

### 1) Default (self-contained ingestion; slower)

```bash
python main.py --mode benchmark --split Accurate_Retrieval
```

### 2) Fast benchmark using prebuilt Chroma collections

```bash
BENCHMARK_USE_PREBUILT_CHROMA=1 \
python main.py --mode benchmark --split Accurate_Retrieval --max-examples 5
```

### Optional extra logging (useful when it looks “silent”)

```bash
BENCHMARK_LOG_EVERY=1 \
BENCHMARK_USE_PREBUILT_CHROMA=1 \
python main.py --mode benchmark --split Accurate_Retrieval --max-examples 5
```

---

## Redis conversation history (Node 4)

Node 4 writes each turn to Redis as a LIST:
- Key: `REDIS_CONV_PREFIX + session_id` (default prefix `conv:`)
- Each element is a JSON object with:
  - `query`, `answer`, `rewritten_query`, `query_type`, `keywords`, `timestamp`

This same `session_id` is also used by LangGraph checkpointing (checkpoint saver).

---

## LangSmith tracing (optional)

Enable by setting in `.env`:
- `LANGCHAIN_TRACING_V2=true`
- `LANGCHAIN_API_KEY=...`
- `LANGCHAIN_PROJECT=...`

In this codebase, the benchmark and the scripts set `thread_id` in the run configuration. Use that `thread_id` to filter traces in the LangSmith UI.

---

## Development Notes / Recommendations

- For VM testing, prefer:
  1. `benchmark/build_chroma.py` (prebuild once)
  2. `scripts/single_prebuilt_query_test.py --trace` (fast iteration)
- Tuning knob:
  - `TOP_K_CHUNKS` controls retrieval depth; higher values improve recall but increase context length and inference time.

---

## GitHub: Publish this code

This repo folder may not be a git repository yet. From the repository root (`Simple_Research_agent/`), run:

```bash
# 1) Initialize git
git init

# 2) Ensure you have a .gitignore (memory-agent/.gitignore already ignores .env and other local artifacts)

# 3) Add code only (recommended: add the whole memory-agent folder)
git add memory-agent

# 4) Commit
git commit -m "Add memory agent (LangGraph + Redis + Chroma) with benchmark and prebuild scripts"

# 5) Add your GitHub remote (replace with your repo URL)
git branch -M main
git remote add origin https://github.com/<YOUR_USERNAME>/<YOUR_REPO>.git

# 6) Push
git push -u origin main
```

If you want, you can share your intended GitHub repo name (no secrets) and I can tailor the commands precisely.

