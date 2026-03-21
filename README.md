# Memory Agent (LangGraph + Redis + Chroma)

Slim memory-agent flow for VM usage:
1. Load HF dataset and ingest haystack sessions into Chroma (one-time).
2. Run the 4-node agent on one dataset question.
3. Print response + latency metrics + ground truth and log JSONL.

## What the system does

The LangGraph pipeline is:
- `intent` -> query rewrite/classification
- `retrieval` -> Chroma semantic search over embedded haystack chunks
- `inference` -> streamed LLM answer generation (TTFT/TTLT measured)
- `memory_store` -> store turn in Redis conversation history

Redis is also used for LangGraph checkpoints (state persistence by `thread_id`).

## Repository layout

- `agents/` - node implementations + `AgentState`
- `graph/pipeline.py` - graph wiring + Redis checkpointer
- `memory/` - Chroma client and embedding helpers
- `benchmark/build_chroma.py` - one-time ingest to Chroma
- `metrics/logger.py` - append-only JSONL logging
- `main.py` - run one dataset query or all questions for one example

## Setup

From `memory-agent/`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Start infra:

```bash
docker compose up -d
```

## One-time ingest: haystack -> embeddings -> Chroma

Each non-empty dataset example is stored in its own collection:
`{CHROMA_COLLECTION_PAPERS}__{split}__{example_id}`

Example:

```bash
python benchmark/build_chroma.py --split Accurate_Retrieval --max-examples 5 --embed-batch-size 128
```

Ingest the full split (all non-empty examples):

```bash
python benchmark/build_chroma.py --split Accurate_Retrieval --max-examples 0 --embed-batch-size 128
```

## Run one dataset query with full metrics

Random non-empty prebuilt example and random question:

```bash
TOP_K_CHUNKS=10 \
python main.py --split Accurate_Retrieval --trace
```

Specific example and question:

```bash
TOP_K_CHUNKS=10 \
python main.py --split Accurate_Retrieval --example-id 18 --question-index 10 --trace
```

Run all questions for one example (each question gets its own `session_id`/`thread_id`):

```bash
TOP_K_CHUNKS=10 \
python main.py --split Accurate_Retrieval --example-id 18 --all-questions --trace
```

Output includes:
- query, predicted answer, ground truth
- `T_time_to_first_token`, `T_time_to_last_token`
- `T_chroma_retrieve`, `T_redis_store`
- cumulative `T_chroma_retrieve_cumulative`, `T_redis_store_cumulative`
- `(node, database)` latency breakdown entries

When `--all-questions` is used, metrics are still logged per question/session as separate JSONL entries.

All runs are appended to `metrics_log.jsonl` (or `METRICS_LOG_PATH`).


