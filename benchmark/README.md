# Benchmark Suite

This folder contains the evaluation tools for the LLM engines.

## Setup

We recommend using [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
uv sync
```

## Running the Benchmark

The benchmark scripts measure throughput, latency, and quality metrics (Perplexity and Trace Length).

### Track 1: Agent
```bash
uv run runner_agent.py --url $ENGINE_URL --data data/track1/train.json --concurrency 16
```

### Track 2: Chat
```bash
uv run runner_chat.py --url $ENGINE_URL --data data/track2/train.jsonl --concurrency 128
```
