# Optimization Plan

> Last updated: 2026-03-17

## Current Goal

Build and optimize a high-throughput vLLM serving engine for Track 2 (Customer Support Chatbot)
targeting maximum throughput and minimum latency at 128 concurrency on a single RTX 5080.

## Hypothesis

The biggest performance gains will come from (in order):
1. **FP8 quantization** — halves model memory, freeing VRAM for more concurrent KV caches
2. **FP8 KV cache** — halves KV cache memory, critical for 128 concurrency
3. **Reduced max_model_len** — inputs are 6-92 chars, outputs max 256 tokens → 1024 is plenty
4. **Exact-match response cache** — 36.1% of training queries are duplicates; temperature=0 makes caching lossless
5. **APC + chunked prefill** — built-in vLLM features for prefix sharing and P99 reduction
6. **Disabling Qwen3 thinking mode** — prevents wasted tokens on `<think>` blocks

## Key Data Insights

- 13,435 total benchmark queries, 8,582 unique (36.1% duplicate rate)
- All single-turn: `[{"role": "user", "content": "..."}]` — no multi-turn sessions
- Benchmark uses `temperature=0` (greedy/deterministic), `max_tokens=256`
- Input lengths: 6-92 characters (~15-40 tokens after tokenization)
- Response format: `{"output": "...", "logprobs": [...]}` (NOT OpenAI format)
- Benchmark computes P99 (not P95 as spec says)

## Architecture

```
Incoming Request (conc. 128)
    │
    ▼
┌─────────────────┐
│  Exact-Match     │──cache hit──▶ Return cached response + logprobs
│  Response Cache  │
└────────┬────────┘
         │ cache miss
         ▼
┌─────────────────────────────────────────────────┐
│  vLLM AsyncLLMEngine                             │
│  ┌────────────────────────────────────────────┐  │
│  │ Model: Qwen3-4B-Instruct-2507 (FP8)       │  │
│  │ KV Cache: FP8, max_model_len=1024          │  │
│  │ Attention: FlashInfer                       │  │
│  │ APC: enabled (shared prefix caching)        │  │
│  │ Chunked Prefill: enabled (P99 reduction)    │  │
│  │ max_num_seqs: 256                           │  │
│  │ gpu_memory_utilization: 0.95                │  │
│  └────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
         │
         ▼
    Return response + logprobs (cache the result)
```

## Optimizations Implemented

### Tier 1 — High Impact, Low Risk

| # | Optimization | Mechanism | Expected Impact |
|---|-------------|-----------|----------------|
| 1 | FP8 model quantization | `quantization="fp8"` | 2x model memory reduction → more KV cache room |
| 2 | FP8 KV cache | `kv_cache_dtype="fp8"` | 2x KV cache memory reduction → 128+ concurrent seqs |
| 3 | Reduced max_model_len | `max_model_len=1024` (from 8192) | 8x less KV cache per seq → fits more concurrent seqs |
| 4 | Exact-match response cache | App-layer SHA256 cache | ~36% requests skip inference entirely |
| 5 | Disable Qwen3 thinking | `enable_thinking=False` in chat template | Prevents wasted tokens on `<think>` blocks |

### Tier 2 — Medium Impact, Low Risk

| # | Optimization | Mechanism | Expected Impact |
|---|-------------|-----------|----------------|
| 6 | Automatic Prefix Caching | `enable_prefix_caching=True` | Reuses KV blocks for shared system prompt prefix |
| 7 | Chunked prefill | `enable_chunked_prefill=True` | Interleaves prefill/decode → reduces P99 latency |
| 8 | GPU memory utilization | `gpu_memory_utilization=0.95` (from 0.9) | 5% more VRAM for KV cache |
| 9 | Batch size tuning | `max_num_seqs=256` | Allows efficient batching of 128+ concurrent requests |

### Tier 3 — Exploration (Saved for Later)

| # | Optimization | Notes |
|---|-------------|-------|
| 10 | Speculative decoding (n-gram) | No extra model; may help decode speed for formulaic responses |
| 11 | max_num_batched_tokens tuning | Controls prefill chunk size, affects throughput/latency tradeoff |
| 12 | Swap space tuning | More CPU swap for KV cache overflow |

### Deprioritized

| # | Optimization | Reason |
|---|-------------|--------|
| - | Semantic/fuzzy cache | Risk of incorrect logprobs → perplexity degradation |
| - | Session-aware scheduler | Benchmark is single-turn, no sessions to track |
| - | Session-TTL KV eviction | No sessions in the benchmark |
| - | CPU KV cache offload | Adds latency, complex implementation |

## Experiment Log

| # | Date | Config Change | P50 (ms) | P99 (ms) | Throughput (req/s) | Perplexity | Verdict |
|---|------|--------------|----------|----------|-------------------|------------|---------|
| 0 | — | baseline (starter kit, no opts) | — | — | — | — | pending |
| 1 | — | full optimized build (all Tier 1+2) | — | — | — | — | pending |

## Discoveries & Surprises

- **36.1% duplicate rate** in training data — exact-match caching is extremely high impact
- **Benchmark is single-turn** (not multi-turn) — session-based optimizations from proposal are irrelevant
- **Benchmark uses P99** (code) not P95 (spec doc) — target P99 reduction
- **temperature=0 always** — makes caching lossless (deterministic outputs)
- **max_tokens=256** is set by benchmark, not configurable from engine side

## Key Techniques & Skills

- FP8 quantization on RTX 5080 (Blackwell) for inference memory savings
- FlashInfer attention backend for CUDA 12.8 compatibility
- Application-layer response caching with SHA256 keys
- vLLM AsyncEngineArgs tuning for high-concurrency serving

## Decisions

- **FP8 over INT4/AWQ**: FP8 has minimal quality loss and native hardware support on RTX 5080
- **max_model_len=1024**: Conservative but safe; can reduce to 512 if memory is tight
- **Skip semantic cache**: Too risky for perplexity; exact-match already covers 36% of requests
- **Skip session-aware scheduling**: No sessions in the actual benchmark

## Dead Ends

- Session-aware scheduler — benchmark has no multi-turn sessions
- Session-TTL KV eviction — same reason
- Semantic/fuzzy caching — risks perplexity degradation for uncertain hit rate improvement

## Next Steps

1. Deploy on Modal (L4 GPU) and test that the engine starts and serves requests
2. Run benchmark against Modal deployment to get initial numbers
3. Deploy on Vast.ai RTX 5080 with Docker image for final benchmarking
4. Tune parameters if needed (max_model_len, max_num_seqs, gpu_memory_utilization)
5. Try speculative decoding (n-gram) if decode is the bottleneck
6. Final Docker image push to GHCR for submission
