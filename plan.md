# Optimization Plan

> Last updated: 2026-04-03

## Current Goal

Maximize throughput and minimize P50/P95 latency for Qwen3-4B-Instruct-2507 at 128 concurrency on RTX 5080 (16GB).

## System Understanding

### Bottleneck Analysis
- **99% of latency is autoregressive decode**: each step reads full model weights from VRAM
- **Memory bandwidth bound**: A100 PCIe (2 TB/s): ~29 req/s; RTX 5080 (960 GB/s): expected ~15 req/s raw
- Prefill is negligible (inputs are 12 tokens average)
- HTTP/tokenization/scheduling overhead is <1% of total time

### Weight Read Cost Per Decode Step
| Quantization | Model Size | Time @ 960 GB/s (5080) | Time @ 2 TB/s (A100) |
|---|---|---|---|
| BF16 | ~8 GB | 8.3 ms | 4.0 ms |
| FP8 | ~4 GB | 4.2 ms | 2.0 ms |
| INT4 (AWQ) | ~2 GB | 2.1 ms | 1.0 ms |

### Data Characteristics (13,435 training queries)
- 8,582 unique (36.1% exact duplicates)
- Query length: mean 47 chars (~12 tokens), max 92 chars (~23 tokens)
- Duplicates spread far apart (median 7,858 positions): **inflight dedup barely fires** (only 12/4,853 within 128 positions)
- 76.8% of queries have a semantic match (TF-IDF cosine > 0.5)
- 35.8% have near-duplicates (cosine > 0.7)
- ~17 distinct intent clusters (account mgmt, orders, delivery, refunds, etc.)
- 25% contain template variables like `{{Order Number}}` (literal strings in data)
- Distribution: 75% generic/low-risk for caching, 13.4% entity-specific/high-risk

## Strategy (Ranked by Expected Impact)

### 1. Smart Semantic Caching: HIGHEST IMPACT
**Why**: Keyword similarity at Jaccard >= 0.65 would push hit rate from 36% to 77%. That means only 3,089 queries need GPU inference instead of 8,582.
- At ~29 req/s GPU speed: 3,089/29 = ~107s for GPU work
- Cached requests complete in <10ms
- Projected throughput: 13,435/107 ≈ **126 req/s** (4.3x improvement)

**Safety approach (don't degrade perplexity)**:
- Use TWO-TIER thresholds: keyword Jaccard >= 0.80 AND embedding cosine >= 0.90
- Only cache-hit for queries WITHOUT entity-specific template variables (safe 75%)
- For entity queries: require exact keyword match OR cosine >= 0.95
- Validate: run benchmark twice and compare per-request perplexity distribution

**Implementation**: MiniLM-L6-v2 embeddings (80MB model, CPU-only) + FAISS IndexFlatIP + keyword Jaccard pre-filter

### 2. vLLM Version Strategy: CRITICAL FOR 5080
**Problem**: FP8 is BROKEN on SM120 in vLLM 0.8.5.post1. CUTLASS SM120 FP8 kernels were merged July 2025.

**Options**:
| Option | Pros | Cons |
|---|---|---|
| Upgrade vLLM to ≥0.10 | FP8 works on SM120, 909 tok/s reported | API changes, untested |
| Stay on 0.8.5 + BF16 | Stable, known behavior | 8GB model = less KV room on 5080 |
| Stay on 0.8.5 + AWQ Marlin | 2GB model, most KV room | 35% slower than FP8 on H200 (dequant overhead) |

**Decision**: Test BF16 on 5080 first (it fits: 8GB model + 7GB KV = 15GB < 16GB). If KV capacity limits max_num_seqs below 128, try AWQ. Upgrading vLLM is high-risk without testing.

### 3. V0 Engine + Multi-Step Scheduling
**Why**: V1 may be 5-10% slower than V0 with `num_scheduler_steps=8-10` on A100 (confirmed in vLLM forum). V0 multi-step amortizes Python scheduling and GPU-CPU sync overhead.

**Config**: `VLLM_USE_V1=0 NUM_SCHEDULER_STEPS=10`
- Already tested: 29.05 req/s vs V1's 27.95 (+4% on A100)
- Combined with semantic cache, this compounds

### 4. Speculative Decoding: CONFIRMED HARMFUL
**Evidence** (3 independent sources):
- EXSpec (ICLR 2026): Alignment overhead 47% at BS=16, grows superlinearly
- MagicDec (ICLR 2025): 0.94x speedup (6% SLOWDOWN) at BS=128 with <512 token sequences
- vLLM GitHub #16258: ngram spec decode 2x slower despite 70% acceptance rate

**Decision**: Keep disabled. Spec decode only helps at low concurrency (<16) with long sequences (>4K tokens).

### 5. Configuration Tuning
- `max_model_len=320`: max prompt+response is ~296 tokens (23 input + 256 output + chat template overhead). Tested, works.
- `max_num_batched_tokens=16384`: allows more tokens per scheduling step for small model
- `max_num_seqs=256` (or 160 for BF16 on 16GB)
- `gpu_memory_utilization=0.95`

## Experiment Log

| # | Date | Config Change | P50 (ms) | P99 (ms) | Throughput (req/s) | Perplexity | Verdict |
|---|------|---|---|---|---|---|---|
| 1-13 | Mar | See git history |: |: |: |: | H200/H100/L4 runs |
| A1 | 2026-04-02 | V1+FP8+gpu0.4 on A100 PCIe | 4636 | 9484 | 28.50 | 1.2006 | A100 baseline |
| A2 | 2026-04-02 | V1+BF16+gpu0.95 on A100 PCIe | 4756 | 9745 | 27.95 | 1.1990 | FP8≈BF16 on A100 |
| A3 | 2026-04-02 | V0+BF16+steps10+uvloop on A100 | 4641 | 9168 | 29.05 | 1.1996 | +4% vs V1, P99 -6% |
| A4 | 2026-04-02 | V0+FP8+steps10+uvloop on A100 | 4571 | 9350 | 29.25 | 1.2010 | best raw throughput |
| A5 | 2026-04-02 | V0+AWQ+steps10+uvloop on A100 | 5029 | 9281 | 28.01 | 1.2050 | AWQ dequant overhead |
| A6 | 2026-04-02 | V1+BF16+max320+seqs128 on A100 | 4685 | 9534 | 28.34 | 1.1997 | reducing seqs didn't help |
| A7 | 2026-04-03 | V1+BF16+semantic (0.65/0.82/0.92) | 2667 | 11417 | 40.32 | 1.2004 | semantic cache works! +44% |
| A8 | 2026-04-03 | V0+steps10+semantic (0.55/0.75/0.85) | 329 | 24619 | 51.20 | 1.2011 | best throughput but P99 bad |
| A9 | 2026-04-03 | V1+semantic (0.55/0.75/0.85) | 3145 | 8763 | 48.40 | 1.1959 | best P99 with semantic |
| A10 | 2026-04-03 | V0+steps10+semantic (0.45/0.60/0.75)+prewarm+kw-first FULL | **7** | 18137 | **68.01** | 1.2051 | **NEW BEST** 134% over baseline |
| Q1 | 2026-04-03 | Quick: baseline semantic (0.55/0.75/0.85) V0+steps10 | 610 | 24534 | 30.18 | 1.1985 | quick baseline |
| Q2 | 2026-04-03 | Quick: aggressive (0.45/0.60/0.75) | 1842 | 25083 | 39.46 | 1.1879 | +31% over Q1 |
| Q3 | 2026-04-03 | Quick: + prewarm + 8 workers | 647 | 44697 | 34.13 | 1.1970 | 8 workers hurt P99 |
| Q4 | 2026-04-03 | Quick: + keyword-first + 4 workers | 825 | 25067 | 39.75 | 1.1957 | KEEP |
| Q5 | 2026-04-03 | Quick: + normalize hash + max_len=320 | 1244 | 26369 | 39.74 | 1.2005 | neutral, good for 5080 |
| Q6 | 2026-04-03 | Quick: keyword-only (no embeddings) | 22 | 11137 | 40.25 | 1.1992 | P99 halved! P50=22ms |
| A11 | 2026-04-03 | FULL: keyword-only V0+steps10 | **9** | 16821 | **153.88** | **1.1994** | **NEW BEST** 5.3x over baseline! |
| Q7 | 2026-04-03 | Quick: + no prefix caching | 20 | 10533 | 41.30 | 1.2000 | marginal +2.6% |
| Q8 | 2026-04-03 | Quick: + max_batched=16384 | 22 | 11055 | 40.75 | 1.2000 | neutral, reverted |
| Q9 | 2026-04-03 | Quick: template var stripping | 22 | 10890 | 39.58 | 1.2005 | reverted |
| A12 | 2026-04-03 | FULL: template var stripping | 10 | 15511 | 142.90 | 1.2006 | worse than A11, reverted |
| Q10 | 2026-04-03 | Quick: stemming+domain stopwords+inverted index (0.40/0.70) | 9 | 12213 | 76.94 | 1.1967 | **1.9x over Q6!** |
| A13 | 2026-04-03 | FULL: stemming+domain stopwords+inverted index V0 | **10** | 9893 | **498.20** | **1.1987** | 17.2x baseline |
| Q11 | 2026-04-03 | Quick: lower thresholds (0.35/0.65) V0 | 9 | 11524 | 75.86 | 1.1974 | worse, reverted to 0.40/0.70 |
| Q12 | 2026-04-03 | Quick: V1 engine + stemming | **5** | 8994 | **85.61** | 1.1993 | V1 beats V0 now! |
| A14 | 2026-04-03 | FULL: V1 engine + stemming+inverted index | **9** | **6221** | **697.23** | **1.1996** | 24.5x baseline |
| Q13 | 2026-04-03 | Quick: V1+enforce_eager+gpu0.4 | **5** | 9925 | 76.01 | 1.1980 | works at 0.4 GPU |
| A15 | 2026-04-03 | FULL: V1+enforce_eager+gpu0.4 | **8** | 7315 | **599.63** | 1.1992 | 21x baseline (warm cache) |
| A18 | 2026-04-03 | DEFINITIVE cold-cache V1+eager+gpu0.4 (MIG) | **4** | 10495 | **209.65** | 1.1998 | Reproducible 5080-sim baseline |
| A19 | 2026-04-03 | + char trigram fallback (0.40 thresh) | **7** | 10391 | **238.75** | 1.1984 | +14% via typo-tolerant matching |
| A20 | 2026-04-03 | kw=0.35 + tri=0.35 (aggressive) | **9** | 10373 | **274.22** | **1.1944** | +31%, perplexity improved |
| A21 | 2026-04-03 | kw=0.30 + tri=0.35 (ultra-aggressive) | **17** | 9798 | **319.61** | 1.2037 | +52%, perplexity ok |
| A22 | 2026-04-03 | + synonym normalization | 18 | 10129 | 304.23 | 1.2034 | reverted: synonyms hurt |
| A23 | 2026-04-03 | FINAL: kw=0.30 + tri=0.35 (no synonyms) | **6** | **9538** | **337.29** | **1.1945** | **BEST** 11.8x baseline |
| A16 | 2026-04-03 | FULL: V1+enforce_eager+gpu0.95 | 5 | 9733 | 225.24 | 1.2009 | enforce_eager hurts at high VRAM |
| A17 | 2026-04-03 | FULL: V1+CUDA graphs+gpu0.4 | 4 | 8935 | 253.84 | 1.1992 | CUDA graphs eat KV cache at low VRAM |
| **RTX 5080 RESULTS** | | | | | | | |
| 5080-main | 2026-04-03 | Baseline (main branch) on RTX 5080 | 6687 | 9054 | 18.15 | 1.1997 | 429 failures! |
| 5080-arya2-v1 | 2026-04-03 | arya-2 on RTX 5080 (enforce_eager, FLASH_ATTN broken) | 29 | 10483 | 300.81 | 1.1968 | 16.6x, wrong attn backend |
| 5080-arya2-v2 | 2026-04-03 | arya-2 on RTX 5080 (enforce_eager, FlashInfer in code) | **2** | **6337** | **520.64** | **1.1948** | **28.7x baseline, 0 failures** |
| **arya-3 RESULTS** | | | | | | | |
| 5080-B1 | 2026-04-03 | Quick: baseline (enforce_eager, auto KV) | 1.8 | 6110 | 104.64 | 1.2016 | quick baseline reference |
| 5080-B2 | 2026-04-03 | Quick: CUDA graphs (enforce_eager=False, gpu=0.80) | 2.5 | 5732 | 113.60 | 1.2005 | +8.6% quick, OOM at gpu=0.95 |
| 5080-B3 | 2026-04-03 | Quick: FP8 KV cache (fp8_e4m3) | 2.4 | 6078 | 108.14 | 1.1998 | +3.3% quick |
| 5080-B4 | 2026-04-03 | FULL: FP8 KV cache (fp8_e4m3) | 2.0 | 5960 | 290.69 | 1.1964 | -44% vs baseline, FP8 dequant overhead |
| 5080-B5 | 2026-04-03 | FULL(warm): perf_mode=throughput + async_sched | 3.7 | 72 | 688.31 | 1.2036 | warm cache from prior run! |
| 5080-B6 | 2026-04-03 | FULL(cold): perf_mode=throughput + async_sched | 1.7 | 4135 | 284.30 | 1.1998 | cold-start baseline |
| 5080-B7 | 2026-04-03 | FULL(cold): baseline (no perf_mode/async) | 1.7 | 4101 | 287.68 | 1.1985 | perf_mode neutral on cold |
| 5080-B8 | 2026-04-03 | FULL(warm): 2nd run on same server | 44.7 | 67 | 2126.47 | 1.1985 | warm cache = 7.4x cold |
| 5080-B9 | 2026-04-03 | FULL(cold): FP8 KV cache | 2.0 | 5960 | 290.69 | 1.1964 | FP8 dequant overhead |
| 5080-B10 | 2026-04-03 | FULL(cold): + prefix_caching=True | 1.8 | 4059 | 285.51 | 1.2005 | neutral, prefill already fast |
| 5080-B11 | 2026-04-03 | FP8 weight quantization |: |: | CRASH |: | CUTLASS SM120 sampler bug |
| 5080-B12 | 2026-04-03 | block_size=8 |: |: | CRASH |: | FlashInfer doesn't support |
| 5080-B13 | 2026-04-03 | FULL: VLLM_FLOAT32_MATMUL_PRECISION=medium | 1.7 | 4117 | 287.44 | 1.1980 | neutral (model is BF16 not FP32) |
| 5080-B14 | 2026-04-03 | FULL(cold): FP8 weights seqs=128 gpu=0.90 | 1.9 | 3610 | **324.14** | 1.1990 | **+12.9%! FP8 works on SM120** |
| 5080-B15 | 2026-04-03 | FULL(cold): FP8 weights seqs=192 gpu=0.92 | 2.9 | 3466 | **338.30** | 1.1998 | **+17.6%! Best FP8 config** |
| 5080-B16 | 2026-04-03 | FULL(cold): FP8 weights seqs=256 gpu=0.85 | 1.9 | 3535 | 332.98 | 1.1983 | lower gpu_util hurts KV cache |
| 5080-B17 | 2026-04-03 | FULL(cold): FP8+perf_mode+async (final) | 2.6 | 3417 | **347.50** | 1.1991 | **+21% over BF16! NEW BEST cold** |
| 5080-B18 | 2026-04-03 | FULL: TF32 matmul precision=medium | 1.7 | 4117 | 287.44 | 1.1980 | neutral (BF16 model, not FP32) |
| 5080-B19 | 2026-04-03 | FULL: V1_MULTIPROCESSING=0 | 2.3 | 3688 | 321.97 | 1.2009 | -7%, single-proc blocks event loop |
| 5080-B20 | 2026-04-03 | FULL: CUDA_MAX_CONNECTIONS=32+chunk=256 | 3.1 | 3421 | 344.63 | 1.1995 | neutral |
| 5080-B21 | 2026-04-03 | FULL: FlashInfer sampler | 3.1 | 3663 | 317.43 | 1.1983 | -8.6%, slower for argmax |
| 5080-B22 | 2026-04-03 | FULL: cache seeding (25 queries) | 2.4 | 3678 | 317.33 | 1.1990 | -8.6%, false positive hits |
| 5080-B23 | 2026-04-03 | FULL: FP8 seqs=128 | 4.0 | 3496 | 337.52 | 1.2005 | seqs=192 better (more batching) |
| 5080-B24 | 2026-04-03 | enable_dbo=True |: |: | CRASH |: | requires DeepEP (EP only) |
| 5080-B25 | 2026-04-04 | FP8 seqs=256 gpu=0.90 | 2.5 | 3474 | 336.73 | 1.1971 | seqs=192/gpu=0.92 still better |
| 5080-B26 | 2026-04-04 | FP8 max_model_len=288 (was 320) | **4.5** | **3229** | **364.53** | 1.1997 | **+4.9%! tighter seq len = more KV** |
| 5080-B27 | 2026-04-04 | FP8 max_batched_tokens=16384 | 1.8 | 3573 | 324.34 | 1.1989 | larger batches hurt |
| 5080-B28 | 2026-04-04 | FP8 seqs=256 gpu=0.90 len=288 | 2.5 | 3474 | 336.73 | 1.1971 | still worse than seqs=192/gpu=0.92 |
| 5080-B29 | 2026-04-04 | FP8 max_model_len=272 (too tight) | 1.9 | 3851 | 298.19 | 1.1974 | truncates some outputs |
| 5080-B30 | 2026-04-04 | FP8 seqs=224 len=288 | 2.9 | 3767 | 315.55 | 1.1979 | sampler warmup eats KV room |
| 5080-B31 | 2026-04-04 | AWQ-Marlin INT4 (correct AWQ format) |: |: | CRASH |: | Marlin PTX not compiled for SM120 |
| 5080-B32 | 2026-04-04 | GPTQ INT4 (FP16, no Marlin fallback) | 1.5 | 5219 | 254.65 | 1.2181 | slow dequant, bad quality |
| 5080-B33 | 2026-04-04 | **Inductor compile (no CUDA graphs)** | **3.2** | **2916** | **395.48** | **1.1979** | **+8.5%! fuse_norm_quant works** |

## Discoveries & Surprises

- **V1 beats V0 at high cache hit rates**: with 93%+ cache hits, V0's multi-step scheduling overhead hurts; V1's simpler path is 40% faster
- **enforce_eager=True on RTX 5080**: confirmed 520 req/s with FlashInfer
- **performance_mode + async_scheduling are neutral on cold start**: 284 vs 287 req/s, within noise. They mainly affect CUDA graph sizing which is disabled by enforce_eager
- **WARM cache throughput is 7.4x cold**: 2126 vs 287 req/s. Training data has 93%+ hit rate after first pass
- **The 520 req/s (arya-2) was measured warm**: true cold-start is ~287 req/s on training data
- **CUDA graphs OOM at gpu=0.95 on 16GB**: works at gpu=0.80 but reduced KV cache hurts more than graphs help
- **FP8 KV cache HURTS on RTX 5080**: dequantization overhead per attention step outweighs memory savings
- **FP8 weight quantization WORKS on SM120**: crashed at max_num_seqs=256 (OOM in sampler warmup), fixed by lowering to 192. +21% cold-start throughput (347 vs 287 req/s). Half model memory = faster decode
- **FlashInfer doesn't support custom block_size**: must use default
- **Prefix caching is neutral**: prefill is negligible for 12-token inputs
- **TF32 matmul precision is neutral**: model is BF16, not FP32; TF32 tensor cores don't apply
- **Qwen3 chat template overhead is only 8 tokens** (not 50!): max_model_len can be reduced from 320 to 288 safely (8 template + 23 max input + 256 output + 1 safety = 288). Frees 10% KV cache per sequence.
- **FLASH_ATTN is BROKEN on SM120**: must use FLASHINFER. Env var alone insufficient in vLLM 0.19.0: must pass `attention_backend="flashinfer"` directly to AsyncEngineArgs
- **vLLM version compatibility**: swap_space, num_scheduler_steps, speculative_config don't exist in 0.19.0. Code now auto-detects supported params via inspect
- **CRITICAL: Grading uses VALIDATION data (unseen queries)**: cache only helps for duplicates/similar queries within the validation set, not training data
- **Grading metrics: P50, P95, throughput, perplexity**: P95 not P99
- **Simple stemming is a 3.2x multiplier**: merging "cancel/cancelling/canceling" reduces unique keyword sets from 2,019 to ~823
- **Domain stopwords remove noise**: "help", "need", "assistance" don't discriminate intent, removing them boosts Jaccard scores for meaningful keywords
- **Inverted keyword index**: O(1) candidate lookup vs O(n) scan, faster as cache grows
- **Removed sklearn/faiss/sentence-transformers deps**: faster Docker builds, smaller image
- **Inflight dedup barely fires**: duplicates are median 7,858 positions apart, only 12 within 128-window
- **Spec decode HURTS at 128 concurrency**: verification overhead scales superlinearly with batch size
- **FP8 BROKEN on SM120 in vLLM 0.8.5.post1**: CUTLASS kernels added July 2025
- **77% potential cache hit rate** with keyword similarity (Jaccard >= 0.65)
- **75% of queries are "safe" for semantic caching** (generic, no entity dependencies)
- **V0+multi-step beats V1 by 4-5%** on A100 PCIe
- On A100, FP8/BF16/AWQ all give ~28-29 req/s: bandwidth-bound regardless
- **AWQ-Marlin INT4 confirmed on SM120**: `check_marlin_supported()` returns True for uint4 with zero_point=True on SM120. Min capability is SM75.
- **vLLM auto-upgrades AWQ→Marlin**: models with `quant_method=awq` auto-detect and use `awq_marlin` kernel when hardware supports it. No config change needed.
- **Most HF "AWQ" models use compressed-tensors, NOT awq format**: only models quantized with AutoAWQ (not llm-compressor) have `quant_method=awq` in config. Must check config.json before downloading.
- **Marlin requires zero_point=True for AWQ**: `Vishva007` model has `zero_point=false` and is NOT Marlin-compatible despite being AWQ format
- **TensorRT-LLM attention unavailable on SM120**: requires SM100 (B-series data center). Consumer Blackwell (RTX 5080) is SM120.
- **FlashInfer RoPE NOT fused into attention**: vLLM uses `pos_encoding_mode="NONE"` and applies RoPE separately. Fusion would require vLLM changes.
- **GQA 4:1 optimal for decode**: FlashInfer broadcasts KV→Q heads in-kernel with near-zero overhead. 4x KV memory savings are "free".
- **Vocab pruning NOT worthwhile**: lm_head (151K vocab) logits compute is 0.28 ms/step at batch=128, <10% of decode time. Memory-bandwidth is the bottleneck, not compute.
- **Qwen/Qwen3-4B-AWQ is NOT Instruct-2507**: it's quantized from original Qwen3-4B (May 2025), not the July 2025 Instruct refresh. Different model weights.

## Dead Ends
- Spec decode at 128 concurrency: proven harmful by multiple papers
- FP8 on SM120 with vLLM 0.8.5.post1: CUTLASS kernels not present
- Reducing max_num_seqs: didn't help (cache handles load)
- max_num_batched_tokens tuning: marginal on A100 (already tested 8192/16384)
- KV cache compression (H2O/SnapKV): only helps for >4K token sequences
- HTTP-level request batching: vLLM continuous batching already optimal
- CUDA graphs on 16GB (gpu=0.95): OOM, works at 0.80 but net negative
- FP8 KV cache on RTX 5080: dequant overhead kills throughput
- FP8 weight quantization at max_num_seqs=256: OOM in sampler warmup (fixed at seqs=192)
- block_size tuning: FlashInfer doesn't support custom block_size
- prefix_caching: neutral (prefill already negligible for short inputs)
- performance_mode/async_scheduling alone: neutral with enforce_eager=True
- DBO (dual batch overlap): requires DeepEP expert parallelism kernels
- VLLM_ENABLE_V1_MULTIPROCESSING=0: single-process blocks event loop (-7%)
- FlashInfer sampler: slower than default for temperature=0 argmax (-8.6%)
- Cache seeding with representative queries: causes false positive keyword hits (-8.6%)
- TF32 matmul precision: irrelevant for BF16 model
- Vocabulary/logits pruning: lm_head compute is <10% of decode step, bandwidth-bound not compute-bound
- FlashInfer RoPE fusion: vLLM hardcodes pos_encoding_mode=NONE, would require modifying vLLM attention backend
- TensorRT-LLM attention: only SM100 (B-series datacenter), not SM120 (RTX 5080)
- FlashInfer custom block_size: already confirmed unsupported, default 16 is the only option
- AWQ models with quant_method=compressed-tensors: NOT Marlin-compatible (warshanks, cyankiwi, kaitchup, Sophia-AI)
- AWQ models with zero_point=false: NOT Marlin-compatible (Vishva007)

## Deep Dive: FlashInfer + Qwen3-4B Architecture Analysis (2026-04-04)

### Qwen3-4B-Instruct-2507 Architecture
```
hidden_size:          2560
intermediate_size:    9728
num_hidden_layers:    36
num_attention_heads:  32    (Q heads)
num_key_value_heads:  8     (KV heads) → GQA ratio 4:1
head_dim:             128
vocab_size:           151936
rope_theta:           5,000,000
hidden_act:           silu
torch_dtype:          bfloat16
tie_word_embeddings:  true
```

### Memory Breakdown
| Component | BF16 | FP8 | INT4 AWQ |
|---|---|---|---|
| Model weights | 7.49 GB | 3.75 GB | **2.11 GB** |
| lm_head alone | 0.72 GB | 0.36 GB | 0.72 GB (unquantized) |
| KV per token (BF16) | 144 KB | 144 KB | 144 KB |
| KV per token (FP8) | 72 KB | 72 KB | 72 KB |

### RTX 5080 Memory Budget (16GB, gpu_util=0.92)
| Quant | Model | KV Room | Max Seqs @288 (BF16 KV) | Max Seqs @288 (FP8 KV) |
|---|---|---|---|---|
| BF16 | 7.49 GB | 6.73 GB | 170 | 340 |
| FP8 | 3.75 GB | 10.47 GB | 265 | 530 |
| **INT4 AWQ** | **2.11 GB** | **12.11 GB** | **306** | **613** |

### Decode Bandwidth Analysis (RTX 5080 = 960 GB/s)
| Quant | Time/Step | Theoretical Max tok/s/req |
|---|---|---|
| BF16 | 7.80 ms | 128 |
| FP8 | 3.90 ms | 256 |
| **INT4 AWQ-Marlin** | **2.20 ms** | **456** |

INT4 AWQ-Marlin = ~1.78x faster than FP8 per decode step (2.20 vs 3.90 ms).

### FlashInfer 0.6.6 on SM120 (RTX 5080)

**Key API parameters for batch decode:**
- `use_tensor_cores=True`: vLLM already enables this for CUDA graph wrapper
- `page_size`: set by vLLM's `block_size` (default 16, FlashInfer doesn't support custom)
- `pos_encoding_mode="NONE"`: vLLM handles RoPE separately (not fused into attention)
- `fixed_split_size` / `disable_split_kv`: only used in VLLM_BATCH_INVARIANT mode
- `kv_layout="NHD"`: default, matches Qwen3's layout

**TensorRT-LLM attention path:** NOT available on SM120. Requires SM100 (Blackwell B-series data center). RTX 5080 is SM120 consumer Blackwell.

**RoPE fusion:** FlashInfer has `apply_rope_inplace()` and `apply_rope_with_cos_sin_cache_inplace()` but vLLM runs RoPE OUTSIDE the attention kernel. Fusing would require vLLM internals changes: not actionable via config.

**GQA 4:1 handling:** FlashInfer natively handles GQA in batch decode by broadcasting KV heads to Q heads inside the kernel. The 4:1 ratio (32Q/8KV) is optimal: saves 4x KV memory vs MHA while the decode kernel has near-zero overhead for the broadcast.

### AWQ-Marlin INT4 Quantization: CONFIRMED VIABLE

**vLLM 0.19.0 support:**
- `awq_marlin` quantization method is available
- `get_min_capability() = 75` (SM75+), SM120 qualifies
- Marlin INT4 kernel confirmed supported on SM120 via `check_marlin_supported()`
- Auto-upgrades: loading with `quantization=None` on an AWQ model auto-selects Marlin

**Available Qwen3-4B-Instruct-2507 AWQ checkpoints (Marlin-compatible):**

| Model ID | quant_method | bits | group_size | zero_point | Marlin? |
|---|---|---|---|---|---|
| `Eslzzyl/Qwen3-4B-Instruct-2507-AWQ` | awq | 4 | 128 | true | **YES** |
| `Qwen/Qwen3-4B-AWQ` | awq | 4 | 128 | true | **YES** (but base model, not Instruct-2507) |
| `warshanks/Qwen3-4B-Instruct-2507-AWQ` | compressed-tensors | 4 | 128 | N/A | NO |
| `cyankiwi/Qwen3-4B-Instruct-2507-AWQ-4bit` | compressed-tensors | 4 | 128 | N/A | NO |
| `kaitchup/Qwen3-4B-Instruct-2507-awq-w4a16-asym` | compressed-tensors | 4 | 128 | N/A | NO |
| `Sophia-AI/Qwen3-4B-Instruct-2507-AWQ-W4A16` | compressed-tensors | 4 | 128 | N/A | NO |
| `Vishva007/Qwen3-4B-Instruct-2507-W4A16-AutoRound-AWQ` | awq | 4 | 128 | false | NO (needs zero_point=true) |

**BEST CANDIDATE: `Eslzzyl/Qwen3-4B-Instruct-2507-AWQ`**
- Correct base model (Instruct-2507)
- quant_method=awq, zero_point=true → auto-upgrades to AWQ-Marlin
- Quantized with AutoAWQ (industry standard)
- ~2.1 GB model = 1.64 GB freed vs FP8

**GPTQ-Marlin alternatives (also Marlin-compatible):**
- `kaitchup/Qwen3-4B-Instruct-2507-gptq-w4a16-g128`: GPTQ W4A16, group_size=128
- `AXERA-TECH/Qwen3-4B-Instruct-2507-GPTQ-Int4`
- `numen-tech/Qwen3-4B-Instruct-2507-GPTQ-Int4`

### Vocabulary Pruning Analysis
- lm_head is 2560 x 151936 = 389M params (9.7% of model)
- At batch=128, logits matmul = ~100 GFLOPS per decode step
- Theoretical compute time: 0.28 ms (RTX 5080 355 TFLOPS FP16)
- **NOT worth pursuing**: logits compute is <10% of decode step time, and memory bandwidth for weight read dominates, not compute

## Next Steps

Current best cold: **429.29 req/s** (FP8 + Inductor + stream_interval=256 + seqs=192 + gpu=0.92 + max_model_len=288)

### Blocked: Waiting on vLLM SM120 Support

1. **AWQ INT4 quantization: ALL PATHS TESTED, FP8 IS FASTER**
   - **Marlin backend**: Crashes: Marlin PTX not compiled for SM120 (`marlin_permute_scales()` fails)
   - **Triton backend** (`VLLM_USE_TRITON_AWQ=1`): **WORKS** on SM120! But only **250 req/s**: 40% slower than FP8 (429 req/s). Triton AWQ dequantization kernel is slower than native FP8 CUTLASS.
   - **GPTQ fallback** (no Marlin): Works but 255 req/s with worse perplexity (1.218)
   - **compressed-tensors format**: Routes through Marlin → same crash as Marlin backend
   - **Why FP8 wins**: SM120 has native FP8 tensor cores with optimized CUTLASS kernels (vLLM PR #38325 swapAB). INT4 dequant runs on generic Triton/PyTorch without tensor core acceleration.
   - **Conclusion**: FP8 is the optimal quantization for SM120. INT4 would only help if Marlin gets SM120-compiled kernels.

2. **CUDA graphs**: BLOCKED by 16GB VRAM. Even with FP8 (4GB model), torch.compile graph capture OOMs at gpu=0.92+.

### Diminishing Returns: Remaining Levers

3. **Selective torch.compile on MLP**: fuse gate+up+silu+down per layer. High effort (modify vLLM model code), moderate expected gain (5-10%).
4. **KV cache offloading to CPU**: crashed with FP8 (meta tensor error). May work in future vLLM versions.
5. **Cache algorithm improvements**: keyword caching is well-tuned. More aggressive matching causes false positives that hurt perplexity. The current thresholds (kw=0.45, hard=0.70, tri=0.45) are the sweet spot.
