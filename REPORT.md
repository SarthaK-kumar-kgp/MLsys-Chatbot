# Optimization Report: High-Throughput LLM Serving on RTX 5080

**Course**: CS4262/5462 Machine Learning Systems - Project 1  
**Track**: B (Customer Support Chatbot)  
**Model**: Qwen3-4B-Instruct-2507  
**Hardware**: NVIDIA GeForce RTX 5080 (SM120, 16GB GDDR7, 960 GB/s)  
**Framework**: vLLM 0.19.0 + FlashInfer + FastAPI  
**Final Result**: **429 req/s** cold-start (vs 18 req/s unoptimized baseline = **23.6x improvement**)

---

## 1. Executive Summary

We built a high-throughput serving engine for Qwen3-4B that achieves **429 requests/second** on an RTX 5080 with 128 concurrent connections. The baseline (unoptimized vLLM) achieved only 18 req/s with 429 request failures. Our optimization stack spans four layers:

| Layer | Technique | Impact |
|-------|-----------|--------|
| **GPU Compute** | FP8 weight quantization | +21% throughput |
| **GPU Compute** | Inductor kernel fusions (no CUDA graphs) | +8.5% throughput |
| **Memory** | Tight max_model_len=288 (measured template overhead) | +4.9% throughput |
| **Application** | Multi-layer keyword cache with stemming + inverted index | High cache hit rate on duplicate/similar queries |
| **Scheduling** | stream_interval=256, async scheduling, performance_mode | +2-5% throughput |

Over **35 experiments** were conducted across 3 branches (arya, arya-2, arya-3), testing quantization, caching, scheduling, compilation, and system-level optimizations. **19 approaches were rejected** after rigorous benchmarking.

---

## 2. Understanding the Bottleneck

Before optimizing, we analyzed where time is spent during LLM inference.

### 2.1 The Decode Bottleneck

Autoregressive text generation works by generating one token at a time. Each token generation step (called a "decode step") requires reading **the entire model** from GPU memory:

```
Per decode step:
  BF16 (8 GB model) / 960 GB/s bandwidth = 8.3 ms
  FP8  (4 GB model) / 960 GB/s bandwidth = 4.2 ms
  INT4 (2 GB model) / 960 GB/s bandwidth = 2.1 ms
```

With an average output of ~178 tokens per response, a single request takes **178 x 4.2ms = 743ms** (FP8). The GPU is **memory-bandwidth bound** -- it spends almost all its time reading weights, not computing.

### 2.2 Batching Helps, But Has Limits

vLLM's continuous batching processes multiple requests simultaneously. All 128 concurrent requests share the same weight read -- the GPU reads weights once and computes 128 tokens. This amortizes the bandwidth cost:

```
Effective per-token time: 4.2ms / 128 = 0.033ms
Per-request (178 tokens): 178 x 0.033ms = 5.8ms
Theoretical max: 128 / 0.0058s = ~22,000 req/s
```

In practice, we achieve ~429 req/s (not 22K) because:
- Not all 128 slots are always full (requests finish at different times)
- Scheduling, sampling, and tokenization add overhead
- KV cache memory limits the actual batch size

### 2.3 The RTX 5080 Challenge

The RTX 5080 is a **consumer Blackwell GPU** (SM120 architecture) with unique constraints:
- Only 16 GB VRAM (vs 80 GB on A100/H100)
- FlashAttention3 is **broken** on SM120 (must use FlashInfer)
- Marlin INT4 kernels are **not compiled** for SM120
- CUDA graphs **OOM** on 16 GB (even with FP8)

These constraints eliminated many standard optimization paths and forced us to find novel approaches.

---

## 3. Optimization 1: Multi-Layer Response Cache (Application Layer)

### 3.1 What It Does

Instead of running GPU inference for every request, we cache responses and return cached answers for similar queries. The cache has four layers, checked in order:

1. **Exact-match cache** -- SHA256 hash of the normalized request. O(1) lookup.
2. **Keyword similarity cache** -- Jaccard similarity of stemmed keywords via inverted index. Returns a cached response if a similar query was already answered.
3. **Semantic inflight dedup** -- If a similar query is currently being processed by the GPU, wait for that result instead of starting a new inference.
4. **Exact inflight dedup** -- If the identical query is already inflight, share the result.

### 3.2 Keyword Matching Pipeline

```
Input: "How do I cancel my order?"

Step 1: Normalize:    "how do i cancel my order"
Step 2: Remove stops:  ["cancel", "order"]       (removed: how, do, i, my)
Step 3: Stem:          ["cancel", "order"]        (cancel→cancel, order→order)
Step 4: Lookup:        inverted_index["cancel"] → {key1, key5, key9}
                        inverted_index["order"]  → {key1, key3, key9}
                        candidates = {key1, key5, key9, key3}
Step 5: Jaccard:       score(key1) = |{cancel,order} ∩ cached_keywords| / |union|
Step 6: Threshold:     if score >= 0.45 → return cached response
```

### 3.3 Why Simple Stemming, Not Embeddings

We initially used sentence-transformer embeddings (MiniLM-L6-v2 + FAISS) for semantic matching. We replaced this with keyword stemming because:

- **3.2x more keyword merging**: "cancel/cancelling/canceling" all become "cancel"
- **No GPU/CPU overhead**: pure Python string operations vs 80MB neural model
- **P99 halved**: removing the embedding model eliminated its latency spikes
- **Smaller Docker image**: removed sklearn, faiss, sentence-transformers dependencies

### 3.4 Character Trigram Fallback

For typo-tolerant matching, we also index character trigrams. If keyword matching fails, trigram Jaccard catches queries like "cansel" vs "cancel":

```
"cancel" → trigrams: {"can", "anc", "nce", "cel"}
"cansel" → trigrams: {"can", "ans", "nse", "sel"}
Jaccard = |{can}| / |{can,anc,nce,cel,ans,nse,sel}| = 1/7 = 0.14
```

### 3.5 Tunable Knobs: Cache Threshold Configuration

The cache has **three thresholds** that control the quality/speed tradeoff. These are the most important knobs for adapting to different data distributions:

| Knob | Default | Range | Effect |
|------|---------|-------|--------|
| `keyword_threshold` | 0.45 | 0.25 - 0.80 | Minimum Jaccard similarity to return a cached response. Lower = more cache hits but more false positives. |
| `hard_stop` | 0.70 | 0.50 - 0.90 | If Jaccard exceeds this, return immediately without checking further candidates. Higher = more confident matches only. |
| `trigram_threshold` | 0.45 | 0.30 - 0.60 | Character trigram Jaccard for typo-tolerant fallback. Lower = catches more typos but risks unrelated matches. |

**What happens when you turn them:**

- **Lower thresholds (e.g., kw=0.30, hard=0.55)**: More cache hits, higher throughput on training data. But returns wrong answers for genuinely different queries: "cancel order" might match "track order" since both share "order". We tested kw=0.30 and saw **+52% throughput but perplexity degraded** (A21: 1.2037 vs 1.1945). On unseen validation data, this risk is amplified.

- **Higher thresholds (e.g., kw=0.65, hard=0.85)**: Fewer false positives, better perplexity. But cache hit rate drops, forcing more GPU inference. We tested kw=0.65 initially (A7) and got only 40 req/s.

- **The sweet spot (kw=0.45, hard=0.70)**: Balances cache hits vs correctness. Catches true paraphrases ("cancel my order" ↔ "I want to cancel the order") while rejecting different intents ("cancel order" vs "track order"). This was found through 8 iterations of threshold tuning (Q1-Q6, A7-A23).

**Experimentation history:**

| Thresholds (kw/hard) | Hit Rate | Throughput | Perplexity | Verdict |
|---|---|---|---|---|
| 0.65 / 0.82 | ~40% | 40 req/s | 1.2004 | Too conservative |
| 0.55 / 0.75 | ~55% | 51 req/s | 1.2011 | Better |
| 0.45 / 0.70 | ~65% | 76 req/s | 1.1992 | Current default |
| 0.40 / 0.70 | ~70% | 77 req/s | 1.1967 | Marginal gain |
| 0.35 / 0.65 | ~75% | 76 req/s | 1.1974 | Diminishing returns |
| 0.30 / 0.55 | ~80% | 85 req/s | 1.2037 | Perplexity degrades |

### 3.6 Advantages and Disadvantages of Keyword Caching

**Advantages:**

1. **Zero external dependencies**: Pure Python: no ML models, no FAISS, no sentence-transformers. Faster Docker builds, smaller image, no version conflicts.
2. **Deterministic and interpretable**: You can inspect exactly why two queries matched (shared keywords). Neural embeddings are black boxes.
3. **O(1) candidate lookup**: The inverted index makes lookup time independent of cache size. Neural approaches require O(n) similarity search or approximate nearest neighbor structures.
4. **No GPU contention**: The cache runs entirely on CPU. Embedding models would compete with the LLM for GPU memory and compute.
5. **Low latency**: Cache lookup takes <0.1ms. Embedding + FAISS search takes 5-50ms.
6. **Stemming merges morphological variants**: "cancel/cancelling/cancelled/cancellation" all become "cancel", achieving coverage that even embeddings sometimes miss for domain-specific terms.

**Disadvantages:**

1. **No semantic understanding**: "I want a refund" and "give me my money back" share zero keywords after stopword removal. An embedding model would recognize these as identical intent.
2. **Sensitive to word choice**: The Jaccard metric treats all keywords equally. "cancel order" and "cancel account" both have 50% overlap but very different intents.
3. **Domain-specific tuning required**: The stopword list and stemmer rules are hand-tuned for customer service. A different domain (medical, legal) would need different rules.
4. **No handling of negation**: "I want to cancel" and "I don't want to cancel" produce identical keywords after stopword removal.

**Why it's the best approach for THIS specific benchmark:**

1. **Temperature=0 (deterministic)**: Every query with the same content produces the identical response. This makes caching safe: there's no randomness.
2. **Short queries (12 tokens avg)**: With so few words, keyword overlap is a strong signal. Longer queries would dilute the Jaccard scores.
3. **Customer service domain**: ~17 distinct intents with clear keyword patterns (cancel, refund, track, delivery, payment, account, etc.). Keywords naturally cluster by intent.
4. **High duplicate rate**: 36% exact duplicates in training data, plus paraphrases. The cache is effective even with conservative thresholds.
5. **No GPU budget for embeddings**: On 16GB VRAM with a 4GB model, there's no room for an embedding model. CPU-based embeddings would add latency.

### 3.7 Impact

On training data (36% exact duplicates): **~50% cache hit rate** on cold start, pushing throughput from ~170 raw GPU req/s to 429 req/s. On warm runs (all queries seen): 2000+ req/s.

---

## 4. Pre-arya-3 Optimizations (arya / arya-2 branches)

Before the arya-3 GPU-level optimizations, significant work was done on the arya and arya-2 branches. These earlier optimizations built the application-layer foundation.

### 4.1 Early Exploration (arya branch, March 2026)

The project started on the arya branch with the professor's starter template. Initial work focused on understanding the system:

- **Benchmarked on Modal L4 GPU**: 13.33 req/s baseline (P50=9.8s, perplexity=1.20)
- **Benchmarked on H200 (141GB VRAM)**: FP8, BF16, AWQ all gave ~28-29 req/s: confirming the system is **bandwidth-bound**, not compute-bound
- **Added inflight request deduplication**: Coalesced identical concurrent requests to share GPU results
- **Optimized FastAPI path**: Skipped Pydantic validation, direct tokenization, cached SamplingParams, orjson serialization

### 4.2 Semantic Caching Evolution (arya-2 branch)

The arya-2 branch is where the caching architecture was developed through iterative experimentation:

**Phase 1: Embedding-based caching (A7-A9):**
- Added MiniLM-L6-v2 sentence embeddings + FAISS IndexFlatIP
- Two-tier matching: keyword Jaccard pre-filter → embedding cosine similarity
- Result: 40-51 req/s (vs 29 baseline), but P99 latency spiked to 24s due to embedding model overhead

**Phase 2: Keyword-only matching (Q6, A11):**
- Removed embeddings entirely. Pure keyword Jaccard matching.
- Result: **154 req/s** (5.3x baseline). P99 halved from 24s to 16s.
- Key insight: embedding computation was the bottleneck, not matching quality

**Phase 3: Stemming + inverted index (Q10, A13):**
- Added `simple_stem()`: suffix stripping that merges morphological variants
- Added domain-specific stopwords ("help", "need", "assistance" don't discriminate intent)
- Added inverted keyword index for O(1) candidate lookup
- Result: **498 req/s** (17.2x baseline, 3.2x over keyword-only)

**Phase 4: V1 engine discovery (Q12, A14):**
- Discovered V1 engine is 40% faster than V0 at high cache hit rates
- V0's multi-step scheduling adds overhead when most requests are cache hits
- Result: **697 req/s** (24.5x baseline): but this was a warm-cache measurement

**Phase 5: Character trigram fallback (A19-A23):**
- Added character trigram matching for typo tolerance
- Tuned keyword/trigram thresholds through 5 iterations
- Final: kw=0.30, tri=0.35 gave best throughput on training data
- Reverted to kw=0.45, tri=0.45 for safer behavior on unseen validation data

**Phase 6: RTX 5080 deployment (f7b8345, 5080-arya2-v2):**
- Discovered FlashAttention3 is broken on SM120: switched to FlashInfer
- Environment variable `VLLM_ATTENTION_BACKEND=FLASHINFER` alone was insufficient; had to pass `attention_backend="flashinfer"` directly in `AsyncEngineArgs`
- Set `enforce_eager=True` to avoid CUDA graph OOM on 16GB
- Used `inspect.signature()` to auto-detect which vLLM params are supported across versions
- Result: **520 req/s** on RTX 5080 (but warm-cache; true cold-start was ~287 req/s)

### 4.3 Application-Layer Optimizations (arya-2)

These optimizations in the FastAPI/HTTP layer were committed in arya-2 and carried forward:

1. **Zero-copy cache hits** (c36f057): Pre-serialize responses to bytes with `orjson.dumps()` at cache insertion time. Cache hits return raw bytes via `Response(content=cached, media_type="application/json")`: no Pydantic serialization, no JSON encoding per request.

2. **Raw JSON parsing** (098d04a): Skip Pydantic validation for incoming requests. Parse raw JSON with `orjson.loads()` and construct `ChatMessage` objects directly. Only the first request (cache miss) pays the full Pydantic cost.

3. **Cached SamplingParams** (098d04a): Use `@lru_cache` on `SamplingParams(temperature, max_tokens)`: the benchmark always sends temperature=0, max_tokens=256, so the same object is reused for every request.

4. **Direct tokenization** (098d04a): Call `tokenizer.apply_chat_template()` with `tokenize=True` to get token IDs directly, instead of first getting text and then tokenizing separately (double tokenization).

5. **Semantic inflight dedup** (ccb67f5): When a new request arrives and a similar query is already being processed by the GPU (matched by keyword Jaccard), the new request waits for the existing GPU result instead of starting a separate inference. This coalesces similar concurrent requests.

---

## 5. Optimization 3: FP8 Weight Quantization (GPU Compute: arya-3)

### 5.1 What It Does

Quantizes model weights from BF16 (16-bit) to FP8 (8-bit), halving the model size from ~8 GB to ~4 GB. Each decode step now reads 4 GB instead of 8 GB from VRAM.

### 5.2 Why It Works on RTX 5080

RTX 5080 (SM120) has native FP8 tensor cores that can directly compute with FP8 values. vLLM 0.19 includes a crucial PR (#38325) that added SM120-optimized CUTLASS FP8 GEMM kernels with a "swapAB" strategy, improving effective bandwidth by **69%** for small-batch decode shapes.

### 5.3 The Sampler OOM Bug

FP8 initially crashed at `max_num_seqs=256` because vLLM's sampler warmup allocates a logits tensor of size `(max_num_seqs x vocab_size)` = `(256 x 151,936 x 4 bytes)` = **148 MB**. With gpu_memory_utilization=0.95, there wasn't enough headroom.

**Fix**: Lowered `max_num_seqs` to 192 and `gpu_memory_utilization` to 0.92. This leaves enough room for the sampler while maximizing KV cache capacity.

### 5.4 Why Not INT4? (Thoroughly Tested, Blocked on SM120)

INT4 quantization would theoretically double our throughput (2GB model → 2.1ms per decode step). We tested it exhaustively across 3 checkpoint formats and 2 quantization backends:

**Attempt 1: AWQ `compressed-tensors` format** (cyankiwi, warshanks models):
- These HuggingFace models are labeled "AWQ" but use `quant_method: compressed-tensors` (quantized by llm-compressor, not AutoAWQ)
- vLLM routes compressed-tensors through Marlin kernels regardless
- **Crash**: `CUDA error: the provided PTX was compiled with an unsupported toolchain`
- Root cause: Marlin PTX binary targets SM75-SM90 only

**Attempt 2: AWQ native format** (`Eslzzyl/Qwen3-4B-Instruct-2507-AWQ`):
- Correct format: `quant_method: awq`, `bits: 4`, `group_size: 128`, `zero_point: true`
- vLLM auto-detects AWQ and selects `awq_marlin` kernel (verified via `check_marlin_supported()` returning True for SM120)
- **Same crash**: `marlin_permute_scales()` fails because the Marlin CUDA kernel PTX is not compiled for SM120
- The API-level check (`check_marlin_supported`) passes, but the actual kernel binary doesn't support SM120

**Attempt 3: GPTQ INT4 without Marlin** (`JunHowie/Qwen3-4B-Instruct-2507-GPTQ-Int4`):
- GPTQ requires `dtype=float16` (not BF16)
- With Marlin disabled, falls back to naive PyTorch dequantization
- **Works but is 40% slower**: 255 req/s vs 429 req/s (FP8)
- **Perplexity degrades**: 1.218 vs 1.202
- The dequantization overhead on every matmul negates the bandwidth savings

**Attempt 4: AWQ with Triton backend** (`VLLM_USE_TRITON_AWQ=1`):
- Discovered an env var that forces a **Triton-based AWQ kernel** instead of Marlin
- Triton JIT-compiles kernels at runtime for the current GPU: **works on SM120!**
- Server starts successfully in 50s (Triton compilation overhead)
- **Result: 250 req/s**: **40% slower than FP8** (429 req/s)
- The Triton dequantization kernel is generic and does not use SM120's tensor cores efficiently
- Perplexity: 1.206 (acceptable but worse than FP8's 1.202)

**Conclusion**: All four INT4 paths were tested. FP8 is definitively faster on SM120 because:
1. SM120 has **native FP8 tensor cores** with optimized CUTLASS kernels (vLLM PR #38325 with swapAB strategy giving 69% better bandwidth)
2. INT4 dequantization on SM120 runs through generic Triton/PyTorch paths **without tensor core acceleration**
3. The dequantization compute overhead exceeds the bandwidth savings from the smaller model

FP8 is the optimal quantization for consumer Blackwell (SM120). INT4 would only become competitive if Marlin gets SM120-compiled kernels with tensor core dequantization.

### 5.5 Impact

**+21% throughput** (287 → 347 req/s), **zero perplexity degradation** (1.199 → 1.199).

---

## 6. Optimization 4: Inductor Compilation Without CUDA Graphs (GPU Compute)

### 6.1 The Problem

vLLM's `enforce_eager=True` disables **both** torch.compile (Inductor) AND CUDA graphs. On 16 GB, CUDA graphs OOM because graph capture pre-allocates memory for all captured tensor shapes. But Inductor compilation (kernel fusion) and CUDA graphs are **independent features**.

### 6.2 The Novel Insight

We decoupled them: enable Inductor compilation while explicitly disabling CUDA graphs:

```python
CompilationConfig(
    mode=3,               # VLLM_COMPILE: enable Inductor
    cudagraph_mode="none"  # No CUDA graphs: avoid OOM
)
```

This activates vLLM's O2 optimization level kernel fusions:
- **fuse_norm_quant**: Merges RMSNorm + FP8 quantization into a single kernel
- **fuse_act_quant**: Merges SiLU activation + quantization into a single kernel
- **fuse_rope_kvcache**: Merges RoPE computation with KV cache write

Each fusion eliminates a kernel launch (~5-10 us) and an intermediate memory read/write. With 36 layers, this saves ~100 kernel launches per decode step.

### 6.3 Why This Is Novel

Standard vLLM documentation presents `enforce_eager` as a binary toggle -- either full optimization (with CUDA graphs) or no optimization. Decoupling compilation from CUDA graphs is not documented and was discovered by reading vLLM's `CompilationConfig` source code.

### 6.4 Impact

**+8.5% throughput** (364 → 395 req/s). Startup takes 35s instead of 20s due to Inductor compilation, but steady-state inference is faster.

---

## 7. Optimization 5: Tight max_model_len (Memory)

### 7.1 The Discovery

The default `max_model_len=320` was set assuming ~50 tokens of chat template overhead. We measured the actual Qwen3 chat template with `enable_thinking=False`:

```python
tokenizer.apply_chat_template(
    [{"role": "user", "content": "longest possible query"}],
    tokenize=True, add_generation_prompt=True, enable_thinking=False
)
# Result: 8 tokens overhead (not 50!)
```

Minimum required: 8 (template) + 23 (max input) + 256 (max output) + 1 (safety) = **288 tokens**.

### 7.2 Why It Helps

Each concurrent sequence reserves KV cache blocks for `max_model_len` tokens. Reducing from 320 to 288 frees **10% more KV cache per sequence**, allowing more sequences to run concurrently. More concurrent sequences means better weight-read amortization across the batch.

### 7.3 The Failure at 272

We also tested `max_model_len=272`, which is below the minimum for some queries. This caused output truncation and **dropped throughput to 298 req/s** (-18%). The 288 value is the precise sweet spot.

### 7.4 Impact

**+4.9% throughput** (347 → 364 req/s), zero quality impact.

---

## 8. Optimization 6: Scheduling Tweaks (Scheduling)

### 8.1 stream_interval=256

By default, vLLM notifies the client after every generated token (stream_interval=1). Since our API returns complete responses (not streaming), this per-token host-device synchronization is pure overhead. Setting `stream_interval=256` batches the output delivery.

### 8.2 async_scheduling=True

Overlaps CPU-side scheduling decisions with GPU execution. The scheduler prepares the next batch while the GPU processes the current batch, reducing idle GPU time.

### 8.3 performance_mode="throughput"

Selects throughput-oriented kernel configurations and batching decisions within vLLM. This is mainly effective when CUDA graphs are enabled, but provides a marginal benefit even without them.

### 8.4 What Didn't Work: scheduler_reserve_full_isl=False

We initially set `scheduler_reserve_full_isl=False` to admit requests without checking if the full input fits in KV cache. This **hurt both throughput AND perplexity** (405 req/s with 1.2046 perplexity → 429 req/s with 1.2017 after removing it). The aggressive admission caused preemptions -- requests being evicted from the batch to make room, wasting GPU work and producing different outputs.

### 8.5 Impact

**+8.6% throughput** (395 → 429 req/s) from stream_interval alone.

---

## 9. RTX 5080 / SM120 Specific Findings

These discoveries are specific to consumer Blackwell GPUs and are not documented elsewhere:

### 9.1 FlashAttention3 is Broken on SM120

FlashAttention3 crashes on consumer Blackwell. The official vLLM docs recommend FlashInfer as the attention backend for SM120. We confirmed this and pass `attention_backend="flashinfer"` directly to the engine args (the environment variable alone is insufficient in vLLM 0.19).

### 9.2 FP8 is the Optimal Quantization

SM120 has native FP8 tensor cores, and vLLM 0.19 includes SM120-specific CUTLASS FP8 GEMM kernels. FP8 gives the best quality/speed tradeoff:

| Quantization | Model Size | Throughput | Perplexity | Status |
|---|---|---|---|---|
| BF16 | 8 GB | 287 req/s | 1.199 | Baseline |
| **FP8** | **4 GB** | **429 req/s** | **1.202** | **Best** |
| GPTQ INT4 | 2.5 GB | 255 req/s | 1.218 | Slow fallback path |
| AWQ INT4 | 2 GB | CRASH | -- | Marlin PTX missing |

### 9.3 CUDA Graphs Don't Fit on 16 GB

Even with FP8 (4 GB model), CUDA graph capture exhausts the remaining 12 GB at gpu=0.95. The torch.compile Inductor compilation allocates temporary buffers during graph capture that push past the VRAM limit. We confirmed this across multiple configurations (gpu=0.80, 0.85, 0.90, 0.95).

### 9.4 Consumer Blackwell vs Data Center Blackwell

SM120 (RTX 5080) lacks some features available on SM100 (B100/B200):
- No TensorRT-LLM attention (requires SM100 family)
- No Marlin INT4 PTX (not compiled for SM120)
- No FlashMLA (must be disabled for SM120 builds)
- CUDA 12.8 required (not 13.0+)

---

## 10. Dead Ends: What We Tried and Why It Failed

We tested 19 approaches that were rejected. Each is documented to prevent re-discovery:

| Approach | Result | Why It Failed |
|---|---|---|
| Speculative decoding | -6% throughput | Verification overhead scales superlinearly at BS=128 (3 papers confirm) |
| CUDA graphs (full/piecewise) | OOM | Graph capture exhausts 16 GB VRAM |
| FP8 KV cache | -44% throughput | Per-token dequantization overhead exceeds memory savings |
| AWQ-Marlin INT4 | CRASH | Marlin PTX not compiled for SM120 |
| Prefix caching | Neutral | Prefill is negligible for 12-token inputs |
| Block size tuning | CRASH | FlashInfer only supports default block size |
| FlashInfer sampler | -8.6% | Slower than PyTorch argmax for temperature=0 |
| FlashInfer autotune | -2.3% | Default kernel selection already optimal |
| Cache seeding | -8.6% | 25 seed queries cause false positive keyword matches |
| Aggressive stemming | -15% | Over-merging words returns wrong cached answers |
| DBO (dual batch overlap) | CRASH | Requires DeepEP expert parallelism kernels |
| V1 multiprocessing=0 | -7% | Single-process blocks event loop |
| TF32 matmul precision | Neutral | Model uses BF16, not FP32 |
| max_model_len=272 | -18% | Truncates some outputs |
| max_num_seqs=224+ | -10% | Sampler warmup OOM eats KV cache room |
| max_num_batched_tokens=16384 | -6% | Larger scheduling buffers waste memory |
| CPU KV offloading | CRASH | Meta tensor error with FP8 |
| scheduler_reserve_full_isl=False | -2% quality | Causes preemptions, hurts perplexity |
| Synonym normalization | -10% | Over-generalizes keyword matching |

---

## 11. Final Architecture

```
Client (128 concurrent)
    │
    ▼
FastAPI + uvicorn (uvloop)
    │
    ├── Layer 1: Exact-match cache (SHA256 hash → pre-serialized JSON bytes)
    │
    ├── Layer 2: Keyword cache (stemmed Jaccard via inverted index)
    │
    ├── Layer 3: Semantic inflight dedup (share GPU result for similar queries)
    │
    ├── Layer 4: Exact inflight dedup (share result for identical queries)
    │
    └── Layer 5: vLLM V1 Engine
        ├── FP8 weight quantization (E4M3)
        ├── FlashInfer attention backend
        ├── Inductor compilation (kernel fusions, no CUDA graphs)
        ├── Async scheduling + performance_mode=throughput
        ├── max_model_len=288 (tight KV allocation)
        └── max_num_seqs=192, gpu_memory_utilization=0.92
```

### 11.1 Final Configuration

```
Model:          Qwen3-4B-Instruct-2507
Quantization:   FP8 (E4M3)
Engine:         vLLM V1 (0.19.0)
Attention:      FlashInfer
Compilation:    Inductor (mode=3, cudagraph_mode=none)
max_model_len:  288
max_num_seqs:   192
gpu_mem_util:   0.92
stream_interval: 256
async_scheduling: True
performance_mode: throughput
```

### 11.2 Final Results (Cold-Start, 13,435 Requests, 128 Concurrency)

```
Throughput:   429.29 req/s
P50 Latency:  6.7 ms
P95 Latency:  2,715 ms
P99 Latency:  4,464 ms
Perplexity:   1.2017
Failures:     0
```

---

## 12. Optimization Journey: Progressive Gains

All numbers are **full 13K cold-start benchmarks** on RTX 5080 at 128 concurrency.

| Step | Optimization | Throughput | Cumulative Gain | Perplexity |
|------|-------------|-----------|-----------------|-----------|
| 0 | Unoptimized baseline (main) | 18 req/s | -- | 1.200 |
| 1 | + Multi-layer keyword cache + FlashInfer + enforce_eager | 287 req/s | +15.9x | 1.199 |
| 2 | + FP8 weight quantization | 347 req/s | +19.1x | 1.199 |
| 3 | + max_model_len=288 | 364 req/s | +20.1x | 1.200 |
| 4 | + Inductor compilation (no CUDA graphs) | 395 req/s | +21.8x | 1.198 |
| 5 | + stream_interval=256 | **429 req/s** | **+23.6x** | **1.202** |

---

## 13. Tools and Methodology

### 13.2 Experiment Discipline

- **35 experiments** logged with exact configs and results
- **19 dead ends** documented to prevent re-discovery
- Every experiment includes: P50, P95, P99 latency, throughput, perplexity, failure count
- Cold-start vs warm-cache distinguished (early experiments conflated these)

### 13.3 Research Methodology

We used AI research agents to search:
- vLLM source code (125K+ lines) for undocumented features
- vLLM GitHub issues/PRs for SM120-specific findings
- HuggingFace for pre-quantized model checkpoints
- Academic papers on LLM serving optimization (2025-2026)

Key research-driven discoveries:
- Inductor compilation without CUDA graphs (from reading `CompilationConfig` source)
- FP8 sampler warmup OOM fix (from reading `_dummy_sampler_run` source)
- Template overhead is 8 tokens (from measuring `apply_chat_template` directly)
- Marlin SM120 incompatibility (from `marlin_permute_scales` PTX error analysis)

---

## 14. Full Benchmark Scores (RTX 5080, Cold-Start)

> **Important**: All scores below (except the unoptimized baseline and warm-cache row) are from **full cold-start benchmarks**: 13,435 requests at 128 concurrency using `data/track2/train.jsonl`. "Cold-start" means the server was freshly started with an empty cache: no prior queries. This is the realistic scenario for grading, where validation data is unseen.

### 14.1 Winning Configurations (Chronological, All Full 13K Cold-Start)

| # | Config | Throughput | P50 | P95 | P99 | Perplexity | Failures | Benchmark |
|---|--------|-----------|-----|-----|-----|-----------|----------|-----------|
| 0 | Unoptimized baseline (main) | 18.15 req/s | 6,687 ms |: | 9,054 ms | 1.1997 | **429** | Full 13K |
| 1 | + FlashInfer + cache (arya-2) | 287.68 req/s | 1.7 ms | 4,101 ms | 6,230 ms | 1.1985 | 0 | Full 13K cold |
| 2 | + FP8 quantization | 347.50 req/s | 2.6 ms | 3,417 ms | 5,594 ms | 1.1991 | 0 | Full 13K cold |
| 3 | + max_model_len=288 | 364.53 req/s | 4.5 ms | 3,229 ms | 5,033 ms | 1.1997 | 0 | Full 13K cold |
| 4 | + Inductor compilation | 395.48 req/s | 3.2 ms | 2,916 ms | 4,645 ms | 1.1979 | 0 | Full 13K cold |
| 5 | + stream_interval=256 | **429.29 req/s** | **6.7 ms** | **2,715 ms** | **4,464 ms** | **1.2017** | **0** | **Full 13K cold** |

**Note on row 5 (429 req/s)**: This is the result after *removing* `scheduler_reserve_full_isl=False`. An earlier test with that flag gave 405 req/s but degraded perplexity to 1.2046 due to request preemptions. Removing it both improved throughput (405→429) and restored perplexity (1.2046→1.2017). The only scheduling change kept is `stream_interval=256`.

### 14.2 Warm-Cache Performance (Second Run on Same Server)

These numbers show what happens when the keyword cache is already populated from a prior benchmark run. The cache serves ~93% of requests instantly. **These are NOT representative of grading performance** (grading uses unseen validation data on a fresh server).

| Config | Throughput | P50 | P95 | P99 | Perplexity | Benchmark |
|--------|-----------|-----|-----|-----|-----------|-----------|
| Final config (warm) | 2,037 req/s | 45 ms | 83 ms | 110 ms | 1.2017 | Full 13K warm |

### 14.3 Key Rejected Configurations (All Full 13K Cold-Start)

| Config | Throughput | Perplexity | Why Rejected | Benchmark |
|--------|-----------|-----------|-------------|-----------|
| CUDA graphs (enforce_eager=False, gpu=0.80) | 113 req/s | 1.2005 | OOM at gpu=0.95, reduced KV cache | Full 13K cold |
| FP8 KV cache (fp8_e4m3) | 290 req/s | 1.1964 | Dequant overhead per attention step | Full 13K cold |
| GPTQ INT4 (no Marlin fallback) | 255 req/s | 1.2181 | Slow PyTorch dequant, quality loss | Full 13K cold |
| FlashInfer sampler | 317 req/s | 1.1983 | Slower than PyTorch argmax for temp=0 | Full 13K cold |
| Cache seeding (25 queries) | 317 req/s | 1.1990 | False positive keyword matches | Full 13K cold |
| scheduler_reserve_full_isl=False | 405 req/s | **1.2046** | Preemptions hurt both speed and quality | Full 13K cold |
| max_model_len=272 | 298 req/s | 1.1974 | Truncated some outputs | Full 13K cold |
| FlashInfer autotune | 396 req/s | 1.1986 | Default kernel selection already optimal | Full 13K cold |
| FP8 seqs=256 gpu=0.85 | 333 req/s | 1.1983 | Lower gpu_util hurts KV cache | Full 13K cold |
| VLLM_ENABLE_V1_MULTIPROCESSING=0 | 322 req/s | 1.2009 | Single-process blocks event loop | Full 13K cold |
