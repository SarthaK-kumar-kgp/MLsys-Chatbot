# MLsys-Chatbot: High-Throughput LLM Serving Engine

**Course**: CS4262/5462 Machine Learning Systems — Project 1: LLM Serving  
**Track 2**: Customer Support Chatbot (Interactive Chat Inference)  
**Model**: Qwen3-4B-Instruct-2507  
**Hardware**: NVIDIA RTX 5080 (SM120, 16 GB GDDR7, 960 GB/s)  
**Result**: **429 req/s** cold-start throughput — a **23.6× improvement** over the 18 req/s unoptimized baseline

**Contributors**: Arya Bhosale · Reiner Anggriawan Jasin · Sarthak Kumar · Thet Su Win

---

## Table of Contents

1. [Intuition First: Why Is LLM Serving Hard?](#1-intuition-first-why-is-llm-serving-hard)
2. [Project Overview](#2-project-overview)
3. [Repository Structure](#3-repository-structure)
4. [The Full Request Lifecycle (End-to-End Workflow)](#4-the-full-request-lifecycle-end-to-end-workflow)
5. [The Caching System — Deep Dive](#5-the-caching-system--deep-dive)
   - [5.1 Why Caching at All?](#51-why-caching-at-all)
   - [5.2 What Makes a Good Cache Key?](#52-what-makes-a-good-cache-key)
   - [5.3 Layer 1 — Exact-Match Cache (SHA256)](#53-layer-1--exact-match-cache-sha256)
   - [5.4 The Normalization Pipeline](#54-the-normalization-pipeline)
   - [5.5 Layer 2 — Keyword Similarity Cache (Jaccard + Inverted Index)](#55-layer-2--keyword-similarity-cache-jaccard--inverted-index)
   - [5.6 Layer 3 — Character Trigram Fallback (Typo Tolerance)](#56-layer-3--character-trigram-fallback-typo-tolerance)
   - [5.7 Layer 4 — Semantic Inflight Dedup](#57-layer-4--semantic-inflight-dedup)
   - [5.8 Layer 5 — Exact Inflight Dedup](#58-layer-5--exact-inflight-dedup)
   - [5.9 Cache Eviction (LRU)](#59-cache-eviction-lru)
   - [5.10 Threshold Tuning: The Quality-Speed Tradeoff](#510-threshold-tuning-the-quality-speed-tradeoff)
   - [5.11 Why Keyword Stemming Beats Embeddings Here](#511-why-keyword-stemming-beats-embeddings-here)
6. [GPU Optimizations](#6-gpu-optimizations)
   - [6.1 The Memory-Bandwidth Bottleneck (with Math)](#61-the-memory-bandwidth-bottleneck-with-math)
   - [6.2 FP8 Weight Quantization](#62-fp8-weight-quantization)
   - [6.3 Inductor Compilation Without CUDA Graphs](#63-inductor-compilation-without-cuda-graphs)
   - [6.4 FlashInfer Attention Backend](#64-flashinfer-attention-backend)
7. [Memory Optimizations](#7-memory-optimizations)
   - [7.1 Tight max_model_len=288](#71-tight-max_model_len288)
   - [7.2 Sampler OOM Fix](#72-sampler-oom-fix)
8. [Scheduling Optimizations](#8-scheduling-optimizations)
9. [Application-Layer Optimizations](#9-application-layer-optimizations)
10. [The Normalization + Stemming Pipeline (Code Walkthrough)](#10-the-normalization--stemming-pipeline-code-walkthrough)
11. [Optimization Journey: Progressive Gains](#11-optimization-journey-progressive-gains)
12. [Dead Ends: 19 Approaches That Didn't Work](#12-dead-ends-19-approaches-that-didnt-work)
13. [Final Architecture](#13-final-architecture)
14. [Configuration Reference](#14-configuration-reference)
15. [Deployment Guide](#15-deployment-guide)
16. [Benchmark Results](#16-benchmark-results)

---

## 1. Intuition First: Why Is LLM Serving Hard?

Before diving into any code, let us build the mental model that motivates every optimization in this project.

### The Autoregressive Tax

A language model generates text one token at a time. To generate token number *k*, it must read the **entire model** — all its weights — from GPU memory. For Qwen3-4B in BF16 format, that is 8 GB per token. The RTX 5080 has 960 GB/s of memory bandwidth, so:

```
Time to generate one token (BF16):  8 GB / 960 GB/s  =  8.3 ms
Average response length:             ~178 tokens
Time for one full response:          178 × 8.3 ms     =  1,477 ms  ≈  1.5 s
```

At 1.5 seconds per request, we could serve at most ~0.7 req/s on a single GPU. That is catastrophically slow.

### Batching Saves the Day

The key insight: **all tokens in a batch share the same weight-read**. If 128 requests are running in parallel, the GPU reads the weights once and computes 128 tokens simultaneously. The cost is amortized:

```
Effective per-token time (128 batch, BF16): 8.3 ms / 128  =  0.065 ms
Per-request (178 tokens):                   178 × 0.065 ms =  11.6 ms
Theoretical throughput:                     128 / 0.0116 s =  ~11,000 req/s
```

So batching turns 0.7 req/s into a theoretical 11,000 req/s — a 15,000× improvement in theory. In practice we hit ~429 req/s because: requests arrive at different times, the KV cache limits how many requests can be simultaneously in-flight, and scheduling/sampling overhead adds up.

### The Caching Multiplier

But there is a more powerful trick: if two requests are asking the same (or similar) question, **we only need to run the GPU once**. For customer support chatbots, this is enormously effective — there are only around 17 distinct intents (cancel order, track shipment, request refund, etc.), and customers phrase them with different words but the same meaning. If we can detect semantic similarity fast enough, we can serve many requests by looking up a stored answer — essentially bypassing the GPU entirely for cache hits.

This is why the caching system is the most impactful single optimization in this project.

---

## 2. Project Overview

This project implements a production-quality LLM serving engine for a customer support chatbot. It serves `Qwen3-4B-Instruct-2507` via a REST API compatible with the OpenAI chat completions format.

**Key requirements** (from the course benchmark):
- Handle 128 simultaneous connections
- Minimize P50 and P99 latency
- Maximize throughput (requests per second)
- Maintain perplexity ≤ 1.25 (quality constraint)
- Zero request failures

**What we built:**
- A 5-layer response cache that avoids GPU inference for repeated/similar queries
- FP8 quantization that halves the model's memory footprint on SM120
- Inductor kernel fusions that eliminate ~100 redundant kernel launches per decode step
- Tight memory allocation that squeezes out 10% more KV cache capacity
- Async scheduling and output batching that reduce CPU-GPU synchronization overhead

---

## 3. Repository Structure

```
MLsys-Chatbot/
├── track2_chat/                    # The serving engine
│   ├── app/
│   │   ├── main.py                 # FastAPI entry point; orchestrates the 5-layer cache
│   │   ├── chat_engine.py          # vLLM async engine + GPU optimizations
│   │   ├── cache.py                # The full multi-layer caching system
│   │   ├── normalize.py            # Text normalization, stemming, keyword extraction
│   │   ├── schemas.py              # Pydantic models (ChatMessage, ChatRequest, ChatResponse)
│   │   └── constants.py            # All tunable parameters (override via env vars)
│   ├── Dockerfile                  # CUDA 12.8 + Python 3.12 container
│   ├── docker-compose.yaml         # GPU-passthrough Docker Compose config
│   ├── pyproject.toml              # Python dependencies (managed by uv)
│   └── modal_deploy.py             # Serverless GPU deployment on Modal
├── benchmark/
│   ├── runner_chat.py              # Async benchmark client (128 concurrency)
│   ├── data/track2/train.jsonl     # 13,435 customer service prompts
│   └── pyproject.toml
├── REPORT.md                       # Full optimization report with experiment data
└── plan.md                         # Experiment tracking log (35 experiments)
```

---

## 4. The Full Request Lifecycle (End-to-End Workflow)

Understanding the flow of a single request through the system is the best way to see how all the pieces connect.

```
Client (HTTP POST /v1/chat/completions)
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│  FastAPI + uvicorn (uvloop event loop)                      │
│                                                             │
│  1. Parse raw JSON with orjson (skip Pydantic validation)   │
│  2. Compute cache key (SHA256 of normalized content)        │
│                                                             │
│  ── LAYER 1: Exact-Match Cache ─────────────────────────── │
│  3. Look up SHA256 key in OrderedDict                       │
│     HIT  → return pre-serialized bytes immediately          │
│     MISS → continue to Layer 2                              │
│                                                             │
│  ── LAYER 2: Keyword Similarity Cache ──────────────────── │
│  4. Extract stemmed keywords from user message              │
│  5. Look up candidates in inverted index                    │
│  6. Compute Jaccard similarity with each candidate          │
│     score ≥ 0.70 → hard stop, return immediately           │
│     score ≥ 0.45 → return best match                        │
│     score < 0.45 → fallback to trigrams                     │
│                                                             │
│  ── LAYER 3: Character Trigram Fallback ─────────────────  │
│  7. Compute character trigrams from normalized text         │
│  8. Jaccard similarity over trigrams                        │
│     score ≥ 0.45 → return cached response (typo-tolerant)  │
│     MISS → continue to Layer 4                              │
│                                                             │
│  ── LAYER 4: Semantic Inflight Dedup ───────────────────── │
│  9. Check if a similar query is currently being processed   │
│     (keyword Jaccard ≥ 0.45 with inflight keyword sets)    │
│     HIT  → await that future, populate cache, return        │
│     MISS → continue to Layer 5                              │
│                                                             │
│  ── LAYER 5: Exact Inflight Dedup ──────────────────────── │
│  10. Check if this exact query is already inflight          │
│      HIT  → await that future, return                       │
│      MISS → claim future as "owner", proceed to GPU        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
             │  (cache miss — only ~50% of requests reach here)
             ▼
┌─────────────────────────────────────────────────────────────┐
│  vLLM V1 Async Engine                                       │
│                                                             │
│  11. Apply Qwen3 chat template (tokenize=True)              │
│  12. Retrieve LRU-cached SamplingParams                     │
│  13. Submit to continuous batching scheduler                │
│                                                             │
│  ── GPU Execution ────────────────────────────────────────  │
│  14. FP8 weight matmuls (E4M3, CUTLASS SM120 kernels)       │
│  15. FlashInfer attention (GQA 4:1, page-based KV cache)    │
│  16. Inductor-fused kernels (norm+quant, act+quant, RoPE)   │
│  17. Autoregressive decode (stream_interval=256)            │
│                                                             │
│  18. Return token IDs + per-token logprobs                  │
└─────────────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│  Response Finalization                                      │
│                                                             │
│  19. Serialize response dict to bytes (orjson.dumps)        │
│  20. Store bytes in cache (exact key + keyword index)       │
│  21. Resolve any waiting futures (inflight dedup)           │
│  22. Return Response(content=bytes, media_type="...")       │
└─────────────────────────────────────────────────────────────┘
```

Every request goes through steps 1–10 (microseconds, pure Python). Only cache misses proceed to steps 11–22 (milliseconds, GPU inference). With ~50% cache hit rate on cold-start data, roughly half of all requests never touch the GPU.

---

## 5. The Caching System — Deep Dive

The cache is the single biggest contributor to throughput. It is implemented in [cache.py](track2_chat/app/cache.py) and orchestrated in [main.py](track2_chat/app/main.py).

### 5.1 Why Caching at All?

The benchmark dataset contains 13,435 customer support queries. Analysis reveals:

- **36% exact duplicates**: Customers ask the same phrasing verbatim
- **~14% semantic duplicates**: Same question, different words ("cancel my order" vs "I'd like to cancel an order")
- **~17 distinct intents**: Cancel, track, refund, payment, account, delivery, etc.

This is not random text — it is a narrow, repetitive domain. A cache that understands semantic equivalence can avoid GPU inference for 50%+ of requests.

**Why temperature=0 makes caching safe**: The benchmark always sends `temperature=0`, meaning the model is deterministic. The same input always produces the same output. So it is correct to serve any cached response that was generated from an equivalent input — there is no sampling randomness to worry about.

### 5.2 What Makes a Good Cache Key?

The cache key must capture everything that determines the model's output: the full conversation history, the temperature setting, and the max tokens limit. But it must also normalize away irrelevant differences (case, trailing whitespace) so "Cancel My Order" and "cancel my order" produce the same key.

```python
# From cache.py: make_key()
def make_key(self, messages, temperature, max_tokens) -> str | None:
    if temperature > 0:
        return None          # Non-deterministic: never cache

    hasher = hashlib.sha256()
    hasher.update(struct.pack("<d", temperature))    # 8 bytes, little-endian double
    hasher.update(struct.pack("<I", max_tokens))     # 4 bytes, little-endian uint
    hasher.update(struct.pack("<I", len(messages)))  # number of turns

    for message in messages:
        role_bytes = message.role.encode("utf-8")
        content_bytes = message.content.lower().strip().encode("utf-8")  # normalize here
        hasher.update(struct.pack("<I", len(role_bytes)))
        hasher.update(role_bytes)
        hasher.update(struct.pack("<I", len(content_bytes)))
        hasher.update(content_bytes)

    return hasher.hexdigest()   # 64-char hex string
```

The `struct.pack` calls include length prefixes before each field. This prevents collisions where, e.g., concatenating two short strings would produce the same bytes as one long string. The key is a 256-bit hash — collisions are astronomically unlikely.

### 5.3 Layer 1 — Exact-Match Cache (SHA256)

**Intuition**: The cheapest kind of match. If the normalized request hashes to something we have seen before, return the stored answer immediately.

```python
# From cache.py: get_by_key()
def get_by_key(self, key: str) -> dict | None:
    if key in self._cache:
        self._cache.move_to_end(key)   # LRU: mark as recently used
        self.hits += 1
        return self._cache[key]        # Pre-serialized bytes
    self.misses += 1
    return None
```

**Data structure**: `OrderedDict[str, bytes]` — insertion-ordered for LRU eviction.

**Lookup cost**: O(1) — dictionary hash table.

**Hit condition**: SHA256 of normalized (lowercase, stripped) content matches exactly.

**What it catches**: The 36% exact duplicate rate in the dataset, plus any request that was already answered with the exact same wording during this server's lifetime.

### 5.4 The Normalization Pipeline

Before keyword extraction and semantic matching, every query passes through a normalization pipeline implemented in [normalize.py](track2_chat/app/normalize.py). Understanding this pipeline is essential for understanding why the cache is effective.

**Step 1 — Lowercase and strip punctuation**:

```python
_NON_ALNUM = re.compile(r"[^a-z0-9\s]")
_MULTI_SPACE = re.compile(r"\s+")

def normalize_text(text: str) -> str:
    text = text.lower()
    text = _NON_ALNUM.sub("", text)          # remove all non-alphanumeric
    text = _MULTI_SPACE.sub(" ", text).strip()
    return text
```

Example:
```
Input:  "How do I cancel my ORDER?!?"
Output: "how do i cancel my order"
```

**Step 2 — Stopword removal**: Function words that carry no intent signal are removed. The stopword list has two tiers:

- *Standard English stopwords*: "a", "the", "is", "and", "of", "to", etc.
- *Domain-specific stopwords*: "help", "need", "please", "want", "know", "assist", "assistance", "gonna", "gotta", "wanna", "damn", "bloody", "freaking"

The domain stopwords are crucial. "I need help with my order" and "I want to cancel my order" both contain "need"/"want"/"help" — those words do not distinguish intent. Removing them lets the content words ("order", "cancel") do the work.

**Step 3 — Suffix stripping (lightweight stemming)**:

```python
def simple_stem(word: str) -> str:
    if len(word) <= 3: return word
    if word.endswith("lling") and len(word) > 6: return word[:-4]   # cancelling → cancel
    if word.endswith("ling")  and len(word) > 5: return word[:-3]   # canceling  → cancel
    if word.endswith("ing")   and len(word) > 5: return word[:-3]   # tracking   → track
    if word.endswith("ment")  and len(word) > 5: return word[:-4]   # shipment   → ship
    if word.endswith("tion")  and len(word) > 5: return word[:-4]   # cancellation → cancel
    if word.endswith("ed")    and len(word) > 4: return word[:-2]   # cancelled  → cancel
    if word.endswith("ly")    and len(word) > 4: return word[:-2]   # quickly    → quick
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]                                              # orders → order
    return word
```

This is not the Porter stemmer — it is a faster, domain-tuned variant. The rules are ordered from most-specific to least-specific, so longer suffixes take priority.

**The merging effect**: Consider all the ways customers phrase "cancel":
```
cancel      → cancel
cancels     → cancel   (strip -s)
cancelled   → cancel   (strip -ed)
cancelling  → cancel   (strip -lling → strip to 'l' → then? no: word[:-4] = cancel)
canceling   → cancel   (strip -ling)
cancellation → cancel  (strip -tion = cancella → no: word[:-4] = cancell → not quite)
```

The practical effect: morphological variants of the same root word all produce the same stem, so a cached answer for "How do I cancel?" also matches "I want to know about cancellation."

**Step 4 — Keyword extraction**:

```python
def extract_keywords(text: str) -> set[str]:
    words = normalize_text(text).split()
    return {simple_stem(w) for w in words if w not in _STOP_WORDS and len(w) > 1}
```

Example:
```
Input:  "How do I cancel my order?"
After normalize_text: "how do i cancel my order"
After split:          ["how", "do", "i", "cancel", "my", "order"]
After stopword remove: ["cancel", "order"]   (how/do/i/my all in STOP_WORDS)
After stem:            {"cancel", "order"}
```

### 5.5 Layer 2 — Keyword Similarity Cache (Jaccard + Inverted Index)

**Intuition**: Two questions are semantically similar if they share most of the same content words after normalization and stemming. We measure this similarity with the Jaccard coefficient, and we use an inverted index to find candidate matches without scanning the entire cache.

#### The Jaccard Similarity Coefficient

Given a query with keyword set *Q* and a cached entry with keyword set *C*:

```
Jaccard(Q, C) = |Q ∩ C| / |Q ∪ C|
```

This value lies in [0, 1]:
- **1.0** — identical keyword sets
- **0.5** — half the keywords are shared
- **0.0** — no shared keywords

**Example 1 — Same intent, different phrasing**:
```
Query:  "cancel my order"   → keywords: {cancel, order}
Cached: "I want to cancel the order" → keywords: {cancel, order}

Jaccard = |{cancel, order} ∩ {cancel, order}| / |{cancel, order} ∪ {cancel, order}|
        = 2 / 2 = 1.0   → perfect match
```

**Example 2 — Related but different intent**:
```
Query:  "cancel order"  → keywords: {cancel, order}
Cached: "track order"   → keywords: {track, order}

Jaccard = |{cancel, order} ∩ {track, order}| / |{cancel, order} ∪ {track, order}|
        = |{order}| / |{cancel, order, track}|
        = 1 / 3 = 0.33   → below threshold 0.45, cache miss
```

**Example 3 — Strong paraphrase**:
```
Query:  "I need to return my item"      → keywords: {return, item}
Cached: "how do I return an item I bought" → keywords: {return, item, bought}

Jaccard = |{return, item}| / |{return, item, bought}|
        = 2 / 3 = 0.67   → above hard_stop 0.70? No. Above threshold 0.45? Yes → cache hit
```

#### The Inverted Index

Naively, computing Jaccard against every entry in a 16,384-entry cache would be O(n) per lookup — too slow. The inverted index reduces this to O(k) where k is the number of keywords in the query (typically 2–5).

**Structure**:
```python
_inverted: dict[str, set[str]]   # keyword → {cache_key_1, cache_key_2, ...}
_key_keywords: dict[str, frozenset[str]]  # cache_key → frozenset(keywords)
```

**Lookup algorithm**:
```
Query keywords: {cancel, order}

Step 1: candidate_keys = inverted["cancel"] ∪ inverted["order"]
        = {key1, key5, key9} ∪ {key1, key3, key9}
        = {key1, key3, key5, key9}

Step 2: For each candidate key, compute Jaccard with the query keywords.
        Only ~4 comparisons instead of 16,384.

Step 3: If any score ≥ hard_stop (0.70): return immediately.
        Else if best score ≥ keyword_threshold (0.45): return best match.
        Else: cache miss.
```

The inverted index makes lookup time proportional to the number of candidates that share at least one keyword with the query — not to the total cache size. For typical queries with 2–5 keywords, this is 10–50 candidates.

**Indexing a new cache entry**:
```python
def _index_entry(self, key: str, query_text: str):
    keywords = frozenset(extract_keywords(query_text))
    self._key_keywords[key] = keywords
    for kw in keywords:
        if kw not in self._inverted:
            self._inverted[kw] = set()
        self._inverted[kw].add(key)
    self._key_trigrams[key] = self._char_trigrams(query_text)
```

Each new cache entry is indexed by all its keywords simultaneously, so the next query containing any of those keywords will find this entry as a candidate.

#### Full Keyword Search Code

```python
def _keyword_search(self, query: str) -> dict | None:
    query_keywords = extract_keywords(query)
    if not query_keywords:
        return None

    # Build candidate set using inverted index
    candidates: dict[str, set[str]] = {}
    for kw in query_keywords:
        for key in self._inverted.get(kw, ()):
            if key not in candidates:
                candidates[key] = self._key_keywords[key]

    if not candidates:
        return None

    best_key = None
    best_score = 0.0

    for key, item_keywords in candidates.items():
        intersection = query_keywords & item_keywords
        score = len(intersection) / len(query_keywords | item_keywords)

        if score >= self._hard_stop:          # ≥ 0.70: high confidence → return now
            self._cache.move_to_end(key)
            self.keyword_hits += 1
            return self._cache[key]

        if score > best_score:
            best_score = score
            best_key = key

    if best_score >= self._keyword_threshold and best_key is not None:
        self._cache.move_to_end(best_key)
        self.keyword_hits += 1
        return self._cache[best_key]

    return None
```

### 5.6 Layer 3 — Character Trigram Fallback (Typo Tolerance)

**Intuition**: The benchmark contains queries with typos — "trfoubles with payment", "cansel my order". Keyword matching fails here because "trfoubles" and "troubles" are different strings. Character trigrams capture the "shape" of a word and tolerate small spelling errors.

**What is a character trigram?** It is every substring of length 3 from the normalized text:

```
"cancel" → {"can", "anc", "nce", "cel"}
```

For a string of length *n*, there are *n − 2* trigrams.

**Jaccard on trigrams**:

```
"cancel" trigrams: {"can", "anc", "nce", "cel"}
"cansel" trigrams: {"can", "ans", "nse", "sel"}

Intersection: {"can"}   (1 shared trigram)
Union:        {"can", "anc", "nce", "cel", "ans", "nse", "sel"}   (7 total)

Jaccard = 1/7 ≈ 0.14   → below threshold (0.45), no match
```

That trigram Jaccard of 0.14 is actually too low for "cancel" vs "cansel" — only one transposition but the trigram overlap is small. The trigram approach is most effective for sentence-level matching, where the overall text has many trigrams and a few misspelled characters barely change the intersection:

```
"how do i cancel my order"  trigrams (from 25 chars): ~23 trigrams including "how", "ow ", "w d", ...
"how do i cansel my order"  trigrams (from 25 chars): ~23 trigrams, differ only around "nse"/"nce"

Jaccard ≈ 21/25 = 0.84   → well above threshold (0.45), cache hit
```

**Why trigrams work for sentences but not isolated words**: A short misspelled word might share only 1 of 4 trigrams with the correct spelling (25%). A long sentence where one word is misspelled might share 21 of 25 trigrams (84%). The trigram layer is specifically designed for sentence-level typo tolerance, not word-level correction.

**Implementation**:
```python
@staticmethod
def _char_trigrams(text: str) -> frozenset[str]:
    text = normalize_text(text)
    if len(text) < 3:
        return frozenset()
    return frozenset(text[i:i+3] for i in range(len(text) - 2))
```

The trigram scan is O(|cache|) — it has no inverted index. This is acceptable because it is only reached after both exact and keyword matching fail, which is relatively rare.

### 5.7 Layer 4 — Semantic Inflight Dedup

**Intuition**: When 128 requests arrive simultaneously, some may be asking similar questions. If request A is already being computed by the GPU, request B (with a similar query) should wait for A's result rather than start a separate GPU inference. This coalesces concurrent similar queries.

**The data structure**:
```python
_inflight: dict[str, asyncio.Future[dict]]        # key → Future being computed
_inflight_keywords: dict[str, frozenset[str]]     # key → keywords of that inflight query
```

**The search**:
```python
def find_similar_inflight(self, query: str) -> asyncio.Future[dict] | None:
    query_kw = frozenset(extract_keywords(query))
    if not query_kw or not self._inflight_keywords:
        return None

    for key, item_kw in self._inflight_keywords.items():
        if not item_kw:
            continue
        inter = query_kw & item_kw
        if not inter:
            continue
        score = len(inter) / len(query_kw | item_kw)
        if score >= self._keyword_threshold:
            future = self._inflight.get(key)
            if future is not None:
                self.dedup_hits += 1
                return future
    return None
```

**Async flow**: The returned `asyncio.Future` is `await`-ed by the incoming request. When the GPU finishes the original request and calls `complete_inflight_by_key`, it calls `future.set_result(response)` — this unblocks all waiting coroutines simultaneously, and each one stores the result under its own cache key:

```python
# In main.py, when similar inflight is found:
similar_inflight = cache.find_similar_inflight(messages[-1].content)
if similar_inflight is not None:
    result = await similar_inflight           # wait for GPU to finish the similar query
    cache.complete_inflight_by_key(cache_key, result)   # store under this query's key too
    return Response(content=result, ...)
```

This is a form of request coalescing — one GPU inference serves many waiting requests.

### 5.8 Layer 5 — Exact Inflight Dedup

**Intuition**: Two identical requests that arrive simultaneously (within the same benchmark batch) should not trigger two separate GPU inferences.

**The future ownership model**:
```python
def claim_inflight_by_key(self, key: str) -> tuple[asyncio.Future[dict], bool]:
    future = self._inflight.get(key)
    if future is not None:
        self.dedup_hits += 1
        return future, False      # is_owner=False → just await the existing future

    future = asyncio.get_running_loop().create_future()
    self._inflight[key] = future
    return future, True           # is_owner=True → you must run the GPU inference
```

**In main.py**:
```python
inflight, is_owner = cache.claim_inflight_by_key(cache_key)
if not is_owner:
    result = await inflight               # not the owner: wait for the owner's result
    return Response(content=result, ...)

# is_owner=True: run GPU inference, then resolve the future for all waiters
response = await engine.generate(request)
response_bytes = orjson.dumps(response)
cache.complete_inflight_by_key(cache_key, response_bytes, query_text=query_text)
return Response(content=response_bytes, ...)
```

**What `complete_inflight_by_key` does**:
```python
def complete_inflight_by_key(self, key, response, query_text=None):
    self._cache[key] = response        # store in cache for future requests
    self._evict_oldest()               # LRU eviction if needed
    future = self._inflight.pop(key, None)
    if future is not None and not future.done():
        future.set_result(response)    # wake up all waiters simultaneously
    self._inflight_keywords.pop(key, None)
    if query_text is not None:
        self._index_entry(key, query_text)  # add to keyword and trigram index
```

One GPU inference → one set_result → N waiters all get the response.

### 5.9 Cache Eviction (LRU)

The cache is bounded at `max_size=16384` entries. When a new entry would exceed this limit, the oldest (least recently used) entry is evicted:

```python
def _evict_oldest(self):
    if len(self._cache) <= self._max_size:
        return

    old_key, _ = self._cache.popitem(last=False)   # remove oldest (front of OrderedDict)

    old_keywords = self._key_keywords.pop(old_key, frozenset())
    for kw in old_keywords:
        bucket = self._inverted.get(kw)
        if bucket is not None:
            bucket.discard(old_key)           # remove from inverted index
            if not bucket:
                del self._inverted[kw]        # clean up empty buckets
    self._key_trigrams.pop(old_key, None)     # remove trigram index
```

LRU order is maintained via Python's `OrderedDict`: every cache hit calls `move_to_end(key)` to mark that entry as recently used. The `popitem(last=False)` call removes the entry at the front, which is the least recently accessed.

The eviction also cleans up the inverted index to prevent stale pointers. An evicted cache entry that still appeared in `_inverted` would cause the keyword search to find it as a candidate, then fail to retrieve it (the key no longer exists in `_cache`).

### 5.10 Threshold Tuning: The Quality-Speed Tradeoff

The cache has three thresholds:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `keyword_threshold` | 0.45 | Minimum Jaccard to accept a keyword cache hit |
| `hard_stop` | 0.70 | Jaccard above this → return immediately without checking other candidates |
| `trigram_threshold` | 0.45 | Minimum trigram Jaccard for the fallback layer |

**What happens when you lower thresholds (e.g., kw=0.30)**:

More queries match cached entries → higher cache hit rate → higher throughput. But "cancel order" (keywords: cancel, order) now matches "track order" (keywords: track, order) with Jaccard = 1/3 = 0.33 — and the customer asking to track their order gets a response about cancellation. Perplexity degrades as the cached response no longer fits the question.

**What happens when you raise thresholds (e.g., kw=0.65)**:

Only near-perfect matches accepted → very few false positives → lower cache hit rate → more GPU work → lower throughput. At kw=0.65, the cache essentially only catches exact paraphrases and provides little benefit beyond the SHA256 layer.

**Threshold search results** (from the experiment log):

| `kw` / `hard_stop` | Hit Rate | Throughput | Perplexity | Verdict |
|---|---|---|---|---|
| 0.65 / 0.82 | ~40% | 40 req/s | 1.2004 | Too conservative |
| 0.55 / 0.75 | ~55% | 51 req/s | 1.2011 | Better |
| **0.45 / 0.70** | **~65%** | **76 req/s** | **1.1992** | **Sweet spot** |
| 0.40 / 0.70 | ~70% | 77 req/s | 1.1967 | Marginal gain |
| 0.35 / 0.65 | ~75% | 76 req/s | 1.1974 | Diminishing returns |
| 0.30 / 0.55 | ~80% | 85 req/s | 1.2037 | Perplexity degrades |

The kw=0.45, hard_stop=0.70 configuration was chosen as the sweet spot: it catches true paraphrases ("cancel my order" ↔ "I want to cancel the order") while rejecting genuinely different queries ("cancel order" vs "track order").

### 5.11 Why Keyword Stemming Beats Embeddings Here

The original approach used sentence-transformer embeddings (MiniLM-L6-v2) with FAISS IndexFlatIP for semantic matching. We replaced it with keyword stemming. Here is why:

| Property | Keyword Jaccard | Sentence Embeddings |
|---|---|---|
| Latency | < 0.1 ms | 5–50 ms |
| GPU usage | None (pure Python) | Competes with LLM on GPU/CPU |
| Docker image size | No new deps | +sklearn, faiss, sentence-transformers |
| Morphological coverage | "cancel/cancelling/cancelled" → same | Depends on training data |
| Interpretability | Inspect shared keywords | Black box 384-dim vector |
| Lookup complexity | O(k) via inverted index | O(n) scan or O(log n) ANN |
| Negation handling | Blind to "don't cancel" | Usually captures negation |
| Semantic generalization | "refund" ≠ "money back" | "refund" ≈ "money back" |

**Why keyword wins for this specific task**:

1. **Temperature=0** means determinism — cached answers are always correct for their original query.
2. **Short queries (~12 tokens avg)** — with so few words, keyword overlap is a strong signal. Long queries would dilute Jaccard.
3. **Narrow domain** — 17 intents with clear keyword patterns. The words "cancel", "refund", "track", "payment" carry almost all the intent signal.
4. **36% exact duplicates** — even the SHA256 layer alone handles a third of all traffic. The semantic layer only needs to cover the paraphrase gap.
5. **No GPU budget** — on 16 GB VRAM with a 4 GB model, loading an 80 MB embedding model would reduce available KV cache.

The embedding model was removed after testing showed **P99 halved** (from 24s to 16s) when the embedding computation was eliminated. The embedding was the bottleneck, not the matching quality.

---

## 6. GPU Optimizations

### 6.1 The Memory-Bandwidth Bottleneck (with Math)

**Intuition**: LLM inference is not limited by how fast the GPU can compute — it is limited by how fast GPU memory can deliver weights to the compute units. This is called being "memory-bandwidth bound."

**Why?** Each token generation (decode step) requires a forward pass through all 36 transformer layers. Each layer's weights must be read from VRAM. For Qwen3-4B in BF16:

```
Model size (BF16):     4B params × 2 bytes/param  =  8 GB
RTX 5080 bandwidth:    960 GB/s
Time per decode step:  8 GB / 960 GB/s            =  8.3 ms
```

That 8.3 ms is spent almost entirely on memory transfers, not arithmetic. The tensor cores are idle most of the time, waiting for data.

**Batching amortizes the cost**: If 128 requests are in-flight simultaneously, all 128 tokens are generated in one forward pass. The 8.3 ms is shared across 128 requests:

```
Per-token time (128 batch): 8.3 ms / 128  =  0.065 ms
Per-request (178 tokens):   178 × 0.065 ms = 11.6 ms
Theoretical throughput:     128 / 0.0116 s ≈ 11,034 req/s
```

We achieve 429 req/s in practice because: not all 128 slots are always occupied (requests finish and new ones are admitted asynchronously), the KV cache limits concurrent sequences, and sampling/scheduling add overhead.

**The implication**: Any optimization that reduces model size directly improves throughput, because it reduces the memory-read time per decode step.

### 6.2 FP8 Weight Quantization

**Intuition**: Instead of storing each model weight as a 16-bit float (BF16), store it as an 8-bit float (FP8). The model becomes half the size, so each decode step reads half as much memory.

```
BF16 model:   8 GB → decode step takes 8.3 ms
FP8 model:    4 GB → decode step takes 4.2 ms
```

Theoretically, FP8 doubles throughput. In practice, the gain is +21% (287 → 347 req/s) because the decode step also includes other operations (attention, KV cache read/write, sampling) that are not reduced by quantization.

**Why FP8 works so well on RTX 5080 (SM120)**:

The RTX 5080 has dedicated FP8 tensor cores that can compute matrix multiplications directly in FP8 without converting to a higher precision first. This is crucially different from INT4, where the weights must be dequantized to BF16 before computation. vLLM 0.19 includes SM120-specific CUTLASS FP8 GEMM kernels (vLLM PR #38325) with a "swapAB" strategy that improved effective bandwidth by **69%** for the small batch sizes typical in decode.

**The FP8 format (E4M3)**:
- 1 sign bit
- 4 exponent bits (range: ±448)
- 3 mantissa bits (precision: ~3 decimal digits)

BF16 has 8 exponent bits and 7 mantissa bits. FP8 E4M3 sacrifices precision for size. For transformer weights, this is acceptable because the model has learned to operate effectively in this narrower dynamic range, and the quantization error is small enough not to affect output quality (perplexity: 1.199 with BF16 vs 1.199 with FP8 — identical).

**The sampler OOM problem**: At `max_num_seqs=256`, the logits tensor (256 × 151,936 vocab × 4 bytes) = 148 MB needs to be allocated during warmup. With FP8 already using more memory than BF16 in some transient paths, this OOMed. Fix: reduce to `max_num_seqs=192` and `gpu_memory_utilization=0.92`.

**Why not INT4?**

INT4 would theoretically halve the model again (2 GB → 2.1 ms decode step), but on SM120 there is no hardware INT4 tensor core acceleration. The available paths are:

- **AWQ-Marlin kernels**: Crash (Marlin PTX not compiled for SM120)
- **GPTQ with PyTorch fallback**: Works but 40% slower than FP8 (dequant overhead)
- **AWQ with Triton**: Works but 40% slower (Triton dequant does not use tensor cores)

The dequantization compute overhead exceeds the memory bandwidth savings on SM120. FP8 is definitively the optimal quantization for this GPU.

### 6.3 Inductor Compilation Without CUDA Graphs

**Intuition**: GPU kernel launches have overhead (~5–10 µs each). In a single transformer layer, there are many small operations: RMSNorm, quantize, matmul, add, SiLU, matmul, KV-cache-write, attention... If these can be fused into fewer larger kernels, we eliminate the launch overhead.

**The standard vLLM approach** uses CUDA graphs: capture the entire decode step as a static computation graph, then replay it. This eliminates launch overhead almost entirely. But CUDA graph capture pre-allocates memory for every tensor in every captured shape — on 16 GB VRAM, this OOMs even after FP8 reduces the model to 4 GB.

**The novel insight**: torch.compile Inductor and CUDA graphs are independent features. `enforce_eager=True` disables both. But you can enable Inductor (kernel fusions) while explicitly disabling CUDA graph capture:

```python
from vllm.config import CompilationConfig

kwargs["compilation_config"] = CompilationConfig(
    mode=3,                    # vLLM "O2": enable Inductor compilation
    cudagraph_mode="none",     # do NOT capture CUDA graphs
)
```

**What Inductor does**: It JIT-compiles PyTorch ops into fused CUDA kernels. The vLLM O2 level enables these specific fusions:

- **`fuse_norm_quant`**: RMSNorm + FP8 quantization → one kernel instead of two
- **`fuse_act_quant`**: SiLU activation + quantization → one kernel instead of two
- **`fuse_rope_kvcache`**: RoPE position encoding + KV cache write → one kernel instead of two

With 36 transformer layers and ~3 fusions per layer:

```
Fused kernel launches eliminated: 36 × 3 = 108 per decode step
Overhead per launch:              ~5 µs
Total overhead eliminated:        108 × 5 µs = 0.54 ms per decode step
```

At 429 req/s, each decode step serves ~128 requests in ~4.2 ms. Eliminating 0.54 ms (13%) of that overhead translates to +8.5% throughput (364 → 395 req/s).

**Startup cost**: Inductor compilation happens on the first forward pass. It takes 35 seconds (vs 20 seconds without compilation), but is a one-time cost.

### 6.4 FlashInfer Attention Backend

**Intuition**: The attention mechanism in a transformer reads the current query and compares it with all past key-value pairs (the KV cache) to compute attention weights. For long sequences with many concurrent requests, this is expensive. FlashInfer is an optimized attention library specifically designed for LLM inference.

**Why FlashInfer instead of FlashAttention3?**

FlashAttention3 crashes on consumer Blackwell (SM120). The official vLLM recommendation for SM120 is FlashInfer. The critical detail: setting the environment variable `VLLM_ATTENTION_BACKEND=FLASHINFER` is insufficient — in vLLM 0.19, you must also pass it directly to the engine:

```python
kwargs["attention_backend"] = "flashinfer"
```

**What FlashInfer handles**: Qwen3-4B uses Grouped Query Attention (GQA) with 32 query heads but only 8 key-value heads (4:1 ratio). FlashInfer handles this natively with efficient page-based KV cache management (block_size=16 tokens per page).

---

## 7. Memory Optimizations

### 7.1 Tight `max_model_len=288`

**Intuition**: vLLM pre-allocates KV cache blocks for up to `max_model_len` tokens per sequence. If this value is larger than necessary, we waste KV cache memory — memory that could have been used to run more concurrent sequences.

**The discovery**: The default configuration used `max_model_len=320`, estimated from a rough assumption of ~50 tokens of chat template overhead. We measured the actual template overhead:

```python
tokenizer.apply_chat_template(
    [{"role": "user", "content": "longest possible query"}],
    tokenize=True, add_generation_prompt=True, enable_thinking=False
)
# Template tokens: 8   (not 50!)
```

Minimum required context window:
```
Template overhead:     8 tokens
Max input length:     23 tokens  (measured from dataset)
Max output length:   256 tokens  (benchmark max_tokens=256)
Safety margin:         1 token
─────────────────────────────────
Total max_model_len: 288 tokens
```

**Why this matters**:

Each sequence in the KV cache occupies:
```
Tokens per sequence: 288
KV cache dtype:      BF16 (2 bytes per value)
KV pairs per token:  2 (key + value)
Attention heads:     8 KV heads
Hidden dim per head: 128
Transformer layers:  36

Memory per sequence: 288 × 2 × 2 × 8 × 128 × 36 / (1024³) ≈ 0.40 MB
```

Reducing from 320 to 288 frees 10% per sequence. With 192 concurrent sequences, that is:
```
Freed memory: 0.10 × 0.40 MB × 192 ≈ 7.7 MB
```

The freed memory allows slightly more KV cache pages, enabling marginally more concurrent sequences and better batch utilization. Impact: +4.9% throughput (347 → 364 req/s).

**The failure at 272**: Testing `max_model_len=272` dropped throughput to 298 req/s (-18%) because some queries exceeded this length, causing truncated outputs and generation failures. 288 is the precise minimum that works for all queries in the dataset.

### 7.2 Sampler OOM Fix

**Intuition**: During initialization, vLLM's sampler allocates a "dummy" logits tensor to warm up the sampling kernel. This tensor is sized `(max_num_seqs × vocab_size × 4 bytes)`.

For Qwen3-4B with vocab_size=151,936:
```
Logits tensor (max_num_seqs=256): 256 × 151,936 × 4 bytes = 155.6 MB
Logits tensor (max_num_seqs=192): 192 × 151,936 × 4 bytes = 116.7 MB
Difference:                        38.9 MB saved
```

With the FP8 model occupying ~4 GB and `gpu_memory_utilization=0.95` leaving only ~11.2 GB for everything else, the 155 MB sampler tensor caused OOM during initialization. Reducing to `max_num_seqs=192` at `gpu_memory_utilization=0.92` provides enough headroom.

This also leaves more room for KV cache pages, partially offsetting the reduction in `max_num_seqs`.

---

## 8. Scheduling Optimizations

### `stream_interval=256`

**Intuition**: By default, vLLM sends a notification to the Python event loop after every generated token (stream_interval=1). Even for non-streaming responses, this means 178 host-device synchronizations per request — each one involves crossing the Python/CUDA boundary and scheduling an event loop callback. Setting `stream_interval=256` delivers tokens in batches of 256, reducing this to 1 synchronization per response.

This is the most impactful scheduling change: **+8.6% throughput** (395 → 429 req/s).

### `async_scheduling=True`

Overlaps CPU-side scheduling (deciding which requests to run next) with GPU execution (running the current batch). Without this, the CPU idles while the GPU runs, then the GPU idles while the CPU schedules. With it, the CPU prepares batch *k+1* while the GPU processes batch *k*.

### `performance_mode="throughput"`

Selects throughput-oriented kernel configurations within vLLM. Primarily effective when CUDA graphs are enabled, provides marginal benefit otherwise.

### What Didn't Work: `scheduler_reserve_full_isl=False`

This flag allows the scheduler to admit new requests without checking whether the full input fits in the KV cache. We tested this expecting more concurrent sequences, but it caused **preemptions** — requests being evicted mid-generation to make room for others, wasting GPU work and producing different (lower quality) outputs:

```
With scheduler_reserve_full_isl=False: 405 req/s, perplexity=1.2046
Without:                               429 req/s, perplexity=1.2017
```

Removing the flag improved both throughput and perplexity. The preemptions were net-negative.

---

## 9. Application-Layer Optimizations

These optimizations reduce per-request Python overhead, which matters when the cache handles ~50% of traffic.

**Zero-copy cache hits**: Responses are serialized to bytes at insertion time:
```python
response_bytes = orjson.dumps(response)
cache.complete_inflight_by_key(cache_key, response_bytes, ...)
```

Cache hits return raw bytes directly:
```python
return Response(content=cached, media_type="application/json")
```

No Pydantic serialization, no JSON encoding — the bytes go straight to the network.

**Raw JSON parsing**: Incoming requests are parsed with `orjson.loads()` directly, bypassing Pydantic validation for the request body. Pydantic is only used to construct `ChatMessage` objects (which have no validation logic).

**LRU-cached SamplingParams**: The benchmark always sends `temperature=0, max_tokens=256`. Using `@lru_cache` on `SamplingParams(temperature, max_tokens)` means the same Python object is reused for every request — no allocation per request.

**Direct tokenization**: `tokenizer.apply_chat_template(messages, tokenize=True, ...)` returns token IDs directly, avoiding the double-pass of first getting text then tokenizing.

**uvloop**: The asyncio event loop is replaced with uvloop, a C-based implementation that is significantly faster for I/O-bound async workloads.

---

## 10. The Normalization + Stemming Pipeline (Code Walkthrough)

The complete pipeline for a query "How do I cancel my Order??":

```
Input: "How do I cancel my Order??"

Step 1 — normalize_text():
  .lower()          → "how do i cancel my order??"
  sub non-alnum     → "how do i cancel my order"
  collapse spaces   → "how do i cancel my order"

Step 2 — split():
  ["how", "do", "i", "cancel", "my", "order"]

Step 3 — filter stopwords:
  "how"    → in STOP_WORDS → discard
  "do"     → in STOP_WORDS → discard
  "i"      → in STOP_WORDS → discard
  "cancel" → not in STOP_WORDS → keep
  "my"     → in STOP_WORDS → discard
  "order"  → not in STOP_WORDS → keep

Step 4 — simple_stem():
  "cancel" → no suffix matches → "cancel"
  "order"  → ends in "r", not "-s"/"-ed"/etc. → "order"

Result: {"cancel", "order"}
```

The same pipeline on "I'd like to cancel an order I placed" yields:
```
After normalize: "id like to cancel an order i placed"
After split: ["id", "like", "to", "cancel", "an", "order", "i", "placed"]
After stopwords: ["cancel", "order", "placed"]
After stem: {"cancel", "order", "placed"}
```

Jaccard with {"cancel", "order"}:
```
Intersection: {cancel, order}  → size 2
Union: {cancel, order, placed} → size 3
Score: 2/3 = 0.67 → above threshold 0.45 → cache hit
```

---

## 11. Optimization Journey: Progressive Gains

All numbers are full 13,435-request cold-start benchmarks on RTX 5080 at 128 concurrency.

| Step | Optimization Added | Throughput | Gain vs Prev | Cumulative | Perplexity |
|------|-------------------|-----------|-------------|-----------|-----------|
| 0 | Unoptimized baseline | 18 req/s | — | 1× | 1.200 |
| 1 | Multi-layer keyword cache + FlashInfer + enforce_eager | 287 req/s | +15.9× | +15.9× | 1.199 |
| 2 | FP8 weight quantization | 347 req/s | +21% | +19.3× | 1.199 |
| 3 | Tight `max_model_len=288` | 364 req/s | +4.9% | +20.2× | 1.200 |
| 4 | Inductor compilation (no CUDA graphs) | 395 req/s | +8.5% | +21.9× | 1.198 |
| 5 | `stream_interval=256` | **429 req/s** | **+8.6%** | **+23.8×** | **1.202** |

**Cold-start vs warm cache**:

"Cold-start" means the server was freshly started with an empty cache. All 13,435 benchmark prompts arrive unseen. The cache gradually fills as requests complete. This is the realistic grading scenario.

"Warm cache" (second benchmark run on the same server) shows what happens when all queries have already been answered. The cache serves ~93% of requests instantly:

```
Warm cache throughput: 2,037 req/s  (4.7× over cold-start)
Warm P50: 45 ms, P95: 83 ms, P99: 110 ms
```

---

## 12. Dead Ends: 19 Approaches That Didn't Work

Each rejected approach is documented with the root cause to prevent re-testing.

| Approach | Result | Root Cause |
|---|---|---|
| Speculative decoding (ngram) | −6% throughput | Verification overhead scales superlinearly at BS=128; the draft tokens compete with the main model for batch slots |
| CUDA graphs (full or piecewise) | OOM | Graph capture pre-allocates for all captured shapes; exhausts 16 GB even with FP8 |
| FP8 KV cache | −44% throughput | Per-token dequantization during attention costs more than the reduced bandwidth |
| AWQ-Marlin INT4 | CRASH | Marlin PTX binary targets SM75–SM90; not compiled for SM120 |
| AWQ with Triton backend | −40% throughput | Generic Triton dequant does not use SM120 tensor cores |
| GPTQ INT4 (no Marlin) | −40% throughput, −0.017 perplexity | PyTorch dequant is slower than FP8 matmul; quality degrades |
| Prefix caching | Neutral | Prefill is negligible for 12-token average inputs; cache warms too slowly to help |
| FlashInfer sampler | −8.6% throughput | FlashInfer sampler is slower than PyTorch `argmax` for temperature=0 (top-1 sampling) |
| FlashInfer autotune | −2.3% throughput | Default kernel selection is already optimal for this workload |
| Cache seeding (25 warmup queries) | −8.6% throughput | Seed queries create false-positive keyword matches for unrelated incoming queries |
| Aggressive stemming | −15% throughput | Over-merging: unrelated words collapse to same stem, returning wrong cached answers |
| DBO (dual batch overlap) | CRASH | Requires DeepEP expert parallelism kernels not available in single-GPU setup |
| V1 multiprocessing disabled | −7% throughput | Single-process mode blocks the asyncio event loop during scheduling |
| TF32 matmul precision | Neutral | Model uses BF16; TF32 only affects FP32 operations |
| `max_model_len=272` | −18% throughput | Below minimum for some queries; causes truncated outputs and generation failures |
| `max_num_seqs=224+` | −10% throughput | Sampler warmup OOM eats into KV cache headroom |
| `max_num_batched_tokens=16384` | −6% throughput | Larger scheduling buffers waste memory that could be KV cache |
| CPU KV offloading | CRASH | Meta tensor error with FP8 quantization |
| `scheduler_reserve_full_isl=False` | −2% quality | Causes request preemptions; wasted GPU work and different (lower quality) outputs |

---

## 13. Final Architecture

```
Client (128 concurrent HTTP connections)
    │
    ▼
FastAPI + uvicorn (uvloop event loop)
    │
    ├─── [Cache Layer 1] SHA256 exact-match → return pre-serialized bytes  O(1)
    │
    ├─── [Cache Layer 2] Keyword Jaccard (stemmed, inverted index)          O(k)
    │         ├── score ≥ 0.70 (hard_stop): return immediately
    │         └── score ≥ 0.45: return best match
    │
    ├─── [Cache Layer 3] Character trigram Jaccard fallback                 O(|cache|)
    │         └── score ≥ 0.45: return (typo-tolerant)
    │
    ├─── [Cache Layer 4] Semantic inflight dedup (keyword Jaccard ≥ 0.45) 
    │         └── await existing GPU future, populate cache for this key
    │
    ├─── [Cache Layer 5] Exact inflight dedup (same SHA256 key)
    │         └── await existing GPU future, return
    │
    └─── GPU Inference (cache miss path, ~50% of requests)
              │
              ▼
         vLLM V1 AsyncLLMEngine
              ├── FP8 E4M3 weight quantization (CUTLASS SM120 kernels)
              ├── FlashInfer attention backend (GQA 4:1, page KV cache)
              ├── Inductor compilation mode=3 (kernel fusions, no CUDA graphs)
              ├── Async scheduling + stream_interval=256
              ├── max_model_len=288, max_num_seqs=192, gpu_util=0.92
              └── Continuous batching (up to 128 concurrent sequences)
```

---

## 14. Configuration Reference

All parameters are in [constants.py](track2_chat/app/constants.py) and can be overridden via environment variables.

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `Qwen/Qwen3-4B-Instruct-2507` | HuggingFace model identifier |
| `MAX_MODEL_LENGTH` | `288` | Maximum token context window (measured, not estimated) |
| `GPU_MEMORY_UTILIZATION` | `0.92` | Fraction of VRAM for model + KV cache (rest for sampler) |
| `QUANTIZATION` | `fp8` | Weight quantization format (`fp8`, `None`) |
| `KV_CACHE_DTYPE` | `auto` | KV cache dtype (`auto`=BF16; FP8 was tested and rejected) |
| `MAX_NUM_SEQS` | `192` | Max concurrent sequences (limited by sampler OOM) |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | Scheduler token buffer size |
| `ENABLE_PREFIX_CACHING` | `false` | vLLM prefix caching (neutral; prefill negligible for short inputs) |
| `ENABLE_CHUNKED_PREFILL` | `true` | Reduce prefill latency spikes |
| `SEMANTIC_CACHE_ENABLED` | `true` | Enable keyword+trigram semantic cache |
| `CACHE_MAX_SIZE` | `16384` | Maximum number of cached responses (LRU eviction) |
| `ENFORCE_EAGER` | `false` | Use Inductor compilation (false = compile; true = no compile) |
| `SPEC_DECODE_ENABLED` | `false` | Speculative decoding (tested, harmful at BS=128) |

Environment variables from Dockerfile:

| Variable | Value | Purpose |
|---|---|---|
| `VLLM_ATTENTION_BACKEND` | `FLASHINFER` | Backup hint (not sufficient alone; must also pass to engine args) |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Reduce memory fragmentation |

---

## 15. Deployment Guide

### Docker (recommended)

```bash
# Build the container
cd track2_chat
docker compose build

# Start the server (GPU passthrough via docker-compose.yaml)
docker compose up -d

# Verify readiness (polls until engine is initialized, ~35s)
until curl -sf http://localhost:8000/ready; do sleep 2; done
echo "Server ready"

# Stop
docker compose stop
```

The Dockerfile uses:
- Base: `nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04`
- Python: 3.12 (via uv)
- Entrypoint: `uvicorn app.main:app --host 0.0.0.0 --port 8000 --loop uvloop --no-access-log`

### Modal (serverless)

```bash
cd track2_chat

# One-time: download model to Modal volume
modal run modal_deploy.py::download_model

# Deploy
modal deploy modal_deploy.py

# Development mode (hot reload)
modal serve modal_deploy.py
```

### Benchmark

```bash
cd benchmark
uv sync
uv run runner_chat.py \
    --url http://localhost:8000 \
    --data data/track2/train.jsonl \
    --concurrency 128
```

The benchmark sends 13,435 prompts with 128 concurrent workers. Each request uses `temperature=0, max_tokens=256`. It reports: throughput (req/s), P50/P95/P99 latency, average perplexity, and failure count.

---

## 16. Benchmark Results

### Final Results (Cold-Start, 13,435 Requests, 128 Concurrency)

```
Throughput:   429.29 req/s
P50 Latency:  6.7 ms
P95 Latency:  2,715 ms
P99 Latency:  4,464 ms
Perplexity:   1.2017
Failures:     0
```

### Progressive Optimization Results (All Cold-Start)

| Config | Throughput | P50 | P95 | P99 | Perplexity | Failures |
|--------|-----------|-----|-----|-----|-----------|----------|
| Unoptimized baseline | 18.15 req/s | 6,687 ms | — | 9,054 ms | 1.1997 | 429 |
| + Cache + FlashInfer | 287.68 req/s | 1.7 ms | 4,101 ms | 6,230 ms | 1.1985 | 0 |
| + FP8 quantization | 347.50 req/s | 2.6 ms | 3,417 ms | 5,594 ms | 1.1991 | 0 |
| + `max_model_len=288` | 364.53 req/s | 4.5 ms | 3,229 ms | 5,033 ms | 1.1997 | 0 |
| + Inductor compilation | 395.48 req/s | 3.2 ms | 2,916 ms | 4,645 ms | 1.1979 | 0 |
| + `stream_interval=256` | **429.29 req/s** | **6.7 ms** | **2,715 ms** | **4,464 ms** | **1.2017** | **0** |

### Warm-Cache Performance (Second Run)

When the cache is already populated from a prior benchmark run:

```
Throughput:   2,037 req/s  (4.7× over cold-start)
P50 Latency:  45 ms
P95 Latency:  83 ms
P99 Latency:  110 ms
```

Note: This represents a ~93% cache hit rate — not representative of grading on unseen validation data.

### Key Rejected Configurations

| Config | Throughput | Perplexity | Why Rejected |
|--------|-----------|-----------|-------------|
| CUDA graphs (gpu_util=0.80) | 113 req/s | 1.2005 | OOM at higher gpu_util; reduced KV cache at 0.80 |
| FP8 KV cache | 290 req/s | 1.1964 | Dequant overhead per attention step |
| GPTQ INT4 | 255 req/s | 1.2181 | Slow PyTorch dequant + quality loss |
| FlashInfer sampler | 317 req/s | 1.1983 | Slower than PyTorch argmax at temp=0 |
| `scheduler_reserve_full_isl=False` | 405 req/s | 1.2046 | Preemptions hurt both speed and quality |
| `max_model_len=272` | 298 req/s | 1.1974 | Truncated outputs |
