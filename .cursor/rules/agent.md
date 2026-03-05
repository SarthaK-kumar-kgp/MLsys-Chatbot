# MLSys Project - Track 2: Customer Support Chatbot

## Project Context

CS4262/5462 Machine Learning Systems - Project 1: LLM Serving
**Track B**: High-throughput LLM serving engine for interactive chat

### Architecture
- **Model**: Qwen/Qwen3-4B-Instruct-2507 (max context 8192)
- **Framework**: vLLM with FlashInfer attention backend
- **API**: FastAPI serving on port 8000
- **Deployment**: Docker (linux/amd64) on NVIDIA RTX 5080
- **Model cache**: `/root/.cache/huggingface` (volume mounted)

### Endpoints
- `GET /health` - liveness check
- `GET /ready` - engine readiness check
- `POST /v1/chat/completions` - chat completion (returns `output` + `logprobs`)

### Performance Targets
- Concurrency: 128 simultaneous requests
- Metrics: P50/P95 latency, throughput (req/s), perplexity
- Optimize for: low TTFT, high throughput, cache eviction strategy

## Monorepo Structure (Nx)

```
apps/
  chat-engine/          # Main serving engine
    app/
      main.py           # FastAPI entrypoint
      chat_engine.py    # Core vLLM engine logic (MAIN OPTIMIZATION TARGET)
      schemas.py        # Pydantic request/response models
      constants.py      # Model name, max length
    Dockerfile          # CUDA 12.8 + uv based
    docker-compose.yaml
    pyproject.toml      # Python deps (fastapi, uvicorn, pydantic, vllm)
    scripts/
  benchmark/            # Benchmark runner
    runner_chat.py
    data/track2/
```

## Key Commands

```bash
# Nx tasks
npx nx run chat-engine:build        # Build Docker image
npx nx run chat-engine:serve        # docker compose up --build -d
npx nx run chat-engine:stop         # docker compose stop
npx nx run chat-engine:logs         # docker compose logs -f
npx nx run chat-engine:lint         # ruff check
npx nx run chat-engine:format       # ruff format
npx nx run chat-engine:push-image   # Build + push to GHCR
npx nx run benchmark:run            # Benchmark at concurrency=128
npx nx run benchmark:run-local      # Benchmark at concurrency=16

# Mise tasks
mise run install / build / serve / benchmark / lint / test
```

## Development Guidelines

- All optimization work happens in `apps/chat-engine/app/chat_engine.py`
- Preserve the HTTP endpoint contract (health, ready, chat/completions)
- Response must include `output` (str) and `logprobs` (list[float])
- Docker image must be linux/amd64 for grading
- Do not add external API calls - outbound network is disabled during eval
- Model weights are pre-cached, do not download at runtime
- Use conventional commits: `feat()`, `fix()`, `opt()`, `chore()`
- Branch naming: `feat/<desc>`, `fix/<desc>`, `opt/<desc>`
- All work goes through PRs against `main`
