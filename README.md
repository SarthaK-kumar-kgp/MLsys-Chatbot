# MLSys - LLM Serving Engine

> CS4262/5462 Machine Learning Systems | Project 1 | **Track B: Customer Support Chatbot**

## Overview

A high-throughput LLM serving engine optimized for interactive chat, built to handle **128 concurrent sessions** with low latency and intelligent KV cache management.

| Spec | Detail |
|------|--------|
| **Model** | Qwen/Qwen3-4B-Instruct-2507 |
| **Max Context** | 8,192 tokens |
| **Target GPU** | NVIDIA RTX 5080 |
| **Platform** | Docker (linux/amd64) |
| **Framework** | vLLM + FlashInfer |

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check |
| `GET` | `/ready` | Engine readiness |
| `POST` | `/v1/chat/completions` | Chat completion (returns `output` + `logprobs`) |

---

## Getting Started

### Prerequisites

- [mise](https://mise.jdx.dev/) - manages Python, Node, uv
- [Docker](https://docs.docker.com/get-docker/) with NVIDIA GPU support
- NVIDIA GPU with CUDA 12.8+

### Setup

```bash
# Install tooling via mise
mise install

# Install project dependencies
mise run install
# or manually:
npm install
cd apps/chat-engine && uv sync
```

### Running the Engine

```bash
# Build and start (Docker Compose)
npx nx run chat-engine:serve

# Check logs
npx nx run chat-engine:logs

# Quick smoke test
npx nx run chat-engine:test

# Stop
npx nx run chat-engine:stop
```

### Running Benchmarks

```bash
# Full benchmark (concurrency=128)
npx nx run benchmark:run

# Local testing (concurrency=16)
npx nx run benchmark:run-local
```

---

## Project Structure

```
mlsys/
├── apps/
│   ├── chat-engine/           # LLM serving engine
│   │   ├── app/
│   │   │   ├── main.py        # FastAPI entrypoint
│   │   │   ├── chat_engine.py # vLLM engine (optimization target)
│   │   │   ├── schemas.py     # Request/response models
│   │   │   └── constants.py   # Model config
│   │   ├── scripts/           # Test scripts
│   │   ├── Dockerfile
│   │   ├── docker-compose.yaml
│   │   └── pyproject.toml
│   └── benchmark/             # Benchmark runner
│       ├── runner_chat.py
│       └── data/track2/
├── mise.toml                  # Tool versions & task runner
├── nx.json                    # Nx workspace config
├── CLAUDE.md                  # AI assistant context
└── .cursor/rules/agent.md     # Cursor AI context
```

---

## Development Workflow

### Branching

| Pattern | Use |
|---------|-----|
| `feat/<desc>` | New features |
| `fix/<desc>` | Bug fixes |
| `opt/<desc>` | Performance optimizations |
| `chore/<desc>` | Tooling, config, maintenance |

All work merges into `main` via Pull Requests.

### Nx Commands Cheatsheet

```bash
npx nx run chat-engine:build        # Build Docker image
npx nx run chat-engine:serve        # Start engine
npx nx run chat-engine:stop         # Stop engine
npx nx run chat-engine:logs         # Tail logs
npx nx run chat-engine:test         # Smoke test
npx nx run chat-engine:lint         # Lint (ruff)
npx nx run chat-engine:format       # Format (ruff)
npx nx run chat-engine:push-image   # Build + push to GHCR
npx nx run benchmark:run            # Benchmark (128 concurrency)
npx nx run benchmark:run-local      # Benchmark (16 concurrency)
```

### Mise Tasks

```bash
mise run install     # Install all deps
mise run build       # Build Docker image
mise run serve       # Start engine
mise run benchmark   # Run benchmark
mise run lint        # Lint all
mise run test        # Test all
```

---

## Deployment (Vast.ai)

```bash
# 1. Login to GHCR
echo $CR_PAT | docker login ghcr.io -u $GITHUB_USERNAME --password-stdin

# 2. Build + push
npx nx run chat-engine:push-image

# 3. Get digest
docker inspect --format='{{index .RepoDigests 0}}' ghcr.io/$GITHUB_USERNAME/chat-engine:latest
```

Then create a Vast.ai template with your image digest and rent an RTX 5080 instance.

---

## Evaluation Criteria

- **Latency**: End-to-end P50 and P95
- **Throughput**: Requests per second at concurrency=128
- **Perplexity**: Output quality (lower is better)
- Optimizations: novelty and effectiveness

---

## Team

Built with vLLM, FastAPI, Nx, and mise.
