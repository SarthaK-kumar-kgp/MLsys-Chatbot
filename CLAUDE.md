# MLSys Project - Track 2: Customer Support Chatbot

## Project Overview

CS4262/5462 Machine Learning Systems - Project 1: LLM Serving
**Track 2**: High-throughput serving engine for interactive chat (Qwen3-4B-Instruct-2507)

### Key Specs
- Model: `Qwen/Qwen3-4B-Instruct-2507` with max context 8192
- Endpoints: `GET /health`, `GET /ready`, `POST /v1/chat/completions`
- Target concurrency: 128 simultaneous requests
- Evaluation: P50/P99 latency, throughput (req/s), perplexity
- Runtime: Single NVIDIA RTX 5080, Docker (amd64), model at `/root/.cache/huggingface`
- Framework: vLLM with FlashInfer attention backend

## Repository Structure

```
track2_chat/           # Serving engine (Python/FastAPI/vLLM/Docker)
  app/
    main.py            # FastAPI entry point with lifespan + caching
    chat_engine.py     # vLLM engine with optimizations
    constants.py       # Configuration constants (env-configurable)
    schemas.py         # Pydantic request/response models
    cache.py           # Exact-match response cache
  Dockerfile
  docker-compose.yaml
  pyproject.toml
  modal_deploy.py      # Modal GPU deployment script
benchmark/             # Benchmark runner (from mlsys_llm_benchmark)
  runner_chat.py       # Track 2 benchmark script
  data/track2/train.jsonl  # 13,435 benchmark prompts
  pyproject.toml
plan.md                # Optimization tracking
```

## Quick Commands

### Engine (Docker)
```bash
cd track2_chat
docker compose up --build -d     # Build & start engine
docker compose stop              # Stop engine
curl http://localhost:8000/ready  # Check readiness
```

### Engine (Modal)
```bash
cd track2_chat
modal run modal_deploy.py::download_model   # Download model (one-time)
modal deploy modal_deploy.py                # Deploy server
modal serve modal_deploy.py                 # Dev mode (hot reload)
```

### Benchmark
```bash
cd benchmark
uv sync
uv run runner_chat.py --url http://localhost:8000 --data data/track2/train.jsonl --concurrency 128
```

## Git Workflow

- NEVER add "Co-Authored-By" lines to commit messages
- `main` branch is protected - all work goes through PRs
- Branch naming: `feat/<description>`, `fix/<description>`, `opt/<description>`
- Use `/create-pr` command below for PR creation

## Planning & Experimentation

All optimization work MUST be tracked in `plan.md` at the repo root. This is the single source of truth for what we've tried, what worked, and what to do next.

### Hard Rules

1. **Before starting work**: Read `plan.md` (if it exists) to understand current state
2. **After EVERY benchmark/experiment run**: IMMEDIATELY update `plan.md` before doing anything else. No exceptions.
3. **After EVERY exploration phase**: Log discoveries, surprises, and decisions
4. **When using subagents**: Save important findings from agent research back to `plan.md`

### What Goes in plan.md

- **Current goal** and hypothesis being tested
- **Experiment log** with configs tried and results (P50, P99, throughput, perplexity)
- **Discoveries & surprises** found during exploration
- **Decisions made** and rationale (why X over Y)
- **Key skills/techniques** learned that may be reusable
- **Next steps** prioritized by expected impact
- **Dead ends** so we don't repeat them

### Benchmark Results Format

When logging benchmark results, always include:
```
| Run | Config Change | P50 (ms) | P99 (ms) | Throughput (req/s) | Perplexity | Notes |
|-----|--------------|----------|----------|-------------------|------------|-------|
```

---

# /plan

Create or update `plan.md` to track the current optimization work.

## Steps

1. **Check if `plan.md` exists**:
   - If yes: read it and understand current state
   - If no: create it with the template below

2. **Understand context**:
   - Read recent git log for what's been done
   - Check current branch and any in-progress work
   - If benchmark results are available, gather them

3. **Create/update plan.md** with this structure:

   ```markdown
   # Optimization Plan

   > Last updated: YYYY-MM-DD

   ## Current Goal
   [What we're trying to achieve right now]

   ## Hypothesis
   [What we think will improve performance and why]

   ## Experiment Log

   | # | Date | Config Change | P50 (ms) | P99 (ms) | Throughput | Perplexity | Verdict |
   |---|------|--------------|----------|----------|------------|------------|---------|
   | 1 | ...  | baseline     | ...      | ...      | ...        | ...        | ...     |

   ## Discoveries & Surprises
   - [Things learned during exploration that weren't expected]

   ## Key Techniques & Skills
   - [Reusable patterns, tools, or approaches discovered]

   ## Decisions
   - [Decision]: [Rationale]

   ## Dead Ends
   - [What was tried and why it didn't work - so we don't repeat it]

   ## Next Steps
   1. [Highest impact item]
   2. [Next item]
   ```

4. **Show the user** the current state of the plan.

---

# /log-result

Log a benchmark or experiment result to `plan.md` immediately after a run.

## Steps

1. **Read `plan.md`** (error if it doesn't exist - run `/plan` first)

2. **Gather result data**:
   - Ask user for results if not provided, or parse from benchmark output
   - Required: what config changed, P50, P99, throughput
   - Optional: perplexity, notes

3. **Append to Experiment Log table** in `plan.md`

4. **Add any discoveries/surprises** mentioned by user

5. **Update "Next Steps"** based on what we learned

6. **Show the updated plan** to the user.

---

# /explore

Systematically explore an optimization approach before implementing.

## Steps

1. **Read `plan.md`** to understand what's been tried

2. **Define the exploration scope**:
   - What are we exploring? (e.g., quantization options, batching strategies)
   - What are the key questions to answer?

3. **Research using subagents** (launch in parallel where possible):
   - Search docs, papers, GitHub issues for the approach
   - Check vLLM/FlashInfer docs for relevant configs
   - Look at what configs others use for similar models

4. **Synthesize findings** into a summary with:
   - Options available and tradeoffs
   - Recommended approach with rationale
   - Expected impact (rough estimate)
   - Risks or gotchas

5. **Update `plan.md`**:
   - Add findings to "Discoveries & Surprises"
   - Add learned techniques to "Key Techniques & Skills"
   - Update "Next Steps" with the exploration results

6. **Present recommendation** to user for approval before implementing.

---

# /review-plan

Review the current plan and suggest next steps.

## Steps

1. **Read `plan.md`**
2. **Analyze experiment history**: trends, what's working, diminishing returns
3. **Check for gaps**: untried approaches, unexplored configs
4. **Suggest prioritized next steps** based on expected impact vs effort
5. **Update `plan.md`** "Next Steps" section with recommendations

---

# /create-pr

Create a new PR for the current branch with a well-crafted title and description.

**Base branch**: Defaults to `main`. User can specify a different base branch (e.g., `/create-pr against develop`).

## Steps

1. **Determine base branch**:
   - Default: `main`
   - If user specifies a branch (e.g., "against develop"), use that instead
   - Store as `BASE_BRANCH` for subsequent commands

2. **Gather context** (run in parallel):
   - `git branch --show-current` - get current branch
   - `git log origin/$BASE_BRANCH..HEAD --oneline` - commits on this branch
   - `git diff origin/$BASE_BRANCH...HEAD --stat` - change summary
   - `gh pr list --head $(git branch --show-current) --json number` - check if PR exists

3. **Check prerequisites**:
   - If PR already exists, suggest `/update-pr` instead
   - Ensure branch is pushed: `git push -u origin HEAD` (ask before pushing)

4. **Analyze changes**:
   - Review the diff: `git diff origin/$BASE_BRANCH...HEAD`
   - Understand what changed and why

5. **Generate PR content**:

   **Title format** (conventional commit):
   - `feat(<scope>): <description>` - new features
   - `fix(<scope>): <description>` - bug fixes
   - `refactor(<scope>): <description>` - restructuring
   - `chore(<scope>): <description>` - maintenance
   - `opt(<scope>): <description>` - optimization

   **Description template**:
   ```markdown
   ## Summary
   - [Key change 1]
   - [Key change 2]

   ## Notes for Reviewers
   [What to focus on, tradeoffs made, follow-up work]

   ## Testing & Confidence
   - **Risk Level**: [Low/Medium/High]
   - **Tested**: [What was tested]
   - **Known Gaps**: [What wasn't tested]
   ```

6. **Create the PR**:
   ```bash
   gh pr create --base $BASE_BRANCH --title "<title>" --body "$(cat <<'EOF'
   <description>
   EOF
   )"
   ```

7. **Return the PR URL**.

---

# /update-pr

Update an existing PR's title and/or description based on new commits.

## Steps
1. Get current PR: `gh pr view --json number,title,body,headRefName`
2. Get new commits since PR was created
3. Update: `gh pr edit <number> --title "<new>" --body "<new>"`

---

# /commit

Stage and commit changes with a conventional commit message.

## Steps
1. `git status` and `git diff` to understand changes
2. Draft commit message following conventional commits
3. Stage relevant files (not `.env`, credentials, model weights)
4. Commit with descriptive message
