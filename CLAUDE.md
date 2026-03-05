# MLSys Project - Track 2: Customer Support Chatbot

## Project Overview

CS4262/5462 Machine Learning Systems - Project 1: LLM Serving
**Track B**: High-throughput serving engine for interactive chat (Qwen3-4B-Instruct-2507)

### Key Specs
- Model: `Qwen/Qwen3-4B-Instruct-2507` with max context 8192
- Endpoints: `GET /health`, `GET /ready`, `POST /v1/chat/completions`
- Target concurrency: 128 simultaneous requests
- Evaluation: P50/P95 latency, throughput (req/s), perplexity
- Runtime: Single NVIDIA RTX 5080, Docker (amd64), model at `/root/.cache/huggingface`
- Framework: vLLM with FlashInfer attention backend

## Monorepo Structure

```
apps/
  chat-engine/    # Main serving engine (Python/FastAPI/vLLM/Docker)
  benchmark/      # Benchmark runner from mlsys_llm_benchmark
```

## Nx Commands

```bash
npx nx run chat-engine:build        # Build Docker image
npx nx run chat-engine:serve        # Start engine (docker compose up)
npx nx run chat-engine:stop         # Stop engine
npx nx run chat-engine:logs         # Tail engine logs
npx nx run chat-engine:test         # Quick smoke test
npx nx run chat-engine:lint         # Lint Python code
npx nx run chat-engine:format       # Format Python code
npx nx run chat-engine:push-image   # Build + tag + push to GHCR
npx nx run benchmark:run            # Run benchmark (concurrency=128)
npx nx run benchmark:run-local      # Run benchmark (concurrency=16)
```

## Mise Tasks

```bash
mise run install     # Install all deps
mise run build       # Build Docker image
mise run serve       # Start engine
mise run benchmark   # Run benchmark
mise run lint        # Lint all
mise run test        # Test all
```

## Git Workflow

- `main` branch is protected - all work goes through PRs
- Branch naming: `feat/<description>`, `fix/<description>`, `opt/<description>`
- Use `/create-pr` command below for PR creation

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
