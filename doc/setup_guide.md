# MLSys Project – Setup & Deployment Guide

**CS4262/5462 Machine Learning Systems – Track 2: Customer Support Chatbot**

---

## Overview

This guide documents the full setup process for running and benchmarking the vLLM serving engine on Vast.ai, and how to test code changes going forward.

### Architecture Summary

```
Mac (code editing + benchmark client)
        │
        │  SSH tunnel: ssh -p 32823 root@155.103.252.90 -L 8000:localhost:8000
        │
Vast.ai RTX 4090 (Instance ID: 33709843) — 155.103.252.90
        │
        └── FastAPI + vLLM v0.11.0 engine (port 8000)
                └── Qwen3-4B-Instruct-2507 model (7.6GB loaded on GPU)
```

---

## Prerequisites

### On Your Mac
- Git
- Python + uv (`pip install uv`)
- SSH client (built-in on Mac)

### Accounts Required
- **GitHub** — github.com/thetsuwin66
- **HuggingFace** — token named `mlsystem_project` (huggingface.co → Settings → Access Tokens)
- **Vast.ai** — account with credits loaded

---

## Part 1: One-Time Vast.ai Instance Setup

This section only needs to be done once when setting up a new instance.

### Step 1 – Rent a GPU Instance on Vast.ai

1. Go to **vast.ai → Search**
2. Set filters:
   - GPU: RTX 5080 or RTX 4090
   - Disk: **80GB minimum** (important — 32GB fills up fast)
   - Template: **vLLM (Serverless)**
3. Click **Rent** on a suitable instance
4. Wait for status to show **Running**

> Our instance: RTX 4090 (48GB VRAM), 32GB disk, AMD EPYC 7502, IP: 155.103.252.90

### Step 2 – Connect via SSH Tunnel

From your Vast.ai dashboard, find the SSH command under **Connect**:

```bash
ssh -p 32823 root@155.103.252.90 -L 8000:localhost:8000
```

The `-L 8000:localhost:8000` tunnels the engine port to your Mac so you can benchmark locally.

Keep this terminal open as long as you need the tunnel.

### Step 3 – Free Up Disk Space

The vLLM template comes pre-loaded with other models (~21GB). Remove them:

```bash
rm -rf /workspace/models
rm -rf /workspace/hf-cache
rm -rf /root/.cache/uv
rm -rf /root/.cache/pip
```

Verify free space (need at least 15GB):
```bash
df -h /workspace
# Expected: ~21GB free after cleanup
```

### Step 4 – Clone the Repository

```bash
cd /workspace
git clone https://github.com/aryabyte21/mlsys.git
cd mlsys/track2_chat
```

If prompted for credentials:
- Username: `thetsuwin66`
- Password: GitHub Personal Access Token (not your GitHub password)

Verify clone:
```bash
ls /workspace/mlsys/track2_chat/
# Expected: Dockerfile  app  docker-compose.yaml  pyproject.toml  scripts
```

### Step 5 – Download the Model

Login to HuggingFace:
```bash
huggingface-cli login
# Paste your HuggingFace token (mlsystem_project) when prompted
# Select Y for git credential
```

Download the model (~8GB, takes 5-10 minutes):
```bash
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 \
  --cache-dir /workspace/hf-cache
```

Expected output when done:
```
Fetching 13 files: 100%|████████| 13/13
/workspace/hf-cache/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554
```

### Step 6 – Set the Model Path in constants.py

Point vLLM directly to the downloaded snapshot (bypasses HuggingFace hub and avoids re-download):

```bash
cat > /workspace/mlsys/track2_chat/app/constants.py << 'EOF'
MODEL_NAME = "/workspace/hf-cache/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"
MAX_MODEL_LENGTH = 8192
EOF
```

### Step 7 – Start the Engine

```bash
cd /workspace/mlsys/track2_chat
export VLLM_ATTENTION_BACKEND=FLASHINFER
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Wait for this message (takes ~2-3 minutes for CUDA graph compilation):
```
Loading safetensors checkpoint shards: 100% Completed | 3/3
Loading weights took 0.98 seconds
Model loading took 7.6065 GiB and 1.163559 seconds
torch.compile takes 30.74 s in total
Available KV cache memory: 34.72 GiB
GPU KV cache size: 252,800 tokens
vLLM Engine initialized and ready.
INFO: Application startup complete.
INFO: Uvicorn running on http://0.0.0.0:8000
```

### Step 8 – Verify the Engine is Running

From your Mac (in a new terminal):
```bash
curl http://localhost:8000/health
# Expected: {"status":"ok"}
```

---

## Part 2: Running the Benchmark

Run this from your **Mac** (not on Vast.ai). Make sure the SSH tunnel is open first.

```bash
cd /Users/thetsu/Documents/thetsuwin/nus_workspace/CS5462_MLSystems/project/mlsys/benchmark
uv sync
uv run runner_chat.py \
  --url http://localhost:8000 \
  --data data/track2/train.jsonl \
  --concurrency 128
```

### Expected Output Format

```
=== Track 2: Performance Check ===
Executing 13435 requests with concurrency 128...
100%|████████████████| 13435/13435

Performance Metrics:
  Throughput: X.XX req/s
  Passed: XXXXX
  Failed: 0
  P50 Latency: X.XXXXs
  P99 Latency: X.XXXXs
  Avg Perplexity: X.XXXX
```

### Metrics to Record

| Metric | Description | Goal |
|--------|-------------|------|
| P50 Latency | Median request time | Lower is better |
| P99 Latency | 99th percentile request time | Lower is better |
| Throughput | Requests per second | Higher is better |
| Perplexity | Response quality | Lower is better |

---

## Part 3: Testing a Code Change

Follow this workflow every time you or a teammate makes a code change.

### Step 1 – Make the change on Mac

Edit files in `track2_chat/app/` on your Mac using VS Code.

### Step 2 – Revert constants.py before committing

The absolute model path in `constants.py` only works on this specific Vast.ai instance. Revert it before pushing:

```bash
cd /Users/thetsu/Documents/thetsuwin/nus_workspace/CS5462_MLSystems/project/mlsys
git checkout track2_chat/app/constants.py
```

### Step 3 – Commit and push

```bash
git add track2_chat/app/<changed-file>.py
git commit -m "opt: describe your change"
git push
```

### Step 4 – Pull on Vast.ai

In the Vast.ai SSH terminal:
```bash
cd /workspace/mlsys/track2_chat
git pull

# Re-apply the absolute model path
cat > /workspace/mlsys/track2_chat/app/constants.py << 'EOF'
MODEL_NAME = "/workspace/hf-cache/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"
MAX_MODEL_LENGTH = 8192
EOF
```

### Step 5 – Restart the engine

Press `Ctrl+C` to stop the current engine, then:
```bash
export VLLM_ATTENTION_BACKEND=FLASHINFER
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Wait for `vLLM Engine initialized and ready.`

### Step 6 – Run the benchmark from Mac

```bash
cd /Users/thetsu/Documents/thetsuwin/nus_workspace/CS5462_MLSystems/project/mlsys/benchmark
uv run runner_chat.py \
  --url http://localhost:8000 \
  --data data/track2/train.jsonl \
  --concurrency 128
```

### Step 7 – Record results in plan.md

Use `/log-result` to log the benchmark numbers into `plan.md`.

---

## Troubleshooting

### Disk full error (`No space left on device`)
```bash
# Check disk usage
df -h /workspace
du -sh /workspace/* 2>/dev/null | sort -rh | head -10

# Clean up (safe to delete these)
rm -rf /root/.cache/huggingface
rm -rf /workspace/hf-cache/xet
rm -rf /root/.cache/uv
rm -rf /root/.cache/pip
rm -rf /workspace/ep_kernels_workspace
```

### GPU memory error (`Free memory less than desired utilization`)
Another vLLM process is using the GPU. Check and kill it:
```bash
nvidia-smi
# Look for VLLM::EngineCore process
kill -9 <PID>
```

### SSH tunnel disconnected
Re-run the tunnel command on Mac:
```bash
ssh -p 32823 root@155.103.252.90 -L 8000:localhost:8000
```

### Model weights not initialized (missing layers error)
The model download was incomplete. Re-download:
```bash
rm -rf /workspace/hf-cache
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 \
  --cache-dir /workspace/hf-cache
```

### vLLM can't find model (LocalEntryNotFoundError)
Make sure `constants.py` uses the absolute path, not the HuggingFace model ID:
```bash
cat /workspace/mlsys/track2_chat/app/constants.py
# Should show the full /workspace/hf-cache/... path, not "Qwen/Qwen3-4B-Instruct-2507"
```

---

## Part 4: Keeping Costs Low (Stop vs Destroy)

**Always STOP the instance when not in use — never Destroy it.**

| State | Cost | What's preserved |
|-------|------|-----------------|
| Running | ~$0.40/hr | Everything |
| **Stopped** | ~$0.03/hr | Everything (disk + model + repo) |
| Destroyed | $0 | Nothing — full setup from scratch |

### When done for the day
Go to **Vast.ai dashboard → your instance → click STOP**

### When resuming work
1. Go to **Vast.ai dashboard → your instance → click START**
2. SSH in with tunnel:
   ```bash
   ssh -p 32823 root@155.103.252.90 -L 8000:localhost:8000
   ```
3. Start the engine:
   ```bash
   cd /workspace/mlsys/track2_chat
   export VLLM_ATTENTION_BACKEND=FLASHINFER
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

No re-download, no re-setup. Everything is preserved on disk.

---

## Quick Reference Card

| Task | Command |
|------|---------|
| SSH into Vast.ai | `ssh -p 32823 root@155.103.252.90 -L 8000:localhost:8000` |
| Start engine | `cd /workspace/mlsys/track2_chat && export VLLM_ATTENTION_BACKEND=FLASHINFER && uvicorn app.main:app --host 0.0.0.0 --port 8000` |
| Check engine health | `curl http://localhost:8000/health` |
| Run benchmark | `cd benchmark && uv run runner_chat.py --url http://localhost:8000 --data data/track2/train.jsonl --concurrency 128` |
| Check disk space | `df -h /workspace` |
| Check GPU usage | `nvidia-smi` |
| Pull latest code | `cd /workspace/mlsys && git pull` |
