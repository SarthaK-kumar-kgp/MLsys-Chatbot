"""Deploy chat-engine on Modal with a GPU.

Usage (run from track2_chat/ directory):
    cd track2_chat
    modal run modal_deploy.py::download_model    # download model (one-time)
    modal deploy modal_deploy.py                 # deploy server
    modal serve modal_deploy.py                  # dev mode (hot reload)
"""

import modal

app = modal.App("chat-engine")

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

# Container image: vLLM + FastAPI with our app code
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm",
        "fastapi[standard]",
        "pydantic",
        "huggingface_hub[hf_xet]",
    )
    .add_local_dir("app", remote_path="/root/app", copy=True)
)

# Persistent volume for model weights
model_volume = modal.Volume.from_name("hf-model-cache", create_if_missing=True)


@app.function(
    image=image,
    volumes={"/root/.cache/huggingface": model_volume},
    timeout=600,
)
def download_model():
    """Pre-download model weights into the persistent volume."""
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_NAME)
    model_volume.commit()
    print(f"Model {MODEL_NAME} downloaded successfully!")


@app.function(
    image=image,
    gpu="L4",
    volumes={"/root/.cache/huggingface": model_volume},
    timeout=3600,
    scaledown_window=600,
)
@modal.concurrent(max_inputs=150)
@modal.asgi_app()
def serve():
    """Serve the chat-engine FastAPI app on an L4 GPU."""
    import logging
    import os
    import sys

    logging.basicConfig(level=logging.INFO)
    sys.path.insert(0, "/root")

    # L4 GPU config — safe baseline (no FP8 initially)
    os.environ["QUANTIZATION"] = ""
    os.environ["KV_CACHE_DTYPE"] = ""
    os.environ["GPU_MEMORY_UTILIZATION"] = "0.90"
    os.environ["MAX_NUM_SEQS"] = "128"
    os.environ["ENABLE_CHUNKED_PREFILL"] = "true"
    os.environ["ENABLE_PREFIX_CACHING"] = "true"

    from app.main import app as fastapi_app

    return fastapi_app
