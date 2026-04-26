import modal

app = modal.App("chat-engine")

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.8.5.post1",
        "transformers<4.53.0",
        "fastapi[standard]",
        "pydantic",
        "huggingface_hub[hf_xet]",
    )
    .add_local_dir("app", remote_path="/root/app", copy=True)
)

model_volume = modal.Volume.from_name("hf-model-cache", create_if_missing=True)


@app.function(
    image=image,
    volumes={"/root/.cache/huggingface": model_volume},
    timeout=600,
)
def download_model():
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_NAME)
    model_volume.commit()
    print(f"Model {MODEL_NAME} downloaded")


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
    import logging
    import os
    import sys

    logging.basicConfig(level=logging.INFO)
    sys.path.insert(0, "/root")

    os.environ["QUANTIZATION"] = ""
    os.environ["KV_CACHE_DTYPE"] = ""
    os.environ["GPU_MEMORY_UTILIZATION"] = "0.90"
    os.environ["MAX_NUM_SEQS"] = "128"
    os.environ["MAX_NUM_BATCHED_TOKENS"] = "8192"
    os.environ["ENABLE_CHUNKED_PREFILL"] = "true"
    os.environ["ENABLE_PREFIX_CACHING"] = "true"
    os.environ["SPEC_DECODE_ENABLED"] = "false"

    from app.main import app as fastapi_app

    return fastapi_app
