"""
Modal deployment for Track 2: Customer Support Chatbot
Usage:
    modal run modal_deploy.py::download_model   # one-time model download
    modal serve modal_deploy.py                 # dev mode (hot reload)
    modal deploy modal_deploy.py                # production deploy
"""

import modal

# --- Image ---
# Match the Dockerfile: CUDA 12.8 + vLLM + FlashInfer
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04",
        add_python="3.12",
    )
    .pip_install("uv")
    .run_commands("pip install vllm fastapi uvicorn pydantic")
    .env({"VLLM_ATTENTION_BACKEND": "FLASHINFER"})
)

# --- Persistent volume for model weights ---
# Downloaded once, reused across deploys
model_volume = modal.Volume.from_name("hf-model-cache", create_if_missing=True)
MODEL_DIR = "/root/.cache/huggingface"
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

app = modal.App("track2-chat-engine", image=image)


@app.function(
    gpu="L4",
    volumes={MODEL_DIR: model_volume},
    timeout=1800,  # 30 min — model download can be slow
)
def download_model():
    """One-time setup: download model weights to the persistent volume."""
    from huggingface_hub import snapshot_download

    print(f"Downloading {MODEL_NAME} to {MODEL_DIR}...")
    snapshot_download(MODEL_NAME, cache_dir=MODEL_DIR)
    model_volume.commit()
    print("Done. Model cached in Modal volume.")


@app.cls(
    gpu="L4",
    volumes={MODEL_DIR: model_volume},
    timeout=600,
    allow_concurrent_inputs=128,  # handle 128 concurrent requests
)
class ChatServer:
    @modal.enter()
    async def startup(self):
        """Load the vLLM engine once when the container starts."""
        import asyncio
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine

        print(f"Loading vLLM engine with model: {MODEL_NAME}")
        engine_args = AsyncEngineArgs(
            model=MODEL_NAME,
            gpu_memory_utilization=0.9,
            max_model_len=8192,
            trust_remote_code=True,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.tokenizer = await self.engine.get_tokenizer()
        self.is_ready = True
        print("Engine ready.")

    @modal.fastapi_endpoint(method="GET", docs=True)
    async def health(self):
        return {"status": "ok"}

    @modal.fastapi_endpoint(method="GET")
    async def ready(self):
        from fastapi import HTTPException
        if self.is_ready:
            return {"status": "ready", "message": "Chat engine is initialized and ready."}
        raise HTTPException(status_code=503, detail="Engine is initializing")

    @modal.fastapi_endpoint(method="POST")
    async def v1_chat_completions(self, request: dict):
        """
        POST /v1/chat/completions
        Body: {"messages": [{"role": "user", "content": "..."}], "temperature": 0, "max_tokens": 256}
        Returns: {"output": "...", "logprobs": [...]}
        """
        from fastapi import HTTPException
        from vllm.sampling_params import SamplingParams
        from vllm.utils import random_uuid

        if not self.is_ready:
            raise HTTPException(status_code=503, detail="Engine is initializing")

        messages = request.get("messages", [])
        temperature = request.get("temperature", 0.7)
        max_tokens = request.get("max_tokens", 128)

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=1,
        )

        final_output = None
        async for output in self.engine.generate(prompt, sampling_params, random_uuid()):
            final_output = output

        if final_output is None:
            raise HTTPException(status_code=500, detail="No output generated")

        output_data = final_output.outputs[0]
        text_output = output_data.text

        logprobs: list[float] = []
        for i, token_id in enumerate(output_data.token_ids):
            if i < len(output_data.logprobs):
                step_logprobs = output_data.logprobs[i]
                if token_id in step_logprobs:
                    logprobs.append(step_logprobs[token_id].logprob)

        return {"output": text_output, "logprobs": logprobs}
