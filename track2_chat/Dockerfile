FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml .
RUN uv sync --no-install-project

COPY . .

RUN uv sync

ENV PORT=8000
EXPOSE 8000

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

ENV VLLM_USE_V1=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV VLLM_NO_USAGE_STATS=1
ENV CUDA_MODULE_LOADING=LAZY
ENV QUANTIZATION="fp8"
ENV ENABLE_PREFIX_CACHING=false
ENV ENFORCE_EAGER=false
ENV GPU_MEMORY_UTILIZATION=0.92
ENV VLLM_ATTENTION_BACKEND=FLASHINFER

CMD ["sh", "-c", "uv run uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --no-access-log --loop uvloop"]
