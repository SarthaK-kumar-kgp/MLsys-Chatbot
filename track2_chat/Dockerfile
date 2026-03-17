FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install dependencies first (layer caching)
COPY pyproject.toml .
RUN uv sync --no-install-project

# Copy application code
COPY . .
RUN uv sync

ENV PORT=8000
EXPOSE 8000

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV VLLM_ATTENTION_BACKEND=FLASHINFER

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
