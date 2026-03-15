import os

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507")
MAX_MODEL_LENGTH = int(os.getenv("MAX_MODEL_LENGTH", "1024"))
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.95"))
QUANTIZATION = os.getenv("QUANTIZATION", "fp8")
KV_CACHE_DTYPE = os.getenv("KV_CACHE_DTYPE", "fp8")
MAX_NUM_SEQS = int(os.getenv("MAX_NUM_SEQS", "256"))
ENABLE_PREFIX_CACHING = os.getenv("ENABLE_PREFIX_CACHING", "true").lower() == "true"
ENABLE_CHUNKED_PREFILL = os.getenv("ENABLE_CHUNKED_PREFILL", "true").lower() == "true"
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "16384"))
