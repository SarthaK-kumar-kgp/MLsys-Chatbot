import os

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507")
MAX_MODEL_LENGTH = int(os.getenv("MAX_MODEL_LENGTH", "288"))
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.92"))
QUANTIZATION = os.getenv("QUANTIZATION", "fp8") or None
KV_CACHE_DTYPE = os.getenv("KV_CACHE_DTYPE", "auto")
MAX_NUM_SEQS = int(os.getenv("MAX_NUM_SEQS", "192"))
MAX_NUM_BATCHED_TOKENS = int(os.getenv("MAX_NUM_BATCHED_TOKENS", "8192"))
ENABLE_PREFIX_CACHING = os.getenv("ENABLE_PREFIX_CACHING", "false").lower() == "true"
ENABLE_CHUNKED_PREFILL = os.getenv("ENABLE_CHUNKED_PREFILL", "true").lower() == "true"
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "16384"))
SWAP_SPACE = int(os.getenv("SWAP_SPACE", "0"))
ENFORCE_EAGER = os.getenv("ENFORCE_EAGER", "false").lower() == "true"

SEMANTIC_CACHE_ENABLED = os.getenv("SEMANTIC_CACHE_ENABLED", "true").lower() == "true"

NUM_SCHEDULER_STEPS = int(os.getenv("NUM_SCHEDULER_STEPS", "1"))

SPEC_DECODE_ENABLED = os.getenv("SPEC_DECODE_ENABLED", "false").lower() == "true"
SPEC_NUM_TOKENS = int(os.getenv("SPEC_NUM_TOKENS", "3"))
SPEC_PROMPT_LOOKUP_MAX = int(os.getenv("SPEC_PROMPT_LOOKUP_MAX", "4"))
SPEC_PROMPT_LOOKUP_MIN = int(os.getenv("SPEC_PROMPT_LOOKUP_MIN", "2"))
