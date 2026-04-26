import logging
import os
from itertools import count
from functools import lru_cache

from app.schemas import ChatRequest
from app.constants import (
    ENABLE_CHUNKED_PREFILL,
    ENABLE_PREFIX_CACHING,
    ENFORCE_EAGER,
    GPU_MEMORY_UTILIZATION,
    KV_CACHE_DTYPE,
    MAX_MODEL_LENGTH,
    MAX_NUM_BATCHED_TOKENS,
    MAX_NUM_SEQS,
    MODEL_NAME,
    NUM_SCHEDULER_STEPS,
    QUANTIZATION,
    SPEC_DECODE_ENABLED,
    SPEC_NUM_TOKENS,
    SPEC_PROMPT_LOOKUP_MAX,
    SPEC_PROMPT_LOOKUP_MIN,
    SWAP_SPACE,
)
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams

logger = logging.getLogger(__name__)


@lru_cache(maxsize=64)
def _get_sampling_params(temperature: float, max_tokens: int) -> SamplingParams:
    return SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        logprobs=1,
    )


class ChatEngine:
    def __init__(self):
        self.engine = None
        self.tokenizer = None
        self.is_ready = False
        self._supports_thinking = None
        self._chat_template_kwargs = None
        self._request_counter = count(1)

    async def initialize(self):
        if self.is_ready:
            return

        speculative_config = None
        if SPEC_DECODE_ENABLED:
            speculative_config = {
                "method": "ngram",
                "prompt_lookup_min": SPEC_PROMPT_LOOKUP_MIN,
                "prompt_lookup_max": SPEC_PROMPT_LOOKUP_MAX,
                "num_speculative_tokens": SPEC_NUM_TOKENS,
                "disable_logprobs": False,
            }

        logger.info(
            "Initializing vLLM: model=%s, max_model_len=%d, quant=%s, kv_dtype=%s, "
            "max_num_seqs=%d, max_batched_tokens=%d, prefix_caching=%s, "
            "chunked_prefill=%s, spec_decode=%s",
            MODEL_NAME, MAX_MODEL_LENGTH, QUANTIZATION, KV_CACHE_DTYPE,
            MAX_NUM_SEQS, MAX_NUM_BATCHED_TOKENS, ENABLE_PREFIX_CACHING,
            ENABLE_CHUNKED_PREFILL, speculative_config,
        )

        import inspect
        valid_params = set(inspect.signature(AsyncEngineArgs.__init__).parameters)

        dtype = os.getenv("DTYPE", "auto")
        kwargs = dict(
            model=MODEL_NAME,
            max_model_len=MAX_MODEL_LENGTH,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            trust_remote_code=True,
            quantization=QUANTIZATION,
            dtype=dtype,
            kv_cache_dtype=KV_CACHE_DTYPE,
            max_num_seqs=MAX_NUM_SEQS,
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
            enable_prefix_caching=ENABLE_PREFIX_CACHING,
            enable_chunked_prefill=ENABLE_CHUNKED_PREFILL,
            disable_log_stats=True,
            enforce_eager=ENFORCE_EAGER,
        )

        if not ENFORCE_EAGER:
            try:
                from vllm.config import CompilationConfig
                cudagraph = os.getenv("CUDAGRAPH_MODE", "none")
                kwargs["compilation_config"] = CompilationConfig(
                    mode=3,
                    cudagraph_mode=cudagraph,
                )
                logger.info("Inductor compilation enabled (no CUDA graphs)")
            except Exception:
                logger.info("CompilationConfig not available, using defaults")

        optional = {
            "attention_backend": "flashinfer",
            "swap_space": SWAP_SPACE,
            "num_scheduler_steps": NUM_SCHEDULER_STEPS,
            "speculative_config": speculative_config,
            "performance_mode": os.getenv("PERFORMANCE_MODE", "throughput"),
            "async_scheduling": True,
            "stream_interval": 256,
        }
        for key, val in optional.items():
            if key in valid_params:
                kwargs[key] = val

        engine_args = AsyncEngineArgs(**kwargs)

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        tokenizer = self.engine.get_tokenizer()
        if hasattr(tokenizer, '__await__'):
            tokenizer = await tokenizer
        self.tokenizer = tokenizer

        try:
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": "test"}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            self._supports_thinking = True
        except TypeError:
            self._supports_thinking = False

        self._chat_template_kwargs = {
            "tokenize": True,
            "add_generation_prompt": True,
        }
        if self._supports_thinking:
            self._chat_template_kwargs["enable_thinking"] = False

        self.is_ready = True
        logger.info("Engine initialization complete")

    async def generate(self, request: ChatRequest) -> dict:
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        token_ids = self.tokenizer.apply_chat_template(
            messages, **self._chat_template_kwargs
        )

        sampling_params = _get_sampling_params(request.temperature, request.max_tokens)

        final_output = None
        request_id = str(next(self._request_counter))
        async for output in self.engine.generate(
            {"prompt_token_ids": token_ids}, sampling_params, request_id
        ):
            final_output = output

        if final_output is None:
            raise RuntimeError("No output generated")
        if not final_output.outputs:
            raise RuntimeError("No output candidates generated")

        output_data = final_output.outputs[0]
        if output_data.logprobs is None:
            raise RuntimeError("logprobs are missing from vLLM output")

        logprobs = []
        append_logprob = logprobs.append
        for token_id, step in zip(output_data.token_ids, output_data.logprobs):
            token_logprob = step.get(token_id)
            if token_logprob is None:
                raise RuntimeError("missing logprob for sampled token")
            append_logprob(token_logprob.logprob)

        return {"output": output_data.text, "logprobs": logprobs}
