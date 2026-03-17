import logging
import uuid

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

from app.constants import (
    ENABLE_CHUNKED_PREFILL,
    ENABLE_PREFIX_CACHING,
    GPU_MEMORY_UTILIZATION,
    KV_CACHE_DTYPE,
    MAX_MODEL_LENGTH,
    MAX_NUM_SEQS,
    MODEL_NAME,
    QUANTIZATION,
)
from app.schemas import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)


class ChatEngine:
    def __init__(self) -> None:
        self.engine: AsyncLLMEngine | None = None
        self.tokenizer = None
        self.is_ready = False

    async def initialize(self) -> None:
        if self.is_ready:
            return

        logger.info(
            "Initializing vLLM: model=%s, max_model_len=%d, quant=%s, kv_dtype=%s, "
            "max_num_seqs=%d, prefix_caching=%s, chunked_prefill=%s",
            MODEL_NAME,
            MAX_MODEL_LENGTH,
            QUANTIZATION,
            KV_CACHE_DTYPE,
            MAX_NUM_SEQS,
            ENABLE_PREFIX_CACHING,
            ENABLE_CHUNKED_PREFILL,
        )

        engine_args = AsyncEngineArgs(
            model=MODEL_NAME,
            max_model_len=MAX_MODEL_LENGTH,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            trust_remote_code=True,
            quantization=QUANTIZATION,
            kv_cache_dtype=KV_CACHE_DTYPE,
            max_num_seqs=MAX_NUM_SEQS,
            enable_prefix_caching=ENABLE_PREFIX_CACHING,
            enable_chunked_prefill=ENABLE_CHUNKED_PREFILL,
            disable_log_stats=True,
        )

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.tokenizer = await self.engine.get_tokenizer()
        self.is_ready = True
        logger.info("Engine initialization complete!")

    async def generate(self, request: ChatRequest) -> ChatResponse:
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        # Disable Qwen3 thinking mode to avoid wasted tokens on <think> blocks
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        sampling_params = SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            logprobs=1,
        )

        request_id = str(uuid.uuid4())
        final_output = None
        async for output in self.engine.generate(prompt, sampling_params, request_id):
            final_output = output

        if final_output is None:
            raise RuntimeError("No output generated")

        output_data = final_output.outputs[0]
        text_output = output_data.text

        if output_data.logprobs is None:
            raise RuntimeError("logprobs are missing from vLLM output")

        logprobs: list[float] = []
        for i, token_id in enumerate(output_data.token_ids):
            if i < len(output_data.logprobs):
                step_logprobs = output_data.logprobs[i]
                if token_id in step_logprobs:
                    logprobs.append(step_logprobs[token_id].logprob)
                else:
                    logprobs.append(0.0)

        return ChatResponse(output=text_output, logprobs=logprobs)
