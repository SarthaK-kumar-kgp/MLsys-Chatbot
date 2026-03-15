import uuid

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

from app.config import (
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


class ChatEngine:
    def __init__(self) -> None:
        self.engine: AsyncLLMEngine | None = None
        self.tokenizer = None
        self.ready = False

    async def initialize(self) -> None:
        engine_args = AsyncEngineArgs(
            model=MODEL_NAME,
            max_model_len=MAX_MODEL_LENGTH,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            trust_remote_code=True,
            # --- Memory optimizations ---
            quantization=QUANTIZATION,
            kv_cache_dtype=KV_CACHE_DTYPE,
            # --- Scheduling optimizations ---
            max_num_seqs=MAX_NUM_SEQS,
            enable_prefix_caching=ENABLE_PREFIX_CACHING,
            enable_chunked_prefill=ENABLE_CHUNKED_PREFILL,
            # --- Misc ---
            disable_log_stats=True,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.tokenizer = await self.engine.get_tokenizer()
        self.ready = True

    async def generate(self, request: ChatRequest) -> ChatResponse:
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        # Disable Qwen3 thinking mode to avoid wasting tokens on <think> blocks
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

        output_data = final_output.outputs[0]
        text_output = output_data.text

        logprobs: list[float] = []
        for i, token_id in enumerate(output_data.token_ids):
            if output_data.logprobs and i < len(output_data.logprobs):
                token_logprob_dict = output_data.logprobs[i]
                if token_id in token_logprob_dict:
                    logprobs.append(token_logprob_dict[token_id].logprob)
                else:
                    logprobs.append(0.0)
            else:
                logprobs.append(0.0)

        return ChatResponse(output=text_output, logprobs=logprobs)
