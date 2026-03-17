from app.schemas import ChatRequest, ChatResponse
from app.constants import MODEL_NAME, MAX_MODEL_LENGTH
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

class ChatEngine:
    """
    This engine uses vLLM's AsyncLLMEngine for high-performance inference.
    """
    def __init__(self):
        self.model_name = MODEL_NAME
        self.engine = None
        self.tokenizer = None
        self.is_ready = False

    async def initialize(self):
        if self.is_ready:
            return

        print(f"Initializing vLLM with model: {self.model_name}...")

        engine_args = AsyncEngineArgs(
            model=self.model_name,
            gpu_memory_utilization=0.9,
            max_model_len=MAX_MODEL_LENGTH,
            trust_remote_code=True,
        )

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

        self.tokenizer = await self.engine.get_tokenizer()

        self.is_ready = True
        print("vLLM Engine initialized and ready.")

    async def generate(self, request: ChatRequest) -> ChatResponse:
        if not self.is_ready:
            raise Exception("Engine is still initializing. Please try again later.")

        messages_dicts = [{"role": m.role, "content": m.content} for m in request.messages]
        prompt = self.tokenizer.apply_chat_template(
            messages_dicts,
            tokenize=False,
            add_generation_prompt=True
        )
        sampling_params = SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            logprobs=1
        )
        results_generator = self.engine.generate(
            prompt,
            sampling_params,
            random_uuid(), # Unique request ID for vLLM tracking
        )

        final_output = None
        async for request_output in results_generator:
            final_output = request_output

        if final_output is None:
            raise Exception("No output generated")

        text_output = final_output.outputs[0].text

        output_data = final_output.outputs[0]
        if output_data.logprobs is None:
            raise RuntimeError("logprobs are missing from vLLM output")

        if not output_data.token_ids:
            raise RuntimeError("token_ids is empty, cannot provide logprobs")

        logprobs: list[float] = []
        for i, token_id in enumerate(output_data.token_ids):
            if i < len(output_data.logprobs):
                step_logprobs = output_data.logprobs[i]
                if token_id in step_logprobs:
                    logprobs.append(step_logprobs[token_id].logprob)
                else:
                    raise RuntimeError(f"Token ID {token_id} not found in logprobs at step {i}")

        return ChatResponse(output=text_output, logprobs=logprobs)
