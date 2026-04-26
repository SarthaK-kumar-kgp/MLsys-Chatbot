import asyncio
import logging
from contextlib import asynccontextmanager

import orjson
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import ORJSONResponse, Response

from app.cache import ResponseCache
from app.chat_engine import ChatEngine
from app.constants import CACHE_MAX_SIZE, SEMANTIC_CACHE_ENABLED
from app.normalize import extract_keywords
from app.schemas import ChatMessage, ChatRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

engine = ChatEngine()
cache = ResponseCache(max_size=CACHE_MAX_SIZE)


async def _init_engine():
    try:
        await engine.initialize()
        logger.info("Sending warmup request")
        warmup_req = ChatRequest(
            messages=[ChatMessage(role="user", content="hello")],
            temperature=0,
            max_tokens=1,
        )
        await engine.generate(warmup_req)
        logger.info("Warmup complete")
    except Exception:
        logger.exception("Engine initialization failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting engine initialization in background")
    asyncio.create_task(_init_engine())
    yield


app = FastAPI(
    title="Track 2: Chat Engine",
    lifespan=lifespan,
    default_response_class=ORJSONResponse,
)


_DEFAULTS = ChatRequest.model_fields


def _default(field: str):
    return _DEFAULTS[field].default


@app.post("/v1/chat/completions")
async def chat_completions(raw_request: Request):
    if not engine.is_ready:
        raise HTTPException(status_code=503, detail="Engine is still initializing")

    body = await raw_request.body()
    data = orjson.loads(body)
    messages = [ChatMessage(role=m["role"], content=m["content"]) for m in data["messages"]]
    temperature = data.get("temperature", _default("temperature"))
    max_tokens = data.get("max_tokens", _default("max_tokens"))

    cache_key = cache.make_key(messages, temperature, max_tokens)
    if cache_key is None:
        request = ChatRequest(messages=messages, temperature=temperature, max_tokens=max_tokens)
        return await engine.generate(request)

    cached = cache.get_by_key(cache_key)
    if cached is not None:
        return Response(content=cached, media_type="application/json")

    if SEMANTIC_CACHE_ENABLED and temperature == 0:
        semantic_cached = cache.semantic_get_keyword_only(messages)
        if semantic_cached is not None:
            cache.complete_inflight_by_key(cache_key, semantic_cached)
            return Response(content=semantic_cached, media_type="application/json")

        similar_inflight = cache.find_similar_inflight(messages[-1].content)
        if similar_inflight is not None:
            result = await similar_inflight
            cache.complete_inflight_by_key(cache_key, result)
            return Response(content=result, media_type="application/json")

    inflight, is_owner = cache.claim_inflight_by_key(cache_key)
    if not is_owner:
        result = await inflight
        return Response(content=result, media_type="application/json")

    query_text = messages[-1].content if SEMANTIC_CACHE_ENABLED else None
    if query_text:
        cache._inflight_keywords[cache_key] = frozenset(extract_keywords(query_text))

    try:
        request = ChatRequest(messages=messages, temperature=temperature, max_tokens=max_tokens)
        response = await engine.generate(request)
        response_bytes = orjson.dumps(response)
        cache.complete_inflight_by_key(cache_key, response_bytes, query_text=query_text)
        return Response(content=response_bytes, media_type="application/json")
    except Exception as e:
        cache.fail_inflight_by_key(cache_key, e)
        raise


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    if engine.is_ready:
        return {"status": "ready", "message": "Chat engine is initialized and ready."}
    raise HTTPException(status_code=503, detail="Engine is initializing")
