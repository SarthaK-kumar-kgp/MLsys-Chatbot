import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.cache import ResponseCache
from app.config import CACHE_MAX_SIZE
from app.engine import ChatEngine
from app.schemas import ChatRequest, ChatResponse

engine = ChatEngine()
cache = ResponseCache(max_size=CACHE_MAX_SIZE)


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(engine.initialize())
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    if engine.ready:
        return {"status": "ready", "message": "Engine is ready to serve requests"}
    raise HTTPException(status_code=503, detail="Engine is not ready")


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest) -> ChatResponse:
    if not engine.ready:
        raise HTTPException(status_code=503, detail="Engine is not ready")

    cached = cache.get(request.messages, request.temperature, request.max_tokens)
    if cached is not None:
        return cached

    response = await engine.generate(request)
    cache.put(request.messages, request.temperature, request.max_tokens, response)
    return response
