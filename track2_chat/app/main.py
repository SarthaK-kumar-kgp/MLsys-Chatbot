import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from app.schemas import ChatRequest, ChatResponse
from app.chat_engine import ChatEngine

engine = ChatEngine()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start the engine initialization in the background
    asyncio.create_task(engine.initialize())
    yield
    # Shutdown: Clean up resources if needed
    pass

app = FastAPI(title="Track 2: Chat Engine", lifespan=lifespan)

@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest):
    if not engine.is_ready:
        raise HTTPException(status_code=503, detail="Engine is still initializing")
    return await engine.generate(request)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ready")
def ready():
    if engine.is_ready:
        return {"status": "ready", "message": "Chat engine is initialized and ready."}
    else:
        raise HTTPException(status_code=503, detail="Engine is initializing")
