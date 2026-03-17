from pydantic import BaseModel

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 128

# Simplified Response Schema
class ChatResponse(BaseModel):
    output: str
    logprobs: list[float]
