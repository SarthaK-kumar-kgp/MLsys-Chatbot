import hashlib
import json
from collections import OrderedDict

from app.schemas import ChatMessage, ChatResponse


class ResponseCache:
    """Exact-match response cache for deterministic (temperature=0) requests.

    With 36% duplicate rate in the training data and temperature=0 (greedy decoding),
    caching identical requests eliminates ~36% of GPU inference calls.
    """

    def __init__(self, max_size: int = 16384):
        self._cache: OrderedDict[str, ChatResponse] = OrderedDict()
        self._max_size = max_size
        self.hits = 0
        self.misses = 0

    def _make_key(
        self, messages: list[ChatMessage], temperature: float, max_tokens: int
    ) -> str | None:
        if temperature > 0:
            return None
        data = json.dumps(
            [{"role": m.role, "content": m.content} for m in messages], sort_keys=True
        )
        key_str = f"{data}|t={temperature}|m={max_tokens}"
        return hashlib.sha256(key_str.encode()).hexdigest()

    def get(
        self, messages: list[ChatMessage], temperature: float, max_tokens: int
    ) -> ChatResponse | None:
        key = self._make_key(messages, temperature, max_tokens)
        if key is None:
            return None
        if key in self._cache:
            self._cache.move_to_end(key)
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        return None

    def put(
        self,
        messages: list[ChatMessage],
        temperature: float,
        max_tokens: int,
        response: ChatResponse,
    ) -> None:
        key = self._make_key(messages, temperature, max_tokens)
        if key is None:
            return
        self._cache[key] = response
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)
