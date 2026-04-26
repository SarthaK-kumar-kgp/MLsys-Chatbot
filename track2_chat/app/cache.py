import asyncio
import hashlib
import logging
import struct
from collections import OrderedDict

from app.normalize import extract_keywords, normalize_text
from app.schemas import ChatMessage

logger = logging.getLogger(__name__)


class ResponseCache:
    def __init__(
        self,
        max_size: int = 16384,
        keyword_threshold: float = 0.45,
        hard_stop: float = 0.70,
    ):
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._inflight: dict[str, asyncio.Future[dict]] = {}
        self._max_size = max_size

        self.hits = 0
        self.misses = 0
        self.dedup_hits = 0
        self.keyword_hits = 0

        self._keyword_threshold = keyword_threshold
        self._hard_stop = hard_stop
        self._trigram_threshold = 0.45

        self._key_keywords: dict[str, frozenset[str]] = {}
        self._inflight_keywords: dict[str, frozenset[str]] = {}
        self._inverted: dict[str, set[str]] = {}
        self._key_trigrams: dict[str, frozenset[str]] = {}

    def make_key(
        self, messages: list[ChatMessage], temperature: float, max_tokens: int
    ) -> str | None:
        if temperature > 0:
            return None

        hasher = hashlib.sha256()
        hasher.update(struct.pack("<d", temperature))
        hasher.update(struct.pack("<I", max_tokens))
        hasher.update(struct.pack("<I", len(messages)))

        for message in messages:
            role_bytes = message.role.encode("utf-8")
            content_bytes = message.content.lower().strip().encode("utf-8")
            hasher.update(struct.pack("<I", len(role_bytes)))
            hasher.update(role_bytes)
            hasher.update(struct.pack("<I", len(content_bytes)))
            hasher.update(content_bytes)

        return hasher.hexdigest()

    def get_by_key(self, key: str) -> dict | None:
        if key in self._cache:
            self._cache.move_to_end(key)
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        return None

    def claim_inflight_by_key(self, key: str) -> tuple[asyncio.Future[dict], bool]:
        future = self._inflight.get(key)
        if future is not None:
            self.dedup_hits += 1
            return future, False

        future = asyncio.get_running_loop().create_future()
        self._inflight[key] = future
        return future, True

    def find_similar_inflight(self, query: str) -> asyncio.Future[dict] | None:
        query_kw = frozenset(extract_keywords(query))
        if not query_kw or not self._inflight_keywords:
            return None

        for key, item_kw in self._inflight_keywords.items():
            if not item_kw:
                continue
            inter = query_kw & item_kw
            if not inter:
                continue
            score = len(inter) / len(query_kw | item_kw)
            if score >= self._keyword_threshold:
                future = self._inflight.get(key)
                if future is not None:
                    self.dedup_hits += 1
                    return future
        return None

    def complete_inflight_by_key(
        self, key: str, response: dict, query_text: str | None = None
    ) -> None:
        self._cache[key] = response
        self._evict_oldest()

        future = self._inflight.pop(key, None)
        if future is not None and not future.done():
            future.set_result(response)

        self._inflight_keywords.pop(key, None)

        if query_text is not None:
            try:
                self._index_entry(key, query_text)
            except Exception:
                logger.debug("Failed to index entry for semantic cache", exc_info=True)

    def _keyword_search(self, query: str) -> dict | None:
        query_keywords = extract_keywords(query)
        if not query_keywords:
            return None

        candidates: dict[str, set[str]] = {}
        for kw in query_keywords:
            for key in self._inverted.get(kw, ()):
                if key not in candidates:
                    candidates[key] = self._key_keywords[key]

        if not candidates:
            return None

        best_key = None
        best_score = 0.0
        hard_stop = self._hard_stop

        for key, item_keywords in candidates.items():
            intersection = query_keywords & item_keywords
            score = len(intersection) / len(query_keywords | item_keywords)

            if score >= hard_stop:
                self._cache.move_to_end(key)
                self.keyword_hits += 1
                return self._cache[key]

            if score > best_score:
                best_score = score
                best_key = key

        if best_score >= self._keyword_threshold and best_key is not None:
            self._cache.move_to_end(best_key)
            self.keyword_hits += 1
            return self._cache[best_key]

        return None

    @staticmethod
    def _char_trigrams(text: str) -> frozenset[str]:
        text = normalize_text(text)
        if len(text) < 3:
            return frozenset()
        return frozenset(text[i:i+3] for i in range(len(text)-2))

    def _trigram_search(self, query: str) -> dict | None:
        query_tri = self._char_trigrams(query)
        if not query_tri:
            return None

        for key, item_tri in self._key_trigrams.items():
            inter = query_tri & item_tri
            if not inter:
                continue
            score = len(inter) / len(query_tri | item_tri)
            if score >= self._trigram_threshold:
                self._cache.move_to_end(key)
                self.keyword_hits += 1
                return self._cache[key]

        return None

    def semantic_get_keyword_only(self, messages: list[ChatMessage]) -> dict | None:
        if not self._cache:
            return None
        query = messages[-1].content
        result = self._keyword_search(query)
        if result is not None:
            return result
        return self._trigram_search(query)

    def _index_entry(self, key: str, query_text: str):
        keywords = frozenset(extract_keywords(query_text))
        self._key_keywords[key] = keywords
        for kw in keywords:
            if kw not in self._inverted:
                self._inverted[kw] = set()
            self._inverted[kw].add(key)
        self._key_trigrams[key] = self._char_trigrams(query_text)

    def _evict_oldest(self):
        if len(self._cache) <= self._max_size:
            return

        old_key, _ = self._cache.popitem(last=False)
        old_keywords = self._key_keywords.pop(old_key, frozenset())
        for kw in old_keywords:
            bucket = self._inverted.get(kw)
            if bucket is not None:
                bucket.discard(old_key)
                if not bucket:
                    del self._inverted[kw]
        self._key_trigrams.pop(old_key, None)

    def fail_inflight_by_key(self, key: str, error: Exception) -> None:
        self._inflight_keywords.pop(key, None)
        future = self._inflight.pop(key, None)
        if future is not None and not future.done():
            future.set_exception(error)

    def log_stats(self):
        total = self.hits + self.misses + self.dedup_hits + self.keyword_hits
        logger.info(
            "Cache stats: exact=%d, keyword=%d, dedup=%d, miss=%d, total=%d, "
            "cache_size=%d, inverted_keys=%d",
            self.hits, self.keyword_hits,
            self.dedup_hits, self.misses, total,
            len(self._cache), len(self._inverted),
        )
