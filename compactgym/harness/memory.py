"""Reversible memory store for OFFLOAD_TO_MEMORY, with BM25 keyword search behind ``recall(query)``."""
from __future__ import annotations

import math
import re
from collections import Counter

from pydantic import BaseModel, Field

_tok = re.compile(r"[a-z0-9@]+(?:[.&'\-][a-z0-9@]+)*")  # inner dots, hyphens and apostrophes stay (emails, ids, don't); trailing punctuation does not


def _terms(text: str) -> list[str]:
    return _tok.findall(text.lower())


class MemoryItem(BaseModel):
    id: int
    text: str
    meta: dict = Field(default_factory=dict)


class MemoryStore:
    def __init__(self):
        self.items: list[MemoryItem] = []
        self._tf: list[Counter] = []
        self._df: Counter = Counter()

    def add(self, text: str, **meta) -> int:
        it = MemoryItem(id=len(self.items), text=text, meta=meta)
        self.items.append(it)
        tf = Counter(_terms(text))
        self._tf.append(tf)
        self._df.update(tf.keys())
        return it.id

    def __len__(self) -> int:
        return len(self.items)

    def search(self, query: str, k: int = 5, k1: float = 1.4, b: float = 0.75) -> list[tuple[float, MemoryItem]]:
        q = _terms(query)
        if not q or not self.items:
            return []
        n = len(self.items)
        avg = sum(sum(t.values()) for t in self._tf) / n or 1.0
        scored = []
        for it, tf in zip(self.items, self._tf):
            dl = sum(tf.values()) or 1
            s = 0.0
            for t in q:
                if t in tf:
                    idf = math.log(1 + (n - self._df[t] + 0.5) / (self._df[t] + 0.5))
                    s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * dl / avg))
            if s > 0:
                scored.append((s, it))
        scored.sort(key=lambda x: (-x[0], x[1].id))
        return scored[:k]

    def recall(self, query: str, k: int = 4, max_chars: int = 3000) -> str:
        hits = self.search(query, k)
        if not hits:
            return "No matching memories."
        out, used = [], 0
        for s, it in hits:
            chunk = f"[memory {it.id} score={s:.1f}]\n{it.text}"
            if used + len(chunk) > max_chars:
                chunk = chunk[: max(0, max_chars - used)] + "…"
            out.append(chunk)
            used += len(chunk)
            if used >= max_chars:
                break
        return "\n\n".join(out)

    def dump(self) -> list[dict]:
        return [i.model_dump() for i in self.items]

    @classmethod
    def load(cls, d: list[dict]) -> "MemoryStore":
        m = cls()
        for x in d:
            m.add(x["text"], **x.get("meta", {}))
        return m
