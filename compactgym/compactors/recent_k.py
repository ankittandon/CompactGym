"""recent_k: keep the system prompt, the first user message and the last 5 turns; drop the rest (no rewriting)."""
from __future__ import annotations

from ..harness.resume import ensure_pinned, split_for_resume
from ..interfaces import CompactionAction, EpisodeState, Message
from .base import BaseCompactor


class RecentK(BaseCompactor):
    name = "recent_k"

    def __init__(self, cfg, llm=None, k: int = 5, **kw):
        super().__init__(cfg, llm, **kw)
        self.n = k

    async def __call__(self, messages: list[Message], state: EpisodeState, budget_tokens: int):
        if not self.over_threshold(state, budget_tokens):
            return self.keep(messages)
        system, older, tail = split_for_resume(messages, self.n)
        first = next((m for m in older if m.role == "user" and not m.meta.get("origin")), None)
        if first is None:
            return self.keep(messages)
        new = system + [first] + tail
        new = ensure_pinned(new, state.pinned, bool(state.settings.get("pinned_tier", True)))
        return self.result(CompactionAction.TRUNCATE_VERBATIM, new, None, None, 0.0, None)
