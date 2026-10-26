"""memory_offload: reversible offloading. Older turns go to the memory store verbatim; the context keeps a one-line index
per turn and the agent gets a ``recall(query)`` tool."""
from __future__ import annotations

import json

from ..harness.resume import group_turns, split_for_resume, transcript
from ..interfaces import CompactionAction, EpisodeState, Message
from .base import BaseCompactor


def _line(g: list[Message], mem_id: int, chars: int) -> str:
    parts = []
    for m in g:
        if m.role == "user":
            parts.append("user: " + (m.content or "")[:chars].replace("\n", " "))
        elif m.role == "assistant":
            calls = ", ".join(f"{tc['function']['name']}({tc['function']['arguments'][:60]})" for tc in (m.tool_calls or []))
            if m.content:
                parts.append("assistant: " + m.content[:chars // 2].replace("\n", " "))
            if calls:
                parts.append("called " + calls)
        elif m.role == "tool":
            parts.append(f"-> {m.meta.get('name', 'tool')} result ({len(m.content or '')} chars)")
    return f"[mem {mem_id}] " + " | ".join(parts)


class MemoryOffload(BaseCompactor):
    name = "memory_offload"
    uses_memory = True

    def __init__(self, cfg, llm=None, index_chars: int = 160, max_index_lines: int = 60, **kw):
        super().__init__(cfg, llm, **kw)
        self.index_chars, self.max_index_lines = index_chars, max_index_lines

    async def __call__(self, messages: list[Message], state: EpisodeState, budget_tokens: int):
        if not self.over_threshold(state, budget_tokens):
            return self.keep(messages)
        system, older, tail = split_for_resume(messages, self.k(state))
        if not older:
            return self.keep(messages)
        lines: list[str] = list(state.settings.get("offload_index", []))
        fresh = [m for m in older if m.meta.get("origin") not in ("pinned",)]
        for g in group_turns(fresh):
            if len(g) == 1 and g[0].meta.get("origin") == "compaction_summary":
                continue
            mid = state.memory.add(transcript(g), turns=[m.meta.get("step") for m in g], users=[m.meta.get("user_turn") for m in g])
            lines.append(_line(g, mid, self.index_chars))
        if len(lines) > self.max_index_lines:
            dropped = len(lines) - self.max_index_lines
            lines = [f"... {dropped} older index lines dropped; their content is still in memory -- use recall(query)"] + lines[-self.max_index_lines:]
        state.settings["offload_index"] = lines
        stub = ("Earlier turns were moved to external memory (nothing is lost). Index of what is stored:\n" + "\n".join(lines)
                + "\nCall recall(query) to retrieve exact details (ids, dates, addresses, instructions the user gave) before relying on them.")
        return self.result(CompactionAction.OFFLOAD_TO_MEMORY, self.resume(messages, state, stub), stub, None, 0.0, None)
