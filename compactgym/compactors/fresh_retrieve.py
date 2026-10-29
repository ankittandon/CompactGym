"""fresh_retrieve: Mainplane-style "continue in a fresh session and retrieve". Nothing is summarized. Every older turn goes to the memory store verbatim,
the context keeps only one instruction (no index, no pinned list), and the agent decides what to look up with ``recall(query)`` as it goes.

This is the "let the model manage its own context" position: the full record stays available, exact text comes back, and what to retrieve is the agent's call.
"""
from __future__ import annotations

from ..harness.resume import group_turns, split_for_resume, transcript
from ..interfaces import CompactionAction, EpisodeState, Message
from .base import BaseCompactor

STUB = ("You are continuing an earlier session. Its full record is saved and nothing in it was paraphrased. "
        "Retrieve key context from it as you go with recall(query): it returns the exact text of the matching earlier turns.")


class FreshRetrieve(BaseCompactor):
    name = "fresh_retrieve"
    uses_memory = True

    async def __call__(self, messages: list[Message], state: EpisodeState, budget_tokens: int):
        if not self.over_threshold(state, budget_tokens):
            return self.keep(messages)
        system, older, tail = split_for_resume(messages, self.k(state))
        if not older:
            return self.keep(messages)
        for g in group_turns([m for m in older if m.meta.get("origin") != "pinned"]):
            if len(g) == 1 and g[0].meta.get("origin") == "compaction_summary":
                continue  # the stub from an earlier compaction
            state.memory.add(transcript(g), turns=[m.meta.get("step") for m in g], users=[m.meta.get("user_turn") for m in g])
        return self.result(CompactionAction.OFFLOAD_TO_MEMORY, self.resume(messages, state, STUB), STUB, None, 0.0, None)
