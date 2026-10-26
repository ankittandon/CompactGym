"""frontier: a closed (or stand-in large) model summarizes at the threshold using CompactionRL's summary instruction."""
from __future__ import annotations

from ..harness.resume import split_for_resume, transcript
from ..interfaces import CompactionAction, EpisodeState, Message
from .base import BaseCompactor, strip_summary_tags

COMPACTIONRL_INSTRUCTION = (
    "Summarize the conversation above so that you can continue the work in a fresh context that contains only your summary plus the last few turns. "
    "Your summary must keep: the original goal; the actions already completed (with ids, dates and addresses); important observations; unresolved errors; "
    "the current state; and plausible next steps. Wrap the summary in <summary></summary>."
)


class Frontier(BaseCompactor):
    name = "frontier"
    role = "frontier"

    async def summarize(self, messages: list[Message], state: EpisodeState, role: str | None = None, temperature: float | None = None):
        system, older, tail = split_for_resume(messages, self.k(state))
        role = role or self.role
        msgs = [{"role": "user", "content": transcript(older)}, {"role": "user", "content": COMPACTIONRL_INSTRUCTION}]
        res, lat = await self.call(getattr(self.cfg, role), role, msgs, tag=f"compact:{self.name}", max_tokens=1500, temperature=temperature)
        return strip_summary_tags(res.text), res, lat, bool(older)

    async def __call__(self, messages: list[Message], state: EpisodeState, budget_tokens: int):
        if not self.over_threshold(state, budget_tokens):
            return self.keep(messages)
        summary, res, lat, had = await self.summarize(messages, state)
        if not had:
            return self.keep(messages)
        return self.result(CompactionAction.DEFER_TO_FRONTIER, self.resume(messages, state, summary), summary, res, lat, self.role)
