"""never / threshold_pi / threshold_anthropic: LLM summarization of older turns when the context is ``threshold_frac`` full."""
from __future__ import annotations

from ..harness.resume import split_for_resume, transcript
from ..third_party.conpact.prompts import summarization_prompt
from ..interfaces import CompactionAction, EpisodeState, Message
from .base import BaseCompactor, strip_summary_tags


class ThresholdSummarizer(BaseCompactor):
    option = "pi_mono"
    role = "main"  # the main model summarizes its own context

    async def __call__(self, messages: list[Message], state: EpisodeState, budget_tokens: int):
        if not self.over_threshold(state, budget_tokens):
            return self.keep(messages)
        system, older, tail = split_for_resume(messages, self.k(state))
        if not older:
            return self.keep(messages)
        prompt = summarization_prompt(self.option, [{"role": "user", "content": transcript(older)}])
        spec = getattr(self.cfg, self.role)
        res, lat = await self.call(spec, self.role, prompt, tag=f"compact:{self.name}", max_tokens=1500)
        summary = strip_summary_tags(res.text)
        new = self.resume(messages, state, summary)
        return self.result(CompactionAction.SUMMARIZE_OLDER, new, summary, res, lat, self.role)


class ThresholdPi(ThresholdSummarizer):
    name = "threshold_pi"
    option = "pi_mono"


class ThresholdAnthropic(ThresholdSummarizer):
    name = "threshold_anthropic"
    option = "anthropic"


class Never(ThresholdPi):
    """Keep until forced (context exceeds the budget), then summarize like threshold_pi."""
    name = "never"

    def __init__(self, cfg, llm=None, **kw):
        super().__init__(cfg, llm, threshold_frac=1.0001, **kw)
