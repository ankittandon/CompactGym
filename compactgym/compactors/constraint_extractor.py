"""constraint_extractor: CONPACT's extractor on top of threshold_pi. User turns are mined for session-scoped constraints; the
registry is pinned (and appended to the summary when the pinned tier is ablated)."""
from __future__ import annotations

import asyncio
import json
import re

from ..third_party.conpact.extractor import build_sc_extraction_prompts
from ..interfaces import CompactionAction, EpisodeState, Message
from .base import BaseCompactor
from .threshold import ThresholdPi


class ConstraintExtractor(ThresholdPi):
    name = "constraint_extractor"

    async def _extract(self, state: EpisodeState, messages: list[Message]):
        reg: list[str] = state.settings.setdefault("sc_registry", [])
        seen: list[int] = state.settings.setdefault("sc_seen_uids", [])
        src = state.full_log if state.full_log is not None else messages
        todo = [(i, m) for i, m in enumerate(src) if m.role == "user" and not m.meta.get("origin") and m.meta.get("uid") not in seen]
        known = list(reg)

        async def one(i, m):
            prev = next((x.content for x in reversed(src[:i]) if x.role == "assistant" and x.content), None)
            sys_p, user_p = build_sc_extraction_prompts(m.content or "", known, prev)
            res, l = await self.call(self.cfg.small, "small", [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}],
                                     tag="compact:extractor", max_tokens=3000, json_mode=True)
            return m, res, l

        outs = await asyncio.gather(*[one(i, m) for i, m in todo])  # user turns are independent given the registry snapshot; dedupe afterwards
        toks_in = toks_out = 0
        lat = 0.0
        for m, res, l in outs:
            toks_in, toks_out, lat = toks_in + res.usage.prompt, toks_out + res.usage.completion, max(lat, l)
            seen.append(m.meta.get("uid"))
            try:
                j = json.loads(re.search(r"\{.*\}", res.text, re.S).group(0))
                for sc in j.get("scs", []):
                    t = sc.get("text", "").strip()
                    if t and t.lower() not in {r.lower() for r in reg}:
                        reg.append(t)
            except Exception:
                pass
        return reg, toks_in, toks_out, lat

    async def __call__(self, messages: list[Message], state: EpisodeState, budget_tokens: int):
        if not self.over_threshold(state, budget_tokens):
            return self.keep(messages)
        reg, ti, to, lat_x = await self._extract(state, messages)
        res = await super().__call__(messages, state, budget_tokens)
        if res.action == CompactionAction.KEEP:
            return res
        # re-resume with the registry as the pinned tier
        summary = res.summary or ""
        res.messages = self.resume(messages, state, summary, pinned=reg)
        res.pinned = list(reg)
        # extractor tokens are small-model tokens: priced via res.calls
        res.calls.append({"role": "small", "tokens_in": ti, "tokens_out": to, "cached": 0})
        res.compactor_tokens_in += ti
        res.compactor_tokens_out += to
        res.compactor_latency_s += lat_x
        return res
