"""Shared compactor plumbing."""
from __future__ import annotations

import re
import time

from ..config import Config, ModelSpec
from ..harness.resume import build_resume, ensure_pinned, transcript
from ..llm import LLM, ChatResult
from ..interfaces import CompactionAction, CompactionResult, EpisodeState, Message


class BaseCompactor:
    name = "base"
    uses_memory = False

    def __init__(self, cfg: Config, llm: LLM | None = None, threshold_frac: float | None = None, **kw):
        self.cfg, self.llm = cfg, llm
        self.threshold_frac = cfg.threshold_frac if threshold_frac is None else threshold_frac
        self.kw = kw

    def over_threshold(self, state: EpisodeState, budget: int) -> bool:
        return state.context_tokens >= self.threshold_frac * budget

    def keep(self, messages: list[Message]) -> CompactionResult:
        return CompactionResult(action=CompactionAction.KEEP, messages=messages, blocking=True)

    def k(self, state: EpisodeState) -> int:
        return state.keep_k

    def resume(self, messages: list[Message], state: EpisodeState, summary: str | None, pinned: list[str] | None = None, *, k: int | None = None,
               first_user: bool = False) -> list[Message]:
        """Resume format (system + pinned + summary + last k turns). With the pinned tier ablated, pinned facts are folded into the summary."""
        allp = list(dict.fromkeys(list(state.pinned) + list(pinned or [])))
        enabled = bool(state.settings.get("pinned_tier", True))
        if not enabled and allp:
            summary = (summary or "") + "\n\nKey user facts:\n" + "\n".join(f"- {p}" for p in allp)
            allp = []
        return build_resume(messages, summary, allp, self.k(state) if k is None else k, first_user=first_user)

    async def call(self, spec: ModelSpec, role: str, msgs: list[dict], *, tag: str, max_tokens: int | None = None,
                   temperature: float | None = None, json_mode: bool = False) -> tuple[ChatResult, float]:
        assert self.llm is not None
        t0 = time.perf_counter()
        res = await self.llm.chat(spec, msgs, tag=tag, max_tokens=max_tokens, temperature=temperature, json_mode=json_mode)
        if not res.text.strip() and res.finish_reason == "length":  # a reasoning model spent its whole budget thinking: retry with more room
            res = await self.llm.chat(spec, msgs, tag=tag + ":retry", max_tokens=(max_tokens or spec.max_tokens) * 2, temperature=temperature, json_mode=json_mode)
        return res, time.perf_counter() - t0

    def result(self, action: CompactionAction, messages: list[Message], summary: str | None, res: ChatResult | None, lat: float, role: str | None,
               pinned: list[str] | None = None, confidence: float | None = None, blocking: bool = True) -> CompactionResult:
        return CompactionResult(action=action, messages=messages, summary=summary, pinned=pinned or [], confidence=confidence,
                                compactor_latency_s=lat, compactor_tokens_in=res.usage.prompt if res else 0,
                                compactor_tokens_out=res.usage.completion if res else 0, blocking=blocking, compactor_model=role,
                                calls=[{"role": role, "tokens_in": res.usage.prompt, "tokens_out": res.usage.completion, "cached": res.usage.cached}] if res else [])


def strip_summary_tags(text: str) -> str:
    m = re.search(r"<summary>(.*?)</summary>", text, re.S)
    return (m.group(1) if m else text).strip()


def full_transcript(messages: list[Message]) -> str:
    return transcript(messages, max_tool_chars=None)
