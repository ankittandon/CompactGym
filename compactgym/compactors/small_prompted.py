"""small_prompted: the small model is prompted every turn to pick an action, write the summary, fill the pinned tier and state its confidence."""
from __future__ import annotations

import json
import re

from ..harness.resume import ensure_pinned, split_for_resume, transcript
from ..interfaces import CompactionAction, CompactionResult, EpisodeState, Message
from .base import BaseCompactor
from .cliff import fallback_compact
from .frontier import Frontier

SYSTEM = """You are the context-compaction controller for a tool-using email and calendar assistant. Before every model call you decide whether to compact the conversation, and how.

Actions:
- keep: do nothing (only if the context comfortably fits).
- truncate_verbatim: drop old tool outputs and shorten old tool calls, no rewriting (cheap, lossy for details).
- clear_tool_outputs: blank out old tool outputs but keep every message.
- summarize_older: replace everything except the last {k} turns with a summary you write.
- rewrite: like summarize_older but keep only the last turn.
- offload_to_memory: move old turns to external memory (reversible; the assistant can recall(query) them) and leave an index.
- defer_to_frontier: ask a large model to write the summary (accurate but slow and expensive).

Compacting breaks the prompt cache (the next call re-reads the whole new context at full price) and may pause the user; waiting costs tokens and risks the context overflowing the budget.
NEVER LOSE: every instruction, restriction, preference, correction, decision, reference number, changed date or person to exclude that the user gave -- types: {types}. Put each of them, with every specific (names, ids, dates, numbers), into "pinned" as a short exact statement. Pinned items survive all future compactions verbatim.

Reply with ONE JSON object only:
{{"action": "<one of the actions>", "summary": "<for summarize_older/rewrite: goal, completed actions with ids, key observations, unresolved errors, current state, next steps>", "pinned": ["..."], "confidence": <0..1 your estimate that the compacted context keeps everything needed to continue correctly>}}"""


def parse_decision(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {"action": "summarize_older", "summary": text.strip(), "pinned": [], "confidence": None, "_parse_failed": True}


class SmallPrompted(BaseCompactor):
    name = "small_prompted"
    uses_memory = True

    def __init__(self, cfg, llm=None, min_fill: float = 0.5, must_compact: bool = False, temperature: float | None = None, **kw):
        super().__init__(cfg, llm, **kw)
        self.min_fill, self.must_compact, self.temperature = min_fill, must_compact, temperature

    def _user_prompt(self, messages: list[Message], state: EpisodeState, older: list[Message], budget: int) -> str:
        pinned = "\n".join(f"- {p}" for p in state.pinned) or "(none yet)"
        return (f"Context size: {state.context_tokens} tokens of a {budget} budget ({100 * state.context_tokens // max(1, budget)}% full). Compactions so far: {state.n_compactions}.\n"
                f"Already pinned:\n{pinned}\n\n"
                f"=== Conversation to be compacted (everything except the last {state.keep_k} turns, which are always kept) ===\n"
                f"{transcript(older, max_tool_chars=700)}\n=== end ===\nChoose the action now. JSON only.")

    async def __call__(self, messages: list[Message], state: EpisodeState, budget_tokens: int):
        if not self.must_compact and state.context_tokens < self.min_fill * budget_tokens:
            return self.keep(messages)
        system, older, tail = split_for_resume(messages, state.keep_k)
        if not older:
            return self.keep(messages)
        types = ", ".join(state.profile.get("never_lose", [])) or "all"
        sys_p = SYSTEM.format(k=state.keep_k, types=types)
        if self.must_compact:
            sys_p += "\nThe context is over budget: you MUST pick an action other than keep."
        res, lat = await self.call(self.cfg.small, "small", [{"role": "system", "content": sys_p}, {"role": "user", "content": self._user_prompt(messages, state, older, budget_tokens)}],
                                   tag="compact:small_prompted", max_tokens=1500, temperature=self.temperature)
        d = parse_decision(res.text)
        try:
            action = CompactionAction(str(d.get("action", "")).strip().lower())
        except ValueError:
            action = CompactionAction.SUMMARIZE_OLDER
        if self.must_compact and action == CompactionAction.KEEP:
            action = CompactionAction.SUMMARIZE_OLDER
        pinned = [str(p) for p in (d.get("pinned") or []) if isinstance(p, (str, int, float))]
        try:
            conf = max(0.0, min(1.0, float(d.get("confidence")))) if d.get("confidence") is not None else None
        except (TypeError, ValueError):
            conf = None
        summary = d.get("summary") if isinstance(d.get("summary"), str) else None
        enabled = bool(state.settings.get("pinned_tier", True))
        base = dict(res=res, lat=lat, role="small", pinned=pinned, confidence=conf)
        if action == CompactionAction.KEEP:
            r = self.result(action, messages, None, res, lat, "small", pinned=pinned, confidence=conf)
            return r
        if action in (CompactionAction.SUMMARIZE_OLDER, CompactionAction.REWRITE):
            if not summary:
                summary = "(no summary provided)"
            k = 1 if action == CompactionAction.REWRITE else state.keep_k
            return self.result(action, self.resume(messages, state, summary, pinned, k=k), summary, res, lat, "small", pinned=pinned, confidence=conf)
        if action == CompactionAction.DEFER_TO_FRONTIER:
            fr = Frontier(self.cfg, self.llm)
            s, fres, flat, _ = await fr.summarize(messages, state)
            out = self.result(action, self.resume(messages, state, s, pinned), s, fres, flat, "frontier", pinned=pinned, confidence=conf)
            out.calls.append({"role": "small", "tokens_in": res.usage.prompt, "tokens_out": res.usage.completion, "cached": res.usage.cached})
            out.compactor_latency_s += lat
            return out
        if action == CompactionAction.OFFLOAD_TO_MEMORY:
            from .memory_offload import MemoryOffload
            mo = MemoryOffload(self.cfg, self.llm, threshold_frac=0.0)
            r = await mo(messages, state, budget_tokens)
            r.pinned, r.confidence, r.action = pinned, conf, action
            r.messages = self.resume(messages, state, r.summary, pinned)
            r.calls = [{"role": "small", "tokens_in": res.usage.prompt, "tokens_out": res.usage.completion, "cached": res.usage.cached}]
            r.compactor_tokens_in, r.compactor_tokens_out, r.compactor_latency_s, r.compactor_model = res.usage.prompt, res.usage.completion, lat, "small"
            return r
        if action == CompactionAction.CLEAR_TOOL_OUTPUTS:
            keep_ids = {id(m) for m in tail}
            new = [m if (m.role != "tool" or id(m) in keep_ids) else m.model_copy(update={"content": "[tool output cleared]"}) for m in messages]
            new = ensure_pinned(new, list(dict.fromkeys(state.pinned + pinned)), enabled)
            return self.result(action, new, None, res, lat, "small", pinned=pinned, confidence=conf)
        # TRUNCATE_VERBATIM
        from .cliff import cliff_messages
        src = [m for m in messages if m.role != "system" and m.meta.get("origin") not in ("pinned", "compaction_summary")]
        out = cliff_messages(src, state.keep_k)
        new = ensure_pinned([m for m in messages if m.role == "system"] + (out or src), list(dict.fromkeys(state.pinned + pinned)), enabled)
        return self.result(action, new, None, res, lat, "small", pinned=pinned, confidence=conf)
