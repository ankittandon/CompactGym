"""cliff: CliffCompaction's algorithm (vendored third_party/cliff), plus the deterministic budget fallback used by the harness.

Rules (unchanged from the paper's code): drop tool results > 500 chars, tool calls -> ~150 char signatures,
reasoning -> 300 chars, keep system prompt + first user message + last K turns verbatim, discard any previous
compacted block instead of compacting it again. Like the real proxy, it compacts the *uncompacted* log
(``state.full_log``), not the already-compacted live context.
"""
from __future__ import annotations

from ..config import Config
from ..harness.resume import ensure_pinned
from ..harness.tokens import count_tokens
from ..third_party.cliff.cliff import compact as _cliff
from ..third_party.cliff.config import Config as CliffConfig
from ..third_party.cliff.dialect_base import SUMMARY_HEADER
from ..third_party.cliff.dialect_openai_chat import DIALECT
from ..interfaces import CompactionAction, EpisodeState, Message
from .base import BaseCompactor


def _cliff_cfg(keep_recent: int, budget_tokens: int = 0) -> CliffConfig:
    return CliffConfig(threshold_tokens=budget_tokens, keep_recent=keep_recent, thinking_max_chars=300, keep_thinking=True,
                       cmd_max_chars=150, result_max_chars=500)


def cliff_messages(messages: list[Message], keep_recent: int) -> list[Message] | None:
    """Run the vendored algorithm over our messages; returns the compacted list (kept messages are the same objects)."""
    dicts, by_id = [], {}
    for m in messages:
        d = m.to_api()
        d["reasoning_content"] = m.meta.get("reasoning") or ""
        dicts.append(d)
        by_id[id(d)] = m
    res = _cliff(dicts, DIALECT, _cliff_cfg(keep_recent))
    if res is None:
        return None
    out: list[Message] = []
    for d in res.messages:
        if id(d) in by_id:
            out.append(by_id[id(d)])
        else:
            out.append(Message(role="user", content=d["content"], meta={"origin": "compaction_summary", "cliff": True}))
    return out


def fallback_compact(history: list[Message], budget: int, tools: list[dict], overhead: int, state: EpisodeState) -> list[Message]:
    """Deterministic last resort when a compactor leaves the context over budget (logged as forced)."""
    fits = lambda ms: count_tokens(ms, tools, overhead) <= budget  # noqa: E731
    system = [m for m in history if m.role == "system"]
    pinned = [m for m in history if m.meta.get("origin") == "pinned"]
    old_summ = [m for m in history if m.meta.get("origin") == "compaction_summary" and not m.meta.get("cliff")]
    rest = [m for m in history if m.role != "system" and m.meta.get("origin") not in ("pinned", "compaction_summary")]
    for k in dict.fromkeys([max(1, state.keep_k), 1]):  # never drop the latest turn
        comp = cliff_messages(rest, k)
        cand = system + pinned + old_summ + (comp if comp is not None else rest)
        if fits(cand):
            return cand
        cand2 = system + pinned + (comp if comp is not None else rest)  # drop the old summary before giving up on a smaller k
        if fits(cand2):
            return cand2
    # hard truncation: shrink the largest message bodies until it fits
    cand = system + pinned + (comp if comp is not None else rest)
    for lim in (1500, 800, 400, 200, 80):
        cand = [m.model_copy(update={"content": (m.content or "")[:lim] + ("…" if len(m.content or "") > lim else "")}) if m.role in ("tool", "assistant") and len(m.content or "") > lim else m for m in cand]
        if fits(cand):
            return cand
    while len(cand) > len(system) + len(pinned) + 1 and not fits(cand):
        del cand[len(system) + len(pinned)]
    return cand


class Cliff(BaseCompactor):
    name = "cliff"

    def __init__(self, cfg: Config, llm=None, keep_recent: int = 3, **kw):
        super().__init__(cfg, llm, **kw)
        self.keep_recent = keep_recent

    async def __call__(self, messages: list[Message], state: EpisodeState, budget_tokens: int):
        if not self.over_threshold(state, budget_tokens):
            return self.keep(messages)
        source = [m for m in (state.full_log if state.full_log is not None else messages) if m.meta.get("origin") not in ("pinned", "compaction_summary")]
        out = cliff_messages(source, self.keep_recent)
        if out is None:
            return self.keep(messages)
        out = ensure_pinned(out, state.pinned, bool(state.settings.get("pinned_tier", True)))
        return self.result(CompactionAction.TRUNCATE_VERBATIM, out, None, None, 0.0, None)
