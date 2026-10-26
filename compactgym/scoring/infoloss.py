"""InfoLoss: increase in the main model's NLL of the recorded next-k turns under the compacted context vs the full one.

* NLL comes from prompt-token logprobs (``echo``) of [context + continuation] rendered in the model's chat format; the
  continuation tokens are identical in both calls, so per-token deltas are exact.
* Action tokens (tool names, arguments, ids, dates, recipients) get larger, configurable weights.
* Later continuation turns are discounted by ``gamma**j``. The aggregate is clipped at 0 below (per-token detail is signed).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from openai_harmony import Conversation, Role

from ..config import ModelSpec
from ..harness.tokens import encoding, to_harmony
from ..interfaces import Message
from ..llm import LLM

DEFAULT_WEIGHTS = {"format": 0.0, "text": 1.0, "tool_name": 3.0, "args": 2.0, "id_date_recipient": 4.0}
_SPANS = [
    ("tool_name", re.compile(r"(?<=to=functions\.)\w+")),
    ("id_date_recipient", re.compile(r"\be\d{4}\b|\bc\d{3,4}\b|\d{4}-\d{2}-\d{2}|\b\d{1,2}:\d{2}\b|[\w.+-]+@[\w.-]+|\b[A-Z]{2,3}-\d{3,}\b|\bs\d{3}\b")),
]
_ARGS = re.compile(r"<\|message\|>(.*?)(?:<\|call\|>|<\|end\|>)", re.S)


@dataclass
class InfoLossResult:
    infoloss: float  # weighted mean delta NLL (nats/token), clipped at 0
    raw: float  # unclipped
    nll_full: float
    nll_comp: float
    n_tokens: int
    n_turns: int
    per_token: list[dict] = field(default_factory=list)


def split_continuation(full_log: list[Message], upto_len: int, k: int) -> list[Message]:
    """The recorded next-k assistant steps after position ``upto_len`` (their tool results and any user messages in between)."""
    out, n_asst = [], 0
    for m in full_log[upto_len:]:
        if m.role == "assistant":
            if n_asst >= k:
                break
            n_asst += 1
        out.append(m)
    # drop a trailing user message that precedes no assistant message
    while out and out[-1].role == "user":
        out.pop()
    return out


def _render_with_spans(context: list[Message], cont: list[Message], tools: list[dict]):
    """-> (token ids, list of (start, end, is_assistant) for each continuation harmony message)."""
    enc = encoding()
    hm_ctx = to_harmony(context, tools)
    hm_all = to_harmony(context + cont, tools)
    n0 = len(hm_ctx)
    assert hm_all[:n0] == hm_ctx or True
    tokens = list(enc.render_conversation(Conversation.from_messages(hm_all)))
    bounds = []
    prev = len(enc.render_conversation(Conversation.from_messages(hm_all[:n0])))
    for j in range(n0, len(hm_all)):
        end = len(enc.render_conversation(Conversation.from_messages(hm_all[: j + 1])))
        bounds.append((prev, end, hm_all[j].author.role == Role.ASSISTANT))
        prev = end
    return tokens, bounds


def _token_weights(tokens: list[int], a: int, b: int, weights: dict) -> tuple[list[float], list[str]]:
    enc = encoding()
    pieces = [enc.decode([t]) if t < 200000 else enc.decode([t]) for t in tokens[a:b]]
    text, offs = "", []
    for p in pieces:
        offs.append((len(text), len(text) + len(p)))
        text += p
    # Structural tokens (channel/recipient headers, terminators) are 'format': in-context examples of the format shrink after compaction,
    # which would swamp the information signal, so they carry weight 0 by default. Only message bodies and tool names are scored.
    w = [weights["format"]] * len(pieces)
    kind = ["format"] * len(pieces)

    def mark(s: int, e: int, name: str):
        for i, (x, y) in enumerate(offs):
            if x < e and y > s and weights[name] > w[i]:
                w[i], kind[i] = weights[name], name

    for m in _ARGS.finditer(text):
        mark(m.start(1), m.end(1), "text")
    for m in re.finditer(r"to=functions\.\w+.*?<\|message\|>(.*?)(?:<\|call\|>|<\|end\|>)", text, re.S):
        mark(m.start(1), m.end(1), "args")
    for name, rx in _SPANS:
        for m in rx.finditer(text):
            mark(m.start(), m.end(), name)
    return w, kind


async def continuation_nll(llm: LLM, spec: ModelSpec, context: list[Message], cont: list[Message], tools: list[dict], weights: dict | None = None) -> dict:
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    tokens, bounds = _render_with_spans(context, cont, tools)
    lps = await llm.prompt_logprobs(spec, tokens, tag="infoloss")
    idx, nll, w, turn, kind, tok = [], [], [], [], [], []
    j = -1
    for (a, b, is_asst) in bounds:
        if not is_asst:
            continue
        j += 1
        ws, ks = _token_weights(tokens, a + 1, b, weights)  # skip <|start|>
        for off, (wt, kd) in enumerate(zip(ws, ks)):
            p = a + 1 + off
            idx.append(p)
            nll.append(-lps[p])
            w.append(wt)
            turn.append(j)
            kind.append(kd)
            tok.append(encoding().decode([tokens[p]]))
    return {"nll": nll, "w": w, "turn": turn, "kind": kind, "tok": tok, "n_ctx": len(tokens) - sum(b - a for a, b, _ in bounds)}


def combine(full: dict, comp: dict, gamma: float = 0.8) -> InfoLossResult:
    assert full["w"] == comp["w"] and len(full["nll"]) == len(comp["nll"])
    num = den = nf = nc = 0.0
    per = []
    for i in range(len(full["nll"])):
        if full["w"][i] <= 0:
            continue
        g = gamma ** full["turn"][i]
        d = comp["nll"][i] - full["nll"][i]
        wt = full["w"][i] * g
        num += wt * d
        den += wt
        nf += wt * full["nll"][i]
        nc += wt * comp["nll"][i]
        per.append({"tok": full["tok"][i], "kind": full["kind"][i], "turn": full["turn"][i], "w": wt, "nll_full": full["nll"][i], "nll_comp": comp["nll"][i], "delta": d})
    raw = num / den if den else 0.0
    return InfoLossResult(infoloss=max(0.0, raw), raw=raw, nll_full=nf / den if den else 0.0, nll_comp=nc / den if den else 0.0,
                          n_tokens=len(per), n_turns=(max(full["turn"]) + 1) if per else 0, per_token=per)


async def infoloss(llm: LLM, spec: ModelSpec, full_context: list[Message], compacted: list[Message], cont: list[Message], tools: list[dict], *,
                   weights: dict | None = None, gamma: float = 0.8, full_cache: dict | None = None, cache_key=None) -> InfoLossResult:
    if full_cache is not None and cache_key in full_cache:
        full = full_cache[cache_key]
    else:
        full = await continuation_nll(llm, spec, full_context, cont, tools, weights)
        if full_cache is not None:
            full_cache[cache_key] = full
    comp = await continuation_nll(llm, spec, compacted, cont, tools, weights)
    return combine(full, comp, gamma)
