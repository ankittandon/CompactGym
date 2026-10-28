"""edit_context: Shelley-style in-place compaction. The model sees a numbered view of the live context and returns edits by message index:

* ``trim``   {"op": "trim", "ids": [6, 10]}                      blank a tool result (its structure stays; the original is saved to memory)
* ``squish`` {"op": "squish", "start": 2, "end": 11, "note": ""}  replace a contiguous range by one short note (original saved to memory)
* ``hide``   {"op": "hide", "ids": [19]}                         drop a whole unit (a user/assistant message, or an assistant tool call with its results)

Every removed original is stored in the memory store, so the note carries a pointer and the agent can ``recall(query)`` it: nothing is lost.
Edits are validated like the real tool rejects bad calls: protected messages (system, pinned, the last k turns), ranges that split a tool call from
its result, notes that are not shorter than what they replace, and overlapping ops are rejected. One retry shows the model its errors; invalid ops
that remain are dropped (the harness' budget fallback still guarantees the budget).
"""
from __future__ import annotations

import json
import re

from ..harness.resume import ensure_pinned, split_for_resume, transcript
from ..interfaces import CompactionAction, CompactionResult, EpisodeState, Message
from .base import BaseCompactor

VIEW_CHARS = 1200  # per-message preview shown to the editing model

SYSTEM = """You are the context editor for a tool-using email and calendar assistant. The conversation below is numbered (#index). Your context is over budget: free space by editing it. You cannot touch messages marked [protected].

Edit operations (reply with ONE JSON object only):
{{"ops": [
  {{"op": "trim", "ids": [<tool result indices whose content is no longer needed>]}},
  {{"op": "squish", "start": <first index>, "end": <last index>, "note": "<short note that keeps what must survive>"}},
  {{"op": "hide", "ids": [<indices of whole units that are no longer needed>]}}
 ],
 "pinned": ["<exact user instruction or fact that must survive every future edit>", ...],
 "confidence": <0..1 that nothing needed to continue correctly was lost>}}

Rules:
- Originals are saved to memory and the note gets a pointer, so the assistant can recall them -- but it will not know to unless your note says what is there. Put every specific (names, ids, dates, numbers, addresses) that later requests could need into the note or into "pinned".
- NEVER LOSE: every instruction, restriction, preference, correction, decision, reference number, changed date or person to exclude that the user gave -- types: {types}. Put each, verbatim with its specifics, into "pinned".
- A squish range must be contiguous and must not cut a tool call off from its result (an assistant message with tool calls and all its tool results go together). A note must be shorter than what it replaces.
- Prefer trimming bulky tool results and squishing finished sub-tasks. Free at least {need} tokens."""


def est_tokens(m: Message) -> int:
    n = len(m.content or "") + sum(len(tc["function"]["arguments"]) + len(tc["function"]["name"]) for tc in (m.tool_calls or []))
    return n // 4 + 4


def protected_indices(messages: list[Message], keep_k: int) -> set[int]:
    """System, pinned, and the last k turns (plus any unanswered user message) cannot be edited."""
    _, _, tail = split_for_resume(messages, keep_k)
    tail_ids = {id(m) for m in tail}
    return {i for i, m in enumerate(messages) if m.role == "system" or m.meta.get("origin") == "pinned" or id(m) in tail_ids}


def render_view(messages: list[Message], protected: set[int]) -> str:
    lines = []
    for i, m in enumerate(messages):
        tag = " [protected]" if i in protected else ""
        body = m.content or ""
        for tc in m.tool_calls or []:
            body += f"\n[tool call] {tc['function']['name']}({tc['function']['arguments']})"
        extra = f" ...(+{len(body) - VIEW_CHARS} chars not shown)" if len(body) > VIEW_CHARS else ""
        label = m.role if m.role != "tool" else f"tool result {m.meta.get('name', '')}".strip()
        lines.append(f"#{i} [{label}] ~{est_tokens(m)} tokens{tag}\n{body[:VIEW_CHARS]}{extra}")
    return "\n\n".join(lines)


def _units(messages: list[Message]) -> dict[int, list[int]]:
    """For each assistant message index with tool calls: [its index] + the indices of its tool results."""
    by_call: dict[str, int] = {}
    out: dict[int, list[int]] = {}
    for i, m in enumerate(messages):
        if m.role == "assistant" and m.tool_calls:
            out[i] = [i]
            for tc in m.tool_calls:
                by_call[tc["id"]] = i
        elif m.role == "tool" and m.tool_call_id in by_call:
            out[by_call[m.tool_call_id]].append(i)
    return out


def apply_edits(messages: list[Message], ops: list[dict], protected: set[int], memory) -> tuple[list[Message], list[str], list[str]]:
    """-> (new messages, applied descriptions, errors). Invalid ops are skipped (and reported); valid ones are applied together."""
    n = len(messages)
    units = _units(messages)
    owner = {j: a for a, js in units.items() for j in js}  # message index -> its unit's assistant index
    claimed: set[int] = set()
    plan: dict[int, tuple] = {}  # index -> ("trim",) | ("hide", unit) | ("squish", start, end, note)
    errors: list[str] = []
    applied: list[str] = []

    def free(idx: list[int], what: str) -> bool:
        bad = [i for i in idx if i < 0 or i >= n]
        if bad:
            errors.append(f"{what}: index {bad[0]} does not exist")
            return False
        prot = [i for i in idx if i in protected]
        if prot:
            errors.append(f"{what}: #{prot[0]} is protected")
            return False
        over = [i for i in idx if i in claimed]
        if over:
            errors.append(f"{what}: #{over[0]} is already used by another op")
            return False
        return True

    for op in ops if isinstance(ops, list) else []:
        if not isinstance(op, dict):
            errors.append("op is not an object")
            continue
        kind = op.get("op")
        if kind == "trim":
            ids = [i for i in op.get("ids", []) if isinstance(i, int)]
            if not ids:
                errors.append("trim: needs valid ids")
                continue
            if not free(ids, "trim"):
                continue
            if any(messages[i].role != "tool" for i in ids):
                errors.append("trim: only tool results can be trimmed (use squish or hide for messages)")
                continue
            for i in ids:
                claimed.add(i)
                plan[i] = ("trim",)
            applied.append(f"trim {ids}")
        elif kind == "squish":
            a, b, note = op.get("start"), op.get("end"), str(op.get("note") or "").strip()
            if not (isinstance(a, int) and isinstance(b, int)) or a > b:
                errors.append("squish: needs integer start <= end")
                continue
            idx = list(range(a, b + 1))
            if not free(idx, "squish"):
                continue
            if not note:
                errors.append("squish: a note is required")
                continue
            if any(owner.get(i) is not None and not set(units[owner[i]]) <= set(idx) for i in idx):
                errors.append(f"squish #{a}-#{b}: cuts a tool call off from its result (take the whole call and all its results)")
                continue
            if len(note) // 4 + 4 >= sum(est_tokens(messages[i]) for i in idx):
                errors.append(f"squish #{a}-#{b}: the note must be shorter than what it replaces")
                continue
            for i in idx:
                claimed.add(i)
            plan[a] = ("squish", a, b, note)
            for i in idx[1:]:
                plan[i] = ("skip",)
            applied.append(f"squish #{a}-#{b}")
        elif kind == "hide":
            ids = [i for i in op.get("ids", []) if isinstance(i, int)]
            if not ids:
                errors.append("hide: needs ids")
                continue
            whole = sorted({j for i in ids for j in (units[owner[i]] if i in owner else [i])})
            if any(i in owner and owner[i] != i for i in ids):
                errors.append("hide: name the assistant message of a tool call to hide the call with its results")
                continue
            if not free(whole, "hide"):
                continue
            for i in whole:
                claimed.add(i)
                plan[i] = ("hide", tuple(whole))
            applied.append(f"hide {whole}")
        else:
            errors.append(f"unknown op {kind!r}")

    out: list[Message] = []
    done_hide: set[tuple] = set()
    for i, m in enumerate(messages):
        step = plan.get(i)
        if step is None:
            out.append(m)
        elif step[0] == "trim":
            mid = memory.add(transcript([m]), turns=[m.meta.get("step")], users=[m.meta.get("user_turn")])
            out.append(m.model_copy(update={"content": f"[trimmed: {len(m.content or '')} chars removed; original saved as mem {mid} -- recall(query) to retrieve]",
                                            "meta": {**m.meta, "trimmed": mid}}))
        elif step[0] == "squish":
            _, a, b, note = step
            mid = memory.add(transcript(messages[a:b + 1]), turns=[x.meta.get("step") for x in messages[a:b + 1]], users=[x.meta.get("user_turn") for x in messages[a:b + 1]])
            out.append(Message(role="user", content=f"[Squished #{a}-#{b}; originals saved as mem {mid} -- recall(query) to retrieve]\n{note}", meta={"origin": "compaction_summary", "edit": True}))
        elif step[0] == "hide":
            if step[1] not in done_hide:
                done_hide.add(step[1])
                memory.add(transcript([messages[j] for j in step[1]]), turns=[messages[j].meta.get("step") for j in step[1]], users=[messages[j].meta.get("user_turn") for j in step[1]])
        # "skip": swallowed by a squish
    return out, applied, errors


def parse_edit_reply(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except Exception:
        return None
    return d if isinstance(d, dict) else None


class EditContext(BaseCompactor):
    """Edits the live context in place by message index. ``role`` picks the editing model ("frontier" = the Shelley-style baseline, "small" = what we train)."""
    name = "edit_context"
    uses_memory = True

    def __init__(self, cfg, llm=None, role: str = "frontier", **kw):
        super().__init__(cfg, llm, **kw)
        self.role = role

    async def __call__(self, messages: list[Message], state: EpisodeState, budget_tokens: int):
        if not self.over_threshold(state, budget_tokens):
            return self.keep(messages)
        prot = protected_indices(messages, self.k(state))
        if len(prot) >= len(messages):
            return self.keep(messages)
        need = max(256, state.context_tokens - int(0.6 * budget_tokens))
        types = ", ".join(state.profile.get("never_lose", [])) or "all"
        pinned = "\n".join(f"- {p}" for p in state.pinned) or "(none yet)"
        user = (f"Context: {state.context_tokens} tokens of a {budget_tokens} budget. Already pinned:\n{pinned}\n\n=== Numbered context ===\n"
                f"{render_view(messages, prot)}\n=== end ===\nReturn the JSON edit now.")
        convo = [{"role": "system", "content": SYSTEM.format(types=types, need=need)}, {"role": "user", "content": user}]
        spec = getattr(self.cfg, self.role)
        calls: list[dict] = []
        lat_total, tin, tout = 0.0, 0, 0
        out, applied, errors, d = messages, [], [], None
        for attempt in range(2):
            res, lat = await self.call(spec, self.role, convo, tag=f"compact:{self.name}", max_tokens=1500)
            lat_total += lat
            tin += res.usage.prompt
            tout += res.usage.completion
            calls.append({"role": self.role, "tokens_in": res.usage.prompt, "tokens_out": res.usage.completion, "cached": res.usage.cached})
            d = parse_edit_reply(res.text)
            if d is None:
                errors = ["the reply was not a JSON object"]
            else:
                out, applied, errors = apply_edits(messages, d.get("ops") or [], prot, state.memory)
                if not errors or attempt == 1:
                    break
            convo = convo + [{"role": "assistant", "content": res.text}, {"role": "user", "content": "Rejected: " + "; ".join(errors[:6]) + ". Fix it and return the full JSON again."}]
        state.settings["edit_log"] = {"applied": applied, "rejected": errors}
        if not applied:
            r = self.keep(messages)
            r.compactor_latency_s, r.compactor_tokens_in, r.compactor_tokens_out, r.compactor_model, r.calls = lat_total, tin, tout, self.role, calls
            return r
        new_pinned = [str(p) for p in ((d or {}).get("pinned") or []) if isinstance(p, (str, int, float))]
        allp = list(dict.fromkeys(list(state.pinned) + new_pinned))
        out = ensure_pinned(out, allp, bool(state.settings.get("pinned_tier", True)))
        try:
            conf = max(0.0, min(1.0, float((d or {}).get("confidence")))) if (d or {}).get("confidence") is not None else None
        except (TypeError, ValueError):
            conf = None
        notes = "\n".join(m.content for m in out if m.meta.get("edit"))
        return CompactionResult(action=CompactionAction.EDIT_CONTEXT, messages=out, summary=notes or None, pinned=new_pinned, confidence=conf,
                                compactor_latency_s=lat_total, compactor_tokens_in=tin, compactor_tokens_out=tout, blocking=True,
                                compactor_model=self.role, calls=calls)
