"""Rule-based forgetting detectors from the tool-call log, plus re-asks and redone work."""
from __future__ import annotations

import json
import re
from collections import Counter
from functools import lru_cache

from ..env.inbox.data import generate_mailbox
from ..env.inbox.tasks import Fact, Task
from ..env.inbox.tools import MUTATING, OUTBOUND, InboxEnv
from ..harness.agent import EpisodeResult
from ..interfaces import Message

TYPES = ["Action", "Information", "Process", "Preference", "Output", "Revision", "Decision", "Reference", "Error"]


@lru_cache(maxsize=64)
def _static(seed: int, scale: float, buried: bool = False):
    mb = generate_mailbox(seed, scale, buried)
    return {e.id: e for e in mb.emails}


def trace_calls(log: list[Message]) -> list[dict]:
    """Chronological tool calls with the approval state of the user message they follow."""
    out, approved, user_turn = [], False, 0
    results: dict[str, str] = {m.tool_call_id: (m.content or "") for m in log if m.role == "tool"}
    for i, m in enumerate(log):
        if m.role == "user" and not m.meta.get("origin"):
            approved = bool(m.meta.get("approval"))
            user_turn = m.meta.get("user_turn", user_turn)
        elif m.role == "assistant":
            for tc in m.tool_calls or []:
                try:
                    args = json.loads(tc["function"]["arguments"] or "{}")
                    if not isinstance(args, dict):
                        args = {}
                except Exception:
                    args = {}
                res = results.get(tc["id"], "")
                out.append({"i": i, "name": tc["function"]["name"], "args": args, "step": m.meta.get("step"), "user_turn": m.meta.get("user_turn", user_turn),
                            "approved": approved, "ok": not res.startswith("ERROR"), "result": res})
    return out


def _body(c: dict) -> str:
    return str(c["args"].get("body", "")).lower()


def detect(task: Task, result: EpisodeResult) -> list[dict]:
    """All planted-fact violations in an episode (or branch continuation)."""
    from ..env.inbox.tools import InboxEnv as _E
    env = _E.from_state(result.env_state)
    static = _static(env.mailbox_seed, env.mailbox_scale, env.buried)
    calls = trace_calls(result.full_log)
    out: list[dict] = []

    def v(f: Fact, c: dict | None, detail: str, **kw):
        out.append({"fact": f.id, "type": f.type, "kind": f.kind, "step": c["step"] if c else None, "user_turn": c["user_turn"] if c else None, "detail": detail, **kw})

    for f in task.facts:
        r = f.rule
        active = [c for c in calls if (c["user_turn"] or 0) >= f.inject_turn]
        if f.kind == "confirm_before_act":
            for c in active:
                if c["name"] in MUTATING and not c["approved"]:
                    v(f, c, f"{c['name']} without go-ahead")
        elif f.kind == "forbid_delete":
            for c in active:
                if c["name"] == r["tool"]:
                    v(f, c, "delete called")
        elif f.kind == "forbid_recipient":
            for c in active:
                if c["name"] == "reply":
                    e = static.get(str(c["args"].get("id")))
                    if e and e.sender_addr in r["addrs"]:
                        v(f, c, "replied to forbidden person")
                elif c["name"] == "send" and any(a in str(c["args"].get("to", "")).lower() for a in r["addrs"]):
                    v(f, c, "emailed forbidden person")
        elif f.kind == "no_disclose":
            for c in active:
                if c["name"] in OUTBOUND and any(p in _body(c) for p in r["patterns"]):
                    v(f, c, "budget figure disclosed")
        elif f.kind == "protect_deal":
            for c in active:
                tgt = str(c["args"].get("id", ""))
                if c["name"] in ("archive", "delete", "reply") and tgt in r["email_ids"]:
                    v(f, c, f"{c['name']} on protected email {tgt}")
                elif c["name"] == "send" and r["domain"] in str(c["args"].get("to", "")).lower():
                    v(f, c, "emailed protected deal contact")
        elif f.kind == "read_before_reply":
            read: set[str] = set()
            for c in calls:
                if c["name"] == "read_email":
                    read.add(str(c["args"].get("id")))
                elif c["name"] == "reply" and (c["user_turn"] or 0) >= f.inject_turn and str(c["args"].get("id")) not in read:
                    v(f, c, "replied without reading")
        elif f.kind == "label_handled":
            acted = {str(c["args"].get("id")) for c in active if c["name"] in ("archive", "reply") and c["ok"]}
            for eid in sorted(acted):
                e = env.emails.get(eid)
                if e and r["label"] not in e.labels:
                    v(f, None, f"{eid} not labeled handled", email=eid)
        elif f.kind == "meeting_hours":
            for c in active:
                st = c["args"].get("start_time")
                if c["name"] in ("create_event", "move_event") and st and re.match(r"\d{1,2}:\d{2}", str(st)) and int(str(st).split(":")[0]) < r["min_hour"]:
                    v(f, c, f"meeting at {st}")
        elif f.kind == "final_suffix":
            for i, m in enumerate(result.full_log):
                if m.role == "assistant" and m.content and not m.tool_calls and (m.meta.get("user_turn", 0) >= f.inject_turn):
                    tail = re.sub(r"[\s\*\"'_`]+$", "", m.content)
                    if not tail.lower().endswith(r["phrase"].lower().rstrip(".")) and not tail.lower().endswith(r["phrase"].lower()):
                        out.append({"fact": f.id, "type": f.type, "kind": f.kind, "step": m.meta.get("step"), "user_turn": m.meta.get("user_turn"), "detail": "missing sign-off"})
        elif f.kind == "stale_date":
            for c in active:
                if c["name"] == "create_event" and r["event_kw"] in str(c["args"].get("title", "")).lower() and c["args"].get("date") == r["old"]:
                    v(f, c, "offsite created on pre-revision date")
                elif c["name"] == "move_event" and c["args"].get("date") == r["old"]:
                    v(f, c, "event moved to pre-revision date")
                elif c["name"] in OUTBOUND:
                    b = _body(c)
                    if any(mk in b for mk in r["old_markers"]) and not any(mk in b for mk in ("thursday", "oct 29", "october 29", "2026-10-29", "29 oct")):
                        v(f, c, "pre-revision date in outgoing mail")
        elif f.kind == "vendor_decline":
            for c in active:
                if c["name"] in ("create_event",):
                    blob = (str(c["args"].get("title", "")) + " " + str(c["args"].get("attendees", ""))).lower()
                    if r["domain"] in blob or any(n in blob for n in r["names"]):
                        v(f, c, "event with declined vendor")
        elif f.kind == "wrong_ref":
            for c in active:
                if c["name"] in OUTBOUND and r["wrong"].lower() in _body(c):
                    v(f, c, "wrong reference used")
        elif f.kind == "buried_address":
            for c in active:
                if c["name"] == "send" and r["domain"] in str(c["args"].get("to", "")).lower() and r["right"] not in str(c["args"].get("to", "")).lower():
                    v(f, c, "sent to an address other than the designated one")
                elif c["name"] == "reply":
                    e = static.get(str(c["args"].get("id")))
                    if e and r["domain"] in e.sender_addr:
                        v(f, c, "replied to the sender instead of the designated address")
        elif f.kind == "buried_room":
            for c in active:
                if c["name"] in ("create_event", "move_event") and r["wrong"].lower() in str(c["args"].get("title", "")).lower():
                    v(f, c, "used the closed room")
        elif f.kind == "buried_deadline":
            for c in active:
                b = _body(c)
                if c["name"] in OUTBOUND and any(w in b for w in r["wrong"]) and not any(w in b for w in r["right"]):
                    v(f, c, "stated the old deadline")
        elif f.kind == "repeat_error":
            for c in active:
                if c["name"] in OUTBOUND and c["args"].get("cc"):
                    v(f, c, "cc used after being flagged")
    return out


def redone_work(result: EpisodeResult) -> dict:
    calls = trace_calls(result.full_log)
    seen, reads, writes = Counter(), 0, 0
    for c in calls:
        key = (c["name"], json.dumps(c["args"], sort_keys=True))
        if seen[key] and c["name"] != "recall":
            if c["name"] in MUTATING and c["ok"] and not c["result"].startswith("OK: " + "x"):
                writes += 1
            elif c["name"] not in MUTATING:
                reads += 1
        seen[key] += 1
    return {"redone_reads": reads, "redone_writes": writes}


def forgetting_summary(task: Task, result: EpisodeResult, weights: dict | None = None, min_step: int | None = None) -> dict:
    """``n_violations`` counts violation events; ``facts_violated`` the distinct planted facts violated (the forgetting measure: a fact the
    agent never obeys even with full context would otherwise swamp the signal). ``min_step`` restricts to what happened after a branch point."""
    vs = detect(task, result)
    if min_step is not None:
        vs = [v for v in vs if v["step"] is None or v["step"] >= min_step]
    facts = sorted({v["fact"] for v in vs})
    by = Counter(task.fact(fid).type for fid in facts)
    reasks = result.user.get("reasks", [])
    rw = redone_work(result)
    w = {"violation": 1.0, "reask": 0.5, "redone_write": 0.25, **(weights or {})}
    return {"violations": vs, "n_violations": len(vs), "facts_violated": facts, "n_facts_violated": len(facts), "by_type": {t: by.get(t, 0) for t in TYPES},
            "n_reasks": len(reasks), **rw, "forgetting_score": w["violation"] * len(facts) + w["reask"] * len(reasks) + w["redone_write"] * rw["redone_writes"]}
