"""Success checker: diff the final mailbox/calendar against the expected state, with partial credit per goal."""
from __future__ import annotations

from ..env.inbox.tasks import Goal, Task
from ..env.inbox.tools import InboxEnv
from ..harness.agent import EpisodeResult


def _texts_in_turn(result: EpisodeResult, turn: int) -> str:
    return " ".join((m.content or "") for m in result.full_log if m.role == "assistant" and not m.tool_calls and m.meta.get("user_turn") == turn).lower()


def check_score(chk: dict, env: InboxEnv, result: EpisodeResult, goal: Goal) -> float:
    t = chk["type"]
    if t == "emails_in_folders":
        ids = chk["ids"]
        return sum(env.emails[i].folder in chk["folders"] for i in ids) / len(ids) if ids else 1.0
    if t == "emails_labeled":
        ids = chk["ids"]
        return sum(chk["label"] in env.emails[i].labels for i in ids) / len(ids) if ids else 1.0
    if t == "sent_to":
        for s in env.sent:
            if chk["addr"].lower() in s["to"].lower():
                if not chk.get("body_any") or any(k in s["body"].lower() for k in chk["body_any"]):
                    return 1.0
        return 0.0
    if t == "not_sent_to":
        return 0.0 if any(chk["addr"].lower() in s["to"].lower() for s in env.sent) else 1.0
    if t == "body_absent":
        mails = [s for s in env.sent if chk["addr"].lower() in s["to"].lower()]
        return 1.0 if mails and not any(k in s["body"].lower() for s in mails for k in chk["none"]) else 0.0
    if t in ("event_exists", "no_event"):
        hits = 0
        for ev in env.events.values():
            if ev.status == "cancelled":
                continue
            if chk.get("date") and ev.date != chk["date"]:
                continue
            title = ev.title.lower()
            att = " ".join(ev.attendees).lower()
            tm = any(k.lower() in title for k in chk.get("title_any", []))
            am = any(k.lower() in att for k in chk.get("attendee_any", []))
            if t == "event_exists":
                hits += tm or am
            else:
                hits += (am or tm) if not chk.get("date") else (tm or am)
        if t == "event_exists":
            return 1.0 if hits else 0.0
        return 0.0 if hits else 1.0
    if t == "event_cancelled":
        ev = env.events.get(chk["id"])
        return 1.0 if ev and ev.status == "cancelled" else 0.0
    if t == "answer_contains":
        txt = _texts_in_turn(result, goal.turn)
        return 1.0 if any(k.lower() in txt for k in chk["any"]) else 0.0
    raise ValueError(t)


def success_summary(task: Task, result: EpisodeResult, reached_turn: int | None = None) -> dict:
    env = InboxEnv.from_state(result.env_state)
    last_turn = max([m.meta.get("user_turn", 0) for m in result.full_log if m.role == "user" and not m.meta.get("origin")] or [0])
    per_goal = {}
    for g in task.goals:
        if g.turn > last_turn:
            per_goal[g.id] = 0.0  # never reached
            continue
        s = [check_score(c, env, result, g) for c in g.checks]
        per_goal[g.id] = sum(s) / len(s) if s else 1.0
    n = len(task.goals)
    return {"success": sum(per_goal.values()) / n if n else 0.0, "per_goal": per_goal, "goals_complete": sum(v >= 0.999 for v in per_goal.values()),
            "n_goals": n, "reached_turn": last_turn}
