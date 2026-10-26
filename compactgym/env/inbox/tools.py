"""Deterministic inbox/calendar tools. All outputs are pure functions of (state, args)."""
from __future__ import annotations

import json
import re
from datetime import date, datetime
from typing import Any

from .data import TODAY, USER_ADDR, USER_NAME, Email, Event, Mailbox, generate_mailbox

MUTATING = {"reply", "send", "archive", "delete", "create_event", "move_event", "cancel_event"}
OUTBOUND = {"reply", "send"}

TOOL_SPECS: list[dict] = [
    {"name": "list_emails", "description": "List emails in a folder, newest first. Returns id, date, sender, subject and a snippet. Paged.",
     "parameters": {"type": "object", "properties": {"folder": {"type": "string", "enum": ["inbox", "archive", "trash", "sent"]},
                                                       "limit": {"type": "integer"}, "offset": {"type": "integer"}, "label": {"type": "string"}}}},
    {"name": "search_emails", "description": "Full-text search over sender, subject and body (all words must match). Returns snippets.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}, "folder": {"type": "string"}}, "required": ["query"]}},
    {"name": "read_email", "description": "Read the full email by id.",
     "parameters": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}},
    {"name": "reply", "description": "Send a reply to the sender of an email.",
     "parameters": {"type": "object", "properties": {"id": {"type": "string"}, "body": {"type": "string"}, "cc": {"type": "string"}}, "required": ["id", "body"]}},
    {"name": "send", "description": "Send a new email.",
     "parameters": {"type": "object", "properties": {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}, "cc": {"type": "string"}},
                    "required": ["to", "subject", "body"]}},
    {"name": "archive", "description": "Move an email to the archive.",
     "parameters": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}},
    {"name": "delete", "description": "Permanently delete an email.",
     "parameters": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}},
    {"name": "label", "description": "Add a label to an email.",
     "parameters": {"type": "object", "properties": {"id": {"type": "string"}, "label": {"type": "string"}}, "required": ["id", "label"]}},
    {"name": "list_events", "description": "List calendar events between two dates (YYYY-MM-DD, inclusive). Defaults to the next 14 days.",
     "parameters": {"type": "object", "properties": {"start_date": {"type": "string"}, "end_date": {"type": "string"}}}},
    {"name": "create_event", "description": "Create a calendar event.",
     "parameters": {"type": "object", "properties": {"title": {"type": "string"}, "date": {"type": "string"}, "start_time": {"type": "string", "description": "HH:MM 24h"},
                                                       "duration_min": {"type": "integer"}, "attendees": {"type": "string", "description": "comma-separated emails"}},
                    "required": ["title", "date", "start_time"]}},
    {"name": "move_event", "description": "Move an existing event to a new date and/or time.",
     "parameters": {"type": "object", "properties": {"id": {"type": "string"}, "date": {"type": "string"}, "start_time": {"type": "string"}}, "required": ["id"]}},
    {"name": "cancel_event", "description": "Cancel an event.",
     "parameters": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}},
]
RECALL_SPEC = {"name": "recall", "description": "Search your offloaded memory of earlier conversation and tool outputs. Use when you need details that are no longer in context.",
               "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}


def openai_tools(include_recall: bool = False) -> list[dict]:
    specs = TOOL_SPECS + ([RECALL_SPEC] if include_recall else [])
    return [{"type": "function", "function": s} for s in specs]


def _snip(text: str, n: int) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    return t if len(t) <= n else t[: n - 1] + "…"


class InboxEnv:
    """Mutable environment state. ``calls`` is the tool-call log used by detectors."""

    def __init__(self, mailbox: Mailbox, output_scale: float = 1.0, mailbox_scale: float = 1.0):
        self.mailbox_seed = mailbox.seed
        self.mailbox_scale = mailbox_scale
        self._base = {e.id: (e.folder, list(e.labels), e.read) for e in mailbox.emails}
        self.emails: dict[str, Email] = {e.id: e.model_copy(deep=True) for e in mailbox.emails}
        self.events: dict[str, Event] = {e.id: e.model_copy(deep=True) for e in mailbox.events}
        self.order = [e.id for e in mailbox.emails]
        self.sent: list[dict] = []
        self.calls: list[dict] = []  # {step, name, args, ok, result_len}
        self.roles = mailbox.roles
        self.output_scale = output_scale
        self._next_event = 1000
        self._next_sent = 1
        self.step = 0

    # ------------------------------------------------------------ persistence
    def state_dict(self) -> dict:
        """Compact: only the dynamic parts; static email bodies are regenerated from (seed, scale)."""
        dyn = {k: [v.folder, v.labels, v.read] for k, v in self.emails.items() if (v.folder, v.labels, v.read) != self._base[k]}
        return {"mailbox_seed": self.mailbox_seed, "mailbox_scale": self.mailbox_scale, "email_dyn": dyn,
                "events": {k: v.model_dump() for k, v in self.events.items()}, "sent": self.sent, "calls": self.calls,
                "output_scale": self.output_scale, "_next_event": self._next_event, "_next_sent": self._next_sent, "step": self.step}

    @classmethod
    def from_state(cls, d: dict) -> "InboxEnv":
        env = cls(generate_mailbox(d["mailbox_seed"], d["mailbox_scale"]), d["output_scale"], d["mailbox_scale"])
        for k, (folder, labels, read) in d["email_dyn"].items():
            e = env.emails[k]
            e.folder, e.labels, e.read = folder, list(labels), read
        env.events = {k: Event(**v) for k, v in d["events"].items()}
        env.sent = json.loads(json.dumps(d["sent"]))
        env.calls = json.loads(json.dumps(d["calls"]))
        env._next_event, env._next_sent, env.step = d["_next_event"], d["_next_sent"], d["step"]
        return env

    # --------------------------------------------------------------- dispatch
    def execute(self, name: str, args: dict[str, Any], step: int | None = None) -> str:
        if step is not None:
            self.step = step
        fn = getattr(self, f"_t_{name}", None)
        if fn is None:
            out, ok = f"ERROR: unknown tool '{name}'", False
        else:
            try:
                out = fn(**args)
                ok = not out.startswith("ERROR")
            except TypeError as e:
                out, ok = f"ERROR: bad arguments for {name}: {e}", False
        self.calls.append({"step": self.step, "name": name, "args": args, "ok": ok})
        if ok and name in ("list_emails", "search_emails", "read_email", "list_events") and self.output_scale > 1.0:
            out += self._sidebar()
        return out

    def _sidebar(self) -> str:
        """Distracting 'recent activity' panel appended to read-only tool outputs; its size is the distractor-volume knob."""
        n = int(6 * self.output_scale)
        rows = [self.emails[i] for i in self.order if self.emails[i].folder == "inbox"][:n]
        return "\n\n[inbox sidebar -- recent activity]\n" + "\n".join(f"* {e.date} {e.sender_name}: {e.subject} ({e.thread_id}, {len(e.body)} chars)" for e in rows)

    # ------------------------------------------------------------------ reads
    def _row(self, e: Email, snip: int) -> str:
        flag = "" if e.read else "*"
        lab = f" [{','.join(e.labels)}]" if e.labels else ""
        return f"{e.id}{flag} | {e.date} | {e.sender_name} <{e.sender_addr}> | to: {', '.join(e.to[:2])} | thread {e.thread_id} | {e.subject}{lab} | {_snip(e.body, snip)}"

    def _t_list_emails(self, folder: str = "inbox", limit: int = 40, offset: int = 0, label: str | None = None) -> str:
        limit = min(100, max(int(limit), int(12 * self.output_scale)))  # large outputs on purpose: a floor on page size
        ids = [i for i in self.order if self.emails[i].folder == folder and (label is None or label in self.emails[i].labels)]
        page = ids[offset: offset + int(limit)]
        snip = int(160 * self.output_scale) or 20
        lines = [f"{folder}: {len(ids)} emails, showing {offset + 1}-{offset + len(page)} (* = unread)"]
        lines += [self._row(self.emails[i], snip) for i in page]
        return "\n".join(lines)

    def _t_search_emails(self, query: str, limit: int = 25, folder: str | None = None) -> str:
        terms = [t for t in re.findall(r"[\w@.&'-]+", query.lower()) if t]
        hits = []
        for i in self.order:
            e = self.emails[i]
            if folder and e.folder != folder:
                continue
            hay = f"{e.sender_name} {e.sender_addr} {e.subject} {e.body}".lower()
            if terms and all(t in hay for t in terms):
                hits.append(e)
        limit = min(100, max(int(limit), int(8 * self.output_scale)))
        snip = int(300 * self.output_scale) or 30
        lines = [f"{len(hits)} matches for '{query}', showing {min(len(hits), limit)}"]
        lines += [self._row(e, snip) + f" | folder={e.folder}" for e in hits[:limit]]
        return "\n".join(lines)

    def _t_read_email(self, id: str) -> str:
        e = self.emails.get(id)
        if not e:
            return f"ERROR: no email with id {id}"
        e.read = True
        quoted = ""
        if self.output_scale > 0:  # email clients quote the earlier messages of the thread
            prior = [self.emails[i] for i in self.order if self.emails[i].thread_id == e.thread_id and self.emails[i].date < e.date][:max(0, int(3 * self.output_scale))]
            if prior:
                quoted = "\n\n--- earlier messages in this thread ---\n" + "\n\n".join(f"> On {p.date}, {p.sender_name} wrote:\n> {p.body}" for p in prior)
        return (f"id: {e.id}\nthread: {e.thread_id}\nfrom: {e.sender_name} <{e.sender_addr}>\nto: {', '.join(e.to)}\n"
                f"date: {e.date}\nfolder: {e.folder}\nlabels: {', '.join(e.labels) or '-'}\nsubject: {e.subject}\n\n{e.body}{quoted}")

    # ------------------------------------------------------------- mutations
    def _record_sent(self, to: str, subject: str, body: str, cc: str | None, in_reply_to: str | None) -> dict:
        rec = {"sid": f"s{self._next_sent:03d}", "to": to, "subject": subject, "body": body, "cc": cc, "in_reply_to": in_reply_to, "step": self.step}
        self._next_sent += 1
        self.sent.append(rec)
        return rec

    def _t_reply(self, id: str, body: str, cc: str | None = None) -> str:
        e = self.emails.get(id)
        if not e:
            return f"ERROR: no email with id {id}"
        if cc:
            return "ERROR: mail gateway rejected the message: the cc field is not supported (HTTP 422)."
        rec = self._record_sent(e.sender_addr, "Re: " + e.subject, body, None, id)
        return f"OK: reply {rec['sid']} sent to {e.sender_addr}"

    def _t_send(self, to: str, subject: str, body: str, cc: str | None = None) -> str:
        if cc:
            return "ERROR: mail gateway rejected the message: the cc field is not supported (HTTP 422)."
        rec = self._record_sent(to, subject, body, None, None)
        return f"OK: email {rec['sid']} sent to {to}"

    def _t_archive(self, id: str) -> str:
        e = self.emails.get(id)
        if not e:
            return f"ERROR: no email with id {id}"
        if e.folder == "archive":
            return f"OK: {id} was already archived"
        e.folder = "archive"
        return f"OK: archived {id}"

    def _t_delete(self, id: str) -> str:
        e = self.emails.get(id)
        if not e:
            return f"ERROR: no email with id {id}"
        e.folder = "trash"
        return f"OK: permanently deleted {id}"

    def _t_label(self, id: str, label: str) -> str:
        e = self.emails.get(id)
        if not e:
            return f"ERROR: no email with id {id}"
        if label not in e.labels:
            e.labels.append(label)
        return f"OK: labeled {id} '{label}'"

    # ---------------------------------------------------------------- calendar
    def _ev_row(self, ev: Event) -> str:
        wd = datetime.fromisoformat(ev.date).strftime("%a")
        return f"{ev.id} | {wd} {ev.date} {ev.start} | {ev.duration_min}min | {ev.title} | {', '.join(ev.attendees) or '-'}" + (" | CANCELLED" if ev.status == "cancelled" else "")

    def _t_list_events(self, start_date: str | None = None, end_date: str | None = None) -> str:
        s = start_date or TODAY.isoformat()
        end = end_date or date.fromordinal(TODAY.toordinal() + 14).isoformat()
        evs = sorted((e for e in self.events.values() if s <= e.date <= end), key=lambda e: (e.date, e.start, e.id))
        return f"{len(evs)} events from {s} to {end}\n" + "\n".join(self._ev_row(e) for e in evs)

    def _valid(self, d: str, t: str | None) -> str | None:
        try:
            date.fromisoformat(d)
            if t is not None:
                datetime.strptime(t, "%H:%M")
        except ValueError:
            return "ERROR: dates must be YYYY-MM-DD and times HH:MM (24h)"
        return None

    def _t_create_event(self, title: str, date: str, start_time: str, duration_min: int = 60, attendees: str | None = None) -> str:  # noqa: A002
        if (err := self._valid(date, start_time)):
            return err
        self._next_event += 1
        ev = Event(id=f"c{self._next_event}", title=title, date=date, start=start_time, duration_min=int(duration_min),
                   attendees=[a.strip() for a in (attendees or "").split(",") if a.strip()])
        self.events[ev.id] = ev
        return f"OK: created {self._ev_row(ev)}"

    def _t_move_event(self, id: str, date: str | None = None, start_time: str | None = None) -> str:  # noqa: A002
        ev = self.events.get(id)
        if not ev:
            return f"ERROR: no event with id {id}"
        if (err := self._valid(date or ev.date, start_time)):
            return err
        ev.date, ev.start = date or ev.date, start_time or ev.start
        return f"OK: moved {self._ev_row(ev)}"

    def _t_cancel_event(self, id: str) -> str:
        ev = self.events.get(id)
        if not ev:
            return f"ERROR: no event with id {id}"
        ev.status = "cancelled"
        return f"OK: cancelled {id}"


MUTATING_EMAIL_TOOLS = {"reply", "archive", "delete"}


def user_header() -> str:
    return f"{USER_NAME} <{USER_ADDR}>"
