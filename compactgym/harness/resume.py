"""Turn grouping and the resume format: system prompt + pinned tier + summary + last k turns (CompactionRL)."""
from __future__ import annotations

from ..interfaces import Message

SUMMARY_PREFIX = "[Summary of the earlier conversation, written by a context compactor]\n"
PINNED_PREFIX = "[Pinned -- exact instructions and facts from the user. They remain in force; follow them.]\n"


def group_turns(messages: list[Message]) -> list[list[Message]]:
    """Groups = [user msgs]* + assistant msg + its tool msgs. A trailing group with no assistant message (an
    unanswered user message) is returned last so callers can always keep it."""
    groups: list[list[Message]] = []
    cur: list[Message] = []
    for m in messages:
        if m.role == "system":
            continue
        if m.role == "assistant":
            if cur and any(x.role in ("assistant", "tool") for x in cur):
                groups.append(cur)
                cur = []
            cur.append(m)
        elif m.role == "tool":
            cur.append(m)
        else:  # user
            if cur and any(x.role in ("assistant", "tool") for x in cur):
                groups.append(cur)
                cur = []
            cur.append(m)
    if cur:
        groups.append(cur)
    return groups


def split_for_resume(messages: list[Message], k: int) -> tuple[list[Message], list[Message], list[Message]]:
    """-> (system msgs, older msgs to compact, tail msgs kept verbatim: last k complete turns + any unanswered user msg)."""
    system = [m for m in messages if m.role == "system"]
    groups = group_turns(messages)
    trailing: list[Message] = []
    if groups and not any(x.role == "assistant" for x in groups[-1]):
        trailing = groups.pop()
    keep = groups[-k:] if k > 0 else []
    older = groups[: len(groups) - len(keep)]
    # a kept turn must not start with an orphan tool message
    return system, [m for g in older for m in g], [m for g in keep for m in g] + trailing


def pinned_message(pinned: list[str]) -> Message | None:
    if not pinned:
        return None
    return Message(role="user", content=PINNED_PREFIX + "\n".join(f"- {p}" for p in pinned), meta={"origin": "pinned"})


def summary_message(summary: str) -> Message:
    return Message(role="user", content=SUMMARY_PREFIX + summary.strip(), meta={"origin": "compaction_summary"})


def build_resume(messages: list[Message], summary: str | None, pinned: list[str], k: int, *, first_user: bool = False) -> list[Message]:
    system, older, tail = split_for_resume(messages, k)
    out = list(system)
    if first_user:
        fu = next((m for m in older + tail if m.role == "user" and not m.meta.get("origin")), None)
        if fu is not None and fu not in tail:
            out.append(fu)
    pm = pinned_message(pinned)
    if pm:
        out.append(pm)
    if summary:
        out.append(summary_message(summary))
    return out + tail


def strip_derived(messages: list[Message]) -> list[Message]:
    """Remove pinned/summary messages (they are re-derived by the harness)."""
    return [m for m in messages if m.meta.get("origin") not in ("pinned",)]


def transcript(messages: list[Message], max_tool_chars: int | None = None) -> str:
    """Plain-text transcript for compactor prompts."""
    lines = []
    for m in messages:
        if m.role == "system":
            continue
        if m.role == "assistant":
            body = m.content or ""
            for tc in m.tool_calls or []:
                body += f"\n[tool call] {tc['function']['name']}({tc['function']['arguments']})"
            lines.append(f"[assistant]: {body.strip()}")
        elif m.role == "tool":
            t = m.content or ""
            if max_tool_chars and len(t) > max_tool_chars:
                t = t[:max_tool_chars] + " …[truncated]"
            lines.append(f"[tool result {m.meta.get('name', '')}]: {t}")
        else:
            lines.append(f"[user]: {m.content}")
    return "\n\n".join(lines)


def ensure_pinned(messages: list[Message], pinned: list[str], enabled: bool = True) -> list[Message]:
    """(Re)place the pinned message right after the system message(s)."""
    out = [m for m in messages if m.meta.get("origin") != "pinned"]
    pm = pinned_message(pinned) if enabled else None
    if pm is None:
        return out
    i = max((j for j, m in enumerate(out) if m.role == "system"), default=-1) + 1
    return out[:i] + [pm] + out[i:]
