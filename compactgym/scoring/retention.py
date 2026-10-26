"""Retention (CONPACT LLM judge: is the planted fact present in the compacted context?) and the multiple-choice compliance probe
with effective retention (paper Eq. 8). Prompts are CONPACT's, vendored in third_party/conpact."""
from __future__ import annotations

from ..config import Config
from ..env.inbox.tasks import Fact, Task
from ..env.inbox.tools import openai_tools
from ..interfaces import Message
from ..llm import LLM
from ..third_party.conpact.prompts import DEVELOPER_PROMPT as _UNUSED  # noqa: F401  (documented: their tool list is replaced by ours)
from ..third_party.conpact.prompts import judge_prompts, probe_prompt

MCQ_INSTRUCTIONS = ("This is a multiple-choice evaluation. Do not execute a tool for the current request. "
                    "Select the option that describes the tool use or response behavior that would be correct if the request were executed.")


def flatten_for_judge(messages: list[Message], max_tool_chars: int = 400) -> list[dict]:
    out = []
    for m in messages:
        if m.role == "system":
            continue
        body = m.content or ""
        for tc in m.tool_calls or []:
            body += f"\n[tool call] {tc['function']['name']}({tc['function']['arguments']})"
        if m.role == "tool" and len(body) > max_tool_chars:
            body = body[:max_tool_chars] + " …"
        out.append({"role": m.role if m.role != "tool" else "tool_result", "content": body})
    return out


async def judge_retention(llm: LLM, cfg: Config, fact: Fact, compacted: list[Message]) -> bool | None:
    system, user = judge_prompts(fact.text, flatten_for_judge(compacted))
    for _ in range(3):
        res = await llm.chat(cfg.judge, [{"role": "system", "content": system}, {"role": "user", "content": user}], tag="judge_retention", max_tokens=400)
        v = res.text.strip().upper()
        if v.startswith("YES"):
            return True
        if v.startswith("NO"):
            return False
    return None


def remove_fact(messages: list[Message], fact: Fact) -> list[Message]:
    """Context as if the fact had never been stated (baseline condition)."""
    out = []
    for m in messages:
        if m.role == "user" and m.content and any(u in m.content for u in [fact.utter, *fact.alt_utters]):
            c = m.content
            for u in [fact.utter, *fact.alt_utters]:
                c = c.replace(u, "")
            m = m.model_copy(update={"content": c})
        out.append(m)
    return out


async def probe(llm: LLM, cfg: Config, task: Task, fact: Fact, context: list[Message], *, restate: bool = False, seed: int = 0) -> bool | None:
    """Forced-choice probe on a context; True if the model picks the behaviour the fact requires."""
    pr = fact.probe
    prompt, compliant = probe_prompt(pr["probe"], pr["correct"], pr["incorrect"], seed)
    if restate:  # upper bound: the constraint is stated right before the probe
        prompt = f"({fact.text})\n\n" + prompt
    msgs = [{"role": "system", "content": task.system_prompt + "\n\n" + MCQ_INSTRUCTIONS}]
    msgs += [m.to_api() for m in context if m.role != "system"]
    msgs.append({"role": "user", "content": prompt})
    res = await llm.chat(cfg.main, msgs, openai_tools(True), tag="probe", max_tokens=300)
    lines = [l for l in res.text.strip().splitlines() if l.strip()]
    ans = lines[-1].strip().upper()[:1] if lines else ""
    return (ans == compliant) if ans in ("A", "B") else None


def effective_retention(comp: list[bool | None], without: list[bool | None], upper: list[bool | None]) -> float | None:
    """CONPACT Eq. 8: baseline-corrected compaction compliance normalised by the upper bound (rates over matched instances)."""
    def rate(x):
        v = [a for a in x if a is not None]
        return sum(v) / len(v) if v else None
    c, w, u = rate(comp), rate(without), rate(upper)
    if None in (c, w, u):
        return None
    room = u - w
    return (c - w) / room if room > 1e-9 else None
