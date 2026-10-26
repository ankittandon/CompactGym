"""Simulated user: an LLM persona driven by the task script (or purely scripted for tests).

* drips planted facts at their scripted turns, never restates them otherwise
* approves when the agent asks for a go-ahead (``meta.approval``)
* records re-asks (keyword match, optional LLM judge for borderline) and reacts
* LLM rewrites are rejected (fallback to canonical text) if they leak an un-scheduled fact or drop key details
"""
from __future__ import annotations

import random
import re
from typing import Any

from ...config import Config, ModelSpec
from ...llm import LLM
from .tasks import Task

CONFIRM_RE = re.compile(r"(go-?ahead|go ahead|confirm|proceed|shall i|should i|would you like me to|want me to|ok to|okay to|approve|approval|is that (ok|okay|right|correct)|before i (send|archive|delete|schedule|create|cancel|move|reply))", re.I)
PERSONA = ("You are Sam Okafor, a busy operations lead. You write short, casual messages to your email assistant. "
           "Rewrite the message below in your own casual voice. Keep EVERY concrete detail (names, dates, numbers, quoted phrases, ids) exactly. "
           "Do not add any new instructions, preferences, constraints or facts. Do not drop anything. Output only the rewritten message.")


def _must_keep(text: str) -> list[str]:
    toks = re.findall(r"\b\d[\w:.-]*\b|\"[^\"]+\"|'[^']+'|\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b", text)
    return [t.strip("\"'") for t in toks if len(t) > 1]


class SimUser:
    def __init__(self, task: Task, cfg: Config, llm: LLM | None = None, mode: str = "scripted", judge_reasks: bool = False, seed: int = 0):
        self.task, self.cfg, self.llm, self.mode, self.judge_reasks = task, cfg, llm, mode, judge_reasks
        self.rng = random.Random(seed)
        self.cur = -1
        self.approvals_this_turn = 0
        self.nudges_this_turn = 0
        self.reasks: list[dict] = []
        self.leaks: list[dict] = []
        self.fallbacks = 0
        self.reask_counts: dict[str, int] = {}

    # ----------------------------------------------------------- persistence
    def state_dict(self) -> dict[str, Any]:
        return {"cur": self.cur, "approvals": self.approvals_this_turn, "nudges": self.nudges_this_turn, "reasks": self.reasks,
                "leaks": self.leaks, "fallbacks": self.fallbacks, "reask_counts": self.reask_counts, "rng": self.rng.getstate()}

    def load_state(self, d: dict[str, Any]) -> None:
        self.cur, self.approvals_this_turn, self.nudges_this_turn = d["cur"], d["approvals"], d["nudges"]
        self.reasks, self.leaks, self.fallbacks, self.reask_counts = d["reasks"], d["leaks"], d["fallbacks"], d["reask_counts"]
        st = d["rng"]
        self.rng.setstate((st[0], tuple(st[1]), st[2]))

    # ------------------------------------------------------------------ script
    def active_facts(self, upto_turn: int | None = None):
        t = self.cur if upto_turn is None else upto_turn
        return [f for f in self.task.facts if f.inject_turn <= t]

    async def _render(self, turn_idx: int) -> tuple[str, dict]:
        ut = self.task.user_turns[turn_idx]
        meta = {"user_turn": turn_idx, "script": True, "fact_ids": ut.fact_ids, "goal_ids": ut.goal_ids}
        if self.mode != "llm" or self.llm is None:
            return ut.text, meta
        try:
            res = await self.llm.chat(self.cfg.user_sim, [{"role": "system", "content": PERSONA}, {"role": "user", "content": ut.text}], tag="user_sim", max_tokens=500)
            out = res.text.strip()
        except Exception:
            out = ""
        low = out.lower()
        scheduled = set(ut.fact_ids)
        leaked = [f.id for f in self.task.facts if f.id not in scheduled and any(k.lower() in low for k in f.leak_keywords if len(k) > 3)]
        missing = [k for k in _must_keep(ut.text) if k.lower() not in low]
        # allow the persona to paraphrase fact wording, but require key tokens (dates, names, quoted phrases) of goals
        if not out or leaked or len(missing) > 0:
            self.fallbacks += 1
            if leaked:
                self.leaks.append({"turn": turn_idx, "facts": leaked})
            meta["llm_fallback"] = True
            return ut.text, meta
        return out, meta

    async def first_message(self) -> tuple[str, dict]:
        self.cur = 0
        self.approvals_this_turn = self.nudges_this_turn = 0
        return await self._render(0)

    async def respond(self, agent_text: str, step: int = 0) -> tuple[str, dict] | None:
        """User reply to the agent's final text; None means the script is finished."""
        low = (agent_text or "").lower()
        asks = "?" in low or bool(CONFIRM_RE.search(low))
        # 1) re-ask of something already given
        if "?" in low:
            for f in self.active_facts():
                if any(k in low for k in f.reask_keywords):
                    n = self.reask_counts.get(f.id, 0)
                    self.reask_counts[f.id] = n + 1
                    self.reasks.append({"fact": f.id, "step": step, "turn": self.cur, "via": "keyword"})
                    if n == 0 or not self.task.spec.restate_on_reask:
                        return "I already told you that earlier -- please check back and carry on.", {"user_turn": self.cur, "reask_reaction": f.id}
                    return f"Again: {f.text}", {"user_turn": self.cur, "reask_reaction": f.id, "restated": f.id}
            g = self.task.goal(self.task.user_turns[self.cur].goal_ids[0]) if self.task.user_turns[self.cur].goal_ids else None
            if g and any(k in low for k in g.reask_keywords):
                self.reasks.append({"fact": None, "goal": g.id, "step": step, "turn": self.cur, "via": "keyword"})
                return "Your call -- use your best judgment and go ahead.", {"user_turn": self.cur, "approval": True}
        # 2) go-ahead request
        if asks and CONFIRM_RE.search(low) and self.approvals_this_turn < 3:
            self.approvals_this_turn += 1
            return "Yes, go ahead -- just with what you described for this request, nothing extra.", {"user_turn": self.cur, "approval": True}
        # 3) other question: judge borderline re-asks, else let the agent decide
        if "?" in low and self.nudges_this_turn < 2:
            self.nudges_this_turn += 1
            if self.judge_reasks and self.llm is not None:
                if await self._judge_reask(agent_text):
                    self.reasks.append({"fact": None, "step": step, "turn": self.cur, "via": "judge"})
                    return "I already gave you that earlier -- check back.", {"user_turn": self.cur, "reask_reaction": "judge"}
            return "Use your best judgment and go ahead.", {"user_turn": self.cur, "approval": True}
        # 4) advance the script
        self.cur += 1
        if self.cur >= len(self.task.user_turns):
            return None
        self.approvals_this_turn = self.nudges_this_turn = 0
        return await self._render(self.cur)

    async def _judge_reask(self, agent_text: str) -> bool:
        given = "\n".join(f"- {u.text}" for u in self.task.user_turns[: self.cur + 1])
        prompt = ("Below is what the user has told the assistant so far, then the assistant's latest message.\n"
                  f"[User said]\n{given}\n\n[Assistant message]\n{agent_text}\n\n"
                  "Does the assistant ask the user for information that the user ALREADY provided above? Answer YES or NO only.")
        try:
            res = await self.llm.chat(self.cfg.judge, [{"role": "user", "content": prompt}], tag="judge_reask", max_tokens=200)
            return res.text.strip().upper().startswith("YES")
        except Exception:
            return False
