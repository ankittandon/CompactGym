"""Test doubles and a scenario builder that executes real tool calls against a real InboxEnv."""
from __future__ import annotations

import json

from compactgym.config import Config
from compactgym.env.inbox.data import generate_mailbox
from compactgym.env.inbox.tasks import TaskSpec, generate_task
from compactgym.env.inbox.tools import InboxEnv
from compactgym.harness.agent import EpisodeResult
from compactgym.interfaces import Message
from compactgym.llm import CallLogger, ChatResult, SpendTracker, Usage


class FakeLLM:
    """Deterministic stand-in for compactgym.llm.LLM. ``handler(kind, spec, messages, tools)`` returns text or ChatResult."""

    def __init__(self, handler=None, logprob_fn=None):
        self.handler = handler or (lambda *a, **k: "OK")
        self.logprob_fn = logprob_fn or (lambda toks: [-0.1] * len(toks))
        self.spend = SpendTracker(1e9)
        self.logger = CallLogger(None)
        self.calls: list[dict] = []

    async def chat(self, spec, messages, tools=None, *, tag="", session=None, max_tokens=None, temperature=None, json_mode=False):
        self.calls.append({"tag": tag, "messages": messages, "tools": tools, "temperature": temperature})
        out = self.handler(tag, spec, messages, tools)
        if isinstance(out, ChatResult):
            return out
        n_in = sum(len(str(m.get("content") or "")) // 4 for m in messages)
        return ChatResult(text=out, usage=Usage(prompt=n_in, cached=0, completion=max(1, len(out) // 4)), latency_s=0.01)

    async def prompt_logprobs(self, spec, token_ids, *, tag="logprobs"):
        self.calls.append({"tag": tag, "n": len(token_ids)})
        return self.logprob_fn(token_ids)

    async def aclose(self):
        pass


def task_for(kind: str, seed: int = 2, turns: int = 8, **kw):
    return generate_task(TaskSpec(seed=seed, n_user_turns=turns, n_facts=1, fact_kinds=[kind], **kw))


class Scenario:
    """Builds a coherent (full_log, env) pair by really executing tool calls."""

    def __init__(self, task):
        self.task = task
        s = task.spec
        self.env = InboxEnv(generate_mailbox(s.seed, s.mailbox_scale, s.buried), s.distractor_volume, s.mailbox_scale)
        self.log: list[Message] = [Message(role="system", content="sys")]
        self.step = 0
        self.turn = 0
        self.n = 0

    def user(self, text="hi", turn=None, approval=False):
        self.turn = self.turn if turn is None else turn
        self.log.append(Message(role="user", content=text, meta={"user_turn": self.turn, "approval": approval}))
        return self

    def call(self, name, **args):
        self.n += 1
        cid = f"c{self.n}"
        self.log.append(Message(role="assistant", content=None, tool_calls=[{"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}],
                                meta={"step": self.step, "user_turn": self.turn}))
        out = self.env.execute(name, args, step=self.step)
        self.log.append(Message(role="tool", content=out, tool_call_id=cid, meta={"name": name}))
        self.step += 1
        self.out = out
        return self

    def say(self, text):
        self.log.append(Message(role="assistant", content=text, meta={"step": self.step, "user_turn": self.turn}))
        self.step += 1
        return self

    def result(self, reasks=None) -> EpisodeResult:
        return EpisodeResult(episode_id="t", task_id=self.task.task_id, compactor="never", cfg={}, full_log=self.log, env_state=self.env.state_dict(),
                             user={"reasks": reasks or []})
