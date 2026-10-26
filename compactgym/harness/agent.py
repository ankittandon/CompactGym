"""Agent loop with a per-turn compaction hook, budget enforcement, background compaction, snapshots and branching."""
from __future__ import annotations

import asyncio
import copy
import gzip
import json
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..config import Config, load_profile
from ..env.inbox.data import generate_mailbox
from ..env.inbox.tasks import Task
from ..env.inbox.tools import InboxEnv, openai_tools
from ..env.inbox.user import SimUser
from ..llm import LLM
from ..interfaces import CompactionAction, CompactionResult, Compactor, EpisodeState, Message
from .memory import MemoryStore
from .resume import pinned_message
from .snapshots import Snapshot, SnapshotStore
from .tokens import count_tokens, message_tokens, msg_hash


class EpisodeResult(BaseModel):
    episode_id: str
    task_id: str
    compactor: str
    cfg: dict
    steps: list[dict] = Field(default_factory=list)
    full_log: list[Message] = Field(default_factory=list)
    final_messages: list[Message] = Field(default_factory=list)
    env_state: dict = Field(default_factory=dict)
    turn_final_text: dict[int, str] = Field(default_factory=dict)
    user: dict = Field(default_factory=dict)
    status: str = "ok"
    error: str | None = None
    wall_s: float = 0.0
    branched_from: tuple[str, int] | None = None
    pinned: list[str] = Field(default_factory=list)
    compaction_events: list[dict] = Field(default_factory=list)  # {step, user_turn, action, confidence, forced, messages(post), pinned}

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(p, "wt") as f:
            f.write(self.model_dump_json())

    @classmethod
    def load(cls, path: str | Path) -> "EpisodeResult":
        with gzip.open(path, "rt") as f:
            return cls.model_validate_json(f.read())


class Episode:
    def __init__(self, cfg: Config, task: Task, compactor: Compactor, llm: LLM, *, episode_id: str, user_mode: str = "scripted",
                 judge_reasks: bool = False, snapshots: SnapshotStore | None = None, background: bool = False,
                 budget_tokens: int | None = None, seed: int = 0):
        self.cfg, self.task, self.compactor, self.llm = cfg, task, compactor, llm
        self.episode_id = episode_id
        self.budget = budget_tokens if budget_tokens is not None else cfg.budget_tokens  # may be huge (full-context recording)
        self.background = background
        self.snapshots = snapshots
        self.profile = load_profile(cfg.profile)
        self.include_recall = bool(getattr(compactor, "uses_memory", False))
        self.tools = openai_tools(self.include_recall)
        self.env = InboxEnv(generate_mailbox(task.spec.seed, task.spec.mailbox_scale, task.spec.buried), task.spec.distractor_volume, task.spec.mailbox_scale)
        self.user = SimUser(task, cfg, llm, mode=user_mode, judge_reasks=judge_reasks, seed=seed)
        self.full_log: list[Message] = []
        self.state = EpisodeState(episode_id=episode_id, profile=self.profile, keep_k=cfg.keep_k, memory=MemoryStore(),
                                  settings={"pinned_tier": cfg.pinned_tier, "budget_tokens": self.budget}, full_log=self.full_log)
        self.history: list[Message] = []
        self.step = 0
        self.uid = 0
        self.user_turn = 0
        self.turn_final_text: dict[int, str] = {}
        self.steps: list[dict] = []
        self.compaction_events: list[dict] = []
        self.prev_hashes: list[str] = []
        self.pending: dict | None = None  # background compaction {task, n, uids, t0}
        self.branched_from: tuple[str, int] | None = None
        self.done = False
        self.status = "ok"
        self.error: str | None = None
        self._sys_overhead = None

    # --------------------------------------------------------------- helpers
    def _msg(self, role: str, content: str | None, **kw) -> Message:
        self.uid += 1
        meta = {"uid": self.uid, "step": self.step, "user_turn": self.user_turn, **kw.pop("meta", {})}
        return Message(role=role, content=content, meta=meta, **kw)

    def _append(self, m: Message) -> None:
        self.history.append(m)
        self.full_log.append(m)

    def ctx_tokens(self, msgs: list[Message] | None = None) -> int:
        return count_tokens(msgs if msgs is not None else self.history, self.tools, self.cfg.prompt_overhead_tokens)

    # ------------------------------------------------------------- snapshots
    def snapshot(self) -> Snapshot:
        st = self.state.model_dump(exclude={"memory", "full_log"})
        return Snapshot(
            episode_id=self.episode_id, turn=self.step, messages=[m.model_copy(deep=True) for m in self.history],
            env_state=self.env.state_dict(), rng_state=self.user.state_dict(),
            full_log=[m.model_copy(deep=True) for m in self.full_log], state=st, memory=self.state.memory.dump(),
            extras={"uid": self.uid, "user_turn": self.user_turn, "turn_final_text": self.turn_final_text, "prev_hashes": self.prev_hashes,
                    "n_steps_logged": len(self.steps), "steps": self.steps})

    @classmethod
    def from_snapshot(cls, snap: Snapshot, cfg: Config, task: Task, compactor: Compactor, llm: LLM, **kw) -> "Episode":
        ep = cls(cfg, task, compactor, llm, episode_id=kw.pop("episode_id", f"{snap.episode_id}@{snap.turn}"), **kw)
        ep.history = [m.model_copy(deep=True) for m in snap.messages]
        ep.full_log = [m.model_copy(deep=True) for m in snap.full_log]
        ep.env = InboxEnv.from_state(json.loads(json.dumps(snap.env_state)))
        ep.user.load_state(json.loads(json.dumps(snap.rng_state, default=list)))
        st = dict(snap.state)
        ep.state = EpisodeState(**{**st, "memory": MemoryStore.load(snap.memory), "episode_id": ep.episode_id,
                                   "settings": {**st.get("settings", {}), "pinned_tier": cfg.pinned_tier, "budget_tokens": ep.budget}, "full_log": ep.full_log})
        x = snap.extras
        ep.uid, ep.user_turn, ep.step = x["uid"], x["user_turn"], snap.turn
        ep.turn_final_text = {int(k): v for k, v in x["turn_final_text"].items()}
        ep.prev_hashes = list(x["prev_hashes"])
        ep.steps = []  # new branch logs only its own steps
        ep.branched_from = (snap.episode_id, snap.turn)
        ep.base_steps = x.get("steps", [])
        return ep

    # --------------------------------------------------------- compaction hook
    async def _apply(self, result: CompactionResult, rec: dict, pause: float, *, base_n: int | None = None) -> None:
        new_msgs = [m for m in result.messages]
        if base_n is not None:  # background result computed on history[:base_n]; keep what arrived since
            new_msgs = new_msgs + self.history[base_n:]
        self.history = new_msgs
        if result.summary is not None:
            self.state.summary = result.summary
        if self.cfg.pinned_tier:
            for p in result.pinned:
                if p and p not in self.state.pinned:
                    self.state.pinned.append(p)
        if result.action != CompactionAction.KEEP:
            self.state.n_compactions += 1

    async def _forced_fallback(self, rec: dict) -> None:
        from ..compactors.cliff import fallback_compact
        t0 = time.perf_counter()
        before = self.ctx_tokens()
        self.history = fallback_compact(self.history, self.budget, self.tools, self.cfg.prompt_overhead_tokens, self.state)
        self.state.n_compactions += 1
        rec["forced"] = True
        rec["forced_tokens_removed"] = before - self.ctx_tokens()
        rec["forced_latency_s"] = time.perf_counter() - t0

    async def _run_compactor(self, msgs: list[Message], state: EpisodeState) -> CompactionResult:
        t0 = time.perf_counter()
        res = await self.compactor(msgs, state, self.budget)
        if not res.compactor_latency_s:
            res.compactor_latency_s = time.perf_counter() - t0
        return res

    async def _compaction_phase(self, rec: dict) -> None:
        self.state.turn, self.state.context_tokens = self.step, self.ctx_tokens()
        rec["ctx_before"] = self.state.context_tokens
        if self.background:
            # 1) apply a finished background result if still valid
            if self.pending and self.pending["task"].done():
                p, self.pending = self.pending, None
                try:
                    res: CompactionResult = p["task"].result()
                    n = p["n"]
                    cur_uids = [m.meta.get("uid") for m in self.history[:n]]
                    if cur_uids != p["uids"] or res.action == CompactionAction.KEEP:
                        rec["bg_stale"] = cur_uids != p["uids"]
                        rec["calls_wasted"] = res.calls
                    else:
                        res = res.model_copy(update={"blocking": False})
                        await self._apply(res, rec, 0.0, base_n=n)
                        self._log_result(rec, res, pause=0.0)
                except Exception as e:  # compactor failure never kills the episode
                    rec["compactor_error"] = repr(e)
            # 2) launch a new background compaction on a copy of the current history
            if self.pending is None:
                msgs = [m.model_copy(deep=True) for m in self.history]
                st = self.state.model_copy(update={"memory": self.state.memory, "pinned": list(self.state.pinned)})
                self.pending = {"task": asyncio.create_task(self._run_compactor(msgs, st)), "n": len(msgs), "uids": [m.meta.get("uid") for m in msgs]}
        else:
            msgs = [m.model_copy(deep=True) for m in self.history]
            t0 = time.perf_counter()
            try:
                res = await self._run_compactor(msgs, self.state)
                pause = time.perf_counter() - t0 if res.blocking else 0.0
                await self._apply(res, rec, pause)
                self._log_result(rec, res, pause=pause)
            except Exception as e:
                rec["compactor_error"] = repr(e)
        # budget enforcement: KEEP (or a too-weak compaction) exceeding the budget forces a fallback
        if self.ctx_tokens() > self.budget:
            await self._forced_fallback(rec)
        rec["ctx_after"] = self.ctx_tokens()
        if (rec.get("action") not in (None, "keep")) or rec.get("forced"):
            self.compaction_events.append({"step": self.step, "user_turn": self.user_turn, "action": rec.get("action") or "forced", "confidence": rec.get("confidence"),
                                           "forced": bool(rec.get("forced")), "ctx_before": rec["ctx_before"], "ctx_after": rec["ctx_after"],
                                           "pinned": list(self.state.pinned), "messages": [m.model_copy(deep=True) for m in self.history]})

    def _log_result(self, rec: dict, res: CompactionResult, pause: float) -> None:
        rec.update(action=res.action.value, confidence=res.confidence, blocking=res.blocking, pause_s=pause,
                   compactor_latency_s=res.compactor_latency_s, compactor_tokens_in=res.compactor_tokens_in,
                   compactor_tokens_out=res.compactor_tokens_out, compactor_model=res.compactor_model,
                   summary_chars=len(res.summary or ""), pinned_n=len(res.pinned), calls=res.calls or
                   ([{"role": res.compactor_model, "tokens_in": res.compactor_tokens_in, "tokens_out": res.compactor_tokens_out, "cached": 0}] if res.compactor_tokens_in else []))

    # ------------------------------------------------------------- cache model
    def _modeled_cached(self, msgs: list[Message], total: int) -> int:
        hs = [msg_hash(m) for m in msgs]
        n = 0
        for a, b in zip(self.prev_hashes, hs):
            if a != b:
                break
            n += 1
        if self._sys_overhead is None:
            self._sys_overhead = count_tokens([m for m in msgs if m.role == "system"], self.tools)
        est = [message_tokens(m) for m in msgs if m.role != "system"]
        non_sys = [m for m in msgs if m.role != "system"]
        sys_n = len(msgs) - len(non_sys)
        matched_est = sum(message_tokens(m) for m in msgs[sys_n:n]) if n > sys_n else 0
        ratio = max(0.1, (total - self._sys_overhead) / max(1, sum(est)))
        return min(total, int((self._sys_overhead if n >= sys_n else 0) + matched_est * ratio)) if n > 0 else 0

    # ----------------------------------------------------------------- tools
    def _run_tool(self, tc: dict) -> str:
        fn = tc.get("function", {})
        name = fn.get("name", "")
        raw = fn.get("arguments", "")
        try:
            args = json.loads(raw) if isinstance(raw, str) and raw.strip() else (raw if isinstance(raw, dict) else {})
            if not isinstance(args, dict):
                raise ValueError("arguments must be an object")
        except Exception as e:
            self.env.calls.append({"step": self.step, "name": name, "args": {"_raw": str(raw)[:200]}, "ok": False})
            return f"ERROR: invalid JSON arguments ({e})"
        if name == "recall":
            self.env.calls.append({"step": self.step, "name": "recall", "args": args, "ok": True})
            return self.state.memory.recall(str(args.get("query", "")))
        return self.env.execute(name, args, step=self.step)

    # ------------------------------------------------------------------ main
    async def run(self, max_steps: int | None = None, stop_at_step: int | None = None) -> EpisodeResult:
        t_start = time.perf_counter()
        max_steps = max_steps or self.cfg.max_agent_steps
        try:
            if not self.history:
                self.history.append(Message(role="system", content=self.task.system_prompt, meta={"uid": 0, "origin": "system"}))
                self.full_log.append(self.history[0])
                text, meta = await self.user.first_message()
                self.user_turn = 0
                self._inject_oracle_pins(meta)
                self._append(self._msg("user", text, meta=meta))
            while not self.done and self.step < max_steps:
                if stop_at_step is not None and self.step >= stop_at_step:
                    break
                if self.snapshots is not None:
                    self.snapshots.save(self.snapshot())
                await self._one_step()
        except Exception as e:  # noqa: BLE001
            self.status, self.error = "error", repr(e)
        finally:
            if self.pending:
                self.pending["task"].cancel()
                self.pending = None
        if not self.done and self.status == "ok" and self.step >= max_steps:
            self.status = "max_steps"
        return EpisodeResult(episode_id=self.episode_id, task_id=self.task.task_id, compactor=getattr(self.compactor, "name", "?"),
                             cfg=self.cfg.model_dump(), steps=self.steps, full_log=self.full_log, final_messages=self.history,
                             env_state=self.env.state_dict(), turn_final_text=self.turn_final_text,
                             user={**self.user.state_dict(), "rng": None}, status=self.status, error=self.error,
                             wall_s=time.perf_counter() - t_start, branched_from=self.branched_from, pinned=list(self.state.pinned),
                             compaction_events=self.compaction_events)

    def _inject_oracle_pins(self, meta: dict) -> None:
        if not self.cfg.pin_oracle or not self.cfg.pinned_tier:
            return
        for fid in meta.get("fact_ids", []):
            f = self.task.fact(fid)
            if f.text not in self.state.pinned:
                self.state.pinned.append(f.text)
        self.history = [m for m in self.history if m.meta.get("origin") != "pinned"]
        pm = pinned_message(self.state.pinned)
        if pm:
            pm.meta["uid"] = -1
            idx = max((i for i, m in enumerate(self.history) if m.role == "system"), default=-1) + 1
            self.history.insert(idx, pm)

    async def _one_step(self) -> None:
        rec: dict[str, Any] = {"step": self.step, "user_turn": self.user_turn, "episode_id": self.episode_id}
        t0 = time.perf_counter()
        await self._compaction_phase(rec)
        msgs = self.history
        total = self.ctx_tokens(msgs)
        rec["prompt_tokens_est"] = total
        rec["cached_modeled"] = self._modeled_cached(msgs, total)
        res = await self.llm.chat(self.cfg.main, [m.to_api() for m in msgs], self.tools, tag=f"main:{self.episode_id}", session=self.episode_id)
        self.prev_hashes = [msg_hash(m) for m in msgs]
        rec.update(prompt_tokens_api=res.usage.prompt, cached_api=res.usage.cached, completion_tokens=res.usage.completion,
                   main_latency_s=res.latency_s, finish_reason=res.finish_reason, n_tool_calls=len(res.tool_calls))
        tcs = [{"id": tc.get("id") or f"call_{self.step}_{i}", "type": "function",
                "function": {"name": tc["function"]["name"], "arguments": tc["function"].get("arguments") or "{}"}} for i, tc in enumerate(res.tool_calls)]
        am = self._msg("assistant", res.text or None, tool_calls=tcs or None, meta={"reasoning": res.reasoning})
        self._append(am)
        if tcs:
            for tc in tcs:
                out = self._run_tool(tc)
                self._append(self._msg("tool", out, tool_call_id=tc["id"], meta={"name": tc["function"]["name"]}))
            rec["tools"] = [tc["function"]["name"] for tc in tcs]
        else:
            self.turn_final_text[self.user_turn] = res.text
            nxt = await self.user.respond(res.text, self.step)
            if nxt is None:
                self.done = True
            else:
                text, meta = nxt
                if "user_turn" in meta:
                    self.user_turn = meta["user_turn"]
                self._inject_oracle_pins(meta)
                self._append(self._msg("user", text, meta=meta))
        rec["step_wall_s"] = time.perf_counter() - t0
        self.steps.append(rec)
        self.step += 1


async def branch(snapshots: SnapshotStore, episode_id: str, turn: int, compactor: Compactor, cfg: Config, task: Task, llm: LLM, *,
                 max_steps: int | None = None, new_episode_id: str | None = None, **kw) -> EpisodeResult:
    """Continue from a snapshot with a different compactor and the same seeds."""
    snap = snapshots.load(episode_id, turn)
    ep = Episode.from_snapshot(snap, cfg, task, compactor, llm, episode_id=new_episode_id or f"{episode_id}@{turn}", **kw)
    return await ep.run(max_steps=max_steps)
