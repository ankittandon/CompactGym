"""Shared helpers for the pilot and the experiment matrix."""
from __future__ import annotations

import os
import asyncio
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from ..compactors import make_compactor
from ..compactors.base import BaseCompactor
from ..config import Config, load_profile
from ..env.inbox.tasks import CONPACT_TYPES, FACT_TYPES, Task, TaskSpec, generate_task
from ..harness.agent import Episode, EpisodeResult
from ..harness.memory import MemoryStore
from ..harness.snapshots import Snapshot, SnapshotStore
from ..interfaces import CompactionAction, CompactionResult, EpisodeState, Message
from ..llm import LLM
from ..scoring.calibration import reliability
from ..scoring.cost import episode_cost
from ..scoring.forgetting import TYPES, forgetting_summary
from ..scoring.retention import judge_retention
from ..scoring.reward import reward
from ..scoring.success import success_summary

INJECTIONS = ["top", "middle", "repeated", "top", "bottom", "middle"]


def hard_spec(seed: int, n_turns: int = 28, n_facts: int = 4, volume: float = 4.0, buried: bool = False) -> TaskSpec:
    """Hard mode: facts planted early, exercised only in the last 40% of the session, unrecoverable once forgotten, and gating success."""
    rng = random.Random(10_000 + seed)
    types = FACT_TYPES[:]
    rng.shuffle(types)
    return TaskSpec(seed=seed, n_user_turns=n_turns, n_facts=n_facts, fact_types=types[: max(n_facts, 5)], injection="early", exercise_after=0.6,
                    restate_on_reask=False, gate_success=True, buried=buried, strict=bool(seed % 2), direct=bool((seed // 2) % 3 != 0), distractor_volume=volume)


def pilot_spec(seed: int, n_turns: int = 28, n_facts: int = 4, volume: float = 4.0) -> TaskSpec:
    if os.environ.get("COMPACTGYM_SPEC") == "hard_buried":  # hard mode + 2 facts that live only in long email bodies
        return hard_spec(seed, n_turns, n_facts, volume, buried=True)
    if os.environ.get("COMPACTGYM_SPEC") == "hard":  # COMPACTGYM_SPEC=hard switches every pilot stage to hard mode
        return hard_spec(seed, n_turns, n_facts, volume)
    rng = random.Random(10_000 + seed)
    types = FACT_TYPES[:]
    rng.shuffle(types)
    return TaskSpec(seed=seed, n_user_turns=n_turns, n_facts=n_facts, fact_types=types[: max(n_facts, 5)], injection=INJECTIONS[seed % len(INJECTIONS)],
                    strict=bool(seed % 2), direct=bool((seed // 2) % 3 != 0), distractor_volume=volume)


class PrecomputedCompactor(BaseCompactor):
    """Applies a pre-computed compaction at its first call (e.g. the best-of-8 pick), then defers to a fallback compactor."""
    name = "best8_infoloss"
    uses_memory = True

    def __init__(self, cfg, llm, result: CompactionResult, then: BaseCompactor):
        super().__init__(cfg, llm)
        self.result, self.then, self.used = result, then, False

    async def __call__(self, messages, state, budget):
        if not self.used:
            self.used = True
            r = self.result.model_copy(deep=True)
            if r.pinned:
                state.pinned = list(dict.fromkeys(state.pinned + r.pinned))
            return r
        return await self.then(messages, state, budget)


def state_from_snapshot(snap: Snapshot, cfg: Config, budget: int, ctx_tokens: int, memory: bool = True) -> EpisodeState:
    st = dict(snap.state)
    return EpisodeState(**{**st, "memory": MemoryStore.load(snap.memory), "full_log": [m.model_copy(deep=True) for m in snap.full_log],
                           "context_tokens": ctx_tokens, "turn": snap.turn, "pinned": list(st.get("pinned", [])),
                           "settings": {**st.get("settings", {}), "pinned_tier": cfg.pinned_tier, "budget_tokens": budget}})


async def retention_for_context(llm: LLM, cfg: Config, task: Task, messages: list[Message], user_turn: int, state: EpisodeState | None = None) -> dict[str, bool | None]:
    """Judge every fact that has been injected by ``user_turn``. Memory-backed contexts also count facts recoverable via recall()."""
    facts = [f for f in task.facts if f.inject_turn <= user_turn]
    res = await asyncio.gather(*[judge_retention(llm, cfg, f, messages) for f in facts])
    out = {}
    for f, r in zip(facts, res):
        if not r and state is not None and state.memory is not None and len(state.memory):
            hit = state.memory.recall(f.probe["probe"], k=4)
            r = bool(r) or (f.utter[:40] in hit)
        out[f.id] = r
    return out


async def score_row(cfg: Config, llm: LLM, task: Task, result: EpisodeResult, *, branch_turn: int | None = None, with_retention: bool = True,
                    profile: str | None = None, baseline_facts: list[str] | None = None) -> dict:
    """One results row for an episode (or branch continuation)."""
    prof = load_profile(profile or cfg.profile)
    fs = forgetting_summary(task, result, prof.get("forgetting_weights"), min_step=branch_turn)
    base = set(baseline_facts or [])
    ss = success_summary(task, result)
    cs = episode_cost(result, cfg)
    row: dict[str, Any] = {
        "episode_id": result.episode_id, "task_id": result.task_id, "compactor": result.compactor, "status": result.status, "budget": cfg.budget_tokens,
        "success": ss["success"], "goals_complete": ss["goals_complete"], "n_goals": ss["n_goals"], "reached_turn": ss["reached_turn"],
        "n_violations": fs["n_violations"], "n_facts_violated": fs["n_facts_violated"], "facts_violated": fs["facts_violated"],
        "n_new_facts_violated": len(set(fs["facts_violated"]) - base), "n_reasks": fs["n_reasks"], "redone_reads": fs["redone_reads"], "redone_writes": fs["redone_writes"],
        "forgetting_score": fs["forgetting_score"], **{f"viol_{t}": fs["by_type"][t] for t in TYPES}, **cs,
        "n_forced": sum(1 for s in result.steps if s.get("forced")), "summary_chars": float(np.mean([s["summary_chars"] for s in result.steps if s.get("summary_chars")] or [0])),
        "mean_infoloss": None, "branch_turn": branch_turn,
    }
    ret_all, cal_c, cal_r = [], [], []
    by_type: dict[str, list] = {}
    if with_retention and result.compaction_events:
        ev_scores = await asyncio.gather(*[retention_for_context(llm, cfg, task, e["messages"], e["user_turn"]) for e in result.compaction_events[:6]])
        for e, sc in zip(result.compaction_events, ev_scores):
            vals = [v for v in sc.values() if v is not None]
            if vals:
                r = sum(vals) / len(vals)
                ret_all.append(r)
                if e.get("confidence") is not None:
                    cal_c.append(e["confidence"])
                    cal_r.append(r)
            for fid, v in sc.items():
                if v is not None:
                    by_type.setdefault(task.fact(fid).type, []).append(v)
    ev = result.compaction_events
    row["retained_ratio"] = float(np.mean([e["ctx_after"] / max(1, e["ctx_before"]) for e in ev])) if ev else 1.0
    row["retention"] = float(np.mean(ret_all)) if ret_all else None
    for t in TYPES:
        row[f"ret_{t}"] = float(np.mean(by_type[t])) if by_type.get(t) else None
    row["conf_ece"] = reliability(cal_c, cal_r)["ece"] if cal_c else None
    r = reward(row["success"], row["pause_s"], row["forgetting_score"], row["cost"], 0.0, prof, retention=row["retention"])
    row.update(r)
    return row


def jl_append(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec, default=float) + "\n")


def jl_read(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path)] if path.exists() else []
