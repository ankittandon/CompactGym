"""Repeated-compaction test: replay a recorded full-context trajectory through a compactor at a small budget so it fires many times in a row;
measure after every event whether the planted facts are still present (or, for memory_offload, recoverable via recall)."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from ..compactors import make_compactor
from ..config import Config
from ..env.inbox.tasks import generate_task
from ..env.inbox.tools import openai_tools
from ..harness.agent import EpisodeResult
from ..harness.memory import MemoryStore
from ..harness.tokens import count_tokens
from ..interfaces import EpisodeState
from ..llm import CallLogger, LLM
from .common import jl_append, jl_read, pilot_spec, retention_for_context

BUDGET = 6000


async def replay(cfg: Config, llm: LLM, seed: int, name: str, root: Path, budget: int = BUDGET):
    ep_id = f"p{seed:03d}"
    res = EpisodeResult.load(root / "full" / f"{ep_id}.json.gz")
    task = generate_task(pilot_spec(seed))
    tools = openai_tools(True)
    comp = make_compactor(name, cfg, llm, threshold_frac=0.8) if name != "small_prompted" else make_compactor(name, cfg, llm, min_fill=0.8)
    log = res.full_log
    prefix = [log[0]]
    live = [log[0]]
    state = EpisodeState(keep_k=cfg.keep_k, memory=MemoryStore(), profile={"never_lose": ["all"]}, settings={"pinned_tier": cfg.pinned_tier}, full_log=prefix)
    events = 0
    out = []
    for m in log[1:]:
        live.append(m)
        prefix.append(m)
        boundary = m.role in ("tool", "user") or (m.role == "assistant" and not m.tool_calls)
        if not boundary:
            continue
        state.context_tokens = count_tokens(live, tools, cfg.prompt_overhead_tokens)
        if state.context_tokens < 0.8 * budget:
            continue
        r = await comp([x.model_copy() for x in live], state, budget)
        if r.action.value == "keep":
            continue
        live = r.messages
        state.n_compactions += 1
        for p in r.pinned:
            if cfg.pinned_tier and p not in state.pinned:
                state.pinned.append(p)
        events += 1
        ut = m.meta.get("user_turn", 0)
        if not any(f.inject_turn <= ut for f in task.facts):
            continue
        ret = await retention_for_context(llm, cfg, task, live, ut, state)
        for fid, v in ret.items():
            if v is not None:
                out.append({"seed": seed, "compactor": name, "event": events, "fact": fid, "type": task.fact(fid).type, "retained": bool(v), "error": 0.0 if v else 1.0,
                            "ctx_after": count_tokens(live, tools, cfg.prompt_overhead_tokens)})
    for r in out:
        jl_append(root / "repeated.jsonl", r)
    print(f"[repeated] {ep_id} {name}: {events} events, final err={1 - (sum(r['retained'] for r in out if r['event'] == events) / max(1, sum(1 for r in out if r['event'] == events))):.2f}", flush=True)


async def amain(a):
    root = Path(a.out)
    cfg = Config(budget_tokens=BUDGET, log_dir=str(root / "logs"))
    cfg.concurrency = 12
    llm = LLM(cfg, CallLogger(root / "logs" / "calls.jsonl"))
    done = {(r["seed"], r["compactor"]) for r in jl_read(root / "repeated.jsonl")}
    sem = asyncio.Semaphore(a.conc)

    async def go(s, n):
        async with sem:
            await replay(cfg, llm, s, n, root)

    await asyncio.gather(*[go(s, n) for s in range(a.n) for n in a.compactors if (s, n) not in done])
    print(f"[spend] ${llm.spend.total:.2f}")
    await llm.aclose()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/pilot")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--conc", type=int, default=6)
    ap.add_argument("--compactors", nargs="+", default=["threshold_pi", "cliff", "memory_offload"])
    asyncio.run(amain(ap.parse_args()))
