"""Experiment matrix: scenario x task x budget x branch point x compactor x seed, run concurrently with retries and spend tracking.

    uv run python -m compactgym.runner.matrix --compactors cliff threshold_pi --budgets 8000 16000 32000 --seeds 100 101 102 --out outputs/matrix
Rows go to results.jsonl as they finish (re-runs skip finished jobs) and are consolidated into results.parquet.
A job is (task seed, injection scenario, budget, compactor, run seed[, branch point]); with ``--branch-from DIR`` jobs resume
from a recorded full-context run's snapshot instead of starting from scratch.
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import time
from pathlib import Path

import pandas as pd

from ..compactors import make_compactor
from ..config import Config
from ..env.inbox.tasks import generate_task
from ..harness.agent import Episode, branch
from ..harness.snapshots import SnapshotStore
from ..llm import CallLogger, LLM
from .common import jl_append, jl_read, pilot_spec, score_row

ALL = ["never", "threshold_pi", "threshold_anthropic", "recent_k", "cliff", "constraint_extractor", "memory_offload", "frontier", "small_prompted"]


def job_key(j: dict) -> str:
    return "|".join(str(j.get(k)) for k in ("task_seed", "budget", "compactor", "run_seed", "injection", "branch_ep", "branch_step", "background", "pinned_tier"))


async def run_job(cfg0: Config, llm: LLM, j: dict, root: Path, sem: asyncio.Semaphore, branch_from: Path | None):
    cfg = cfg0.model_copy(update={"budget_tokens": j["budget"], "pinned_tier": j.get("pinned_tier", True)})
    spec = pilot_spec(j["task_seed"])
    if j.get("injection"):
        spec = spec.model_copy(update={"injection": j["injection"]})
    task = generate_task(spec)
    comp = make_compactor(j["compactor"], cfg, llm)
    async with sem:
        t0 = time.time()
        eid = f"m{j['task_seed']}-{j['compactor']}-{j['budget']}-{j['run_seed']}"
        if branch_from is not None and j.get("branch_ep"):
            res = await branch(SnapshotStore(branch_from / "snapshots"), j["branch_ep"], j["branch_step"], comp, cfg, task, llm, new_episode_id=eid,
                               user_mode="llm", budget_tokens=j["budget"], seed=j["run_seed"])
        else:
            ep = Episode(cfg, task, comp, llm, episode_id=eid, user_mode="llm", judge_reasks=False, background=bool(j.get("background")), budget_tokens=j["budget"], seed=j["run_seed"])
            res = await ep.run()
        if res.status == "error":
            print(f"[matrix] {eid} FAILED {res.error}", flush=True)
            return
        res.compactor = j["compactor"]
        row = await score_row(cfg, llm, task, res, branch_turn=j.get("branch_step"))
        row.update({**j, "injection": spec.injection, "wall_s": time.time() - t0, "n_steps": len(res.steps)})
        jl_append(root / "results.jsonl", row)
        print(f"[matrix] {eid:34s} success={row['success']:.2f} viol={row['n_violations']} cost=${row['cost']:.3f} comp={row['n_compactions']} forced={row['n_forced']} spend=${llm.spend.total:.2f}", flush=True)


def consolidate(root: Path) -> pd.DataFrame:
    df = pd.DataFrame(jl_read(root / "results.jsonl"))
    if len(df):
        df.to_parquet(root / "results.parquet")
    return df


async def amain(a):
    root = Path(a.out)
    cfg = Config(log_dir=str(root / "logs"), max_agent_steps=a.max_steps, max_spend_usd=a.max_spend, concurrency=a.http_conc)
    llm = LLM(cfg, CallLogger(root / "logs" / "calls.jsonl"))
    jobs = [dict(task_seed=s, budget=b, compactor=c, run_seed=r, injection=None, pinned_tier=pt, background=bg)
            for s, b, c, r, pt, bg in itertools.product(a.seeds, a.budgets, a.compactors, range(a.reps), [not a.no_pinned_tier], [a.background])]
    done = {job_key(r) for r in jl_read(root / "results.jsonl")}
    todo = [j for j in jobs if job_key(j) not in done]
    print(f"[matrix] {len(todo)} jobs ({len(jobs) - len(todo)} already done)")
    sem = asyncio.Semaphore(a.conc)
    try:
        await asyncio.gather(*[run_job(cfg, llm, j, root, sem, None) for j in todo])
    finally:
        consolidate(root)
        print(f"[spend] ${llm.spend.total:.2f}")
        await llm.aclose()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/matrix")
    ap.add_argument("--compactors", nargs="+", default=ALL)
    ap.add_argument("--budgets", nargs="+", type=int, default=[8000, 16000, 32000])
    ap.add_argument("--seeds", nargs="+", type=int, default=[100, 101, 102])
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--conc", type=int, default=4)
    ap.add_argument("--http-conc", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=140)
    ap.add_argument("--max-spend", type=float, default=25.0)
    ap.add_argument("--no-pinned-tier", action="store_true")
    ap.add_argument("--background", action="store_true", help="run compaction in the background (applied next turn, blocking=False)")
    asyncio.run(amain(ap.parse_args(argv)))


if __name__ == "__main__":
    main()
