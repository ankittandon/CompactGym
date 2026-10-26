"""Pilot (go/no-go): record full-context runs -> score compaction candidates at branch points -> resume branches live -> report.

    uv run python -m compactgym.runner.pilot record  --n 50
    uv run python -m compactgym.runner.pilot score   --n 50
    uv run python -m compactgym.runner.pilot resume  --n-resume 10
    uv run python -m compactgym.runner.pilot report
Every stage is resumable (results are appended to JSONL and skipped on re-run).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

from ..compactors import make_compactor
from ..config import Config
from ..env.inbox.tasks import CONPACT_TYPES, Task, generate_task
from ..env.inbox.tools import openai_tools
from ..harness.agent import Episode, EpisodeResult, branch
from ..harness.snapshots import SnapshotStore
from ..harness.tokens import count_tokens
from ..interfaces import CompactionResult
from ..llm import CallLogger, LLM, SpendLimitExceeded
from ..scoring.infoloss import continuation_nll, combine, split_continuation
from ..scoring.retention import probe, remove_fact
from ..scoring.forgetting import forgetting_summary
from .common import PrecomputedCompactor, jl_append, jl_read, pilot_spec, retention_for_context, score_row, state_from_snapshot

BUDGET = 16_000
K_CONT = 3  # recorded next-k turns for InfoLoss
CORE = ["small_prompted", "frontier", "threshold_pi", "cliff"]  # + best-of-8 (small samples, chosen by InfoLoss)
ALL_BASELINES = ["never", "threshold_pi", "threshold_anthropic", "threshold_codex", "recent_k", "cliff", "constraint_extractor", "memory_offload", "frontier", "small_prompted"]
N_SAMPLES = 8
SAMPLE_TEMP = 0.8


def paths(root: Path) -> dict[str, Path]:
    return {"full": root / "full", "snap": root / "snapshots", "full_rows": root / "full_rows.jsonl", "bps": root / "bps.json",
            "cands": root / "candidates.jsonl", "detail": root / "infoloss_detail.jsonl", "picks": root / "picks", "branches": root / "branches.jsonl",
            "episodes": root / "episodes", "spend": root / "spend.json"}


def mk_cfg(root: Path, **kw) -> Config:
    cfg = Config(budget_tokens=BUDGET, max_agent_steps=140, log_dir=str(root / "logs"), **kw)
    return cfg


def mk_compactor(name: str, cfg: Config, llm: LLM, **kw):
    """Compactor instance that compacts at the branch point regardless of fill (threshold 0)."""
    if name in ("small_prompted",):
        return make_compactor(name, cfg, llm, must_compact=True, **kw)
    if name == "never":
        return make_compactor(name, cfg, llm)  # compacts at >=100% (the branch context is over budget)
    return make_compactor(name, cfg, llm, threshold_frac=0.0, **kw)


# ----------------------------------------------------------------------------- 1. record
async def record_one(cfg: Config, llm: LLM, seed: int, P: dict[str, Path], sem: asyncio.Semaphore):
    ep_id = f"p{seed:03d}"
    out = P["full"] / f"{ep_id}.json.gz"
    if out.exists():
        return
    async with sem:
        task = generate_task(pilot_spec(seed))
        store = SnapshotStore(P["snap"])
        ep = Episode(cfg, task, make_compactor("never", cfg, llm), llm, episode_id=ep_id, user_mode="llm", judge_reasks=True, budget_tokens=10**9,
                     snapshots=store, seed=seed)
        t0 = time.time()
        res = await ep.run()
        if res.status == "error":  # do not persist failed runs: a re-run resumes them
            print(f"[record] {ep_id} FAILED: {res.error}", flush=True)
            return
        res.save(out)
        row = await score_row(cfg, llm, task, res, with_retention=False)
        row.update(seed=seed, wall_s=time.time() - t0, n_steps=len(res.steps), max_ctx=max((s["ctx_after"] for s in res.steps), default=0),
                   user_fallbacks=ep.user.fallbacks, user_leaks=len(ep.user.leaks), tool_errors=sum(1 for m in res.full_log if m.role == "tool" and (m.content or "").startswith("ERROR")),
                   bad_args=sum(1 for m in res.full_log if m.role == "tool" and (m.content or "").startswith("ERROR: invalid JSON")),
                   empty_replies=sum(1 for s in res.steps if s.get("finish_reason") != "tool_calls" and not res.turn_final_text.get(s["user_turn"])))
        jl_append(P["full_rows"], row)
        print(f"[record] {ep_id} {res.status} steps={len(res.steps)} maxctx={row['max_ctx']} success={row['success']:.2f} viol={row['n_violations']} {time.time() - t0:.0f}s spend=${llm.spend.total:.2f}", flush=True)


async def stage_record(cfg, llm, n, P, conc, start=0):
    sem = asyncio.Semaphore(conc)
    await asyncio.gather(*[record_one(cfg, llm, s, P, sem) for s in range(start, n)])


# ------------------------------------------------------------------------- branch points
def choose_bps(res: EpisodeResult, task: Task, n_bp: int = 4, budget: int = BUDGET) -> list[int]:
    first_inj = min(f.inject_turn for f in task.facts)
    elig = [s["step"] for s in res.steps if s["ctx_before"] >= budget and s["step"] + K_CONT + 1 < len(res.steps) and s["user_turn"] >= first_inj + 1
            and res.full_log[_log_index(res, s["step"])].role in ("user", "tool")]
    if not elig:
        return []
    if len(elig) <= n_bp:
        return elig
    idx = np.linspace(0, len(elig) - 1, n_bp).round().astype(int)
    return [elig[i] for i in sorted(set(idx))]


def _log_index(res: EpisodeResult, step: int) -> int:
    """Index in full_log where step ``step`` begins (== len(snapshot.full_log)) minus one, i.e. the last message before the call."""
    n = 0
    for i, m in enumerate(res.full_log):
        if m.role == "assistant" and m.meta.get("step") == step:
            return i - 1
        n = i
    return n


def log_len_at(res: EpisodeResult, step: int) -> int:
    for i, m in enumerate(res.full_log):
        if m.role == "assistant" and m.meta.get("step") == step:
            return i
    return len(res.full_log)


async def stage_select(cfg, n, P, n_bp, n_resume_bps):
    bps = {}
    for seed in range(n):
        ep_id = f"p{seed:03d}"
        f = P["full"] / f"{ep_id}.json.gz"
        if not f.exists():
            continue
        res = EpisodeResult.load(f)
        task = generate_task(pilot_spec(seed))
        b = choose_bps(res, task, n_bp)
        if b:
            bps[ep_id] = b
    # resume subset: the first eligible branch point of the first episodes with >=2 bps
    resume = []
    for ep_id, b in bps.items():
        if len(resume) < n_resume_bps:
            resume.append([ep_id, b[0]])  # earliest eligible: leaves the longest horizon to observe forgetting
    P["bps"].write_text(json.dumps({"bps": bps, "resume": resume}, indent=1))
    print(f"[select] {sum(len(v) for v in bps.values())} branch points in {len(bps)} episodes; {len(resume)} for live resume")
    return bps, resume


# ------------------------------------------------------------------------------ 2. score
def active_facts(task: Task, user_turn: int):
    return [f for f in task.facts if f.inject_turn <= user_turn]


async def score_bp(cfg: Config, llm: LLM, ep_id: str, step: int, P: dict[str, Path], names: list[str], sem: asyncio.Semaphore, probes: bool = True):
    seed = int(ep_id[1:])
    async with sem:
        task = generate_task(pilot_spec(seed))
        res = EpisodeResult.load(P["full"] / f"{ep_id}.json.gz")
        store = SnapshotStore(P["snap"])
        snap = store.load(ep_id, step)
        full_ctx = snap.messages
        tools = openai_tools(True)
        ctx_tokens = count_tokens(full_ctx, tools, cfg.prompt_overhead_tokens)
        cont = split_continuation(res.full_log, len(snap.full_log), K_CONT)
        user_turn = res.steps[step]["user_turn"]
        facts = active_facts(task, user_turn)
        base_nll = await continuation_nll(llm, cfg.main, full_ctx, cont, tools)
        pf = sorted(facts, key=lambda f: (f.type not in CONPACT_TYPES, f.id))[:2]
        without = {}
        if probes:
            vals = await asyncio.gather(*[probe(llm, cfg, task, f, remove_fact(full_ctx, f), seed=step) for f in pf])
            without = {f.id: v for f, v in zip(pf, vals)}

        async def one(name: str, sample_idx: int | None = None, is_best_pool: bool = False):
            st = state_from_snapshot(snap, cfg, BUDGET, ctx_tokens)
            if name == "small_sample":
                comp = make_compactor("small_prompted", cfg, llm, must_compact=True, temperature=SAMPLE_TEMP)
            else:
                comp = mk_compactor(name, cfg, llm)
            t0 = time.perf_counter()
            r = await comp([m.model_copy(deep=True) for m in full_ctx], st, BUDGET)
            lat = time.perf_counter() - t0
            il = combine(base_nll, await continuation_nll(llm, cfg.main, r.messages, cont, tools))
            ret = await retention_for_context(llm, cfg, task, r.messages, user_turn, st)
            vals = [v for v in ret.values() if v is not None]
            row = {"episode_id": ep_id, "step": step, "user_turn": user_turn, "ctx_tokens": ctx_tokens, "candidate": name, "sample": sample_idx,
                   "action": r.action.value, "confidence": r.confidence, "infoloss": il.infoloss, "infoloss_raw": il.raw, "nll_full": il.nll_full, "nll_comp": il.nll_comp,
                   "n_scored_tokens": il.n_tokens, "ctx_after": count_tokens(r.messages, tools, cfg.prompt_overhead_tokens), "summary_chars": len(r.summary or ""),
                   "pinned_n": len(r.pinned), "tokens_in": sum(c["tokens_in"] for c in r.calls) if r.calls else r.compactor_tokens_in,
                   "tokens_out": sum(c["tokens_out"] for c in r.calls) if r.calls else r.compactor_tokens_out, "latency_s": lat, "calls": r.calls,
                   "retention": float(np.mean(vals)) if vals else None, "ret_by_fact": ret, "n_facts_active": len(facts),
                   "ret_by_type": {t: float(np.mean([v for fid, v in ret.items() if task.fact(fid).type == t and v is not None])) for t in {task.fact(fid).type for fid in ret if ret[fid] is not None}}}
            return row, r, il, st

        out = []
        singles = await asyncio.gather(*[one(n) for n in names])
        samples = await asyncio.gather(*[one("small_sample", i) for i in range(N_SAMPLES)]) if "small_prompted" in names else []
        for row, r, il, st in singles:
            out.append((row, r, il))
        if samples:
            best = min(samples, key=lambda x: x[0]["infoloss"])
            for row, r, il, st in samples:
                row["is_pick"] = row is best[0]
                out.append((row, r, il))
            brow = dict(best[0], candidate="best8_infoloss", is_pick=True)
            out.append((brow, best[1], best[2]))
            (P["picks"]).mkdir(parents=True, exist_ok=True)
            (P["picks"] / f"{ep_id}_{step}.json").write_text(best[1].model_dump_json())
            # oracle upper bound: best sample by retention
            rbest = max(samples, key=lambda x: (x[0]["retention"] if x[0]["retention"] is not None else -1, -x[0]["infoloss"]))
            out.append((dict(rbest[0], candidate="best8_retention", is_pick=False), rbest[1], rbest[2]))
        # compliance probes (cheap complement to live detection) for the core candidates
        if probes:
            targets = {n: next(r for row, r, il in out if row["candidate"] == n) for n in CORE + ["best8_infoloss"] if any(row["candidate"] == n for row, _, _ in out)}
            async def pr(n, r):
                comp = await asyncio.gather(*[probe(llm, cfg, task, f, r.messages, seed=step) for f in pf])
                up = await asyncio.gather(*[probe(llm, cfg, task, f, r.messages, restate=True, seed=step) for f in pf])
                return n, {f.id: {"comp": c, "upper": u, "without": without.get(f.id)} for f, c, u in zip(pf, comp, up)}
            pres = dict(await asyncio.gather(*[pr(n, r) for n, r in targets.items()]))
            for row, r, il in out:
                if row["candidate"] in pres:
                    row["probes"] = pres[row["candidate"]]
        (P["picks"] / "all").mkdir(parents=True, exist_ok=True)  # every candidate's output, for SFT target selection (scripts/build_sft.py)
        for row, r, il in out:
            if row["candidate"] not in ("best8_infoloss", "best8_retention"):
                (P["picks"] / "all" / f"{ep_id}_{step}_{row['candidate']}_{row['sample'] if row['sample'] is not None else 'x'}.json").write_text(r.model_dump_json())
            jl_append(P["cands"], row)
            if row["candidate"] in CORE + ["best8_infoloss"]:
                top = sorted(il.per_token, key=lambda p: -p["delta"] * p["w"])[:25]
                jl_append(P["detail"], {"episode_id": ep_id, "step": step, "candidate": row["candidate"], "top": top})
        print(f"[score] {ep_id}@{step} ctx={ctx_tokens} " + " ".join(f"{r['candidate'][:8]}={r['infoloss']:.3f}/{r['retention'] if r['retention'] is None else round(r['retention'], 2)}" for r, _, _ in out if r.get("sample") is None)
              + f" spend=${llm.spend.total:.2f}", flush=True)


async def stage_score(cfg, llm, P, conc, limit=None, resume_only=False):
    meta = json.loads(P["bps"].read_text())
    done = {(r["episode_id"], r["step"]) for r in jl_read(P["cands"])}
    resume_set = {(e, s) for e, s in meta["resume"]}
    jobs = [(e, s) for e, ss in meta["bps"].items() for s in ss if (e, s) not in done]
    if resume_only:
        jobs = [j for j in jobs if j in resume_set]
    if limit:
        jobs = jobs[:limit]
    sem = asyncio.Semaphore(conc)
    async def safe(e, s):
        try:
            await score_bp(cfg, llm, e, s, P, ALL_BASELINES_NO_DUP if (e, s) in resume_set else CORE, sem)
        except SpendLimitExceeded:
            raise
        except Exception as ex:  # one failed branch point must not kill the stage; a re-run picks it up
            print(f"[score] {e}@{s} FAILED: {ex!r}"[:300], flush=True)

    await asyncio.gather(*[safe(e, s) for e, s in jobs])


ALL_BASELINES_NO_DUP = [n for n in ALL_BASELINES if n != "never"] + ["never"]


# ---------------------------------------------------------------------------- 3. resume
async def resume_one(cfg: Config, llm: LLM, ep_id: str, step: int, name: str, P: dict[str, Path], sem: asyncio.Semaphore):
    key = {"episode_id": ep_id, "step": step, "compactor": name}
    if any(r["episode_id"] == ep_id and r["step"] == step and r["compactor"] == name for r in jl_read(P["branches"])):
        return
    async with sem:
        seed = int(ep_id[1:])
        task = generate_task(pilot_spec(seed))
        store = SnapshotStore(P["snap"])
        if name == "best8_infoloss":
            pick = CompactionResult.model_validate_json((P["picks"] / f"{ep_id}_{step}.json").read_text())
            comp = PrecomputedCompactor(cfg, llm, pick, make_compactor("small_prompted", cfg, llm))
        elif name == "small_prompted":
            comp = make_compactor(name, cfg, llm)  # live policy: decides every turn once >=50% full
        else:
            comp = make_compactor(name, cfg, llm)
        t0 = time.time()
        res = await branch(store, ep_id, step, comp, cfg, task, llm, new_episode_id=f"{ep_id}@{step}:{name}", user_mode="llm", budget_tokens=BUDGET, seed=seed)
        res.compactor = name
        P["episodes"].mkdir(parents=True, exist_ok=True)
        res.save(P["episodes"] / f"{ep_id}_{step}_{name}.json.gz")
        full = EpisodeResult.load(P["full"] / f"{ep_id}.json.gz")
        base = forgetting_summary(task, full, min_step=step)["facts_violated"]  # what the full-context agent got wrong anyway after this point
        row = await score_row(cfg, llm, task, res, branch_turn=step, baseline_facts=base)
        row["baseline_facts_violated"] = base
        row.update(step=step, episode_id=ep_id, compactor=name, wall_s=time.time() - t0, n_steps=len(res.steps))
        jl_append(P["branches"], row)
        print(f"[resume] {ep_id}@{step} {name:20s} success={row['success']:.2f} viol={row['n_violations']} cost=${row['cost']:.3f} compactions={row['n_compactions']} forced={row['n_forced']} {time.time() - t0:.0f}s spend=${llm.spend.total:.2f}", flush=True)


async def stage_resume(cfg, llm, P, conc, names):
    meta = json.loads(P["bps"].read_text())
    sem = asyncio.Semaphore(conc)
    async def safe(e, s, n):
        try:
            await resume_one(cfg, llm, e, s, n, P, sem)
        except SpendLimitExceeded:
            raise
        except Exception as ex:
            print(f"[resume] {e}@{s} {n} FAILED: {ex!r}"[:300], flush=True)

    await asyncio.gather(*[safe(e, s, n) for e, s in meta["resume"] for n in names])


# ----------------------------------------------------------------------------------- CLI
async def amain(args):
    root = Path(args.out)
    P = paths(root)
    cfg = mk_cfg(root)
    cfg.max_spend_usd = args.max_spend
    cfg.concurrency = args.http_conc
    llm = LLM(cfg, CallLogger(root / "logs" / "calls.jsonl"))
    try:
        if args.stage in ("record", "all"):
            await stage_record(cfg, llm, args.n, P, args.conc, args.start)
        if args.stage in ("select", "record", "all"):
            await stage_select(cfg, args.n, P, args.n_bp, args.n_resume)
        if args.stage in ("score", "all"):
            await stage_score(cfg, llm, P, args.conc, args.limit, args.resume_only)
        if args.stage in ("resume", "all"):
            await stage_resume(cfg, llm, P, args.conc, args.arms or ALL_BASELINES + ["best8_infoloss"])
    finally:
        P["spend"].parent.mkdir(parents=True, exist_ok=True)
        P["spend"].write_text(json.dumps({"total_usd": llm.spend.total, "by_tag": llm.spend.by_tag, "n_calls": llm.spend.n_calls}, indent=1))
        print(f"[spend] ${llm.spend.total:.2f} over {llm.spend.n_calls} calls")
        await llm.aclose()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["record", "select", "score", "resume", "all"])
    ap.add_argument("--out", default="outputs/pilot")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n-bp", type=int, default=4)
    ap.add_argument("--n-resume", type=int, default=24)
    ap.add_argument("--conc", type=int, default=6)
    ap.add_argument("--http-conc", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume-only", action="store_true")
    ap.add_argument("--arms", nargs="+", default=None, help="resume only these compactors (default: all baselines + best8_infoloss)")
    ap.add_argument("--max-spend", type=float, default=25.0)
    asyncio.run(amain(ap.parse_args(argv)))


if __name__ == "__main__":
    main()
