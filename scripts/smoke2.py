import asyncio, sys, time, json
from compactgym.config import Config
from compactgym.llm import LLM
from compactgym.env.inbox.tasks import TaskSpec, generate_task
from compactgym.harness.agent import Episode
from compactgym.compactors import make_compactor
from compactgym.scoring.forgetting import forgetting_summary
from compactgym.scoring.success import success_summary
from compactgym.scoring.cost import episode_cost

async def main(comp="never", turns=14, facts=3, budget=10**9, seed=1, vol=2.0, inj="top", mode="scripted"):
    cfg = Config(); cfg.log_dir = "outputs/smoke"
    llm = LLM(cfg)
    task = generate_task(TaskSpec(seed=seed, n_user_turns=turns, n_facts=facts, distractor_volume=float(vol), injection=inj))
    ep = Episode(cfg, task, make_compactor(comp, cfg, llm), llm, episode_id=f"smoke2-{comp}-{seed}", user_mode=mode, budget_tokens=budget)
    t = time.time()
    r = await ep.run()
    print(r.status, r.error, "steps", len(r.steps), "wall", round(time.time()-t,1), "spend", round(llm.spend.total,4))
    ctx = [s.get("ctx_after") for s in r.steps]; print("ctx tokens:", ctx[::4], "max", max(ctx))
    print("facts:", [(f.id, f.kind, f.inject_turn) for f in task.facts])
    fs = forgetting_summary(task, r); ss = success_summary(task, r)
    print("viol", fs["n_violations"], fs["by_type"], "reasks", fs["n_reasks"], fs["redone_reads"], fs["redone_writes"])
    for v in fs["violations"][:8]: print("  V", v)
    print("success", round(ss["success"],3), ss["per_goal"], "reached", ss["reached_turn"])
    print("cost", episode_cost(r, cfg))
    print("forced/compactions:", sum(1 for s in r.steps if s.get("forced")), sum(1 for s in r.steps if s.get("action") not in (None,"keep")))
    r.save(f"outputs/smoke/{comp}-{seed}.json.gz")
    await llm.aclose()
asyncio.run(main(*[int(a) if a.isdigit() else a for a in sys.argv[1:]]))
