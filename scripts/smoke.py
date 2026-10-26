import asyncio, sys, time, json
from compactgym.config import Config
from compactgym.llm import LLM
from compactgym.env.inbox.tasks import TaskSpec, generate_task
from compactgym.harness.agent import Episode
from compactgym.compactors import make_compactor

async def main(comp="never", turns=4, facts=2, budget=10**9, seed=1, mode="scripted"):
    cfg = Config(); cfg.log_dir = "outputs/smoke"
    llm = LLM(cfg)
    task = generate_task(TaskSpec(seed=seed, n_user_turns=turns, n_facts=facts))
    ep = Episode(cfg, task, make_compactor(comp, cfg, llm), llm, episode_id=f"smoke-{comp}", user_mode=mode, budget_tokens=budget)
    t = time.time()
    r = await ep.run()
    print(r.status, r.error, "steps", len(r.steps), "wall", round(time.time()-t,1), "spend", round(llm.spend.total,4))
    for s in r.steps:
        print(s["step"], s["user_turn"], s.get("ctx_before"), s.get("prompt_tokens_api"), s.get("cached_api"), s.get("cached_modeled"), s.get("tools"), s.get("action"), round(s["main_latency_s"],1))
    print([ (m.role, (m.content or "")[:90]) for m in r.full_log if m.role in ("user",)])
    print(r.turn_final_text)
    r.save(f"outputs/smoke/{comp}.json.gz")
    await llm.aclose()
asyncio.run(main(*[int(a) if a.isdigit() else a for a in sys.argv[1:]]))
