import asyncio, json
from pathlib import Path
from compactgym.runner.pilot import *
async def main():
    root = Path("outputs/pilot"); P = paths(root); cfg = mk_cfg(root); cfg.concurrency = 8
    llm = LLM(cfg, CallLogger(root / "logs" / "calls.jsonl"))
    meta = json.loads(P["bps"].read_text()); sem = asyncio.Semaphore(3)
    await asyncio.gather(*[score_bp(cfg, llm, e, s, P, ["constraint_extractor"], sem, probes=False) for e, s in meta["resume"]])
    print("spend", llm.spend.total); await llm.aclose()
asyncio.run(main())
