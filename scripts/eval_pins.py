"""Which models can extract the facts worth pinning? Zero-shot, from the transcript up to a branch point, scored against the planted facts.

    COMPACTGYM_SPEC=hard uv run python scripts/eval_pins.py --root outputs/hard --models glm-5p3 glm-5p3-flash gpt-oss-120b [--n 36] [--max-spend 2]

Gold = the canonical text of every planted fact stated by the branch point. A fact counts as recalled when most of its content words appear in one predicted
item (scripts/eval_sft.py ``covers``). Precision is the share of predicted items that match a gold fact, so listing one-off task requests lowers it.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eval_sft import covers, parse_reply  # noqa: E402

from compactgym.config import FW, Config, ModelSpec
from compactgym.env.inbox.tasks import generate_task
from compactgym.harness.resume import transcript
from compactgym.harness.snapshots import SnapshotStore
from compactgym.llm import CallLogger, LLM
from compactgym.runner.common import pilot_spec

SYSTEM = """You keep the pinned list for an email and calendar assistant. The pinned list is shown to the assistant at all times, even after the rest of its history is compacted away.

Read the conversation and list every standing instruction, restriction, preference, correction, decision, reference number, changed date or person to leave out that the user gave and that still applies. Include every specific (names, dates, numbers, ids). Write each as one short, exact statement. Leave out one-off task requests.

Reply with one JSON object only: {"pinned": ["...", "..."]}"""


async def run(args) -> None:
    root = Path(args.root)
    meta = json.loads((root / "bps.json").read_text())
    bps = [(e, b[0]) for e, b in meta["bps"].items()][: args.n]  # first branch point of each episode
    store = SnapshotStore(root / "snapshots")
    items = []
    for ep, step in bps:
        snap = store.load(ep, step)
        task = generate_task(pilot_spec(int(ep[1:])))
        ut = snap.extras["user_turn"]
        items.append({"ep": ep, "step": step, "text": transcript([m for m in snap.messages if m.role != "system"], max_tool_chars=700),
                      "gold": [f.text for f in task.facts if f.inject_turn <= ut]})
    cfg = Config(max_spend_usd=args.max_spend)
    llm = LLM(cfg, CallLogger(Path("outputs/eval_pins_calls.jsonl")))
    sem = asyncio.Semaphore(args.conc)
    out = Path(args.out)
    out.write_text("")

    async def one(model: str, it: dict) -> dict:
        spec = ModelSpec(model=FW + model, max_tokens=4500 if "nemotron" in model else 2500)
        async with sem:
            try:
                res = await llm.chat(spec, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": it["text"] + "\n\nReturn the JSON now."}], tag="eval_pins", max_tokens=spec.max_tokens)
                text = res.text
            except Exception as e:
                text = f"ERROR {e!r}"
        d = parse_reply(text) or {}
        pred = [str(p) for p in (d.get("pinned") or []) if isinstance(p, (str, int, float))]
        gold = it["gold"]
        rec = sum(any(covers(p, g) for p in pred) for g in gold) / len(gold) if gold else 1.0
        prec = sum(any(covers(p, g) for g in gold) for p in pred) / len(pred) if pred else (1.0 if not gold else 0.0)
        row = {"model": model, "ep": it["ep"], "step": it["step"], "n_gold": len(gold), "n_pred": len(pred), "valid": bool(d), "recall": rec, "precision": prec, "pred": pred}
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")
        return row

    try:
        rows = await asyncio.gather(*[one(m, it) for m in args.models for it in items])
    finally:
        await llm.aclose()
    print(f"{len(items)} branch points, {sum(len(i['gold']) for i in items)} gold facts, spend ${llm.spend.total:.2f}\n")
    print(f"{'model':28s} valid  recall  precision  preds/bp  perfect-recall bps")
    for m in args.models:
        r = [x for x in rows if x["model"] == m]
        n = len(r)
        print(f"{m:28s} {sum(x['valid'] for x in r) / n:5.0%}  {sum(x['recall'] for x in r) / n:6.2f}  {sum(x['precision'] for x in r) / n:9.2f}  {sum(x['n_pred'] for x in r) / n:8.1f}  {sum(x['recall'] == 1.0 for x in r)}/{n}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/hard")
    ap.add_argument("--models", nargs="+", default=["glm-5p3", "glm-5p3-flash", "gpt-oss-120b"])
    ap.add_argument("--n", type=int, default=36)
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--max-spend", type=float, default=2.0)
    ap.add_argument("--out", default="outputs/pins_eval.jsonl")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
