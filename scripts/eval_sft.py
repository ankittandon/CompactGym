"""Score a compaction controller against a held-out SFT file (written by build_sft.py).

    # score replies you already have: JSONL of {"episode_id", "step", "reply"} (reply = the model's raw text)
    uv run python scripts/eval_sft.py outputs/hard/sft_val.jsonl --predictions replies.jsonl
    # or have a Fireworks model produce the replies first (spends API credit: one call per example)
    uv run python scripts/eval_sft.py outputs/hard/sft_val.jsonl --generate accounts/fireworks/models/<model> [--save replies.jsonl]
    # reference points that need no model
    uv run python scripts/eval_sft.py outputs/hard/sft_val.jsonl --baseline always_truncate|majority|oracle

Metrics, per example then averaged:
* valid_json      the reply parses to an object with the four fields (action, summary, pinned, confidence)
* valid_action    the action is one of the seven known actions
* action_acc      the action equals the target action (the live-outcome winner's action)
* pinned_recall   share of the target's pinned facts that appear in the reply's pinned list (a fact counts when >= 70% of its content words do)
* pinned_prec     share of the reply's pinned items that match some target fact (low = invented or padded facts)
* summary_ok      a summary is present exactly when the action needs one (summarize_older / rewrite)
* conf_err        |confidence - target confidence| (the target confidence is the winner's measured retention)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import Counter
from pathlib import Path

ACTIONS = {"keep", "truncate_verbatim", "clear_tool_outputs", "summarize_older", "rewrite", "offload_to_memory", "defer_to_frontier"}
NEEDS_SUMMARY = {"summarize_older", "rewrite"}
STOP = {"the", "and", "for", "you", "your", "not", "any", "all", "with", "that", "this", "are", "but", "has", "have", "from", "his", "her", "its", "out", "per", "use", "can", "exact", "phrase"}
MATCH = 0.7


def parse_reply(text: str) -> dict | None:
    """The first JSON object in the reply, or None."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except Exception:
        return None
    return d if isinstance(d, dict) else None


def words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9@]+(?:[.\-'][a-z0-9@]+)*", s.lower()) if len(w) > 2 and w not in STOP}


def covers(pred: str, gold: str) -> bool:
    """The predicted pinned item states the gold fact: most of the gold's content words are present."""
    g = words(gold)
    return bool(g) and len(g & words(pred)) / len(g) >= MATCH


def score_one(reply_text: str, target: dict) -> dict:
    d = parse_reply(reply_text)
    out = {"valid_json": False, "valid_action": False, "action_acc": False, "pinned_recall": 0.0, "pinned_prec": 0.0, "summary_ok": False, "conf_err": None, "action": None}
    if d is None or not {"action", "summary", "pinned", "confidence"} <= d.keys():
        return out
    out["valid_json"] = True
    act = str(d.get("action", "")).strip().lower()
    out["action"] = act
    out["valid_action"] = act in ACTIONS
    out["action_acc"] = act == target["action"]
    pinned = [str(p) for p in (d.get("pinned") or []) if isinstance(p, (str, int, float))]
    gold = target.get("pinned") or []
    out["pinned_recall"] = sum(any(covers(p, g) for p in pinned) for g in gold) / len(gold) if gold else 1.0
    out["pinned_prec"] = sum(any(covers(p, g) for g in gold) for p in pinned) / len(pinned) if pinned else (1.0 if not gold else 0.0)
    has = bool(str(d.get("summary") or "").strip())
    out["summary_ok"] = has == (act in NEEDS_SUMMARY)
    try:
        out["conf_err"] = abs(float(d["confidence"]) - float(target["confidence"]))
    except (TypeError, ValueError):
        pass
    return out


def summarize(rows: list[dict], targets: list[dict]) -> dict:
    n = len(rows)
    mean = lambda k: sum(bool(r[k]) if isinstance(r[k], bool) else r[k] for r in rows) / n  # noqa: E731
    errs = [r["conf_err"] for r in rows if r["conf_err"] is not None]
    conf = Counter((t["action"], r["action"] or "invalid") for t, r in zip(targets, rows))
    return {"n": n, "valid_json": mean("valid_json"), "valid_action": mean("valid_action"), "action_acc": mean("action_acc"), "pinned_recall": mean("pinned_recall"),
            "pinned_prec": mean("pinned_prec"), "summary_ok": mean("summary_ok"), "conf_err": sum(errs) / len(errs) if errs else None,
            "confusion": {f"{a} -> {b}": c for (a, b), c in sorted(conf.items())}}


def baseline_reply(name: str, targets: list[dict]) -> list[str]:
    if name == "oracle":  # the target itself: the ceiling, and a check that the scorer agrees with the data
        return [json.dumps(t) for t in targets]
    act = "truncate_verbatim" if name == "always_truncate" else Counter(t["action"] for t in targets).most_common(1)[0][0]
    return [json.dumps({"action": act, "summary": "" if act not in NEEDS_SUMMARY else "x", "pinned": [], "confidence": 0.9}) for _ in targets]


async def generate(examples: list[dict], model: str, max_tokens: int, conc: int) -> list[str]:
    from compactgym.config import Config, ModelSpec
    from compactgym.llm import CallLogger, LLM
    cfg = Config(max_spend_usd=5.0)
    llm = LLM(cfg, CallLogger(Path("outputs/eval_calls.jsonl")))
    spec = ModelSpec(model=model, max_tokens=max_tokens)
    sem = asyncio.Semaphore(conc)

    async def one(ex: dict) -> str:
        async with sem:
            try:
                return (await llm.chat(spec, ex["messages"][:2], tag="eval_sft", max_tokens=max_tokens)).text
            except Exception as e:  # a failed call counts as an invalid reply, not a crash
                return f"ERROR: {e!r}"

    try:
        return await asyncio.gather(*[one(ex) for ex in examples])
    finally:
        await llm.aclose()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("val", help="sft_val.jsonl from build_sft.py")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--predictions", help="JSONL of {episode_id, step, reply}")
    g.add_argument("--generate", metavar="MODEL", help="Fireworks model id to produce the replies (spends credit)")
    g.add_argument("--baseline", choices=["always_truncate", "majority", "oracle"])
    ap.add_argument("--save", help="with --generate: write the replies here")
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--conc", type=int, default=4)
    a = ap.parse_args(argv)

    examples = [json.loads(l) for l in open(a.val)]
    targets = [json.loads(ex["messages"][2]["content"]) for ex in examples]
    if a.baseline:
        replies = baseline_reply(a.baseline, targets)
    elif a.predictions:
        by = {(r["episode_id"], r["step"]): r["reply"] for r in map(json.loads, open(a.predictions))}
        replies = [by.get((ex["episode_id"], ex["step"]), "") for ex in examples]
    else:
        replies = asyncio.run(generate(examples, a.generate, a.max_tokens, a.conc))
        if a.save:
            Path(a.save).write_text("".join(json.dumps({"episode_id": ex["episode_id"], "step": ex["step"], "reply": r}) + "\n" for ex, r in zip(examples, replies)))
    rows = [score_one(r, t) for r, t in zip(replies, targets)]
    res = summarize(rows, targets)
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()
