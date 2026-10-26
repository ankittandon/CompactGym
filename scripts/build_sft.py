"""Best-of-N SFT dataset (NEXT_STEPS section 2): one example per branch point.

Input  = exactly what ``small_prompted`` shows the controller (system prompt + conversation to compact).
Target = the winning candidate's controller JSON {action, summary, pinned, confidence}; the winner is the argmin of
         ``lam * InfoLoss - retention`` over every scored candidate, and ``confidence`` is its *measured* retention.

    uv run python scripts/build_sft.py --root outputs/sft_trial --out outputs/sft_trial/sft.jsonl [--lam 1.0]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from compactgym.compactors.small_prompted import SYSTEM, SmallPrompted
from compactgym.harness.resume import split_for_resume
from compactgym.harness.snapshots import SnapshotStore
from compactgym.harness.tokens import count_tokens
from compactgym.env.inbox.tools import openai_tools
from compactgym.interfaces import CompactionResult
from compactgym.runner.common import jl_read
from compactgym.runner.pilot import BUDGET, mk_cfg, paths, state_from_snapshot

SKIP = {"best8_infoloss", "best8_retention"}  # duplicates of a sample row


def cand_path(P: dict[str, Path], row: dict) -> Path:
    s = row["sample"] if row["sample"] is not None else "x"
    return P["picks"] / "all" / f"{row['episode_id']}_{row['step']}_{row['candidate']}_{s}.json"


def parse_failed(P: dict[str, Path], row: dict) -> bool:
    """The small model's JSON did not parse, so the raw reply was stored as the summary."""
    summ = (json.loads(cand_path(P, row).read_text()).get("summary") or "").lstrip()
    return summ.startswith("{") and '"action"' in summ[:200]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/sft_trial")
    ap.add_argument("--out", default=None)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--select", choices=["proxy", "outcome"], default="proxy",
                    help="proxy: argmin lam*InfoLoss - retention over scored candidates; outcome: argmax live branch reward over the arms that were run live")
    ap.add_argument("--min-margin", type=float, default=0.0, help="outcome mode: drop branch points where the best arm beats the runner-up by less than this")
    args = ap.parse_args()
    root = Path(args.root)
    P = paths(root)
    cfg = mk_cfg(root)
    ctl = SmallPrompted(cfg, None, must_compact=True)
    tools = openai_tools(True)
    store = SnapshotStore(P["snap"])

    by_bp: dict[tuple[str, int], list[dict]] = {}  # rows carry "_path" (the candidate's CompactionResult) and "_value" (higher is better)
    cands = jl_read(P["cands"])
    if args.select == "proxy":
        for r in cands:
            if r["candidate"] not in SKIP and r["retention"] is not None and cand_path(P, r).exists() and not parse_failed(P, r):
                by_bp.setdefault((r["episode_id"], r["step"]), []).append({**r, "_path": cand_path(P, r), "_value": -(args.lam * r["infoloss_raw"] - r["retention"])})
    else:
        single = {(r["episode_id"], r["step"], r["candidate"]): r for r in cands if r["sample"] is None and r["retention"] is not None}
        for b in jl_read(P["branches"]):
            r = single.get((b["episode_id"], b["step"], b["compactor"]))
            if r is None:
                continue
            path = P["picks"] / f"{b['episode_id']}_{b['step']}.json" if b["compactor"] == "best8_infoloss" else cand_path(P, r)
            if path.exists() and not (b["compactor"] != "best8_infoloss" and parse_failed(P, r)):
                by_bp.setdefault((b["episode_id"], b["step"]), []).append({**r, "_path": path, "_value": b["reward"], "_success": b["success"], "_forgot": b["n_new_facts_violated"]})

    out, chosen, gain, margins = [], Counter(), [], []
    for (ep, step), rows in sorted(by_bp.items()):
        ranked = sorted(rows, key=lambda r: -r["_value"])
        win = ranked[0]
        margin = win["_value"] - ranked[1]["_value"] if len(ranked) > 1 else 0.0
        if args.select == "outcome" and (len(ranked) < 2 or margin < args.min_margin):
            continue
        margins.append(margin)
        res = CompactionResult.model_validate_json(win["_path"].read_text())
        snap = store.load(ep, step)
        st = state_from_snapshot(snap, cfg, BUDGET, count_tokens(snap.messages, tools, cfg.prompt_overhead_tokens))
        _, older, _ = split_for_resume(snap.messages, st.keep_k)
        types = ", ".join(st.profile.get("never_lose", [])) or "all"
        sys_p = SYSTEM.format(k=st.keep_k, types=types) + "\nThe context is over budget: you MUST pick an action other than keep."
        target = {"action": res.action.value, "summary": res.summary or "", "pinned": res.pinned, "confidence": round(win["retention"], 3)}
        out.append({"episode_id": ep, "step": step, "candidate": win["candidate"], "sample": win["sample"], "infoloss": win["infoloss"], "retention": win["retention"], "value": win["_value"], "margin": margin, "select": args.select,
                    "messages": [{"role": "system", "content": sys_p}, {"role": "user", "content": ctl._user_prompt(snap.messages, st, older, BUDGET)},
                                 {"role": "assistant", "content": json.dumps(target)}]})
        chosen[win["candidate"]] += 1
        base = [r for r in rows if r["candidate"] == "small_prompted"]
        if base:
            gain.append((base[0]["infoloss"], win["infoloss"], base[0]["retention"], win["retention"], base[0]["_value"], win["_value"]))

    dst = Path(args.out or root / "sft.jsonl")
    dst.write_text("".join(json.dumps(x) + "\n" for x in out))
    print(f"{len(out)} examples -> {dst}; winners by candidate: {dict(chosen)}")
    if args.select == "outcome" and by_bp:
        arms = sorted({r["candidate"] for rows in by_bp.values() for r in rows})
        full = [rows for rows in by_bp.values() if len(rows) == len(arms)]
        print(f"\nlive reward per arm over {len(full)} branch points with every arm (winner's curse: the max of noisy single runs is optimistic):")
        for a in arms:
            print(f"  {a:15s} mean reward {sum(next(r for r in rows if r['candidate'] == a)['_value'] for rows in full) / max(1, len(full)):+.3f}")
        print(f"  {'oracle pick':15s} mean reward {sum(max(r['_value'] for r in rows) for rows in full) / max(1, len(full)):+.3f}   median margin {sorted(margins)[len(margins) // 2]:.3f}" if margins else "")
    if gain and args.select == "proxy":
        n = len(gain)
        print(f"vs single small_prompted (mean over {n}): infoloss {sum(g[0] for g in gain)/n:.3f} -> {sum(g[1] for g in gain)/n:.3f}, "
              f"retention {sum(g[2] for g in gain)/n:.3f} -> {sum(g[3] for g in gain)/n:.3f}")


if __name__ == "__main__":
    main()
