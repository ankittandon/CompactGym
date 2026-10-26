"""Markdown report + plots from pilot (and optional matrix) results.

    uv run python -m compactgym.runner.report --pilot outputs/pilot --matrix outputs/matrix --out outputs/pilot_report.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ..env.inbox.tasks import generate_task
from ..scoring.calibration import reliability
from . import plots
from .common import jl_read, pilot_spec

RNG = np.random.default_rng(0)


def boot_ci(x, n=4000, fn=np.mean):
    x = np.asarray([v for v in x if v is not None and not np.isnan(v)], float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    idx = RNG.integers(0, len(x), (n, len(x)))
    s = np.array([fn(x[i]) for i in idx])
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def f(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def md_table(df: pd.DataFrame, floatfmt=3) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(f(v, floatfmt) if isinstance(v, (float, np.floating)) else str(v) for v in r.values) + " |")
    return "\n".join(out)


def spearman(a, b):
    d = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(d) < 4 or d["a"].nunique() < 2 or d["b"].nunique() < 2:
        return float("nan"), float("nan"), len(d)
    r, p = stats.spearmanr(d["a"], d["b"])
    return float(r), float(p), len(d)


# ------------------------------------------------------------------------------ loading
def load(root: Path):
    full = pd.DataFrame(jl_read(root / "full_rows.jsonl"))
    cands = pd.DataFrame(jl_read(root / "candidates.jsonl"))
    br = pd.DataFrame(jl_read(root / "branches.jsonl"))
    rep = pd.DataFrame(jl_read(root / "repeated.jsonl"))
    spend = json.loads((root / "spend.json").read_text()) if (root / "spend.json").exists() else {}
    return full, cands, br, rep, spend


def fact_table(cands: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cache = {}
    for _, r in cands.iterrows():
        seed = int(r["episode_id"][1:])
        if seed not in cache:
            spec = pilot_spec(seed)
            cache[seed] = (generate_task(spec), spec.injection)
        task, loc = cache[seed]
        for fid, v in (r["ret_by_fact"] or {}).items():
            if v is None:
                continue
            f_ = task.fact(fid)
            rows.append({"candidate": r["candidate"], "sample": r.get("sample"), "episode_id": r["episode_id"], "step": r["step"], "fact": fid, "type": f_.type, "kind": f_.kind,
                         "location": loc if f_.restate_turns == [] or loc != "repeated" else "repeated", "retained": float(bool(v))})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------- sections
def headroom(cands: pd.DataFrame) -> tuple[str, dict]:
    c = cands
    key = ["episode_id", "step"]
    single = c[c.candidate == "small_prompted"].set_index(key)
    pick = c[c.candidate == "best8_infoloss"].set_index(key)
    orc = c[c.candidate == "best8_retention"].set_index(key)
    samples = c[c.candidate == "small_sample"].groupby(key).agg(il_mean=("infoloss", "mean"), il_min=("infoloss", "min"), ret_mean=("retention", "mean"))
    j = (single[["infoloss", "retention"]].join(pick[["infoloss", "retention"]].add_suffix("_pick")).join(samples).join(orc[["retention"]].add_suffix("_oracle"))
         .dropna(subset=["infoloss", "infoloss_pick"]))
    if j.empty:
        return "_no data_", {}
    d_il = j["infoloss"] - j["infoloss_pick"]
    d_ret = (j["retention_pick"] - j["retention"]).dropna()
    d_il_s = j["il_mean"] - j["infoloss_pick"]
    d_ret_s = (j["retention_pick"] - j["ret_mean"]).dropna()
    rows = [
        ["InfoLoss (lower better)", j["infoloss"].mean(), j["il_mean"].mean(), j["infoloss_pick"].mean(), j["il_min"].mean(), (j["infoloss"] - j["infoloss_pick"]).mean(), *boot_ci(d_il), (d_il > 0).mean()],
        ["Retention of planted facts", j["retention"].mean(), j["ret_mean"].mean(), j["retention_pick"].mean(), j["retention_oracle"].mean(), d_ret.mean(), *boot_ci(d_ret), (d_ret > 0).mean()],
    ]
    t = pd.DataFrame(rows, columns=["metric", "single prompted (T=0)", "mean of 8 samples", "best-of-8 pick (by InfoLoss)", "oracle best-of-8", "pick - single (paired)", "CI lo", "CI hi", "frac. pick better"])
    info = {"n": len(j), "il_single": j["infoloss"].mean(), "il_pick": j["infoloss_pick"].mean(), "il_rel": 1 - j["infoloss_pick"].mean() / max(1e-9, j["infoloss"].mean()),
            "il_ci": boot_ci(d_il), "ret_single": j["retention"].mean(), "ret_pick": j["retention_pick"].mean(), "ret_ci": boot_ci(d_ret), "ret_gain": d_ret.mean(),
            "il_vs_samples_ci": boot_ci(d_il_s), "ret_vs_samples": d_ret_s.mean(), "ret_vs_samples_ci": boot_ci(d_ret_s)}
    return f"{len(j)} branch points.\n\n" + md_table(t), info


def validity(cands: pd.DataFrame, br: pd.DataFrame, full: pd.DataFrame):
    res = {}
    out = []
    # offline: across the 8 samples of the same branch point, does lower InfoLoss mean higher retention?
    s = cands[cands.candidate == "small_sample"]
    rs = []
    for _, g in s.groupby(["episode_id", "step"]):
        if g["retention"].nunique() > 1 and g["infoloss"].nunique() > 1:
            rs.append(stats.spearmanr(g["infoloss"], g["retention"])[0])
    res["within_bp_il_ret"] = (float(np.nanmean(rs)) if rs else float("nan"), len(rs))
    out.append(f"* Offline, within the 8 small-model samples of a branch point: mean Spearman(InfoLoss, retention) = **{f(res['within_bp_il_ret'][0], 2)}** over {len(rs)} branch points with variation (expected negative).")
    a = cands[cands["sample"].isna()]
    a = a[~a.candidate.isin(["best8_infoloss", "best8_retention"])]
    r, p, n = spearman(a["infoloss"], a["retention"])
    res["all_il_ret"] = (r, p, n)
    out.append(f"* Offline, all candidates pooled: Spearman(InfoLoss, retention) = **{f(r, 2)}** (p={f(p, 3)}, n={n}).")
    # live: InfoLoss of the compaction vs what happened after the branch point
    if len(br):
        ci = cands[cands["sample"].isna() & ~cands.candidate.isin(["best8_retention"])][["episode_id", "step", "candidate", "infoloss", "infoloss_raw", "retention", "ctx_tokens", "ctx_after"]].rename(columns={"candidate": "compactor", "retention": "retention_bp", "ctx_after": "ctx_after_bp"})
        b = br.merge(ci, on=["episode_id", "step", "compactor"], how="inner")
        b = b.merge(full[["episode_id", "success"]].rename(columns={"success": "success_full"}), on="episode_id", how="left")
        b["success_drop"] = b["success_full"] - b["success"]
        b["violations_after"] = b["n_new_facts_violated"]
        res["pairs"] = b
        for nm, y in [("forgetting (planted facts violated after the branch point that the full-context run obeyed)", "violations_after"), ("failure (success drop vs full-context continuation)", "success_drop"), ("forgetting score incl. re-asks", "forgetting_score")]:
            r, p, n = spearman(b["infoloss"], b[y])
            # within-branch-point mean (controls for task difficulty)
            ws = [stats.spearmanr(g["infoloss"], g[y])[0] for _, g in b.groupby(["episode_id", "step"]) if g[y].nunique() > 1 and g["infoloss"].nunique() > 1]
            res[y] = (r, p, n, float(np.nanmean(ws)) if ws else float("nan"), len(ws))
            rr, pr_, nr = spearman(b["infoloss_raw"], b[y])
            res[y + "_raw"] = (rr, pr_, nr)
            out.append(f"* Spearman(InfoLoss, {nm}) pooled = **{f(r, 2)}** (p={f(p, 3)}, n={n}); mean within-branch-point = {f(res[y][3], 2)} ({len(ws)} bps); unclipped (signed) InfoLoss: ρ={f(rr, 2)} (p={f(pr_, 3)}).")
        r, p, n = spearman(b["retention_bp"], b["violations_after"])
        res["ret_forget"] = (r, p, n)
        out.append(f"* For comparison, Spearman(retention at the branch point, violations after) = {f(r, 2)} (p={f(p, 3)}, n={n}).")
    return "\n".join(out), res


def leaderboard(br: pd.DataFrame, full: pd.DataFrame, cands: pd.DataFrame):
    if br.empty:
        return "_no live branches_", None
    b = br.merge(full[["episode_id", "success", "n_violations"]].rename(columns={"success": "success_full", "n_violations": "viol_full"}), on="episode_id", how="left")
    b["d_success"] = b["success"] - b["success_full"]
    ci = cands[cands["sample"].isna()][["episode_id", "step", "candidate", "infoloss", "retention", "ctx_tokens", "ctx_after"]].rename(columns={"candidate": "compactor", "retention": "retention_bp", "ctx_after": "ctx_after_bp"})
    b = b.merge(ci, on=["episode_id", "step", "compactor"], how="left")
    b["success_per_usd"] = b["success"] / b["cost"].clip(lower=1e-6)
    g = b.groupby("compactor").agg(n=("success", "size"), success=("success", "mean"), d_success_vs_full=("d_success", "mean"), facts_forgotten=("n_new_facts_violated", "mean"), violations=("n_violations", "mean"), reasks=("n_reasks", "mean"),
                                    cost_usd=("cost", "mean"), pause_s=("pause_s", "mean"), compactions=("n_compactions", "mean"), forced=("n_forced", "mean"),
                                    summary_chars=("summary_chars", "mean"), infoloss_at_bp=("infoloss", "mean"), retention_at_bp=("retention_bp", "mean")).reset_index()
    g["success_per_usd"] = g["success"] / g["cost_usd"].clip(lower=1e-6)
    g = g.sort_values("success", ascending=False)
    return md_table(g), b


def vs_cliff(b: pd.DataFrame) -> str:
    rows = []
    cl = b[b.compactor == "cliff"].set_index(["episode_id", "step"])
    for name, g in b.groupby("compactor"):
        if name == "cliff":
            continue
        gg = g.set_index(["episode_id", "step"])
        idx = gg.index.intersection(cl.index)
        if len(idx) < 3:
            continue
        row = [name, len(idx)]
        for col in ["success", "n_violations", "cost", "pause_s"]:
            d = (gg.loc[idx, col] - cl.loc[idx, col]).values
            lo, hi = boot_ci(d)
            row += [float(np.mean(d)), f"[{f(lo, 3)}, {f(hi, 3)}]"]
        rows.append(row)
    cols = ["compactor", "n pairs"]
    for c in ["success", "violations", "cost", "pause"]:
        cols += [f"Δ{c} vs cliff", "95% CI"]
    return md_table(pd.DataFrame(rows, columns=cols)) if rows else "_n/a_"


def replication(ft: pd.DataFrame) -> str:
    if ft.empty:
        return "_n/a_"
    ft = ft[ft["sample"].isna() | (ft.candidate != "small_sample")]
    cp = ft[ft.type.isin(["Action", "Information", "Process", "Preference", "Output"])]
    t = cp.pivot_table(index="candidate", columns="type", values="retained", aggfunc="mean")
    t["CONPACT-type mean"] = cp.groupby("candidate")["retained"].mean()
    t["all-type mean"] = ft.groupby("candidate")["retained"].mean()
    t["n facts"] = ft.groupby("candidate")["retained"].size()
    return md_table(t.reset_index().round(3))


def verification_section(full: pd.DataFrame, root: Path) -> str:
    if full.empty:
        return "_n/a_"
    lines = [f"* {len(full)} full-context episodes, median {int(full['n_steps'].median())} agent steps (max {int(full['n_steps'].max())}), median peak context {int(full['max_ctx'].median())} tokens (max {int(full['max_ctx'].max())}); **{int((full['n_steps'] >= 30).sum())}/{len(full)} sessions ≥ 30 turns**.",
             f"* Tool calling: {int(full['bad_args'].sum())} invalid-JSON tool calls and {int(full['empty_replies'].sum())} empty replies in {int(full['n_steps'].sum())} main-model calls; {int(full['tool_errors'].sum())} tool-level errors in total (incl. deliberate cc rejections / bad ids), status counts: {full['status'].value_counts().to_dict()}.",
             f"* Simulated user: {int(full['user_fallbacks'].sum())} LLM rewrites rejected (fallback to the canonical script), **{int(full['user_leaks'].sum())} leaked-fact rewrites** caught and replaced; mean episode success with full context {f(full['success'].mean(), 2)}; mean planted-fact violations with full context {f(full['n_violations'].mean(), 2)} per episode (the model's base error rate)."]
    # tokenizer + caching from the episode files
    from ..harness.agent import EpisodeResult
    from ..harness.tokens import TOOL_MSG_OVERHEAD
    diffs, raw, rep, mod, tot = [], [], 0, 0, 0
    for p in sorted((root / "full").glob("*.json.gz"))[:12]:
        r = EpisodeResult.load(p)
        idx = {m.meta.get("step"): i for i, m in enumerate(r.full_log) if m.role == "assistant"}
        for s in r.steps:
            if s.get("prompt_tokens_api") and s["step"] in idx:
                n_tool = sum(m.role == "tool" for m in r.full_log[: idx[s["step"]]])
                raw.append(s["prompt_tokens_api"] - s["prompt_tokens_est"])
                diffs.append(s["prompt_tokens_api"] - s["prompt_tokens_est"] - TOOL_MSG_OVERHEAD * n_tool)  # est already has the constant 11
                rep += s.get("cached_api", 0)
                mod += s.get("cached_modeled", 0)
                tot += s["prompt_tokens_api"]
    if diffs:
        lines.append(f"* Tokenizer: gpt-oss-120b uses `o200k_harmony`; contexts are rendered with `openai_harmony`. API `prompt_tokens` = harmony count + 11 (constant header gap) + {TOOL_MSG_OVERHEAD} per tool message; residual after this calibration over {len(diffs)} calls (contexts up to ~45K tokens): mean {np.mean(diffs):.2f}, max |residual| {max(abs(d) for d in diffs)} tokens (uncalibrated gap was {np.mean(raw):.0f} ± {np.std(raw):.0f}). Context budgets therefore count exactly.")
        lines.append(f"* Caching: the API reports `prompt_tokens_details.cached_tokens`. Over {tot:,} prompt tokens it reported {rep / tot:.0%} cached vs {mod / tot:.0%} under our deterministic prefix model; reported hits are noisy (occasional full misses on identical prefixes), so scoring uses the prefix model and logs both.")
    return "\n".join(lines)


def build(pilot: Path, matrix: Path | None, out: Path):
    full, cands, br, rep, spend = load(pilot)
    plots_dir = pilot / "plots"
    md = ["# CompactGym pilot report\n"]
    sec = {}
    ft = fact_table(cands) if len(cands) else pd.DataFrame()
    hr_txt, hr = headroom(cands) if len(cands) else ("_no data_", {})
    val_txt, val = validity(cands, br, full) if len(cands) else ("_no data_", {})
    lb_txt, lbdf = leaderboard(br, full, cands) if len(br) else ("_no live branches_", None)
    # ---------- plots
    figs = {}
    if lbdf is not None and len(lbdf):
        df = lbdf.assign(budget=16000)
        figs["1"] = plots.plot_success_vs_cost(df, plots_dir / "1_success_vs_cost.png")
        figs["2"] = plots.plot_success_vs_pause(df, plots_dir / "2_success_vs_pause.png")
        d7 = lbdf.assign(budget=16000, retained_ratio=lbdf["ctx_after_bp"] / lbdf["ctx_tokens"])
        figs["7"] = plots.plot_budget_axis(d7, plots_dir / "7_budget_axis.png")
    if matrix is not None and (matrix / "results.jsonl").exists():
        m = pd.DataFrame(jl_read(matrix / "results.jsonl"))
        figs["1m"] = plots.plot_success_vs_cost(m, plots_dir / "1_success_vs_cost_matrix.png")
        figs["2m"] = plots.plot_success_vs_pause(m, plots_dir / "2_success_vs_pause_matrix.png")
        figs["7m"] = plots.plot_budget_axis(m, plots_dir / "7_budget_axis_matrix.png")
    if len(rep):
        figs["3"] = plots.plot_repeated(rep, plots_dir / "3_repeated_compaction.png")
    if len(ft):
        figs["4a"], figs["4b"] = plots.plot_retention_by_type(ft[ft.candidate.isin(["cliff", "frontier", "threshold_pi", "small_prompted", "best8_infoloss"])], plots_dir / "4a_retention_by_type.png", plots_dir / "4b_retention_by_location.png")
    if "pairs" in val and len(val["pairs"]):
        figs["5"] = plots.plot_infoloss_validity(val["pairs"], plots_dir / "5_infoloss_validity.png")
    sp = cands[(cands.candidate == "small_prompted") | (cands.candidate == "small_sample")] if len(cands) else cands
    rel = None
    if len(sp) and sp["confidence"].notna().any():
        d = sp.dropna(subset=["confidence", "retention"])
        rel = reliability(d["confidence"].tolist(), d["retention"].tolist())
        rel2 = reliability(d["confidence"].tolist(), (1 - d["infoloss"].clip(0, 1)).tolist())
        figs["6"] = plots.plot_reliability(rel, plots_dir / "6_confidence_reliability.png")
    else:
        rel2 = None
    rel_s = lambda p: str(Path(p).relative_to(out.parent)) if p else ""  # noqa: E731
    img = lambda k, cap: f"![{cap}]({rel_s(figs[k])})\n\n*{cap}*\n" if k in figs else f"_({cap}: no data)_\n"  # noqa: E731

    # ---------- criteria
    crit = []
    if hr:
        c1 = hr["il_ci"][0] > 0 and hr["il_rel"] >= 0.15 and hr["ret_ci"][0] > 0
        crit.append(("Best-of-8 clearly beats a single prompted small model (InfoLoss ≥15% lower, CI>0; retention gain CI>0)", c1,
                     f"InfoLoss {f(hr['il_single'])}→{f(hr['il_pick'])} ({hr['il_rel']:.0%}, CI {f(hr['il_ci'][0])}..{f(hr['il_ci'][1])}); retention {f(hr['ret_single'], 2)}→{f(hr['ret_pick'], 2)} (Δ CI {f(hr['ret_ci'][0], 2)}..{f(hr['ret_ci'][1], 2)})"))
    if val and "violations_after" in val:
        v1, v2 = val["violations_after"], val["success_drop"]
        c2 = (v1[0] >= 0.2 and v1[1] < 0.1) and (v2[0] >= 0.2 and v2[1] < 0.1)
        crit.append(("InfoLoss tracks forgetting and failure (Spearman ≥0.2, p<0.1, both)", c2, f"forgetting ρ={f(v1[0], 2)} (p={f(v1[1], 3)}), failure ρ={f(v2[0], 2)} (p={f(v2[1], 3)})"))
    cl = cands[cands.candidate == "cliff"]["infoloss"].mean() if len(cands) else float("nan")
    crit_c3 = None
    if lbdf is not None and "best8_infoloss" in set(lbdf.compactor) and "cliff" in set(lbdf.compactor):
        row = lambda n: lbdf[lbdf.compactor == n]  # noqa: E731
        crit_c3 = (f"cliff success {f(row('cliff').success.mean(), 3)} @ ${f(row('cliff').cost.mean(), 4)} (={f(row('cliff').success.mean() / row('cliff').cost.mean(), 1)}/$) vs best8 {f(row('best8_infoloss').success.mean(), 3)} @ ${f(row('best8_infoloss').cost.mean(), 4)} (={f(row('best8_infoloss').success.mean() / row('best8_infoloss').cost.mean(), 1)}/$); mean InfoLoss: cliff {f(cl)} vs best8 pick {f(hr.get('il_pick', float('nan')))}")

    # ---------- document
    md.append("## 1. Summary and recommendation\n")
    rec_file = pilot / "recommendation.md"
    md.append(rec_file.read_text() if rec_file.exists() else "_(recommendation pending: see criteria below)_\n")
    md.append("\n### Go criteria (computed)\n")
    md.append("| criterion | met | evidence |\n|---|---|---|")
    for t_, ok, ev in crit:
        md.append(f"| {t_} | {'**yes**' if ok else '**no**'} | {ev} |")
    if crit_c3:
        md.append(f"| A plausible path exists to beat `cliff` on success per dollar | _judgement_ | {crit_c3} |")
    md.append("\n## 2. Verification of the setup\n")
    md.append(verification_section(full, pilot))
    md.append("\n## 3. Headroom: best-of-8 vs a single prompted small model\n")
    md.append(hr_txt)
    md.append("\n`best-of-8 pick` = the sample (temperature 0.8) with the lowest InfoLoss; its retention is then judged independently (no selection on retention). `oracle best-of-8` picks by retention and is an upper bound.\n")
    md.append("\n## 4. Validity: does InfoLoss predict forgetting and failure?\n")
    md.append(val_txt + "\n")
    md.append(img("5", "InfoLoss vs forgetting / failure (one point per branch point × compactor)"))
    md.append("\n## 5. Replication: do small prompted compactors lose constraints here?\n")
    md.append(replication(ft) + "\n\nFraction of planted facts judged present (CONPACT's YES/NO retention judge) after compaction at a ≥16K-token context.\n")
    md.append(img("4a", "Retention by fact type"))
    md.append(img("4b", "Retention by injection location"))
    md.append("\n## 6. The bar: `cliff` vs everything else\n")
    md.append(vs_cliff(lbdf) if lbdf is not None else "_n/a_")
    md.append("\n## 7. Leaderboard at 16K (live branches, continued to the end of the session)\n")
    md.append(lb_txt + "\n\n`d_success_vs_full` is relative to the recorded full-context run of the same episode; cost covers the continuation after the branch point only (cache-aware prefix model); pause counts blocking compactor wall-clock.\n")
    md.append(img("1", "Success vs cost per task"))
    md.append(img("2", "Success vs total pause time"))
    md.append(img("7", "Tokens retained per token of history vs success"))
    if "1m" in figs:
        md.append(img("1m", "Matrix: success vs cost across budgets (8K/16K/32K)"))
        md.append(img("2m", "Matrix: success vs pause across budgets"))
        md.append(img("7m", "Matrix: budget-axis curve"))
    md.append("\n## 8. Repeated compaction\n")
    if len(rep):
        r2 = rep[rep.event <= 20]
        t = r2.pivot_table(index="event", columns="compactor", values="error", aggfunc="mean").round(3).reset_index()
        t["event"] = t["event"].astype(int)
        t["n (facts)"] = r2.groupby("event")["error"].size().values
        md.append("Mean error (1 - fact retention) after the k-th consecutive compaction at a 6K budget, first 20 events (`cliff` is stateless and re-fires on every threshold crossing, so it reaches 100+ events; later events stay flat, see the plot).\n\n" + md_table(t) + "\n")
    md.append(img("3", "Error (1 - fact retention) vs number of compaction events"))
    md.append("\n## 9. Confidence calibration\n")
    if rel:
        md.append(f"small_prompted stated a confidence in {int(sp['confidence'].notna().sum())} of {len(sp)} compactions. ECE vs realized retention = **{f(rel['ece'], 3)}**; vs (1 − InfoLoss) = {f(rel2['ece'], 3) if rel2 else 'n/a'}.\n")
    else:
        md.append("The small model did not report usable confidences.\n")
    md.append(img("6", "Confidence reliability"))
    md.append("\n## 10. Cost and timing\n")
    if spend:
        md.append(f"Total spend **${spend['total_usd']:.2f}** over {spend['n_calls']:,} model calls (prices are config placeholders, see NEXT_STEPS). By tag (top 10):\n")
        top = sorted(spend["by_tag"].items(), key=lambda kv: -kv[1])[:10]
        md.append(md_table(pd.DataFrame([(k, v) for k, v in top], columns=["tag", "usd"])))
    if len(full):
        md.append(f"\nRecording: mean {full['wall_s'].mean():.0f}s per 28-turn episode (4 concurrent). Candidate scoring mean compactor latency (s): " +
                  (", ".join(f"{k} {v:.1f}" for k, v in cands.groupby('candidate')['latency_s'].mean().items() if k in ['small_prompted', 'frontier', 'threshold_pi', 'cliff']) if len(cands) else "n/a"))
    out.write_text("\n".join(md))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", default="outputs/pilot")
    ap.add_argument("--matrix", default=None)
    ap.add_argument("--out", default="outputs/pilot_report.md")
    a = ap.parse_args()
    print(build(Path(a.pilot), Path(a.matrix) if a.matrix else None, Path(a.out)))
