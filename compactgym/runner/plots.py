"""The seven plots. Palette: dataviz reference categorical order (fixed per compactor, never cycled); >8 series add marker shapes as a
secondary encoding and every point is directly labelled. Data frames are the results tables (see report.py for their sources)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h", "*", "<"]
ORDER = ["cliff", "frontier", "threshold_pi", "threshold_anthropic", "recent_k", "constraint_extractor", "memory_offload", "small_prompted", "best8_infoloss", "never",
         "best8_retention", "small_sample"]
STYLE = {n: (SERIES[i % 8], MARKERS[i % len(MARKERS)]) for i, n in enumerate(ORDER)}
TYPES = ["Action", "Information", "Process", "Preference", "Output", "Revision", "Decision", "Reference", "Error"]


def _ax(w=6.4, h=4.4, title="", xl="", yl=""):
    fig, ax = plt.subplots(figsize=(w, h), dpi=140)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    ax.set_xlabel(xl, fontsize=9, color=MUTED)
    ax.set_ylabel(yl, fontsize=9, color=MUTED)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    return fig, ax


def _style(name):
    return STYLE.get(name, ("#52514e", "o"))


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def _label(ax, x, y, text, color):
    ax.annotate(text, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=7.2, color=INK)


def scatter_by_compactor(df, x, y, path, title, xl, yl, group="budget"):
    """One point per compactor (mean over tasks); one panel-marker size per budget when several budgets are present."""
    fig, ax = _ax(6.8, 4.6, title, xl, yl)
    sizes = {b: 28 + 22 * i for i, b in enumerate(sorted(df[group].unique()))} if group in df else {}
    for name, g in df.groupby("compactor"):
        c, m = _style(name)
        gg = g.groupby(group)[[x, y]].mean().reset_index() if group in g else g[[x, y]].mean().to_frame().T
        for _, r in gg.iterrows():
            ax.scatter(r[x], r[y], s=sizes.get(r.get(group), 40), color=c, marker=m, edgecolor="white", linewidth=1.2, zorder=3)
            _label(ax, r[x], r[y], name + (f" {int(r[group] // 1000)}K" if len(sizes) > 1 else ""), c)
    return _save(fig, path)


def plot_success_vs_cost(df, path):
    return scatter_by_compactor(df, "cost", "success", path, "Success vs cost per task", "cost per task (USD, cache-aware)", "task success")


def plot_success_vs_pause(df, path):
    return scatter_by_compactor(df, "pause_s", "success", path, "Success vs total pause time", "blocking pause per task (s)", "task success")


def plot_repeated(rep: pd.DataFrame, path):
    """error (1 - retention of planted facts) against the number of compaction events."""
    fig, ax = _ax(6.4, 4.4, "Repeated-compaction test", "compaction events so far", "error (1 - fact retention)")
    for name, g in rep.groupby("compactor"):
        c, m = _style(name)
        s = g.groupby("event")["error"].agg(["mean", "count", "std"]).reset_index()
        s = s[(s["count"] >= 12) & (s["event"] <= 40)]
        ax.plot(s["event"], s["mean"], color=c, marker=m, lw=2, ms=6, mec="white", label=name)
        se = s["std"].fillna(0) / np.sqrt(s["count"])
        ax.fill_between(s["event"], s["mean"] - 1.96 * se, s["mean"] + 1.96 * se, color=c, alpha=0.12, lw=0)
        if len(s):
            _label(ax, s["event"].iloc[-1], s["mean"].iloc[-1], name, c)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8, frameon=False)
    return _save(fig, path)


def plot_retention_by_type(cands: pd.DataFrame, path_type, path_loc):
    """cands: one row per (candidate, fact) with columns candidate, type, location, retained."""
    names = [n for n in ["cliff", "frontier", "threshold_pi", "small_prompted", "best8_infoloss"] if n in set(cands["candidate"])]
    fig, ax = _ax(9.0, 4.4, "Retention by fact type", "", "fraction of planted facts present after compaction")
    w = 0.8 / max(1, len(names))
    for i, n in enumerate(names):
        g = cands[cands["candidate"] == n].groupby("type")["retained"].mean().reindex(TYPES)
        ax.bar(np.arange(len(TYPES)) + i * w, g.values, w * 0.92, color=_style(n)[0], label=n)
    ax.set_xticks(np.arange(len(TYPES)) + 0.4 - w / 2)
    ax.set_xticklabels(TYPES, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, frameon=False, ncol=3)
    p1 = _save(fig, path_type)
    locs = ["top", "middle", "repeated", "bottom"]
    fig, ax = _ax(7.2, 4.2, "Retention by injection location", "", "fraction of planted facts present")
    for i, n in enumerate(names):
        g = cands[cands["candidate"] == n].groupby("location")["retained"].mean().reindex(locs)
        ax.bar(np.arange(len(locs)) + i * w, g.values, w * 0.92, color=_style(n)[0], label=n)
    ax.set_xticks(np.arange(len(locs)) + 0.4 - w / 2)
    ax.set_xticklabels(locs)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, frameon=False, ncol=3)
    return p1, _save(fig, path_loc)


def plot_infoloss_validity(pairs: pd.DataFrame, path):
    """pairs: one row per (branch point, compactor): infoloss, violations_after, success_drop."""
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), dpi=140)
    for ax, y, yl in zip(axes, ["violations_after", "success_drop"], ["planted-fact violations after the branch point", "success drop vs full-context continuation"]):
        ax.set_title(f"InfoLoss vs {'forgetting' if y == 'violations_after' else 'failure'}", loc="left", fontsize=10.5, color=INK)
        ax.set_xlabel("InfoLoss of the compaction (nats / weighted token)", fontsize=9, color=MUTED)
        ax.set_ylabel(yl, fontsize=9, color=MUTED)
        ax.grid(True, color=GRID, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8.5)
        rng = np.random.default_rng(0)
        for name, g in pairs.groupby("compactor"):
            c, m = _style(name)
            ax.scatter(g["infoloss"], g[y] + rng.normal(0, 0.03, len(g)), color=c, marker=m, s=26, alpha=0.8, edgecolor="white", linewidth=0.6, label=name)
        d = pairs.dropna(subset=["infoloss", y])
        if len(d) > 3:
            z = np.polyfit(d["infoloss"], d[y], 1)
            xs = np.linspace(d["infoloss"].min(), d["infoloss"].max(), 20)
            ax.plot(xs, np.polyval(z, xs), color=INK, lw=1.2, ls="--")
    axes[0].legend(fontsize=7, frameon=False, ncol=2)
    return _save(fig, path)


def plot_reliability(rel: dict, path, title="Confidence reliability (small_prompted)"):
    fig, ax = _ax(4.8, 4.6, title, "stated confidence", "realized retention")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    b = rel.get("bins", [])
    if b:
        ax.plot([x["mean_conf"] for x in b], [x["mean_realized"] for x in b], color=SERIES[0], marker="o", lw=2, mec="white")
        for x in b:
            ax.annotate(f"n={x['n']}", (x["mean_conf"], x["mean_realized"]), textcoords="offset points", xytext=(5, -10), fontsize=7)
    ax.text(0.02, 0.95, f"ECE = {rel['ece']:.2f}  (n={rel['n']})" if rel.get("ece") is not None else "no confidence reported", transform=ax.transAxes, fontsize=8.5, color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return _save(fig, path)


def plot_budget_axis(df, path):
    """tokens retained per token of history (x) against success (y), one point per compactor."""
    return scatter_by_compactor(df, "retained_ratio", "success", path, "Success vs tokens retained per token of history", "context after compaction / context before", "task success", group="budget")
