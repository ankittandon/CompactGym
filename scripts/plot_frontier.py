"""Where we stand against the frontier baseline: forgetting and success against cost, same 12 hard-mode branch points for every arm.

Data: outputs/hard, outputs/hard_fr, outputs/hard_pin (paired by branch point, 8K budget). The "oracle pins" arms pin the planted facts, so they are
an upper bound on what a pin extractor could do, not a deployable method.

    uv run python scripts/plot_frontier.py [out.png]
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# arm: (success, forgotten facts per branch, cost $ per branch, pause s, kind)
ARMS = {
    "frontier":                 (0.843, 0.25, 0.074, 119.6, "frontier"),
    "cliff":                    (0.754, 0.50, 0.033, 0.1, "baseline"),
    "pi-mono summary":          (0.759, 0.67, 0.037, 35.3, "baseline"),
    "small (untrained)":        (0.758, 0.33, 0.034, 157.7, "baseline"),
    "best-of-8":                (0.786, 0.42, 0.038, 179.5, "baseline"),
    "memory offload":           (0.764, 0.92, 0.032, 0.0, "baseline"),
    "fresh session + recall":   (0.693, 1.33, 0.022, 0.0, "baseline"),
    "cliff + oracle pins":      (0.799, 0.17, 0.034, 0.1, "oracle"),
    "recall + oracle pins":     (0.790, 0.08, 0.020, 0.0, "oracle"),
}
BLUE, ORANGE, GREY, INK, MUTED, RULE = "#2a78d6", "#eb6834", "#8d99a6", "#14202b", "#566574", "#d6dee5"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})


def panel(ax, yi, ylabel, ylim, better_low):
    fc, fy = ARMS["frontier"][2], ARMS["frontier"][yi]
    # the efficiency win zone: cheaper than frontier and no worse on this metric
    if better_low:
        ax.fill_between([0.015, fc], 0, fy, color=BLUE, alpha=0.08, lw=0)
    else:
        ax.fill_between([0.015, fc], fy, ylim[1], color=BLUE, alpha=0.08, lw=0)
    ax.axvline(fc, color=ORANGE, lw=1, ls=(0, (4, 3)), alpha=0.7)
    ax.axhline(fy, color=ORANGE, lw=1, ls=(0, (4, 3)), alpha=0.7)
    for name, v in ARMS.items():
        x, y, kind = v[2], v[yi], v[4]
        if kind == "frontier":
            ax.scatter(x, y, s=120, color=ORANGE, zorder=5)
        elif kind == "oracle":
            ax.scatter(x, y, s=110, facecolor="none", edgecolor=BLUE, lw=2.2, zorder=5)
        else:
            ax.scatter(x, y, s=70, color=GREY, zorder=4, edgecolor="white", lw=1)
    ax.set_xlim(0.015, 0.085)
    ax.set_ylim(*ylim)
    ax.set_xlabel("cost per branch (US$, placeholder prices)")
    ax.set_ylabel(ylabel)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(RULE)
    ax.spines["bottom"].set_color(RULE)
    ax.grid(color=RULE, lw=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    return fc, fy


def label(ax, name, yi, dx, dy, color=INK, ha="left", weight="normal"):
    v = ARMS[name]
    ax.annotate(name, (v[2], v[yi]), (v[2] + dx, v[yi] + dy), fontsize=8.5, color=color, ha=ha, va="center", fontweight=weight)


def lead(ax, name, yi, x, y, color=MUTED, ha="left", weight="normal"):
    v = ARMS[name]
    ax.annotate(name, (v[2], v[yi]), (x, y), fontsize=8.5, color=INK if color == MUTED else color, ha=ha, va="center", fontweight=weight,
                arrowprops=dict(arrowstyle="-", color=color, lw=0.8, shrinkA=2, shrinkB=5))


fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.6), facecolor="white")
panel(a, 1, "facts forgotten per branch (lower is better)", (-0.05, 1.45), True)
panel(b, 0, "task success (higher is better)", (0.66, 0.88), False)

for ax, yi in ((a, 1), (b, 0)):
    label(ax, "frontier", yi, 0.0012, 0.045 if yi == 1 else 0.008, ORANGE, "left", "bold")
label(a, "cliff", 1, 0.0013, 0.0)
label(a, "pi-mono summary", 1, 0.0013, 0.05)
lead(a, "small (untrained)", 1, 0.0185, 0.42)
label(a, "best-of-8", 1, 0.0013, -0.06)
label(a, "memory offload", 1, 0.0013, 0.0)
label(a, "fresh session + recall", 1, 0.0013, 0.0)
label(a, "cliff + oracle pins", 1, 0.0013, 0.07, BLUE, weight="bold")
label(a, "recall + oracle pins", 1, 0.0013, 0.0, BLUE, weight="bold")
label(b, "best-of-8", 0, 0.0013, 0.0)
label(b, "fresh session + recall", 0, 0.0013, 0.0)
label(b, "cliff + oracle pins", 0, 0.0013, 0.006, BLUE, weight="bold")
lead(b, "recall + oracle pins", 0, 0.0215, 0.8105, BLUE, weight="bold")
lead(b, "memory offload", 0, 0.0475, 0.779)
lead(b, "small (untrained)", 0, 0.0475, 0.7515)
lead(b, "cliff", 0, 0.0335, 0.7405, ha="center")
lead(b, "pi-mono summary", 0, 0.0475, 0.7655)
a.text(0.0445, 0.02, "cheaper and forgets no more than frontier", fontsize=8, color=BLUE, va="bottom")
b.text(0.0165, 0.867, "cheaper and succeeds at least as often", fontsize=8, color=BLUE, va="top")

fig.suptitle("A cheap harness matches the frontier summarizer on forgetting only with perfect pins", x=0.01, ha="left", fontsize=14, fontweight="bold")
fig.text(0.01, 0.905, "Hollow blue markers use oracle pins (the planted facts), an upper bound. A real extractor has to earn them. Success is still below frontier. 12 branch points, 8K budget.",
         fontsize=9, color=MUTED, ha="left")
fig.text(0.01, 0.01, "Frontier here is glm-5p3, an open large model standing in for a closed one. Frontier also blocks the user for about 120 s per session; the other arms block for under 1 s, except small (untrained) and best-of-8 at 150-180 s.",
         fontsize=8, color=MUTED, ha="left")
fig.tight_layout(rect=(0, 0.04, 1, 0.9))
out = sys.argv[1] if len(sys.argv) > 1 else "figures/frontier_gap.png"
fig.savefig(out, dpi=200, facecolor="white")
print("wrote", out)
