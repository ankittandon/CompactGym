"""Confidence calibration: predicted confidence vs realized retention / (1 - InfoLoss)."""
from __future__ import annotations

import numpy as np


def reliability(conf: list[float], realized: list[float], bins: int = 5) -> dict:
    c, r = np.asarray(conf, float), np.asarray(realized, float)
    ok = ~np.isnan(c) & ~np.isnan(r)
    c, r = c[ok], r[ok]
    if len(c) == 0:
        return {"ece": None, "n": 0, "bins": []}
    edges = np.linspace(0, 1, bins + 1)
    out, ece = [], 0.0
    for i in range(bins):
        m = (c >= edges[i]) & ((c < edges[i + 1]) if i < bins - 1 else (c <= edges[i + 1]))
        if m.any():
            out.append({"lo": edges[i], "hi": edges[i + 1], "n": int(m.sum()), "mean_conf": float(c[m].mean()), "mean_realized": float(r[m].mean())})
            ece += m.mean() * abs(c[m].mean() - r[m].mean())
    return {"ece": float(ece), "n": int(len(c)), "bins": out, "mae": float(np.abs(c - r).mean())}
