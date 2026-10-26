import asyncio

import numpy as np
import pandas as pd
import pytest

from compactgym.config import Config
from compactgym.runner import plots
from compactgym.runner.common import PrecomputedCompactor, pilot_spec, score_row
from compactgym.runner.pilot import choose_bps, log_len_at
from compactgym.runner.report import boot_ci, md_table, spearman
from compactgym.runner.matrix import job_key
from compactgym.env.inbox.tasks import generate_task
from compactgym.harness.agent import Episode
from compactgym.compactors import make_compactor
from compactgym.interfaces import CompactionAction, CompactionResult, EpisodeState, Message
from helpers import FakeLLM
from test_harness import convo, main_handler


def test_stats_helpers():
    r, p, n = spearman([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
    assert r == pytest.approx(1.0) and n == 5
    assert np.isnan(spearman([1, 1, 1, 1], [1, 2, 3, 4])[0])
    lo, hi = boot_ci([1, 2, 3, 4, 5])
    assert lo < 3 < hi
    assert md_table(pd.DataFrame({"a": [1.23456], "b": ["x"]})).count("|") > 4


def test_pilot_specs_vary_and_are_deterministic():
    specs = [pilot_spec(s) for s in range(12)]
    assert {s.injection for s in specs} == {"top", "middle", "repeated", "bottom"}
    assert pilot_spec(3) == pilot_spec(3)
    t = generate_task(pilot_spec(3))
    assert len(t.facts) == 4 and len(t.user_turns) == 28


def test_score_row_end_to_end_with_fake_llm():
    cfg = Config(budget_tokens=2500)
    llm = FakeLLM(main_handler)
    task = generate_task(pilot_spec(2, n_turns=6, n_facts=2, volume=1.0))
    ep = Episode(cfg, task, make_compactor("never", cfg, llm), llm, episode_id="e", budget_tokens=2500)
    res = asyncio.run(ep.run())
    row = asyncio.run(score_row(cfg, FakeLLM(lambda *a: "YES"), task, res))
    for k in ("success", "n_facts_violated", "cost", "pause_s", "n_compactions", "retention", "retained_ratio", "reward", "conf_ece", "mean_infoloss"):
        assert k in row
    assert row["n_forced"] > 0 and row["retention"] == 1.0 and 0 <= row["retained_ratio"] < 1


def test_precomputed_compactor_applies_once_then_defers():
    cfg = Config()
    msgs = convo(3)
    pre = CompactionResult(action=CompactionAction.SUMMARIZE_OLDER, messages=msgs[:2], summary="S", pinned=["p"])
    then = make_compactor("recent_k", cfg, None)
    c = PrecomputedCompactor(cfg, None, pre, then)
    st = EpisodeState(keep_k=2, context_tokens=100)
    r1 = asyncio.run(c(msgs, st, 1000))
    assert r1.summary == "S" and st.pinned == ["p"]
    r2 = asyncio.run(c(msgs, st, 1000))
    assert r2.action == CompactionAction.KEEP  # `then` defers (below its threshold)


def test_matrix_job_key_distinguishes_jobs():
    a = dict(task_seed=1, budget=8000, compactor="cliff", run_seed=0)
    assert job_key(a) != job_key({**a, "budget": 16000}) and job_key(a) == job_key(dict(a))


def test_plots_render(tmp_path):
    rng = np.random.default_rng(0)
    names = ["cliff", "threshold_pi", "small_prompted", "frontier", "memory_offload"]
    df = pd.DataFrame({"compactor": np.repeat(names, 6), "budget": np.tile([8000, 16000], 15), "success": rng.random(30), "cost": rng.random(30) / 10,
                       "pause_s": rng.random(30) * 20, "retained_ratio": rng.random(30)})
    for fn in (plots.plot_success_vs_cost, plots.plot_success_vs_pause, plots.plot_budget_axis):
        assert (tmp_path / "x.png").parent.exists() and fn(df, tmp_path / f"{fn.__name__}.png")
    rep = pd.DataFrame({"compactor": np.repeat(["cliff", "threshold_pi", "memory_offload"], 40), "event": np.tile(np.repeat(np.arange(1, 5), 10), 3), "error": rng.random(120) > 0.5})
    rep["error"] = rep["error"].astype(float)
    assert plots.plot_repeated(rep, tmp_path / "rep.png")
    ft = pd.DataFrame({"candidate": np.repeat(["cliff", "small_prompted"], 40), "type": np.tile(plots.TYPES[:5], 16), "location": np.tile(["top", "middle", "repeated", "bottom"], 20),
                       "retained": rng.random(80) > 0.4})
    a, b = plots.plot_retention_by_type(ft, tmp_path / "a.png", tmp_path / "b.png")
    pairs = pd.DataFrame({"compactor": np.repeat(names, 8), "infoloss": rng.random(40), "violations_after": rng.integers(0, 3, 40), "success_drop": rng.random(40) - 0.3})
    assert plots.plot_infoloss_validity(pairs, tmp_path / "v.png")
    assert plots.plot_reliability({"ece": 0.2, "n": 10, "bins": [{"mean_conf": 0.8, "mean_realized": 0.5, "n": 10}]}, tmp_path / "r.png")
