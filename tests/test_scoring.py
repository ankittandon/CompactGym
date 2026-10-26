import asyncio

import numpy as np
import pytest

from compactgym.config import Config
from compactgym.harness.agent import EpisodeResult
from compactgym.harness.tokens import render_prompt
from compactgym.interfaces import Message
from compactgym.llm import CallLogger, SpendTracker, SpendLimitExceeded, Usage
from compactgym.scoring import infoloss as il
from compactgym.scoring.calibration import reliability
from compactgym.scoring.cost import episode_cost
from compactgym.scoring.reward import reward
from compactgym.scoring.success import success_summary
from helpers import FakeLLM, Scenario, task_for


# ---------------------------------------------------------------- success
def test_success_partial_credit():
    t = task_for("forbid_delete", turns=8)
    s = Scenario(t)
    goals = t.goals
    # reach the last turn so every goal is "reached"
    s.user("x", turn=len(t.user_turns) - 1)
    g = next(g for g in goals if g.kind == "archive_promos")
    ids = g.checks[0]["ids"]
    for i in ids[: len(ids) // 2]:
        s.call("archive", id=i)
    r = s.result()
    ss = success_summary(t, r)
    assert ss["per_goal"][g.id] == pytest.approx((len(ids) // 2) / len(ids))
    assert 0 < ss["success"] < 1
    # deleting is not an accepted way to finish when delete is forbidden
    s2 = Scenario(t)
    s2.user("x", turn=len(t.user_turns) - 1)
    for i in ids:
        s2.call("delete", id=i)
    assert success_summary(t, s2.result())["per_goal"][g.id] == 0.0
    for i in ids:
        s.call("archive", id=i)
    assert success_summary(t, s.result())["per_goal"][g.id] == 1.0


def test_success_unreached_goals_score_zero():
    t = task_for("forbid_delete", turns=8)
    s = Scenario(t)
    s.user("x", turn=0)
    ss = success_summary(t, s.result())
    assert ss["success"] <= 1 and ss["reached_turn"] == 0 and any(v == 0 for v in ss["per_goal"].values())


def test_success_calendar_and_mail_checks():
    t = task_for("stale_date", turns=8)
    s = Scenario(t)
    s.user("x", turn=7)
    s.call("create_event", title="Team Offsite", date="2026-10-29", start_time="14:00", attendees="priya.banerjee@northwind.example")
    pre = next(g for g in t.goals if g.kind == "offsite_create")
    assert success_summary(t, s.result())["per_goal"][pre.id] == 1.0
    s.call("create_event", title="Team Offsite", date="2026-10-28", start_time="14:00")  # stale duplicate
    assert success_summary(t, s.result())["per_goal"][pre.id] == 0.5


# --------------------------------------------------------------- infoloss
def _msgs(n_old):
    ms = [Message(role="system", content="You are a helper."), Message(role="user", content="start " * 3)]
    for i in range(n_old):
        ms.append(Message(role="assistant", content=None, tool_calls=[{"id": f"o{i}", "type": "function", "function": {"name": "list_emails", "arguments": "{}"}}]))
        ms.append(Message(role="tool", content="old output " * 30, tool_call_id=f"o{i}", meta={"name": "list_emails"}))
    return ms


def _cont():
    return [Message(role="assistant", content=None, tool_calls=[{"id": "n1", "type": "function", "function": {"name": "archive", "arguments": '{"id": "e0123"}'}}]),
            Message(role="tool", content="OK: archived e0123", tool_call_id="n1", meta={"name": "archive"}),
            Message(role="assistant", content="Archived it.")]


def test_infoloss_toy_constant_gap():
    full_ctx, comp_ctx, cont = _msgs(6), _msgs(1), _cont()
    n_full = len(render_prompt(full_ctx + cont, []))
    # the model is 0.5 nats/token worse whenever the (shorter) compacted context is used
    llm = FakeLLM(logprob_fn=lambda toks: [-0.1] * len(toks) if len(toks) >= n_full - 2 else [-0.6] * len(toks))
    res = asyncio.run(il.infoloss(llm, Config().main, full_ctx, comp_ctx, cont, [], gamma=0.8))
    assert res.infoloss == pytest.approx(0.5, abs=1e-6) and res.n_turns == 2 and res.n_tokens > 5
    assert {"tool_name", "args", "id_date_recipient", "text"} >= {p["kind"] for p in res.per_token}  # format tokens carry no weight


def test_infoloss_clipped_at_zero_but_raw_signed():
    full_ctx, comp_ctx, cont = _msgs(6), _msgs(1), _cont()
    n_full = len(render_prompt(full_ctx + cont, []))
    llm = FakeLLM(logprob_fn=lambda toks: [-0.6] * len(toks) if len(toks) >= n_full - 2 else [-0.1] * len(toks))
    res = asyncio.run(il.infoloss(llm, Config().main, full_ctx, comp_ctx, cont, [], gamma=0.8))
    assert res.infoloss == 0.0 and res.raw == pytest.approx(-0.5, abs=1e-6)


def test_infoloss_weights_and_discount():
    cont = _cont()
    tokens, bounds = il._render_with_spans(_msgs(1), cont, [])
    asst = [(a, b) for a, b, is_a in bounds if is_a]
    w, kinds = il._token_weights(tokens, asst[0][0] + 1, asst[0][1], il.DEFAULT_WEIGHTS)
    assert "tool_name" in kinds and "id_date_recipient" in kinds and "args" in kinds
    assert max(w) == il.DEFAULT_WEIGHTS["id_date_recipient"]
    # discount: identical deltas, loss concentrated in the later turn counts less than in the earlier turn
    def part(delta_turn):
        full = {"nll": [1.0, 1.0], "w": [1.0, 1.0], "turn": [0, 1], "kind": ["text"] * 2, "tok": ["a", "b"]}
        comp = {**full, "nll": [1.0 + (delta_turn == 0), 1.0 + (delta_turn == 1)]}
        return il.combine(full, comp, gamma=0.5).infoloss
    assert part(0) > part(1) > 0
    # configurable weights change the aggregate
    full = {"nll": [1.0, 1.0], "w": [1.0, 4.0], "turn": [0, 0], "kind": ["text", "id_date_recipient"], "tok": ["a", "b"]}
    comp = {**full, "nll": [2.0, 1.0]}
    assert il.combine(full, comp).infoloss == pytest.approx(0.2)


def test_infoloss_full_baseline_cached():
    full_ctx, comp_ctx, cont = _msgs(4), _msgs(1), _cont()
    llm = FakeLLM()
    cache: dict = {}
    for _ in range(2):
        asyncio.run(il.infoloss(llm, Config().main, full_ctx, comp_ctx, cont, [], full_cache=cache, cache_key="k"))
    assert sum(1 for c in llm.calls if c["tag"] == "infoloss") == 3  # 1 full + 2 compacted


# ------------------------------------------------------------------- cost
def test_cost_model_prices_uncached_cached_output_separately():
    cfg = Config()
    cfg.main = cfg.main.model_copy(update={"price_in": 1.0, "price_cached": 0.2, "price_out": 5.0})
    cfg.small = cfg.small.model_copy(update={"price_in": 0.5, "price_cached": 0.1, "price_out": 2.0})
    steps = [
        {"prompt_tokens_api": 1000, "cached_modeled": 0, "completion_tokens": 100},
        {"prompt_tokens_api": 1200, "cached_modeled": 1100, "completion_tokens": 50},
        {"prompt_tokens_api": 500, "cached_modeled": 100, "completion_tokens": 50, "action": "summarize_older", "blocking": True,
         "compactor_latency_s": 2.0, "pause_s": 2.0, "calls": [{"role": "small", "tokens_in": 2000, "tokens_out": 300, "cached": 0}]},
    ]
    r = EpisodeResult(episode_id="e", task_id="t", compactor="x", cfg={}, steps=steps)
    c = episode_cost(r, cfg, cache_model="prefix")
    main = (1000 * 1.0 + 0 + 100 * 5) + (100 * 1.0 + 1100 * 0.2 + 50 * 5) + (400 * 1.0 + 100 * 0.2 + 50 * 5)
    comp = 2000 * 0.5 + 300 * 2.0
    assert c["main_cost"] == pytest.approx(main / 1e6) and c["compactor_cost"] == pytest.approx(comp / 1e6)
    assert c["cost"] == pytest.approx((main + comp) / 1e6) and c["pause_s"] == pytest.approx(2.0)
    # the compaction broke the cache: step 3 re-read (1250 expected - 100 cached) at the uncached premium
    assert c["reread_cost"] == pytest.approx(min(1250, 500) * 0 + max(0, min(1250, 500) - 100) * 0.8 / 1e6)
    assert c["n_compactions"] == 1


def test_non_blocking_compaction_has_no_pause():
    cfg = Config()
    steps = [{"prompt_tokens_api": 100, "cached_modeled": 0, "completion_tokens": 10, "action": "summarize_older", "blocking": False,
              "compactor_latency_s": 5.0, "pause_s": 0.0, "calls": []}]
    assert episode_cost(EpisodeResult(episode_id="e", task_id="t", compactor="x", cfg={}, steps=steps), cfg)["pause_s"] == 0.0


def test_reward_retention_and_signed_infoloss():
    prof = {"w_p": 0.0, "w_a": 0.0, "w_c": 0.0, "lambda": 1.0, "w_r": 0.5}
    base = reward(success=0.5, pause_s=0, forgetting=0, cost=0, infoloss=0.0, profile=prof, retention=0.8)
    assert base["reward"] == pytest.approx(0.5 + 0.5 * 0.8)
    assert reward(0.5, 0, 0, 0, -0.1, prof, retention=0.8)["reward"] == pytest.approx(base["reward"] + 0.1)  # signed InfoLoss is a bonus, not clipped
    assert reward(0.5, 0, 0, 0, 0.0, prof)["r_retention"] == 0.0  # no retention given -> no term
    assert reward(0.5, 0, 0, 0, 0.0, {"w_p": 0, "w_a": 0, "w_c": 0, "lambda": 1.0}, retention=1.0)["r_retention"] == 0.0  # profiles without w_r are unchanged


def test_reward_and_calibration_and_spend_limit():
    r = reward(success=0.8, pause_s=10, forgetting=2, cost=0.1, infoloss=0.2, profile={"w_p": 0.01, "w_a": 0.1, "w_c": 1.0, "lambda": 0.5}, lam=1.0)
    assert r["reward"] == pytest.approx(0.8 - 0.1 - 0.2 - 0.1 - 0.2)
    rel = reliability([0.9, 0.9, 0.1, 0.1], [1, 1, 0, 0])
    assert rel["ece"] == pytest.approx(0.1)
    assert reliability([], [])["n"] == 0
    st = SpendTracker(0.001)
    with pytest.raises(SpendLimitExceeded):
        st.add(Config().main, Usage(prompt=10_000_000, cached=0, completion=0), "x")
