"""Cost model: uncached / cached / output prices separately; modeled prefix caching so compaction's cache break is priced.

total_cost = main-model cost (prefix-cache model, includes the re-read of each compacted context once uncached)
           + compactor tokens priced at the compactor model's price.
``reread`` itemises the part of main cost caused by cache breaks. ``pause`` = wall-clock seconds of blocking compactor calls.
"""
from __future__ import annotations

from ..config import Config, ModelSpec
from ..harness.agent import EpisodeResult


def role_spec(cfg: Config, role: str | None) -> ModelSpec:
    return getattr(cfg, role or "main") if role in ("main", "small", "frontier", "judge", "user_sim") else cfg.main


def price_main(spec: ModelSpec, prompt: int, cached: int, completion: int) -> float:
    return ((prompt - cached) * spec.price_in + cached * spec.price_cached + completion * spec.price_out) / 1e6


def episode_cost(result: EpisodeResult, cfg: Config, cache_model: str | None = None, steps: list[dict] | None = None) -> dict:
    cache_model = cache_model or cfg.cache_model
    main = cfg.main
    steps = steps if steps is not None else result.steps
    main_cost = comp_cost = reread = 0.0
    pause = 0.0
    prompt_tok = cached_tok = out_tok = comp_tok_in = comp_tok_out = 0
    prev_total = 0
    n_comp = 0
    for s in steps:
        p = s.get("prompt_tokens_api") or s.get("prompt_tokens_est", 0)
        c = s.get("cached_modeled", 0) if cache_model == "prefix" else s.get("cached_api", 0)
        c = min(c, p)
        o = s.get("completion_tokens", 0)
        main_cost += price_main(main, p, c, o)
        prompt_tok, cached_tok, out_tok = prompt_tok + p, cached_tok + c, out_tok + o
        if s.get("action") not in (None, "keep") or s.get("forced"):
            n_comp += 1
            expected = prev_total
            lost = max(0, min(expected, p) - c)
            reread += lost * (main.price_in - main.price_cached) / 1e6
        prev_total = p + o
        calls = list(s.get("calls") or []) + list(s.get("calls_wasted") or [])
        for call in calls:
            sp = role_spec(cfg, call.get("role"))
            ci, co, cc = call.get("tokens_in", 0), call.get("tokens_out", 0), call.get("cached", 0)
            comp_cost += price_main(sp, ci, cc, co)
            comp_tok_in, comp_tok_out = comp_tok_in + ci, comp_tok_out + co
        if s.get("blocking", True) and s.get("compactor_latency_s"):
            pause += s.get("pause_s", s.get("compactor_latency_s", 0.0)) or 0.0
        pause += s.get("forced_latency_s", 0.0) or 0.0
    return {"cost": main_cost + comp_cost, "main_cost": main_cost, "compactor_cost": comp_cost, "reread_cost": reread, "pause_s": pause,
            "prompt_tokens": prompt_tok, "cached_tokens": cached_tok, "output_tokens": out_tok, "compactor_tokens_in": comp_tok_in,
            "compactor_tokens_out": comp_tok_out, "n_compactions": n_comp, "n_steps": len(steps)}
