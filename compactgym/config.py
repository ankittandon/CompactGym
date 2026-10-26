"""Configuration. Model ids, prices and weights are config values, never hardcoded in logic."""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

FW = "accounts/fireworks/models/"


class ModelSpec(BaseModel):
    provider: str = "fireworks"  # fireworks | openai | anthropic
    model: str
    temperature: float = 0.0
    max_tokens: int = 2048
    reasoning_effort: str | None = None
    # $ per 1M tokens
    price_in: float = 0.15
    price_cached: float = 0.03  # default: 20% of uncached (5x ratio; Cliff reference)
    price_out: float = 0.60
    seed: int | None = 0

    @property
    def api_key_env(self) -> str:
        return {"fireworks": "FIREWORKS_API_KEY", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}[self.provider]


def _fw(name: str, **kw) -> ModelSpec:
    return ModelSpec(provider="fireworks", model=FW + name, **kw)


class Config(BaseModel):
    # Prices are UNVERIFIED placeholders (Fireworks inference price page is not machine readable);
    # token counts are logged so cost can be re-priced post hoc (scoring/cost.py).
    main: ModelSpec = Field(default_factory=lambda: _fw("gpt-oss-120b", reasoning_effort="low", max_tokens=1024))
    small: ModelSpec = Field(default_factory=lambda: _fw("nemotron-lightning-3p5-30b-a3b", max_tokens=4500,  # reasoning model: needs room to think + answer
                                                         price_in=0.06, price_cached=0.012, price_out=0.24))
    # NOTE: no ANTHROPIC/OPENAI key in this environment -> default frontier stand-in is a large open model.
    frontier: ModelSpec = Field(default_factory=lambda: _fw("glm-5p3", reasoning_effort="low", max_tokens=3000,
                                                            price_in=0.60, price_cached=0.12, price_out=2.20))
    user_sim: ModelSpec = Field(default_factory=lambda: _fw("gpt-oss-120b", reasoning_effort="low", max_tokens=400))
    judge: ModelSpec = Field(default_factory=lambda: _fw("gpt-oss-120b", reasoning_effort="low", max_tokens=300))
    embed_model: str = FW + "qwen3-embedding-8b"

    budget_tokens: int = 16000
    keep_k: int = 2
    max_agent_steps: int = 80
    threshold_frac: float = 0.8
    pinned_tier: bool = True  # ablation switch
    pin_oracle: bool = False  # diagnostic: pin ground-truth fact texts
    tool_output_scale: float = 1.0
    profile: str = "default"
    cache_model: str = "prefix"  # "prefix" (deterministic modeled) | "reported" (API usage)

    log_dir: str = "outputs/logs"
    max_spend_usd: float = 25.0
    concurrency: int = 8
    prompt_overhead_tokens: int = 11  # constant gap harmony-render vs Fireworks chat template; plus 5 per tool message (tokens.TOOL_MSG_OVERHEAD) -> exact

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        if path is None:
            return cls()
        return cls.model_validate(json.loads(Path(path).read_text()))


def load_profile(name: str) -> dict:
    p = Path(__file__).parent / "profiles" / f"{name}.json"
    return json.loads(p.read_text())
