"""Async model client (Fireworks/OpenAI-compatible + Anthropic), JSONL call log, spend tracker."""
from __future__ import annotations

import asyncio
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

from .config import Config, ModelSpec

BASE_URLS = {
    "fireworks": "https://api.fireworks.ai/inference/v1",
    "openai": "https://api.openai.com/v1",
}


class SpendLimitExceeded(RuntimeError):
    pass


class Usage(BaseModel):
    prompt: int = 0
    cached: int = 0
    completion: int = 0

    def cost(self, spec: ModelSpec) -> float:
        return ((self.prompt - self.cached) * spec.price_in + self.cached * spec.price_cached
                + self.completion * spec.price_out) / 1e6


class ChatResult(BaseModel):
    text: str = ""
    reasoning: str = ""
    tool_calls: list[dict] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    latency_s: float = 0.0
    finish_reason: str | None = None
    model: str = ""


class SpendTracker:
    def __init__(self, limit_usd: float):
        self.limit = limit_usd
        self.total = 0.0
        self.by_tag: dict[str, float] = {}
        self.n_calls = 0

    def add(self, spec: ModelSpec, usage: Usage, tag: str) -> float:
        c = usage.cost(spec)
        self.total += c
        self.n_calls += 1
        self.by_tag[tag] = self.by_tag.get(tag, 0.0) + c
        if self.total > self.limit:
            raise SpendLimitExceeded(f"spend ${self.total:.2f} exceeded limit ${self.limit:.2f}; ask the user before continuing")
        return c


class CallLogger:
    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def log(self, rec: dict) -> None:
        if not self.path:
            return
        async with self._lock:
            with open(self.path, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")


class LLM:
    def __init__(self, cfg: Config, logger: CallLogger | None = None, spend: SpendTracker | None = None):
        self.cfg = cfg
        self.logger = logger or CallLogger(Path(cfg.log_dir) / "calls.jsonl")
        self.spend = spend or SpendTracker(cfg.max_spend_usd)
        self.sem = asyncio.Semaphore(cfg.concurrency)
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=20.0),
                                             limits=httpx.Limits(max_connections=64))
        return self._client

    async def aclose(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    def _key(self, spec: ModelSpec) -> str:
        k = os.environ.get(spec.api_key_env)
        if not k:
            raise RuntimeError(f"{spec.api_key_env} is not set")
        return k

    async def _post(self, url: str, headers: dict, body: dict, retries: int = 14) -> tuple[dict, float]:
        """POST with patient retries: 429/5xx/timeouts back off exponentially (honouring Retry-After) for several minutes before giving up."""
        err: Exception | None = None
        for i in range(retries):
            t0 = time.perf_counter()
            delay = min(60.0, 1.6 ** i + random.random())
            try:
                async with self.sem:
                    # hard wall-clock cap per attempt: stale keep-alive connections (after the machine slept or the network flapped) can hang past httpx timeouts
                    r = await asyncio.wait_for(self.client.post(url, headers=headers, json=body), timeout=300)
                dt = time.perf_counter() - t0
                if r.status_code == 200:
                    return r.json(), dt
                if r.status_code in (408, 409, 425, 429) or r.status_code >= 500:
                    err = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                    ra = r.headers.get("retry-after")
                    if ra and ra.replace(".", "", 1).isdigit():
                        delay = max(delay, min(90.0, float(ra)))
                else:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:400]}")
            except (httpx.TimeoutException, httpx.TransportError, asyncio.TimeoutError) as e:
                err = e
                if self._client is not None:  # drop possibly dead pooled connections
                    old, self._client = self._client, None
                    asyncio.ensure_future(old.aclose())
            await asyncio.sleep(delay)
        raise RuntimeError(f"request failed after {retries} tries: {err}")

    # ------------------------------------------------------------------ chat
    async def chat(self, spec: ModelSpec, messages: list[dict], tools: list[dict] | None = None, *,
                   tag: str = "", session: str | None = None, max_tokens: int | None = None,
                   temperature: float | None = None, json_mode: bool = False) -> ChatResult:
        if spec.provider == "anthropic":
            res = await self._chat_anthropic(spec, messages, max_tokens, temperature)
        else:
            res = await self._chat_openai(spec, messages, tools, session, max_tokens, temperature, json_mode)
        cost = self.spend.add(spec, res.usage, tag)
        await self.logger.log({
            "ts": time.time(), "kind": "chat", "tag": tag, "model": spec.model, "session": session,
            "n_messages": len(messages), "input": messages, "tools": [t["function"]["name"] for t in tools or []],
            "output": res.text, "reasoning": res.reasoning[:2000], "tool_calls": res.tool_calls,
            "prompt_tokens": res.usage.prompt, "cached_tokens": res.usage.cached,
            "completion_tokens": res.usage.completion, "latency_s": res.latency_s, "cost_usd": cost,
        })
        return res

    async def _chat_openai(self, spec, messages, tools, session, max_tokens, temperature, json_mode) -> ChatResult:
        body: dict[str, Any] = {
            "model": spec.model, "messages": messages,
            "max_tokens": max_tokens or spec.max_tokens,
            "temperature": spec.temperature if temperature is None else temperature,
        }
        if spec.seed is not None:
            body["seed"] = spec.seed
        if spec.reasoning_effort:
            body["reasoning_effort"] = spec.reasoning_effort
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self._key(spec)}", "Content-Type": "application/json"}
        if session and spec.provider == "fireworks":
            headers["x-session-affinity"] = session
        d, dt = await self._post(BASE_URLS[spec.provider] + "/chat/completions", headers, body)
        ch = d["choices"][0]
        m = ch["message"]
        u = d.get("usage") or {}
        return ChatResult(
            text=m.get("content") or "", reasoning=m.get("reasoning_content") or m.get("reasoning") or "",
            tool_calls=m.get("tool_calls") or [],
            usage=Usage(prompt=u.get("prompt_tokens", 0),
                        cached=(u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0,
                        completion=u.get("completion_tokens", 0)),
            latency_s=dt, finish_reason=ch.get("finish_reason"), model=spec.model)

    async def _chat_anthropic(self, spec, messages, max_tokens, temperature) -> ChatResult:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        msgs = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] in ("user", "assistant")]
        body = {"model": spec.model, "max_tokens": max_tokens or spec.max_tokens, "messages": msgs,
                "temperature": spec.temperature if temperature is None else temperature}
        if system:
            body["system"] = system
        headers = {"x-api-key": self._key(spec), "anthropic-version": "2023-06-01", "content-type": "application/json"}
        d, dt = await self._post("https://api.anthropic.com/v1/messages", headers, body)
        u = d.get("usage") or {}
        text = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")
        cached = u.get("cache_read_input_tokens", 0) or 0
        return ChatResult(text=text, usage=Usage(prompt=u.get("input_tokens", 0) + cached, cached=cached,
                                                 completion=u.get("output_tokens", 0)),
                          latency_s=dt, finish_reason=d.get("stop_reason"), model=spec.model)

    # -------------------------------------------------------------- logprobs
    async def prompt_logprobs(self, spec: ModelSpec, token_ids: list[int], *, tag: str = "logprobs") -> list[float]:
        """Per-token logprobs of the prompt itself (echo). Index 0 is 0.0 (no BOS conditioning)."""
        body = {"model": spec.model, "prompt": token_ids, "max_tokens": 1, "echo": True, "logprobs": 1, "temperature": 0}
        headers = {"Authorization": f"Bearer {self._key(spec)}", "Content-Type": "application/json"}
        d, dt = await self._post(BASE_URLS[spec.provider] + "/completions", headers, body)
        lp = d["choices"][0]["logprobs"]["token_logprobs"]
        u = d.get("usage") or {}
        usage = Usage(prompt=u.get("prompt_tokens", len(token_ids)),
                      cached=(u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0, completion=1)
        cost = self.spend.add(spec, usage, tag)
        await self.logger.log({"ts": time.time(), "kind": "logprobs", "tag": tag, "model": spec.model,
                               "prompt_tokens": usage.prompt, "cached_tokens": usage.cached,
                               "latency_s": dt, "cost_usd": cost})
        # echo returns len(prompt)+1 tokens (the sampled one is last)
        return [x if x is not None else 0.0 for x in lp[: len(token_ids)]]

    # ------------------------------------------------------------- embeddings
    async def embed(self, texts: list[str]) -> list[list[float]]:
        spec = ModelSpec(model=self.cfg.embed_model)
        headers = {"Authorization": f"Bearer {self._key(spec)}", "Content-Type": "application/json"}
        d, _ = await self._post(BASE_URLS["fireworks"] + "/embeddings", headers, {"model": spec.model, "input": texts})
        return [x["embedding"] for x in d["data"]]
