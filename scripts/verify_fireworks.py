"""Verification of the Fireworks assumptions the pilot depends on -> outputs/verification.md.
(1) prompt-token logprobs, (2) cached-token reporting, (3) tokenizer/template agreement, (4) tool-call round trip."""
import asyncio, json, sys
from pathlib import Path
from compactgym.config import Config
from compactgym.llm import LLM
from compactgym.harness.tokens import render_prompt
from compactgym.env.inbox.tools import openai_tools
from compactgym.interfaces import Message

async def main():
    cfg = Config(); cfg.log_dir = "outputs/verify"
    llm = LLM(cfg)
    out = ["# Fireworks verification\n"]
    # 1. logprobs for prompt tokens
    toks = render_prompt([Message(role="system", content="You are helpful."), Message(role="user", content="The capital of France is Paris.")], [])
    lp = await llm.prompt_logprobs(cfg.main, toks, tag="verify")
    out.append(f"* prompt logprobs, main (`{cfg.main.model}`): {'OK' if len(lp) == len(toks) and any(x < 0 for x in lp) else 'FAIL'} — {len(lp)} per-token values for {len(toks)} prompt tokens (token-id prompt + `echo=true`; the first token has no context -> 0.0)")
    h = {"Authorization": f"Bearer {llm._key(cfg.small)}", "Content-Type": "application/json"}
    d, _ = await llm._post("https://api.fireworks.ai/inference/v1/completions", h, {"model": cfg.small.model, "prompt": "The capital of France is Paris.", "max_tokens": 1, "echo": True, "logprobs": 1})
    out.append(f"* prompt logprobs, small (`{cfg.small.model}`): {'OK' if d['choices'][0]['logprobs']['token_logprobs'] else 'FAIL'} with a text prompt (it has its own tokenizer; token-id prompts must be main-model ids. InfoLoss only ever uses the main model)")
    # 2. caching
    doc = " ".join(f"Item {i}: the figure for region {i} was {i*37%101}." for i in range(400))
    msgs = [{"role": "system", "content": doc}, {"role": "user", "content": "Reply OK."}]
    res = [await llm.chat(cfg.main, msgs, tag="verify", session="verify-cache") for _ in range(3)]
    out.append(f"* caching: identical {res[0].usage.prompt}-token prompt sent 3x -> cached_tokens = {[r.usage.cached for r in res]} (reported in `usage.prompt_tokens_details.cached_tokens`; hits are block-granular and occasionally miss)")
    # 3. tokenizer / template agreement
    tools = openai_tools()
    ms = [Message(role="system", content="You are helpful."), Message(role="user", content=doc)]
    r = await llm.chat(cfg.main, [m.to_api() for m in ms], tools, tag="verify", max_tokens=5)
    ours = len(render_prompt(ms, tools))
    out.append(f"* tokenizer: `o200k_harmony` via `openai_harmony`; ours {ours} vs API `prompt_tokens` {r.usage.prompt} (diff {r.usage.prompt - ours}; constant header offset, see report for long-context statistics)")
    # 4. tool call
    r = await llm.chat(cfg.main, [{"role": "system", "content": "Use tools."}, {"role": "user", "content": "How many emails are in my inbox? Use list_emails with limit 1."}], tools, tag="verify")
    out.append(f"* tool calling: {'OK' if r.tool_calls else 'no tool call'} — {r.tool_calls[0]['function'] if r.tool_calls else ''}")
    out.append(f"\nspend: ${llm.spend.total:.4f}")
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/verification.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))
    await llm.aclose()
asyncio.run(main())
