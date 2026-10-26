# CompactGym

An RL environment for training a small, fast **per-turn compaction controller**. On every turn of an agent's run the controller decides
whether to compact, how (keep / verbatim truncation / clear tool outputs / summarize / rewrite / offload to memory / defer to a frontier model)
and what must survive. It is scored on what the user experiences: task success, forgotten instructions, pauses and cache-aware cost.

This repo is the core build + pilot (go/no-go). Training (SFT on best-of-N, then RL) is the next handoff: see `NEXT_STEPS.md`.
The pilot result is in `outputs/pilot_report.md` (regenerate with `uv run python -m compactgym.runner.report`).

## Layout

```
compactgym/
  interfaces.py        Message, CompactionAction, CompactionResult, EpisodeState, Compactor  (stable; training builds on them)
  config.py            models (main / small / frontier / user_sim / judge) and prices are config values; profiles/*.json hold reward weights
  llm.py               async client (Fireworks/OpenAI-compatible + Anthropic), JSONL call log, spend tracker, patient retries
  env/inbox/           data.py (seeded mailbox+calendar), tools.py (deterministic tools), tasks.py (planted facts, goals, script), user.py (simulated user)
  harness/             agent.py (loop, per-turn hook, budget, background compaction, branch), tokens.py (exact harmony token counts),
                       resume.py (system + pinned + summary + last k turns), memory.py (BM25 + recall), snapshots.py
  compactors/          one file per strategy + registry (llmlingua2 intentionally skipped: needs a local model)
  scoring/             infoloss, cost, forgetting, success, retention, reward, calibration
  runner/              pilot.py, matrix.py, repeated.py, plots.py, report.py
  third_party/         vendored CONPACT + CliffCompaction code (MIT, licenses and NOTICE.md alongside)
tests/                 55 offline tests (a fake LLM), run with `uv run pytest`
outputs/               gitignored
```

## Running

```bash
export FIREWORKS_API_KEY=...   # ANTHROPIC_API_KEY / OPENAI_API_KEY only if you switch the frontier compactor to a closed model
uv sync --extra dev
uv run pytest
uv run python scripts/verify_fireworks.py                   # logprobs / caching / tokenizer / tool-call checks -> outputs/verification.md
uv run python -m compactgym.runner.pilot record --n 50      # 1. full-context runs (+ snapshots at every turn)
uv run python -m compactgym.runner.pilot select             #    choose branch points (context >= 16K)
uv run python -m compactgym.runner.pilot score              # 2. InfoLoss / retention / compliance of candidates at each branch point
uv run python -m compactgym.runner.pilot resume             # 3. continue branches live at 16K under every baseline
uv run python -m compactgym.runner.repeated                 #    repeated-compaction test
uv run python -m compactgym.runner.matrix --budgets 8000 16000 32000   # live matrix (plots 1, 2, 7 across budgets)
uv run python -m compactgym.runner.report --pilot outputs/pilot --matrix outputs/matrix
```

Every stage is resumable. A spend cap (`--max-spend`, default $25) hard-stops a run; ask before raising it.

## Key design decisions

* **Turn** = one main-model call. A *user turn* ends when the agent replies without a tool call; the simulated user then sends the next scripted message.
* **Compaction hook** runs before every main-model call. A compactor that leaves the context over budget is overridden by a deterministic fallback
  (logged as `forced`). `background=True` runs the compactor concurrently with the agent's turn and applies it next turn (`blocking=False`); a result
  whose input prefix changed in the meantime is discarded as stale.
* **Pinned tier**: facts a compactor returns in `pinned` are kept verbatim outside any summary and survive repeated compaction (`Config.pinned_tier` ablates it;
  `pin_oracle` pins ground truth for upper-bound diagnostics).
* **Cost** uses a deterministic prefix-cache model (everything unchanged since the previous request is cached), so a compaction's cache break is priced; the
  API-reported cached tokens are logged too (`Config.cache_model`). Prices are config values - re-price any run post hoc from the logged token counts.
* **InfoLoss** scores the model's own recorded next-k turns under compacted vs full context using prompt-token logprobs; only message bodies and tool names
  are scored (channel/recipient header tokens are excluded because in-context examples of the format shrink after compaction and would swamp the signal).
* **Forgetting** counts *distinct planted facts violated* (a fact the agent ignores even with full context would otherwise swamp the signal); branch
  comparisons use facts newly violated relative to the full-context run.
