# CompactGym

**An RL environment for learning when and how an agent should forget.**

An email-and-calendar assistant works through a long session. Its context fills up, so something has to be compacted: truncated, summarized or moved to memory. CompactGym scores that choice by what the agent forgets and how the task ends, not by how the summary reads.

| | |
|---|---|
| **Domain** | Long-horizon tool use: mock inbox and calendar |
| **Policy being trained** | A small *compaction controller* (not the main agent, which stays frozen) |
| **Episode** | One multi-turn user session, ~80-90 agent steps, contexts of 25-50K tokens |
| **Reward** | Task success, minus pause, forgetting, cost and information loss |
| **Verifier** | Rule-based: mailbox and calendar diff, plus planted-fact detectors (no LLM judge needed for success) |
| **Baselines** | 10 compactors, including the prompts used by Anthropic, Codex and pi-mono |
| **Status** | Pilot done; SFT data pipeline validated on 10 seeds; RL not started |

## How it works

```
 user session (simulated user, scripted goals + planted facts)
        │
        ▼
 ┌─────────────────────────┐   context over budget / threshold
 │  frozen main agent      │ ───────────────────────────────┐
 │  (email + calendar tools)│                                │
 └─────────────────────────┘                                ▼
        ▲                                  ┌──────────────────────────────┐
        │  compacted context               │  COMPACTION CONTROLLER       │
        └──────────────────────────────────│  (the policy you train)      │
                                           └──────────────────────────────┘
        │
        ▼
 success check (mailbox/calendar diff) + forgetting detectors → reward
```

1. **Reset.** A seeded mailbox and calendar are generated, plus a task and a list of facts the simulated user will state during the session (instructions, corrections, revised dates, decisions, reference numbers).
2. **Step.** The frozen main agent works through the user's turns using deterministic tools.
3. **Compaction decision.** Before a model call, if the context hits the threshold, the controller observes the state and acts.
4. **Resume.** The agent continues from the compacted context: system prompt, pinned facts, summary, and the last *k* turns.
5. **Score.** At the end, the verifier checks whether the goal was met, whether any planted facts were violated, and what it all cost.

## Observation space (what the controller sees)

- The conversation to be compacted: everything except the last *k* turns, which are always kept.
- Context size against the token budget (percent full), and the number of compactions so far.
- The facts already pinned.

## Action space

The controller replies with one JSON object:

```json
{"action": "summarize_older", "summary": "...", "pinned": ["exact fact", "..."], "confidence": 0.9}
```

| `action` | Effect |
|---|---|
| `keep` | Do nothing |
| `truncate_verbatim` | Drop long tool outputs, shorten old tool calls, no rewriting |
| `clear_tool_outputs` | Blank old tool outputs, keep every message |
| `summarize_older` | Replace all but the last *k* turns with a written summary |
| `rewrite` | Like summarize, but keep only the last turn |
| `offload_to_memory` | Move old turns to external memory; the agent can `recall(query)` them |
| `defer_to_frontier` | Ask a large model to write the summary (accurate, slow, expensive) |

`pinned` is a verbatim tier that survives every future compaction. `confidence` is the controller's estimate that nothing needed was lost; it is calibrated against the measured retention.

## Reward

```
R = success − w_p · pause − w_a · forgetting − w_c · cost − λ · InfoLoss + w_r · retention
```

| Term | Measured by |
|---|---|
| `success` | Mailbox and calendar diff against the goal |
| `pause` | Seconds the user waits while compaction blocks |
| `forgetting` | Planted facts *newly* violated relative to the full-context run |
| `cost` | Uncached, cached and output tokens, priced per model |
| `InfoLoss` | How much worse the agent predicts its next turns from the compacted context than from the full one. Signed: the compacted context can be easier to continue from, and that is kept as a small bonus |
| `retention` | Dense term: share of planted facts a probe still answers correctly right after compaction. Gives the controller an immediate signal instead of waiting for the episode outcome |

Weights come from a profile (`default`, `cost_sensitive`, `safety_first`) so the same environment can train different tradeoffs.

## What makes it a good gym

- **Counterfactual branching.** Any step can be snapshotted and continued under *G* different compactions with identical seeds. The outcome difference is the causal effect of the compaction itself, which gives a clean group-relative baseline for RL.
- **Facts you can audit.** Forgetting is checked against facts we planted, not judged by a model.
- **Latency and cache are in the reward.** Summaries block the user and break the prompt cache; pure quality metrics miss that.
- **Repeated compaction.** The environment measures how error compounds across compactions (40 events deep).
- **Cheap to run.** About $0.07 to record a full episode, and about $0.24 per seed for a full best-of-8 scoring pass, at the placeholder prices in the code.

## Hard mode

`COMPACTGYM_SPEC=hard COMPACTGYM_BUDGET=8000` switches every runner stage to a harder task, because at 16K almost nothing was forgotten:

- facts are planted in turns 1-3 and only exercised in the last 40% of the session, so they have to survive many compactions (about 17 per branch at 8K);
- a forgotten fact is unrecoverable: the simulated user never restates it when asked again;
- a goal scores 0 if the fact it exercises was violated in that turn, so forgetting changes success.

On 60 live branches this raised newly forgotten facts from 0.2 to 0.43 per branch (33% of branches forget at least one).

## Baselines (the leaderboard to beat)

| Arm | Idea |
|---|---|
| `cliff` | Structure-preserving truncation, no rewriting. The strongest so far |
| `threshold_pi` / `threshold_anthropic` / `threshold_codex` | Summarize at 80% full with the pi-mono, Anthropic-docs and Codex CLI prompts |
| `recent_k` | Keep the last 5 turns, drop the rest |
| `constraint_extractor` | `threshold_pi` plus pinned user constraints |
| `memory_offload` | Reversible offload plus a `recall` tool |
| `frontier` | A large model writes the summary |
| `small_prompted` | The untrained small model as controller |
| `never` | Compact only when forced |

## Pilot results (honest summary)

- Best-of-8 compactions cut information loss 41% against one small-model attempt in the pilot, and 25% in the 10-seed trial.
- Task success barely differs between arms (0.79-0.87, not significant).
- `cliff` has the lowest information loss, the highest retention, zero pause and the lowest cost.
- InfoLoss correlates weakly with live forgetting and failure (Spearman 0.18 across branch points, 0.03 within), so it is not yet a trustworthy reward on its own.
- In hard mode (60 live branches) a single compaction's InfoLoss correlated *negatively* with the facts the whole branch later forgot (rho -0.33, 95% CI -0.54 to -0.08), and its retention not at all (+0.01). One compaction's score does not predict a 17-compaction trajectory, so the default profile now sets `lambda` to 0 and relies on the measured outcome (success, newly violated facts) instead.

**Verdict:** train a router with SFT first, and validate the reward before running RL.

## Quickstart

```bash
export FIREWORKS_API_KEY=...
uv sync --extra dev
uv run pytest                                              # 61 offline tests
uv run python -m compactgym.runner.pilot record --start 1000 --n 1010
uv run python -m compactgym.runner.pilot score  --start 1000 --n 1010   # 8 samples + baselines per branch point
uv run python scripts/build_sft.py --root outputs/pilot                 # best-of-N SFT targets
```

## Known limits

- The frontier arm is a stand-in (`glm-5p3`), not a closed model.
- Prices in the code are placeholders; re-price from the logged token counts.
- Live-branch coverage is thin (84 of 240 planned).
- At a 16K budget compaction rarely changes the outcome. The next version should use 8K and longer horizons.
