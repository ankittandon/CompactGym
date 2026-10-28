# Experiment design: budgeted compaction on LoCoMo / LongMemEval

**Status: design only, nothing built or run.** Dataset facts below come from search summaries, not the official dataset cards. Verify them before relying on them.

## Question

When a conversation is longer than the model's window, which compaction policy keeps the most answerable information under a fixed token budget, and at what cost in tokens and waiting time?

This is the setting where `cliff` should have no advantage. Dialogue has no long tool outputs to drop, so a method that wins by truncating tool results should lose its edge (a hypothesis, not a finding).

## Data

| Dataset | Use | Notes |
|---|---|---|
| **LoCoMo** | Test set | Long dialogues (reported 300-600 turns, about 9K-26K tokens, 19-35 sessions) with questions in categories such as single-hop, multi-hop, temporal and open-domain. Small, so evaluation only. |
| **LongMemEval** | Training set and second test | 500 questions over scalable chat histories. Includes **knowledge updates** (our Revision facts) and abstention. |

Split by conversation, never by question. LoCoMo is widely used, so the reader model may have seen it. Report that as a caveat.

## Setup

1. Stream the conversation in order, session by session. Speaker A plays the user and speaker B the assistant.
2. Whenever the context passes the threshold, the compactor acts and produces the compacted context.
3. After the last session, ask every question with **only the compacted context** (at most `B` tokens) as memory.
4. One fixed frozen reader model answers every question for every arm, so arms differ only in what they kept.
5. Budgets: 2K, 4K and 8K. Conversations are 1-13 times larger than the budget.

## Arms

| Arm | Notes |
|---|---|
| Full context | Upper bound: no budget |
| Last-k turns (`recent_k`) | Lower bound: pure truncation |
| `cliff` | Expected to behave like truncation here, since there are no tool results to drop. Report that. |
| `threshold_pi`, `threshold_anthropic`, `threshold_codex` | Summarize older turns with each published prompt |
| `frontier` | Large model writes the summary |
| `small_prompted` | Small model as controller, with pinned facts |
| `memory_offload` | Offload plus a `recall(query)` tool: effectively retrieval |
| Published memory systems | Take reported numbers only as context, not as a comparison (see below) |

## Metrics

Reported in the same shape as the memory-system papers, so results can be read next to theirs:

- **Answer quality:** token F1 against the gold answer (LoCoMo's own metric) **and** an LLM-judge accuracy, per question category and overall.
- **Cost:** tokens read and written by the compactor plus the reader, in dollars.
- **Latency:** compaction pause, in seconds.
- **Ours, on top:**
  - accuracy against the number of compactions;
  - accuracy on knowledge-update questions, since stale-value errors matter most;
  - the share of questions answered with a *stale* value.

Confidence intervals by bootstrap over conversations, then over questions within them.

## Controls and pitfalls

- Published numbers disagree widely: full-context is reported at 18.7 F1 in one paper and as the top method in another, and there is a reported methodology dispute. **Re-run every baseline ourselves** with identical reader, prompts and judge, and publish the prompts.
- Judge noise: use two judges or spot-check 50 answers by hand.
- Run the reader with no memory at all, to measure how much is answerable from world knowledge or guessing.
- Fix the reader and judge versions before the first run.

## Reward for training (later)

`R = answer correctness (judge or F1) − w_c · cost − w_p · pause`, with correctness as the main term.

- Outcome arrives only at the end, after all compactions, so use branching: snapshot the stream, continue it under G different compactions, and use the group mean as the baseline.
- Train on LongMemEval, test on LoCoMo.
- Skip InfoLoss. It correlated the wrong way in every hard-mode run.

## What to build (about 300 lines)

1. A loader for the LoCoMo JSON and LongMemEval files.
2. A dialogue stream adapter that turns a conversation into our message format and calls the existing compactors.
3. A question-answering runner with F1 and an LLM judge.
4. A runner and report that print the metric table above.

## Rough cost (an estimate, from our logged per-call prices)

A 20K-token dialogue needs a handful of compactor calls, then one short reader call per question at about 4K tokens. That puts one arm on one conversation at roughly $0.10-0.20, so **about $10-20 for all arms on a 10-conversation first pass**, plus judge calls. Cheaper than our email runs.

## What would make this inconclusive

- `cliff` equals truncation and every method scores near the lower bound, which would show the budget is too small to learn anything.
- The reader already knows the dataset and the no-memory control scores well.
- Differences between arms are inside the bootstrap intervals, since LoCoMo has few conversations.
