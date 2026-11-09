# Thread: CompactGym

Numbers come from our own runs (hard mode, 8K budget, 12-14 branches per method). Tweet 1 carries your screen recording. Swap [link] for the site URL.

## 1 (attach your video)  (242 chars)

Does this give you anxiety?

That's an agent compacting its own memory so it can keep going. Minutes of waiting, then a summary of everything it did.

What did it forget? You find out 20 turns later.

We built a gym to measure exactly that. 🧵

## 2  (230 chars)

Context windows fill up, so agents compact: summarize or trim the history and carry on.

If "don't email Jonas" gets lost, nothing errors. The agent just does the wrong thing, much later. A silent failure that is hard to test for.

## 3  (275 chars)

CompactGym: a simulated email + calendar assistant works through 28 requests. Early on the user plants facts (leave Jonas out, the offsite moved to Thursday). Late requests need them.

A rule-based verifier diffs the mailbox, so no LLM judge decides what counts as forgotten.

## 4  (263 chars)

The trick is branching. Snapshot a session, then replay the same moment under different compactors with the same seeds. The difference is the compaction's causal effect.

10 baselines, including the compaction prompts from Anthropic's docs, Codex CLI and pi-mono.

## 5  (208 chars)

Result 1. At an 8K budget, hard mode:

frontier summarizer: forgets 0.25 facts per session, $0.074, blocks you ~2 min
truncation (cliff): forgets 0.50, $0.033, ~0 s

Better memory costs 2x and a coffee break.

## 6  (264 chars)

Result 2. We tried InfoLoss (how much worse the model predicts the next turns after compaction) as a training signal.

It correlates the wrong way with real forgetting: rho -0.33, 95% CI -0.54 to -0.08.

Scoring one compaction is not scoring the 17 that follow it.

## 7  (256 chars)

Result 3. "Don't summarize. Keep the full log and let the agent search it." We built that.

It forgot 1.33 facts per session, 5x the frontier summarizer. The agent called recall about once per session. You can't search for a rule you don't know you forgot.

## 8  (259 chars)

What worked: pin the standing instructions so they are always visible, then truncate or retrieve the rest.

With the true facts pinned (an upper bound):
cliff + pins: 0.17 forgotten, $0.034, 0 s
recall + pins: 0.08, $0.020, 0 s

frontier: 0.25, $0.074, ~2 min

## 9  (253 chars)

The catch: oracle pins are a ceiling, and task success is still ~0.05 below frontier.

Zero-shot, open flash models find about half the facts worth pinning (recall 0.54, precision 0.4). Big headroom, exact labels to train on. That is the model to build.

## 10  (250 chars)

We also published what we got wrong: a test confounded by our own prompt, a bad re-read claim, a flawed first reward check.

Caveats: 12-14 branches per method, an open model standing in for frontier, placeholder prices.

Site, code and plots: [link]

## Alternative openers for tweet 1

- "Does this give you anxiety? Four minutes of \"compacting our conversation\" and you don't know what the agent just forgot. We measured it."
- "Every long agent session ends the same way: a progress bar, and a prayer. We built a gym for the prayer."
- "This progress bar is an agent deciding what to forget. We built a gym to grade the decision. Thread."

## Optional tweet between 3 and 4 (on-device angle)

Small windows make this constant. Apple's on-device model has a 4,096-token window shared between prompt and answer. For an agent running there, compaction isn't a feature, it's the whole game.

## Reply to pin under the thread

Everything is reproducible: `uv run pytest`, then the `pilot` runner. Corrections page included, because the first version of our reward was wrong.
