# NEXT_STEPS

## 0. Pilot status

All pilot stages have run (50 full-context episodes, 200 scored branch points, 240 live branches, repeated-compaction test, 81-run budget matrix, background and pinned-tier smoke runs). `outputs/pilot_report.md` is final; the recommendation (conditional GO, InfoLoss not yet validated) is in `outputs/pilot/recommendation.md`.
Tracked spend with the placeholder prices is **~$33.5** in total (pilot $27.6 incl. dry-run retries, matrix $4.6, other $1.3); individual stages stayed under the $25 gate but the cumulative figure did not - check the real Fireworks bill. If the Mac sleeps on battery during a long run (it did, twice), re-run the same command: stages skip finished work. `llm.py` now caps each request at 300 s and rebuilds the connection pool after a timeout.

Known issues:
* **`constraint_extractor` pinned nothing** (fixed: cap raised to 3000; it now pins 0-5 facts per compaction and retention rose 0.62 -> 0.84) (every candidate row has `pinned_n=0`; it scored identical to `never`). ~50% of extractor calls came back empty because the small reasoning model exhausted `max_tokens=600`; the cap is now 3000 (patched, untested against the API) and the CONPACT extractor prompt is conservative ("default to empty") - check it fires on our short imperative facts, otherwise adapt the prompt. Re-score that candidate (`candidates.jsonl` rows with `candidate == "constraint_extractor"`) before reading the leaderboard.
* The frontier compactor is `glm-5p3` on Fireworks (no `ANTHROPIC_API_KEY`/`OPENAI_API_KEY` was set). Set `Config.frontier` to a closed model to replace the stand-in; the Anthropic path in `llm.py` is implemented but has never been called.
* Prices in `config.py` are placeholders (Fireworks' price page lists only training rates; its training API charges cached prefill at 20% of prefill, hence the 5x default). Token counts are logged, so re-price post hoc.
* InfoLoss is clipped at 0 per spec, which flattens 8-19% of compactions (the compacted context is sometimes *easier* to continue from). Consider scoring the signed value, or a margin.
* 38 simulated-user rewrites leaked unscheduled facts across 50 episodes and were caught and replaced by the canonical script (leak rate 0 in the recorded transcripts); 30% of persona rewrites fall back to the script. Cheap improvement: a stricter persona prompt.
* `label_handled` and `confirm_before_act` are violated often even with full context (base rate): forgetting is therefore measured as *facts newly violated relative to the full-context run*.

## 1. Datasets: keep them off the laptop

Hermes Agent and OpenResearcher long contexts (CONPACT's `hermes` / `openresearcher` sources, tens of GB across the 10K-220K variants) must **not** be downloaded locally.
Nothing large has been downloaded so far (the repo clones are 24 MB in `/tmp/cg_clones`; `.venv` is 343 MB). Proposed: a cloud box or cloud Claude Code session in the user's account
with >= 200 GB disk, `uv sync`, `HF_HOME` on the big volume, and the generation scripts (`conpact` package: `ConPact(source="hermes", length=...).generate()`) run there; only the
sampled episodes (JSONL, a few hundred MB) are copied back or kept in an object store. No cloud environment or SSH host is configured in this app right now, so this needs a choice (see the hand-back note).

## 2. SFT data generation: best-of-N on the inbox

1. Record full-context trajectories (done for 50; scale to ~2,000 seeds: ~$0.15 each).
2. At every eligible turn (context >= budget, >= 1 fact injected) sample N=8 compactions from the small prompted model at T=0.8 (+ the `cliff`, `frontier` and `threshold_pi` outputs as extra candidates).
3. Score each with the pilot's recipe (InfoLoss on the recorded next-3 turns, retention judge, optionally the MCQ probe) and keep the argmin of `lambda*InfoLoss - retention` among those that do not lose a pinned fact. The pilot's `best8_infoloss` is the baseline for this selection rule; `best8_retention` (0.98) is the oracle ceiling.
4. SFT target = the controller's full JSON output (action, summary, pinned, confidence). For `confidence`, regress on the *measured* retention of the chosen candidate so calibration is learned, not asserted (the pilot's small model rarely states a usable confidence).
5. Add Hermes Agent and OpenResearcher contexts (section 1) with CONPACT's SC injection for the retention side; there is no tool-use environment for them, so use InfoLoss on the recorded continuation only and mask the success/forgetting terms.

## 3. RL on Fireworks

* Policy = the small compactor; the main agent stays frozen (as in CompactionRL, where swapping only the summarizer moved SWE-bench Verified 49.0 -> 55.5).
* Reward per episode = `success - w_p*pause - w_a*forgetting - w_c*cost - lambda*infoloss` (`scoring/reward.py`, weights in `profiles/*.json`); calibrate `lambda` so the InfoLoss term is ~10-20% of reward variance on the pilot data, and verify on the live branches that it points the same way as forgetting/failure before using it.
* **Token-level loss normalization**: summaries vary 10x in length (and CompactionRL saw them grow ~50% with no cost term), so normalize the policy-gradient loss by the number of generated tokens over the whole batch (DAPO-style token-level mean) rather than per sequence, and keep the explicit cost/pause terms so length growth is priced.
* **Cross-segment credit for the online stage**: one episode contains k compactions; the outcome (success, later forgetting) arrives only at the end. Treat each compaction as a segment: (a) group-relative baselines from *branching*, i.e. continue the same snapshot with the same seeds under G different compactions (the harness already does this: `harness.agent.branch`) so advantage = branch outcome - group mean, which isolates the compaction's causal effect; (b) credit a segment with its own forgetting detected after it and discount later segments' effects by the repeated-compaction survival curve (plot 3) instead of blaming the last compactor; (c) pinned facts get an immediate dense reward from the retention judge so the controller is not trained purely on delayed outcomes.
* Infra: Fireworks has a serverless training API (sample + train primitives; rates on its pricing page are for training, not inference). Start with LoRA SFT, then RL with G=8 branches per snapshot. Sampling concurrency is the bottleneck - the serverless endpoint rate-limited us (HTTP 429) at ~8 concurrent requests and the account hit its monthly limit; plan a dedicated deployment for the main model during RL.

## 4. What the pilot says should change

* Make `cliff` (structure-preserving, no rewriting) the default fallback in the harness (it already is) and train the controller to *choose between* cliff-style truncation, summarization and offload per turn, not to replace them: `cliff` has the lowest InfoLoss and highest retention of any baseline, and costs nothing.
* The controller's value is probably in *when*: summaries cost 8-30 s of blocking pause and break the cache. The live branches (stage 3) are needed to price this; plots 1/2/7 and the matrix will show whether any policy beats `cliff` on success per dollar at 8K.
* Keep forgetting as a per-fact binary relative to the full-context baseline; do not count events.
* Consider longer horizons after the branch point (the earliest eligible branch point is used for live resume) and a harder 8K budget, where compaction fires 4-6 times per session.
