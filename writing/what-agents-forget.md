# I built a gym to see what AI agents forget, and most of my first answers were wrong

*2026-11-09 · 13 min read · AI agents, Context, Evaluation*

## TL;DR

I wanted to train a small open model to compact an agent's memory as well as a big model does, for a fraction of the cost. To do that I first needed a way to measure what an agent forgets when its history gets squeezed. So I built CompactGym: a simulated email assistant, a set of planted facts, and a checker that diffs the mailbox at the end.

Most of my first measurements turned out to be wrong in ways that didn't throw errors. A quality metric I planned to train on pointed the wrong way. A test I designed to break the cheap method didn't test what I thought. In the end, the thing that worked best was not a better summary. It was keeping the user's instructions in a short pinned list and cutting everything else. I haven't trained the small model yet. I now know what it should learn.

## Why I tried this

I was deep into a long session with an AI agent when a progress bar came up: *Compacting our conversation so we can keep chatting…* It sat at 93% for about four minutes. Then it finished and the agent carried on as if nothing had happened.

That made me anxious in a very specific way. Somewhere in that conversation I had told it things that still mattered. Don't email this person. The date moved. This is the right reference number. I had no idea which of those had survived. I would only find out when the agent did something wrong, probably twenty turns later.

When an agent's context window fills up, it has to compact: summarize the old history, trim it, or move it somewhere else. Every coding agent and chat assistant does some version of this. And the failure is silent. If "don't email Jonas" gets dropped, nothing errors. The agent just keeps working, confidently.

I also kept seeing how good small open models had become. If a small model could do this one job as well as a frontier model, compaction would be cheaper and faster everywhere, and it would matter most on devices with tiny windows. Apple's on-device model, for example, has a 4,096-token window shared between the prompt and the answer.

So the plan was: build a way to measure forgetting, find the best compaction method, then train a small model to match it.

## First, what was I measuring?

I needed a setting where I knew exactly what the agent should remember. Real logs don't give you that, so I built a simulated one.

An email and calendar assistant works through 28 requests from a simulated user. It has real tools over a generated mailbox: search, read, reply, archive, create an event. Early in the session, the user drops in facts that matter later. "Don't email Farah about this, keep her out of the loop." "The offsite moved to Thursday Oct 29." "The PO number is this, their email has a typo." Later requests depend on them.

At the end, a rule-based checker diffs the mailbox and calendar against what should have happened. Did the agent email the person it was told to leave out? Did it use the old date? I deliberately didn't use a language model as the judge. I wanted forgetting to be something I could count.

One detail took me a while to get right. The full-context agent, with nothing compacted at all, also gets some facts wrong. If I counted every violation, I would mostly be measuring how good the base agent is. So a fact only counts as forgotten if the full-context run got it right and the compacted run got it wrong.

The most useful thing I built was branching. I record a session once with full context, then snapshot it at any step. From that snapshot I can replay the rest of the session under a different compaction method, with the same random seeds. The difference between two branches is the effect of the compaction itself, and not of a different user or a different mailbox.

I compared ten methods. Some used published compaction prompts: Anthropic's from its API docs, Codex CLI's, and pi-mono's. One was a plain rule-based truncation called cliff, which drops long tool results and keeps what the user wrote. One had a frontier model write the summary. That "frontier" was glm-5p3, a large open model, because I didn't have keys for a closed one. The main agent doing the email work was gpt-oss-120b throughout.

## The first results said almost nothing

The first pilot ran 50 sessions with a budget of 16,000 tokens. Task success across all ten methods landed between 0.79 and 0.87, and the differences were not significant. Every method looked roughly the same.

That should have been my first warning. At 16K, the sessions barely overflowed, so compaction rarely had a chance to lose anything. I was measuring a problem that mostly wasn't happening.

## The training data had bugs I couldn't see

I moved on anyway and built the pipeline to generate fine-tuning data. At each point where compaction happens, it samples eight compactions from the small model, scores them, and keeps the best one as a training example.

The 10-seed trial was cheap, about $0.24 per seed. It produced 40 examples. They looked fine until I read them.

- The first run only saved the winning summary, not the other seven, so I couldn't change how winners were picked without rerunning everything.
- When the small model's JSON didn't parse, the raw reply was stored as the summary. Some training targets were a summary that contained a whole JSON blob.
- In one batch, the median target was 79 characters long. When cliff won, which was about half the time, the target was just `{"action": "truncate_verbatim", "summary": ""}`. A model trained on that would mostly learn to always truncate.
- The `pinned` field, the list of facts to keep visible, was empty in every single example. Nothing in the pipeline ever filled it.

None of this threw an error. The data generated, the scores came out, and the setup was still wrong.

## The metric I wanted to train on pointed the wrong way

My original plan was to pick winners with InfoLoss. It measures how much worse the main model predicts the next few turns from the compacted context than from the full context. It is cheap to compute and sounds exactly right: less information lost should mean less forgetting.

So I checked it against what actually happened in live branches. At 16K the correlation was basically zero. That check had a flaw I only noticed later: I resumed each session from its earliest eligible point, but some sessions plant their facts late, so a few branches had only two or three turns left to forget anything.

I built a harder version to fix that. In hard mode, facts are planted in the first three turns and only asked for in the last 40% of the session. The user never repeats a fact, so a forgotten one is gone for good. A goal scores zero if the fact behind it was violated. With an 8,000-token budget, a session now compacts about 17 times.

That worked as an environment. Newly forgotten facts went from 0.2 per session to about 0.43, and a third of sessions forgot at least one.

Then I ran the check again. InfoLoss correlated with real forgetting at −0.33 (95% interval −0.54 to −0.08). Negative means that compactions with *higher* InfoLoss went with *fewer* forgotten facts. It held on a second dataset too. Fact retention, which I had planned as a second signal, showed nothing.

I think I understand why. InfoLoss scores one compaction. The outcome depends on the roughly seventeen that come after it. I set its weight to zero.

## Why the cheap method kept winning

For a while the plain truncation, cliff, kept coming out on top, which annoyed me because it does nothing clever.

| Method (hard mode, 8K, same 12 branch points) | Facts forgotten | Task success | Cost | User waits |
|---|---|---|---|---|
| frontier summarizer | 0.25 | 0.84 | $0.074 | ~120 s |
| cliff (truncation) | 0.50 | 0.75 | $0.033 | 0.1 s |

The prices are placeholders from my config, so compare the ratios, not the dollars.

The frontier summarizer forgets less. It also costs about twice as much and blocks the user for around two minutes per session. My reward charged 0.002 per second of waiting, which sounds small. At two minutes it takes 0.24 off the score, while the whole spread in task success between methods was about 0.08.

When I rescored the same branches with that waiting term removed, the frontier summarizer won clearly. So the headline depended on one weight I had picked. That is not a finding about compaction. It is a finding about my reward.

## I designed a test to break cliff, and it didn't

cliff drops any tool result longer than 500 characters. So I assumed it would lose any fact that only lives inside a long email. I built three of those: an email saying to send the signed form to a different address than the invoice shows, one saying the meeting room changed, and one saying a deadline moved. In each case the user only asks the agent to read the email early. The fact is needed much later.

cliff forgot those facts at exactly the same rate as the frontier summarizer: 0.17 per session.

The first explanation I got was that cliff re-opened the email 88% of the time. That turned out to be a miscount. It counted the agent's first read as a re-read. Counting properly, cliff re-read zero of eleven.

The likelier answer was in my own prompt. I had asked the agent to "give me the short version" of each email. So it wrote the fact into its own reply, and cliff keeps assistant text. The agent restated the room change in 11 of 11 cases and the deadline in 6 of 6. The test was measuring my prompt.

## Then I tried other people's ideas

Two posts changed what I built next.

The first was about Shelley, an agent from exe.dev that edits its own history with a tool: trim particular tool outputs, or squish a range of messages into a short note with a pointer back to the original. I rebuilt that. The model sees the context numbered and replies with edits by index. A frontier model used it cleanly. The untrained small model failed the protocol in most compactions and fell back to plain truncation in 7 of 8 and 8 of 13 cases. It was also slow. It is a reasoning model, and it spent 500 to 570 seconds per session thinking, against about 100 for the frontier model. So "small model is faster" was not automatic.

The second was a thread from the author of Mainplane, titled "compaction is a mistake." The argument: summaries are lossy and slow, so keep the full record and start a fresh session with one instruction, "retrieve key context from it as you go." It is a good argument, and it reported speed and cost, not accuracy. So I built it. All old turns go into a store verbatim, the context keeps only that one instruction, and the agent has a search tool.

It forgot 1.33 facts per session, five times the frontier summarizer. The agent called search about once per session, and in 4 of 12 sessions it never searched at all. The facts it lost were standing rules, like who to leave off an email. You can't search for a rule you don't know you've forgotten.

While building that, I found a bug in my own memory search. It kept trailing punctuation on words, so "Jonas." in "Don't email Jonas." never matched a search for "jonas". It had been quietly hurting every retrieval method I'd tested.

To be fair to the thread, my agent was gpt-oss-120b at low reasoning effort, and theirs was a much stronger model on a 400K-token session. A stronger agent may search on its own. I can only say what happened with mine.

## What actually worked: pin the rules

Both failures pointed at the same gap. The standing instructions needed to be visible all the time, and nothing was keeping them there.

So I tried the simplest version of that. I kept a short pinned list of the user's facts at the top of the context and compacted everything else cheaply. To see the ceiling, I gave it the true planted facts.

| Same 12 branch points | Facts forgotten | Task success | Cost | User waits |
|---|---|---|---|---|
| retrieval only | 1.33 | 0.69 | $0.022 | 0 s |
| retrieval + pinned facts | 0.08 | 0.79 | $0.020 | 0 s |
| cliff + pinned facts | 0.17 | 0.80 | $0.034 | 0.1 s |
| frontier summarizer | 0.25 | 0.84 | $0.074 | ~120 s |

With the pins, both cheap methods forgot fewer facts than the frontier summarizer, at a half to a quarter of the cost, with no waiting.

I want to be careful about what this shows. These pins are an oracle: I handed the harness the exact facts for free. A real system has to find them. Task success is also still about 0.05 below the frontier summarizer. So this is a ceiling, not a result.

## Can a model find the pins?

That became the real question. I gave six open models the conversation up to a compaction point and asked them to list the standing instructions. Then I scored their lists against the planted facts.

The best two, glm-5p3-flash and minimax-m3p1-flash, found about half the facts (recall 0.54). The frontier stand-in got 0.48, and 8 of its 36 replies weren't valid JSON. One model returned unusable output 26 times out of 36.

My scorer is also too strict. Here is a real output from glm-5p3-flash, for a fact my scorer marked as missed:

> Gold: *The offsite moved to Thursday Oct 29, not Oct 28 -- please fix it on the calendar.*
>
> Model: *Team offsite (event c1001) moved to Thursday Oct 29 at 14:00, 3 hours, attendees Priya, Jonas, Mei and Carlos…*

It found the fact. It just said it in its own words, with extra detail, and word overlap didn't count it. So real recall is higher than 0.54, and I need a better scorer before I trust the number.

Still, this is the job I want to train a small model for. It is narrow, it can be checked against exact labels, and the zero-shot models already get halfway.

## What I took away from it

The biggest lesson was how much of this was measurement. I came in wanting to train a model. Most of my time went into finding out that my metrics, my tests, and my training data were wrong in ways that looked fine.

Claude wrote most of the code with me. It also got things wrong, including the re-read miscount and an early claim that success differences weren't significant, which they were once paired properly. I had expected it to catch more of this before I ran anything. I still needed to understand the task well enough to know what to check and what the results meant.

I also stopped thinking of compaction as one thing. Remembering the user's rules, keeping the task's progress, and keeping the cost and the wait down are separate problems. The best summary handled the first two reasonably and failed the third. Truncation handled the third and failed the first. A pinned list handles the first and doesn't cost much of the third.

One more practical thing. My inference account got suspended for hitting its monthly spending limit halfway through a 120-branch run, so only 7 branches finished. Every individual run in this project cost a few dollars at most, and it still added up.

I don't want to make this sound like a finished result. I haven't trained the small model, my samples are 12 to 14 branches per method, my frontier is an open model standing in for a closed one, and the environment is a simulation with short, explicit instructions.

## What's next

Next, I want to train a small, non-reasoning model to write the pinned list, with a scorer that understands paraphrase. Then put it in front of the same live branches as the frontier summarizer, with enough branch points to say something real.

I also want to try the same question on public long-conversation benchmarks like LoCoMo and LongMemEval at a 4K budget. There are no tool outputs to drop there, so cliff's main trick won't exist, and the ranking could change.

And I want to redesign the buried-fact test, so the fact can't leak into the agent's own reply or be found with one easy search.

## Tools and references

The methods and models here come from the people building them. The results, and the mistakes, are mine.

- CONPACT (compaction-integrity), for the planted-constraint taxonomy and extractor I started from.
- CliffCompaction, the rule-based truncation I called cliff.
- The compaction prompts from Anthropic's API docs, Codex CLI and pi-mono.
- CompactionRL, for training with compaction inside the RL loop.
- Shelley from exe.dev, for in-place compaction with trim and squish edits.
- Mainplane, for the "compaction is a mistake" argument and the fresh-session-plus-retrieval design.
- Exo, for the idea that the durable log and the prompt don't have to be the same thing.
- Fireworks for inference: gpt-oss-120b as the agent, glm-5p3 as the frontier stand-in, and nemotron-lightning as the small model.

Code, plots and an interactive walkthrough are at github.com/ankittandon/CompactGym.
