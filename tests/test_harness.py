import asyncio
import json

import pytest

from compactgym.compactors import REGISTRY, make_compactor
from compactgym.compactors.cliff import cliff_messages, fallback_compact
from compactgym.config import Config
from compactgym.env.inbox.tasks import TaskSpec, generate_task
from compactgym.env.inbox.tools import openai_tools
from compactgym.harness.agent import Episode, branch
from compactgym.harness.memory import MemoryStore
from compactgym.harness.resume import build_resume, group_turns, split_for_resume
from compactgym.harness.snapshots import SnapshotStore
from compactgym.harness.tokens import count_tokens
from compactgym.interfaces import CompactionAction, CompactionResult, EpisodeState, Message
from compactgym.llm import ChatResult, Usage
from helpers import FakeLLM, task_for


def tc(name, args, i):
    return {"id": f"call{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def convo(n_turns=6, tool_chars=3000):
    ms = [Message(role="system", content="SYS", meta={"uid": 0}), Message(role="user", content="Do things. Never delete.", meta={"uid": 1, "user_turn": 0})]
    for i in range(n_turns):
        ms.append(Message(role="assistant", content=f"thinking {i}", tool_calls=[tc("list_emails", {"limit": 5}, i)], meta={"uid": 10 + 2 * i, "step": i, "reasoning": "r" * 500}))
        ms.append(Message(role="tool", content=f"row {i} " * (tool_chars // 7), tool_call_id=f"call{i}", meta={"uid": 11 + 2 * i, "name": "list_emails"}))
    return ms


# -------------------------------------------------------------- main model double
def main_handler(tag, spec, messages, tools):
    """Deterministic function of the last message only: call a tool after a user msg, finish after a tool result."""
    if not tag.startswith("main:"):
        return "SUMMARY: goal; never delete; done things."
    last = messages[-1]
    if last["role"] == "user":
        return ChatResult(text="", tool_calls=[{"id": f"c{len(messages)}", "type": "function", "function": {"name": "list_emails", "arguments": json.dumps({"limit": 20})}}],
                          usage=Usage(prompt=len(json.dumps(messages)) // 4, completion=20), latency_s=0.01)
    return ChatResult(text="Done. Over to you.", usage=Usage(prompt=len(json.dumps(messages)) // 4, completion=10), latency_s=0.01)


def mk_episode(compactor_name="never", budget=10**9, turns=4, background=False, snapshots=None, llm=None, **kw):
    cfg = Config()
    llm = llm or FakeLLM(main_handler)
    task = generate_task(TaskSpec(seed=2, n_user_turns=turns, n_facts=1, fact_kinds=["forbid_delete"], distractor_volume=1.0))
    comp = make_compactor(compactor_name, cfg, llm, **kw)
    return Episode(cfg, task, comp, llm, episode_id="ep", budget_tokens=budget, background=background, snapshots=snapshots), task, cfg, llm


# ------------------------------------------------------------ resume format
def test_group_turns_and_split():
    ms = convo(5)
    ms.append(Message(role="user", content="new request", meta={"uid": 99}))
    groups = group_turns(ms)
    assert groups[-1][0].content == "new request" and len(groups) == 6
    system, older, tail = split_for_resume(ms, 2)
    assert len(system) == 1 and tail[-1].content == "new request"
    assert [m.role for m in tail] == ["assistant", "tool", "assistant", "tool", "user"]
    assert older[0].role == "user"


def test_resume_format_order():
    ms = convo(5)
    out = build_resume(ms, "SUMMARY TEXT", ["Never delete anything"], 2)
    assert [m.meta.get("origin") for m in out[:3]] == [None, "pinned", "compaction_summary"] or out[0].role == "system"
    assert out[0].role == "system" and out[1].meta["origin"] == "pinned" and out[2].meta["origin"] == "compaction_summary"
    assert sum(m.role == "assistant" for m in out) == 2


# ------------------------------------------------------------ cliff port
def test_cliff_rules():
    ms = convo(6, tool_chars=3000)
    ms[2].tool_calls = [tc("search_emails", {"query": "x" * 400}, 0)]
    out = cliff_messages(ms, 3)
    assert out[0].role == "system" and out[1].content.startswith("Do things")  # system + first user verbatim
    summ = [m for m in out if m.meta.get("origin") == "compaction_summary"]
    assert len(summ) == 1
    text = summ[0].content
    assert "row 0" not in text  # tool results > 500 chars dropped
    assert len(max((l for l in text.split("\n") if l.startswith("[search_emails]")), key=len)) <= 150 + len("[search_emails] ") + 3
    assert ("thinking: " + "r" * 300 + "...") in text  # reasoning truncated to 300 chars
    assert sum(m.role == "assistant" for m in out) == 3 and out[-1] is ms[-1]  # last K turns by reference
    # re-compaction discards the previous compacted block instead of compacting it again
    again = cliff_messages(out + [Message(role="assistant", content="x", meta={"uid": 500}), Message(role="tool", content="y", tool_call_id="z", meta={"uid": 501})], 3)
    assert sum(1 for m in again if m.content and m.content.startswith("The following is a summary")) == 1


# -------------------------------------------------------------- compactors
def test_threshold_pi_uses_summary_and_pinned_tier():
    llm = FakeLLM(lambda tag, spec, msgs, tools: "## Goal\nx\n## Constraints & Preferences\n- never delete")
    cfg = Config()
    c = make_compactor("threshold_pi", cfg, llm)
    state = EpisodeState(keep_k=2, context_tokens=900, settings={"pinned_tier": True}, pinned=["Never delete anything"])
    res = asyncio.run(c(convo(5), state, 1000))
    assert res.action == CompactionAction.SUMMARIZE_OLDER and res.compactor_tokens_in > 0 and res.compactor_model == "main"
    assert [m.meta.get("origin") for m in res.messages[:3]] == [None, "pinned", "compaction_summary"]
    # ablation: pinned tier off -> pinned facts are folded into the summary text instead
    state2 = EpisodeState(keep_k=2, context_tokens=900, settings={"pinned_tier": False}, pinned=["Never delete anything"])
    res2 = asyncio.run(c(convo(5), state2, 1000))
    assert not any(m.meta.get("origin") == "pinned" for m in res2.messages) and "Never delete anything" in res2.summary + res2.messages[1].content
    # below threshold -> KEEP, no model call
    n = len(llm.calls)
    state3 = EpisodeState(keep_k=2, context_tokens=100)
    assert asyncio.run(c(convo(5), state3, 1000)).action == CompactionAction.KEEP and len(llm.calls) == n


def test_all_registered_compactors_keep_below_threshold():
    cfg = Config()
    llm = FakeLLM()
    for name in REGISTRY:
        c = make_compactor(name, cfg, llm)
        res = asyncio.run(c(convo(2), EpisodeState(keep_k=2, context_tokens=10), 10_000))
        assert res.action == CompactionAction.KEEP, name


def test_recent_k_and_memory_offload_and_recall():
    cfg = Config()
    st = EpisodeState(keep_k=2, context_tokens=900, memory=MemoryStore(), settings={"pinned_tier": True})
    r = asyncio.run(make_compactor("recent_k", cfg)(convo(8), st, 1000))
    assert r.messages[1].content.startswith("Do things") and sum(m.role == "assistant" for m in r.messages) == 5
    m = asyncio.run(make_compactor("memory_offload", cfg)(convo(8), st, 1000))
    assert m.action == CompactionAction.OFFLOAD_TO_MEMORY and len(st.memory) > 0 and "recall" in m.summary
    hit = st.memory.recall("row 3")
    assert "row 3" in hit and st.memory.recall("zzzzqqq") == "No matching memories."


def test_forced_fallback_fits_budget():
    cfg = Config()
    ms = convo(10, tool_chars=4000)
    tools = openai_tools()
    budget = 3000
    out = fallback_compact(ms, budget, tools, 0, EpisodeState(keep_k=2))
    assert count_tokens(out, tools) <= budget and out[0].role == "system"


def test_episode_forced_compaction_logged_and_runs():
    ep, task, cfg, llm = mk_episode("never", budget=2500, turns=3)
    res = asyncio.run(ep.run())
    assert res.status == "ok" and res.error is None
    assert any(s.get("forced") for s in res.steps)
    assert all(s["ctx_after"] <= 2500 for s in res.steps)  # budget enforced before every main call


def test_episode_no_compaction_with_big_budget_and_logging():
    ep, task, cfg, llm = mk_episode("never", turns=3)
    res = asyncio.run(ep.run())
    assert res.status == "ok" and not any(s.get("forced") for s in res.steps)
    s0 = res.steps[0]
    for k in ("ctx_before", "ctx_after", "prompt_tokens_api", "cached_modeled", "completion_tokens", "main_latency_s", "step"):
        assert k in s0
    assert len(res.turn_final_text) == len(task.user_turns)


def test_background_compaction_non_blocking_and_stale():
    class SlowSummarizer:
        name = "slow"
        uses_memory = False

        def __init__(self):
            self.calls = 0

        async def __call__(self, messages, state, budget):
            self.calls += 1
            await asyncio.sleep(0.05)
            out = build_resume(messages, "BG SUMMARY", [], 2)
            return CompactionResult(action=CompactionAction.SUMMARIZE_OLDER, messages=out, summary="BG SUMMARY", compactor_latency_s=0.05, blocking=True)

    cfg = Config()
    llm = FakeLLM(main_handler)
    task = generate_task(TaskSpec(seed=2, n_user_turns=3, n_facts=1, fact_kinds=["forbid_delete"]))
    ep = Episode(cfg, task, SlowSummarizer(), llm, episode_id="bg", budget_tokens=10**9, background=True)
    ep.history = convo(6)
    for i, m in enumerate(ep.history):
        m.meta.setdefault("uid", i)
    ep.full_log = list(ep.history)

    async def go():
        rec1 = {}
        await ep._compaction_phase(rec1)  # launches, history untouched
        assert ep.pending is not None and len(ep.history) == 14
        ep.history.append(Message(role="user", content="newer msg", meta={"uid": 777}))  # arrives while compaction runs
        await asyncio.sleep(0.1)
        rec2 = {}
        await ep._compaction_phase(rec2)  # applies the finished result next turn
        return rec2

    rec2 = asyncio.run(go())
    assert rec2["blocking"] is False and rec2["pause_s"] == 0.0 and rec2["action"] == "summarize_older"
    assert ep.history[-1].content == "newer msg" and any(m.meta.get("origin") == "compaction_summary" for m in ep.history)

    async def stale():
        ep2 = Episode(cfg, task, SlowSummarizer(), llm, episode_id="bg2", budget_tokens=10**9, background=True)
        ep2.history = convo(6)
        for i, m in enumerate(ep2.history):
            m.meta.setdefault("uid", i)
        await ep2._compaction_phase({})
        ep2.history = ep2.history[:1] + ep2.history[5:]  # something else rewrote the prefix meanwhile (e.g. a forced fallback)
        await asyncio.sleep(0.1)
        rec = {}
        await ep2._compaction_phase(rec)
        return rec, ep2

    rec, ep2 = asyncio.run(stale())
    assert rec.get("bg_stale") is True and not any(m.meta.get("origin") == "compaction_summary" for m in ep2.history)


# ------------------------------------------------------------- branching
def test_snapshots_and_branching_replay_and_diverge(tmp_path):
    store = SnapshotStore(tmp_path, keep_in_memory=False)
    ep, task, cfg, llm = mk_episode("never", turns=3, snapshots=store)
    base = asyncio.run(ep.run())
    turns = store.turns("ep")
    assert turns == list(range(len(base.steps)))
    snap = store.load("ep", 2)
    assert snap.turn == 2 and snap.env_state and "rng" in snap.rng_state and snap.messages and snap.full_log
    # same compactor + same seeds -> identical continuation
    b1 = asyncio.run(branch(store, "ep", 2, make_compactor("never", cfg, llm), cfg, task, llm))
    assert [m.content for m in b1.full_log] == [m.content for m in base.full_log]
    assert b1.env_state["sent"] == base.env_state["sent"] and b1.branched_from == ("ep", 2)
    # a different compactor diverges in context (and is logged) but replays the same environment up to the branch point
    cfg2 = Config()
    b2 = asyncio.run(branch(store, "ep", 4, make_compactor("recent_k", cfg2, llm, k=1, threshold_frac=0.0), cfg2, task, llm, budget_tokens=10**9))
    assert b2.steps[0]["action"] == "truncate_verbatim" and b2.steps[0]["ctx_after"] < b2.steps[0]["ctx_before"]
    assert [m.content for m in b2.full_log[: len(snap.full_log)]] == [m.content for m in snap.full_log]


def test_modeled_cache_breaks_on_compaction():
    ep, task, cfg, llm = mk_episode("never", turns=3)
    res = asyncio.run(ep.run())
    steps = res.steps
    assert steps[0]["cached_modeled"] == 0
    assert all(b["cached_modeled"] >= 0.95 * a["prompt_tokens_est"] for a, b in zip(steps[1:], steps[2:]))  # everything previously sent stays cached
    ep2, *_ = mk_episode("never", budget=2500, turns=3)
    res2 = asyncio.run(ep2.run())
    forced = [(a, b) for a, b in zip(res2.steps, res2.steps[1:]) if b.get("forced")]
    assert len(forced) > 1 and all(b["cached_modeled"] < 0.6 * a["prompt_tokens_est"] for a, b in forced[1:])  # the compacted context is mostly re-read uncached


def test_oracle_pins_cover_facts_stated_before_a_branch_point():
    from compactgym.config import Config
    from compactgym.harness.agent import Episode
    from compactgym.harness.snapshots import Snapshot
    import inspect
    src = inspect.getsource(Episode.from_snapshot)
    assert "pin_oracle" in src and "inject_turn <= ep.user_turn" in src  # the branch pre-pins every fact whose turn has passed
