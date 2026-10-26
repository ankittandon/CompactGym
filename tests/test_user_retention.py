import asyncio
import json

import pytest

from compactgym.compactors import make_compactor
from compactgym.compactors.small_prompted import SmallPrompted, parse_decision
from compactgym.config import Config
from compactgym.env.inbox.tasks import TaskSpec, generate_task
from compactgym.env.inbox.user import SimUser
from compactgym.harness.memory import MemoryStore
from compactgym.interfaces import CompactionAction, EpisodeState, Message
from compactgym.scoring.retention import effective_retention, flatten_for_judge, judge_retention, probe, remove_fact
from compactgym.third_party.conpact.prompts import probe_prompt
from helpers import FakeLLM
from test_harness import convo


def mk_task():
    return generate_task(TaskSpec(seed=2, n_user_turns=6, n_facts=2, fact_kinds=["confirm_before_act", "stale_date"]))


def test_sim_user_script_approvals_reasks_and_end():
    task = mk_task()
    u = SimUser(task, Config(), None, mode="scripted")
    text, meta = asyncio.run(u.first_message())
    assert meta["user_turn"] == 0 and task.facts[0].utter in text
    r = asyncio.run(u.respond("I can archive these 14 emails. Shall I proceed?"))
    assert r[1].get("approval") and "go ahead" in r[0].lower()
    # re-ask of a given fact: keyword match, recorded, first reaction does not restate, second does
    u.cur = 5
    fact = task.facts[1]
    q = f"Quick question: {fact.reask_keywords[0]}?"
    r1 = asyncio.run(u.respond(q, step=3))
    r2 = asyncio.run(u.respond(q, step=4))
    assert len(u.reasks) == 2 and u.reasks[0]["fact"] == fact.id
    assert "already told you" in r1[0] and fact.text in r2[0] and r2[1].get("restated") == fact.id
    # script advances and eventually ends
    u2 = SimUser(task, Config(), None)
    asyncio.run(u2.first_message())
    seen = 1
    while (nxt := asyncio.run(u2.respond("Done."))) is not None:
        seen += 1
        assert nxt[1]["user_turn"] == seen - 1
    assert seen == len(task.user_turns)


def test_sim_user_state_roundtrip():
    task = mk_task()
    u = SimUser(task, Config(), None, seed=3)
    asyncio.run(u.first_message())
    u.rng.random()
    d = json.loads(json.dumps(u.state_dict(), default=list))
    u2 = SimUser(task, Config(), None, seed=99)
    u2.load_state(d)
    assert u2.rng.random() == u.rng.random() and u2.cur == u.cur


def test_sim_user_llm_mode_rejects_leaks_and_dropped_details():
    task = mk_task()
    cfg = Config()
    stale = task.facts[1]  # injected later (turn >= 2): must not leak into turn 0
    leak = FakeLLM(lambda *a: f"hey, btw {stale.leak_keywords[0]} " + task.user_turns[0].text)
    u = SimUser(task, cfg, leak, mode="llm")
    text, meta = asyncio.run(u.first_message())
    assert meta.get("llm_fallback") and text == task.user_turns[0].text and u.leaks and u.leaks[0]["facts"] == [stale.id]
    drop = FakeLLM(lambda *a: "do the thing")
    u = SimUser(task, cfg, drop, mode="llm")
    assert asyncio.run(u.first_message())[1].get("llm_fallback")
    good = FakeLLM(lambda tag, spec, msgs, tools: msgs[-1]["content"])  # faithful rewrite
    u = SimUser(task, cfg, good, mode="llm")
    assert not asyncio.run(u.first_message())[1].get("llm_fallback") and not u.leaks


def test_retention_judge_wrapper_and_effective_retention():
    task = mk_task()
    fact = task.facts[0]
    llm = FakeLLM(lambda tag, spec, msgs, tools: "YES" if fact.text in msgs[-1]["content"].split("[Compacted Context]")[1] else "NO")
    cfg = Config()
    with_fact = [Message(role="system", content="s"), Message(role="user", content="[Summary] " + fact.text)]
    without = [Message(role="system", content="s"), Message(role="user", content="[Summary] nothing here")]
    assert asyncio.run(judge_retention(llm, cfg, fact, with_fact)) is True
    assert asyncio.run(judge_retention(llm, cfg, fact, without)) is False
    assert asyncio.run(judge_retention(FakeLLM(lambda *a: "maybe"), cfg, fact, with_fact)) is None
    fl = flatten_for_judge(convo(2, tool_chars=2000))
    assert all(len(m["content"]) < 700 for m in fl) and all(m["role"] != "system" for m in fl)
    assert effective_retention([True, True, False, False], [False] * 4, [True] * 4) == pytest.approx(0.5)
    assert effective_retention([True, True], [True, True], [True, True]) is None  # no room to improve
    assert effective_retention([None], [True], [True]) is None


def test_remove_fact_and_probe_letter():
    task = mk_task()
    fact = task.facts[0]
    ms = [Message(role="user", content=f"hi. {fact.utter} next.")]
    assert fact.utter not in remove_fact(ms, fact)[0].content
    _, letter = probe_prompt(fact.probe["probe"], fact.probe["correct"], fact.probe["incorrect"], 0)
    llm = FakeLLM(lambda *a: letter)
    assert asyncio.run(probe(llm, Config(), task, fact, ms, seed=0)) is True
    other = "A" if letter == "B" else "B"
    assert asyncio.run(probe(FakeLLM(lambda *a: other), Config(), task, fact, ms, seed=0)) is False
    assert asyncio.run(probe(FakeLLM(lambda *a: "I think so"), Config(), task, fact, ms, seed=0)) is None
    sent = llm.calls[0]["messages"]
    assert sent[-1]["role"] == "user" and fact.probe["probe"] in sent[-1]["content"]
    asyncio.run(probe(llm, Config(), task, fact, ms, restate=True, seed=0))
    assert fact.text in llm.calls[-1]["messages"][-1]["content"]


def test_small_prompted_parses_and_executes_actions():
    d = parse_decision('noise {"action": "summarize_older", "summary": "S", "pinned": ["never delete"], "confidence": 0.8} tail')
    assert d["action"] == "summarize_older" and d["pinned"] == ["never delete"]
    assert parse_decision("free text")["_parse_failed"]
    cfg = Config()

    def run(payload, **kw):
        llm = FakeLLM(lambda *a: payload)
        c = SmallPrompted(cfg, llm, **kw)
        st = EpisodeState(keep_k=2, context_tokens=900, memory=MemoryStore(), profile={"never_lose": ["Action"]}, settings={"pinned_tier": True})
        return asyncio.run(c(convo(6), st, 1000)), st

    r, _ = run('{"action": "summarize_older", "summary": "S", "pinned": ["never delete"], "confidence": 0.7}')
    assert r.action == CompactionAction.SUMMARIZE_OLDER and r.pinned == ["never delete"] and r.confidence == 0.7
    assert r.messages[1].meta["origin"] == "pinned" and r.compactor_model == "small"
    r, st = run('{"action": "offload_to_memory", "pinned": [], "confidence": 0.4}')
    assert r.action == CompactionAction.OFFLOAD_TO_MEMORY and len(st.memory) > 0
    r, _ = run('{"action": "clear_tool_outputs", "pinned": []}')
    assert any(m.content == "[tool output cleared]" for m in r.messages) and r.messages[-1].content != "[tool output cleared]"
    r, _ = run('{"action": "truncate_verbatim"}')
    assert r.action == CompactionAction.TRUNCATE_VERBATIM
    r, _ = run('{"action": "keep"}')
    assert r.action == CompactionAction.KEEP
    r, _ = run('{"action": "keep"}', must_compact=True)
    assert r.action != CompactionAction.KEEP
    # below min_fill the controller does not even call the model
    llm = FakeLLM()
    c = SmallPrompted(cfg, llm, min_fill=0.5)
    asyncio.run(c(convo(2), EpisodeState(keep_k=2, context_tokens=100), 1000))
    assert not llm.calls


def test_constraint_extractor_pins_registry():
    cfg = Config()

    def handler(tag, spec, msgs, tools):
        if tag == "compact:extractor":
            return json.dumps({"scs": [{"text": "Never delete anything.", "evidence": "x"}]})
        return "## Goal\nx"
    llm = FakeLLM(handler)
    c = make_compactor("constraint_extractor", cfg, llm)
    ms = convo(5)
    st = EpisodeState(keep_k=2, context_tokens=900, settings={"pinned_tier": True}, full_log=ms)
    r = asyncio.run(c(ms, st, 1000))
    assert r.pinned == ["Never delete anything."] and any(m.meta.get("origin") == "pinned" for m in r.messages)
    assert {c_["role"] for c_ in r.calls} == {"main", "small"}
