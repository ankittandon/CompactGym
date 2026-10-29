import asyncio
import json

from compactgym.compactors.edit_context import EditContext, apply_edits, est_tokens, protected_indices, render_view
from compactgym.config import Config
from compactgym.harness.memory import MemoryStore
from compactgym.interfaces import CompactionAction, EpisodeState, Message
from helpers import FakeLLM


def call(cid, name="search_emails", args="{}"):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}


def convo():
    """0 system | 1 user | 2 asst+call c1 | 3 tool c1 (big) | 4 asst | 5 user | 6 asst+call c2 | 7 tool c2 (big) | 8 asst | 9 user | 10 asst | 11 user | 12 asst"""
    big = "email list " * 400
    return [
        Message(role="system", content="sys"), Message(role="user", content="Don't email Jonas. Find promo emails."),
        Message(role="assistant", content=None, tool_calls=[call("c1")]), Message(role="tool", content=big, tool_call_id="c1", meta={"name": "search_emails"}),
        Message(role="assistant", content="Found 12 promo emails."), Message(role="user", content="Archive them."),
        Message(role="assistant", content=None, tool_calls=[call("c2", "archive")]), Message(role="tool", content=big, tool_call_id="c2", meta={"name": "archive"}),
        Message(role="assistant", content="Archived all 12."), Message(role="user", content="Count DevWeekly."),
        Message(role="assistant", content="There are 5."), Message(role="user", content="Thanks."), Message(role="assistant", content="Welcome."),
    ]


def valid_structure(ms):
    seen = set()
    for m in ms:
        if m.role == "assistant":
            seen |= {tc["id"] for tc in (m.tool_calls or [])}
        if m.role == "tool" and m.tool_call_id not in seen:
            return False
    return True


def test_trim_squish_hide_save_originals_and_keep_structure():
    ms, mem = convo(), MemoryStore()
    prot = protected_indices(ms, keep_k=2)
    assert {0, 9, 10, 11, 12} <= prot and 3 not in prot  # system and the last two turns are protected
    out, applied, errors = apply_edits(ms, [{"op": "trim", "ids": [3]}, {"op": "squish", "start": 5, "end": 8, "note": "Archived 12 promo emails (c2)."}], prot, mem)
    assert not errors and len(applied) == 2
    assert out[3].content.startswith("[trimmed") and out[3].role == "tool"  # structure kept, original in memory
    assert any(m.meta.get("edit") and "Archived 12 promo" in m.content and "mem" in m.content for m in out)
    assert valid_structure(out) and sum(map(est_tokens, out)) < sum(map(est_tokens, ms)) // 2
    assert len(mem) == 2 and "email list" in mem.recall("email list")
    out2, applied2, errors2 = apply_edits(ms, [{"op": "hide", "ids": [6]}], prot, MemoryStore())
    assert not errors2 and len(out2) == len(ms) - 2 and valid_structure(out2)  # the call and its result leave together


def test_invalid_edits_are_rejected():
    ms = convo()
    prot = protected_indices(ms, keep_k=2)
    bad = [
        {"op": "trim", "ids": [0]},                                   # system
        {"op": "trim", "ids": [12]},                                  # protected tail
        {"op": "trim", "ids": [4]},                                   # not a tool result
        {"op": "squish", "start": 2, "end": 2, "note": "x" * 1},      # cuts the call from its result
        {"op": "squish", "start": 1, "end": 4, "note": ""},           # no note
        {"op": "squish", "start": 3, "end": 3, "note": "long " * 3000},  # note not shorter than what it replaces
        {"op": "hide", "ids": [7]},                                   # a tool result alone
        {"op": "trim", "ids": [99]},                                  # does not exist
        {"op": "explode"},
    ]
    for op in bad:
        out, applied, errors = apply_edits(ms, [op], prot, MemoryStore())
        assert errors and not applied and [m.content for m in out] == [m.content for m in ms], op
    out, applied, errors = apply_edits(ms, [{"op": "trim", "ids": [3]}, {"op": "squish", "start": 3, "end": 3, "note": "n"}], prot, MemoryStore())
    assert errors and applied == ["trim [3]"]  # overlap: the second op is rejected, the first still applies


def state_for(ms):
    return EpisodeState(memory=MemoryStore(), context_tokens=sum(map(est_tokens, ms)), keep_k=2, profile={"never_lose": ["Action"]})


def test_compactor_end_to_end_with_retry_and_pinned():
    ms = convo()
    replies = iter([
        json.dumps({"ops": [{"op": "squish", "start": 2, "end": 2, "note": "bad"}], "pinned": []}),  # rejected: splits a call
        json.dumps({"ops": [{"op": "trim", "ids": [3, 7]}], "pinned": ["Don't email Jonas."], "confidence": 0.8}),
    ])
    llm = FakeLLM(lambda *a, **k: next(replies))
    comp = EditContext(Config(), llm, role="frontier", threshold_frac=0.0)
    st = state_for(ms)
    res = asyncio.run(comp(ms, st, 2000))
    assert res.action == CompactionAction.EDIT_CONTEXT and res.pinned == ["Don't email Jonas."] and res.confidence == 0.8
    assert len(llm.calls) == 2 and "Rejected" in llm.calls[1]["messages"][-1]["content"]  # the second call saw the error
    assert any("Don't email Jonas." in (m.content or "") and m.meta.get("origin") == "pinned" for m in res.messages)
    assert valid_structure(res.messages) and len(res.calls) == 2 and res.compactor_tokens_in >= 0
    assert st.settings["edit_log"]["applied"] == ["trim [3, 7]"]


def test_compactor_keeps_when_nothing_valid_and_below_threshold():
    ms = convo()
    llm = FakeLLM(lambda *a, **k: "no json at all")
    res = asyncio.run(EditContext(Config(), llm, threshold_frac=0.0)(ms, state_for(ms), 2000))
    assert res.action == CompactionAction.KEEP and len(llm.calls) == 2  # tried twice, then handed back to the harness fallback
    res2 = asyncio.run(EditContext(Config(), FakeLLM(), threshold_frac=0.99)(ms, state_for(ms), 10**6))
    assert res2.action == CompactionAction.KEEP  # below the threshold: no call
    assert "#3 [tool result search_emails]" in render_view(ms, protected_indices(ms, 2))


def test_fresh_retrieve_keeps_only_a_stub_and_stores_the_record_verbatim():
    from compactgym.compactors.fresh_retrieve import FreshRetrieve, STUB
    ms = convo()
    st = state_for(ms)
    res = asyncio.run(FreshRetrieve(Config(), None, threshold_frac=0.0)(ms, st, 2000))
    assert res.action == CompactionAction.OFFLOAD_TO_MEMORY and res.summary == STUB
    assert [m.content for m in res.messages][1:2] == [STUB.join(["[Summary of the earlier conversation, written by a context compactor]\n", ""])]
    assert sum(map(est_tokens, res.messages)) < sum(map(est_tokens, ms)) // 2
    assert "Don't email Jonas" in st.memory.recall("Jonas")  # exact text comes back, nothing paraphrased
    assert valid_structure(res.messages)


def test_memory_search_ignores_trailing_punctuation():
    m = MemoryStore()
    m.add("[user]: Don't email Jonas. Send the form to contracts-desk@orion.example, not billing.")
    assert "Jonas" in m.recall("jonas") and "contracts-desk@orion.example" in m.recall("contracts-desk@orion.example")
    assert "Jonas" in m.recall("Don't email")
