import json

from compactgym.env.inbox.data import generate_mailbox
from compactgym.env.inbox.tasks import FACT_TYPES, KINDS_BY_TYPE, TaskSpec, generate_task
from compactgym.env.inbox.tools import InboxEnv


def env(seed=1, scale=1.0):
    return InboxEnv(generate_mailbox(seed), scale)


def test_mailbox_size_and_determinism():
    a, b = generate_mailbox(5), generate_mailbox(5)
    assert len(a.emails) >= 300 and a.model_dump() == b.model_dump()
    assert generate_mailbox(6).model_dump() != a.model_dump()


def test_tools_deterministic_and_large():
    e1, e2 = env(), env()
    for q in ("list_emails", "search_emails"):
        args = {} if q == "list_emails" else {"query": "renewal"}
        assert e1.execute(q, args) == e2.execute(q, args)
    assert len(e1.execute("list_emails", {})) > 6000  # large on purpose


def test_mutations_and_errors():
    e = env()
    assert e.execute("archive", {"id": "e0001"}).startswith("OK")
    assert e.emails["e0001"].folder == "archive"
    assert "already archived" in e.execute("archive", {"id": "e0001"})
    assert e.execute("delete", {"id": "e0002"}).startswith("OK") and e.emails["e0002"].folder == "trash"
    assert e.execute("read_email", {"id": "nope"}).startswith("ERROR")
    assert e.execute("label", {"id": "e0003", "label": "x"}).startswith("OK") and "x" in e.emails["e0003"].labels
    assert e.execute("reply", {"id": "e0003", "body": "hi", "cc": "a@b.c"}).startswith("ERROR: mail gateway")
    assert not e.sent
    assert e.execute("send", {"to": "a@b.c", "subject": "s", "body": "b"}).startswith("OK") and len(e.sent) == 1
    assert e.execute("bogus", {}).startswith("ERROR")


def test_calendar_tools():
    e = env()
    out = e.execute("create_event", {"title": "X", "date": "2026-10-22", "start_time": "14:00", "duration_min": 30, "attendees": "a@b.c"})
    eid = out.split()[2]
    assert e.execute("move_event", {"id": eid, "date": "2026-10-23"}).startswith("OK") and e.events[eid].date == "2026-10-23"
    assert e.execute("create_event", {"title": "X", "date": "bad", "start_time": "14:00"}).startswith("ERROR")
    assert e.execute("cancel_event", {"id": eid}).startswith("OK") and e.events[eid].status == "cancelled"
    assert "events from" in e.execute("list_events", {})


def test_state_roundtrip():
    e = env()
    e.execute("archive", {"id": "e0001"})
    e.execute("send", {"to": "a@b.c", "subject": "s", "body": "b"})
    e.execute("create_event", {"title": "X", "date": "2026-10-22", "start_time": "14:00"})
    d = json.loads(json.dumps(e.state_dict()))
    assert len(json.dumps(d)) < 20000  # compact snapshot
    e2 = InboxEnv.from_state(d)
    assert e2.emails["e0001"].folder == "archive" and len(e2.sent) == 1 and len(e2.events) == len(e.events)
    assert e2.execute("list_emails", {}) == e.execute("list_emails", {})


def test_task_determinism_and_structure():
    spec = TaskSpec(seed=3, n_user_turns=12, n_facts=4)
    a, b = generate_task(spec), generate_task(spec)
    assert a.model_dump() == b.model_dump()
    assert len(a.user_turns) == 12 and len(a.facts) == 4
    for f in a.facts:
        assert f.rule is not None and f.probe and f.type in FACT_TYPES  # machine-checkable consequence + MCQ probe
    for g in a.goals:
        assert g.checks


def test_all_kinds_generate():
    for t, kinds in KINDS_BY_TYPE.items():
        for k in kinds:
            task = generate_task(TaskSpec(seed=2, n_user_turns=8, n_facts=1, fact_kinds=[k]))
            assert task.facts[0].kind == k and task.facts[0].type == t
            assert any(g.fact_id == task.facts[0].id for g in task.goals)


def test_injection_locations():
    for loc in ("top", "middle", "bottom", "repeated"):
        t = generate_task(TaskSpec(seed=4, n_user_turns=12, n_facts=2, injection=loc, fact_types=["Action", "Output"]))
        inj = [f.inject_turn for f in t.facts]
        if loc == "top":
            assert inj == [0, 0]
        if loc == "bottom":
            assert min(inj) >= 6
        if loc == "repeated":
            assert all(len(f.restate_turns) == 2 for f in t.facts)


def test_framings():
    strict = generate_task(TaskSpec(seed=4, n_facts=1, fact_kinds=["forbid_delete"], strict=True, direct=True))
    plain = generate_task(TaskSpec(seed=4, n_facts=1, fact_kinds=["forbid_delete"], strict=False, direct=False))
    assert strict.facts[0].utter.lower().startswith("this is an important constraint") and plain.facts[0].utter == plain.facts[0].text


def test_latent_association():
    mb = generate_mailbox(2)  # even seed -> Henderson & Co plays the deal
    deal = mb.roles["deal"]
    assert deal["name"] == "Henderson & Co"
    mentions = [e for e in mb.emails if "henderson" in (e.subject + e.body).lower()]
    assert len(mentions) == 1 and mentions[0].tags.get("deal")  # only one signature line names the company
    t = generate_task(TaskSpec(seed=2, n_facts=1, fact_kinds=["protect_deal"]))
    f = t.facts[0]
    assert "Henderson" in f.utter and f.latent
    assert all("H&Co" not in f.utter for _ in [0])
    deal_ids = {e.id for e in mb.emails if e.tags.get("deal")}
    assert set(f.rule["email_ids"]) == deal_ids


def test_hard_mode_flags():
    from compactgym.runner.common import hard_spec
    from compactgym.env.inbox.user import SimUser
    from compactgym.config import Config
    import asyncio
    t = generate_task(hard_spec(2000))
    assert all(1 <= f.inject_turn <= 3 for f in t.facts)  # early
    T = len(t.user_turns)
    assert all(g.turn >= int(T * 0.6) for g in t.goals if g.fact_id and g.kind != "offsite_create")  # exercised late
    # defaults are unchanged: the hard flags are off
    d = TaskSpec(seed=1)
    assert d.exercise_after is None and d.restate_on_reask and not d.gate_success
    # unrecoverable: asking again never restates a fact
    f = next(x for x in t.facts if x.reask_keywords)
    u = SimUser(t, Config(), None)
    u.cur = max(x.inject_turn for x in t.facts) + 1
    replies = [asyncio.run(u.respond(f"Sorry, {f.reask_keywords[0]}?"))[0] for _ in range(3)]
    assert all("already told you" in r and f.text not in r for r in replies)


def test_dates_are_strict():
    env = InboxEnv(generate_mailbox(1))
    assert env.execute("create_event", {"title": "x", "date": "2022022022", "start_time": "10:00"}).startswith("ERROR")
