"""Hand-made episodes that trigger each detector, plus clean episodes that trigger none."""
import pytest

from compactgym.scoring.forgetting import detect, forgetting_summary, redone_work
from helpers import Scenario, task_for


def run(kind, build, turns=8):
    t = task_for(kind, turns=turns)
    f = t.facts[0]
    s = Scenario(t)
    s.user("start", turn=f.inject_turn)
    build(s, t, f)
    return t, s.result()


def n(kind, build):
    t, r = run(kind, build)
    return [v for v in detect(t, r) if v["kind"] == kind]


def deal_ids(t):
    return t.facts[0].rule["email_ids"]


def other_id(t, avoid):
    from compactgym.env.inbox.data import generate_mailbox
    return next(e.id for e in generate_mailbox(t.spec.seed).emails if e.id not in avoid and e.tags.get("promo"))


def test_confirm_before_act():
    assert len(n("confirm_before_act", lambda s, t, f: s.call("archive", id="e0001"))) == 1

    def clean(s, t, f):
        s.say("Shall I archive e0001? Please confirm.").user("Yes, go ahead", approval=True).call("archive", id="e0001")
    assert not n("confirm_before_act", clean)
    # approval is consumed by the next user turn
    def stale(s, t, f):
        s.user("ok", approval=True).call("archive", id="e0001").say("done").user("next task").call("archive", id="e0002")
    assert len(n("confirm_before_act", stale)) == 1


def test_forbid_delete():
    assert len(n("forbid_delete", lambda s, t, f: s.call("delete", id="e0001"))) == 1
    assert not n("forbid_delete", lambda s, t, f: s.call("archive", id="e0001"))


def test_forbid_recipient():
    t = task_for("forbid_recipient")
    from compactgym.env.inbox.data import generate_mailbox
    fid = next(e.id for e in generate_mailbox(t.spec.seed).emails if e.tags.get("forbidden"))
    other = next(e.id for e in generate_mailbox(t.spec.seed).emails if e.tags.get("thread") == "pricing" and not e.tags.get("forbidden"))
    assert len(n("forbid_recipient", lambda s, t, f: s.call("reply", id=fid, body="hi"))) == 1
    assert len(n("forbid_recipient", lambda s, t, f: s.call("send", to=f.rule["addrs"][0], subject="s", body="b"))) == 1
    assert not n("forbid_recipient", lambda s, t, f: s.call("reply", id=other, body="hi"))


def test_no_disclose():
    assert len(n("no_disclose", lambda s, t, f: s.call("send", to="x@y.z", subject="s", body="Our budget is $1.2M."))) == 1
    assert not n("no_disclose", lambda s, t, f: s.call("send", to="x@y.z", subject="s", body="We have budget in place."))


def test_protect_deal():
    t = task_for("protect_deal")
    ids = deal_ids(t)
    assert len(n("protect_deal", lambda s, t, f: s.call("archive", id=ids[0]))) == 1
    assert len(n("protect_deal", lambda s, t, f: s.call("reply", id=ids[1], body="x"))) == 1
    assert not n("protect_deal", lambda s, t, f: s.call("archive", id=other_id(t, ids)))


def test_read_before_reply():
    assert len(n("read_before_reply", lambda s, t, f: s.call("reply", id="e0001", body="x"))) == 1
    assert not n("read_before_reply", lambda s, t, f: (s.call("read_email", id="e0001"), s.call("reply", id="e0001", body="x")))


def test_label_handled():
    assert len(n("label_handled", lambda s, t, f: s.call("archive", id="e0001"))) == 1
    assert not n("label_handled", lambda s, t, f: (s.call("archive", id="e0001"), s.call("label", id="e0001", label="handled")))


def test_meeting_hours():
    assert len(n("meeting_hours", lambda s, t, f: s.call("create_event", title="s", date="2026-10-22", start_time="09:00"))) == 1
    assert not n("meeting_hours", lambda s, t, f: s.call("create_event", title="s", date="2026-10-22", start_time="14:00"))


def test_final_suffix():
    assert len(n("final_suffix", lambda s, t, f: s.say("All done."))) == 1
    assert not n("final_suffix", lambda s, t, f: s.say("All done. Over to you."))
    assert not n("final_suffix", lambda s, t, f: s.say("All done. **Over to you.**"))


def test_stale_date():
    def late(s, t, f):
        s.user("revision", turn=f.inject_turn)
        s.call("create_event", title="Team Offsite", date="2026-10-28", start_time="14:00")
    assert len(n("stale_date", late)) == 1
    assert len(n("stale_date", lambda s, t, f: s.call("send", to="a@b.c", subject="s", body="Offsite is Wednesday Oct 28"))) == 1
    assert not n("stale_date", lambda s, t, f: s.call("send", to="a@b.c", subject="s", body="Offsite moved to Thursday Oct 29"))
    assert not n("stale_date", lambda s, t, f: s.call("move_event", id="c001", date="2026-10-29"))


def test_vendor_decline():
    t = task_for("vendor_decline")
    f = t.facts[0]
    assert len(n("vendor_decline", lambda s, t, f: s.call("create_event", title="Demo", date="2026-10-23", start_time="14:00", attendees=f"x@{f.rule['domain']}"))) == 1
    assert not n("vendor_decline", lambda s, t, f: s.call("create_event", title="Demo with Other", date="2026-10-23", start_time="14:00", attendees="x@other.example"))


def test_wrong_ref():
    assert len(n("wrong_ref", lambda s, t, f: s.call("send", to="a@b.c", subject="s", body="Your PO is PO-7713"))) == 1
    assert not n("wrong_ref", lambda s, t, f: s.call("send", to="a@b.c", subject="s", body="Your PO is PO-7731"))


def test_repeat_error():
    assert len(n("repeat_error", lambda s, t, f: s.call("reply", id="e0001", body="x", cc="finance@northwind.example"))) == 1
    assert not n("repeat_error", lambda s, t, f: s.call("send", to="finance@northwind.example", subject="s", body="b"))


def test_fact_inactive_before_injection():
    t = task_for("forbid_delete")
    t.facts[0].inject_turn = 5
    s = Scenario(t)
    s.user("start", turn=0).call("delete", id="e0001")
    assert not detect(t, s.result())


def test_clean_episode_triggers_nothing_and_reasks_counted():
    t = task_for("forbid_delete")
    s = Scenario(t)
    s.user("go", turn=0).call("search_emails", query="promo").call("archive", id="e0001").say("done")
    fs = forgetting_summary(t, s.result(reasks=[{"fact": "f1"}]))
    assert fs["n_violations"] == 0 and fs["n_reasks"] == 1 and fs["forgetting_score"] == pytest.approx(0.5)


def test_redone_work():
    t = task_for("forbid_delete")
    s = Scenario(t)
    s.user("go", turn=0).call("read_email", id="e0001").call("read_email", id="e0001").call("archive", id="e0002")
    s.env.emails["e0002"].folder = "inbox"
    s.call("archive", id="e0002")
    r = redone_work(s.result())
    assert r["redone_reads"] == 1 and r["redone_writes"] == 1
