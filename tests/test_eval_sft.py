import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from eval_sft import baseline_reply, covers, parse_reply, score_one, summarize  # noqa: E402

TARGET = {"action": "summarize_older", "summary": "s", "confidence": 0.8,
          "pinned": ["Don't email Farah Varga anything about this -- keep Farah out of the loop.", "End every message with the exact phrase \"Over to you.\""]}


def reply(**kw):
    return json.dumps({"action": "summarize_older", "summary": "text", "pinned": [], "confidence": 0.7, **kw})


def test_parse_and_validity():
    assert parse_reply("noise {\"a\": 1} tail") == {"a": 1}
    assert parse_reply("no json") is None
    assert not score_one("not json", TARGET)["valid_json"]
    assert not score_one(json.dumps({"action": "summarize_older"}), TARGET)["valid_json"]  # missing fields
    r = score_one(reply(action="shrink"), TARGET)
    assert r["valid_json"] and not r["valid_action"] and not r["action_acc"]


def test_pinned_matching_is_about_content_not_wording():
    assert covers("Do not email Farah Varga about this, keep Farah out of the loop", TARGET["pinned"][0])
    assert not covers("Archive the promo emails", TARGET["pinned"][0])
    good = score_one(reply(pinned=["Keep Farah Varga out of the loop - don't email Farah anything about this", "End every message with 'Over to you.'"]), TARGET)
    assert good["pinned_recall"] == 1.0 and good["pinned_prec"] == 1.0
    padded = score_one(reply(pinned=["Keep Farah Varga out of the loop - don't email Farah anything about this", "Sam likes tea"]), TARGET)
    assert padded["pinned_recall"] == 0.5 and padded["pinned_prec"] == 0.5


def test_summary_and_confidence():
    assert score_one(reply(summary=""), TARGET)["summary_ok"] is False  # summarize action with no summary
    assert score_one(reply(action="truncate_verbatim", summary=""), TARGET)["summary_ok"] is True
    assert score_one(reply(action="truncate_verbatim", summary="oops"), TARGET)["summary_ok"] is False
    assert abs(score_one(reply(confidence=0.5), TARGET)["conf_err"] - 0.3) < 1e-9


def test_baselines_bracket_the_scorer():
    targets = [TARGET, dict(TARGET, action="truncate_verbatim", summary="", pinned=[])]
    oracle = summarize([score_one(r, t) for r, t in zip(baseline_reply("oracle", targets), targets)], targets)
    assert oracle["action_acc"] == 1.0 and oracle["pinned_recall"] == 1.0 and oracle["conf_err"] == 0.0
    trunc = summarize([score_one(r, t) for r, t in zip(baseline_reply("always_truncate", targets), targets)], targets)
    assert trunc["action_acc"] == 0.5 and trunc["valid_json"] == 1.0
