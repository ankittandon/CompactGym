"""R = success - w_p*pause - w_a*forgetting - w_c*cost - lambda*infoloss + w_r*retention (weights from the profile; lambda is a setting).

* ``retention`` is the dense term: fraction of planted facts the compacted context still answers correctly (probe-based, available right
  after every compaction), so the controller is not trained on delayed outcomes alone. In the pilot it was the more direct signal.
* ``infoloss`` should be the *signed* value (``InfoLossResult.raw``): clipping at 0 discards the cases where the compacted context
  is easier to continue from, and the signed term keeps those as a (small) bonus.
"""
from __future__ import annotations

from ..config import load_profile


def reward(success: float, pause_s: float, forgetting: float, cost: float, infoloss: float, profile: dict | str = "default", lam: float | None = None,
           retention: float | None = None) -> dict:
    p = load_profile(profile) if isinstance(profile, str) else profile
    lam = p.get("lambda", 0.5) if lam is None else lam
    terms = {"success": success, "pause": -p["w_p"] * pause_s, "forgetting": -p["w_a"] * forgetting, "cost": -p["w_c"] * cost, "infoloss": -lam * infoloss,
             "retention": p.get("w_r", 0.0) * (retention if retention is not None else 0.0)}
    return {"reward": sum(terms.values()), **{f"r_{k}": v for k, v in terms.items()}}
