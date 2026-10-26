"""R = success - w_p*pause - w_a*forgetting - w_c*cost - lambda*infoloss (weights from the profile; lambda is a setting)."""
from __future__ import annotations

from ..config import load_profile


def reward(success: float, pause_s: float, forgetting: float, cost: float, infoloss: float, profile: dict | str = "default", lam: float | None = None) -> dict:
    p = load_profile(profile) if isinstance(profile, str) else profile
    lam = p.get("lambda", 0.5) if lam is None else lam
    terms = {"success": success, "pause": -p["w_p"] * pause_s, "forgetting": -p["w_a"] * forgetting, "cost": -p["w_c"] * cost, "infoloss": -lam * infoloss}
    return {"reward": sum(terms.values()), **{f"r_{k}": v for k, v in terms.items()}}
