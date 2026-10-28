"""One file per strategy. ``make_compactor(name, cfg, llm)`` builds a fresh (per-episode) instance."""
from __future__ import annotations

from .base import BaseCompactor
from .cliff import Cliff
from .constraint_extractor import ConstraintExtractor
from .edit_context import EditContext
from .frontier import Frontier
from .memory_offload import MemoryOffload
from .recent_k import RecentK
from .small_prompted import SmallPrompted
from .threshold import Never, ThresholdAnthropic, ThresholdCodex, ThresholdPi

REGISTRY: dict[str, type[BaseCompactor]] = {
    "never": Never, "threshold_pi": ThresholdPi, "threshold_anthropic": ThresholdAnthropic,
    "threshold_codex": ThresholdCodex, "recent_k": RecentK,
    "cliff": Cliff, "constraint_extractor": ConstraintExtractor, "memory_offload": MemoryOffload,
    "frontier": Frontier, "small_prompted": SmallPrompted,
}
# Shelley-style in-place editing: the frontier model (baseline) or the small model (what we train) edits the live context by message index
REGISTRY["edit_context"] = EditContext
REGISTRY["edit_context_small"] = type("EditContextSmall", (EditContext,), {"name": "edit_context_small", "__init__": lambda self, cfg, llm=None, **kw: EditContext.__init__(self, cfg, llm, role="small", **kw)})
# llmlingua2 is intentionally not registered: it needs torch + a local xlm-roberta model (no local models on this machine).
SKIPPED = {"llmlingua2": "needs torch and a local model; excluded (no local models allowed). Add a Fireworks-hosted equivalent if desired."}


def make_compactor(name: str, cfg, llm=None, **kw) -> BaseCompactor:
    if name not in REGISTRY:
        raise KeyError(f"unknown compactor {name!r}; skipped: {SKIPPED}")
    return REGISTRY[name](cfg, llm, **kw)
