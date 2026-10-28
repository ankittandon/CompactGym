"""Shared interfaces. Keep stable: training builds on them."""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    meta: dict = Field(default_factory=dict)  # turn, tokens, origin ("compaction_summary", "pinned", ...)

    def to_api(self) -> dict:
        d: dict[str, Any] = {"role": self.role, "content": self.content if self.content is not None else ""}
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d


class CompactionAction(str, Enum):
    KEEP = "keep"
    TRUNCATE_VERBATIM = "truncate_verbatim"  # Cliff-style, no rewriting
    CLEAR_TOOL_OUTPUTS = "clear_tool_outputs"
    SUMMARIZE_OLDER = "summarize_older"  # summary + last k turns
    REWRITE = "rewrite"
    OFFLOAD_TO_MEMORY = "offload_to_memory"  # reversible; agent can recall()
    DEFER_TO_FRONTIER = "defer_to_frontier"
    EDIT_CONTEXT = "edit_context"  # in-place edits by message index (trim / squish / hide), originals saved to memory


class CompactionResult(BaseModel):
    action: CompactionAction
    messages: list[Message]
    summary: str | None = None
    pinned: list[str] = Field(default_factory=list)  # exact "never lose" facts kept in a pinned tier
    confidence: float | None = None  # predicted retention / (1 - InfoLoss), in [0,1]
    compactor_latency_s: float = 0.0
    compactor_tokens_in: int = 0
    compactor_tokens_out: int = 0
    blocking: bool = True
    # extras (not part of the frozen interface; default so old callers still work)
    compactor_model: str | None = None  # which model's price applies to compactor tokens
    forced: bool = False  # produced by the budget-fallback path
    stale: bool = False
    calls: list[dict] = Field(default_factory=list)  # per-call cost breakdown: {role, tokens_in, tokens_out, cached}


class EpisodeState(BaseModel):
    """Mutable per-episode state handed to compactors."""
    episode_id: str = ""
    turn: int = 0  # main-model call index
    profile: dict = Field(default_factory=dict)
    goal: str = ""
    pending_subtasks: list[str] = Field(default_factory=list)
    memory: Any = None  # harness.memory.MemoryStore (kept Any to avoid import cycle)
    full_log: Any = Field(default=None, exclude=True)  # uncompacted trajectory (list[Message]); stateless compactors (cliff) read it
    pinned: list[str] = Field(default_factory=list)  # pinned tier
    summary: str | None = None  # current compaction summary text (if any)
    context_tokens: int = 0
    n_compactions: int = 0
    keep_k: int = 2
    settings: dict = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}


class Compactor(Protocol):
    name: str

    async def __call__(self, messages: list[Message], state: EpisodeState, budget_tokens: int) -> CompactionResult: ...
