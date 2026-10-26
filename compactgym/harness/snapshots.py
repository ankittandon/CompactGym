"""Per-turn snapshots (episode_id, turn, messages, env_state, rng_state, ...) so any turn can be branched and replayed."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

from pydantic import BaseModel, Field

from ..interfaces import Message


class Snapshot(BaseModel):
    episode_id: str
    turn: int  # main-model call index; snapshot is taken BEFORE this turn's compaction hook
    messages: list[Message]  # live context
    env_state: dict
    rng_state: dict  # simulated-user state incl. python RNG state
    full_log: list[Message] = Field(default_factory=list)  # complete uncompacted trajectory so far
    state: dict = Field(default_factory=dict)  # EpisodeState (minus memory)
    memory: list[dict] = Field(default_factory=list)
    extras: dict = Field(default_factory=dict)  # counters: uid, user_turn, prev request hashes, final texts ...


class SnapshotStore:
    def __init__(self, root: str | Path | None = None, keep_in_memory: bool = False):
        self.root = Path(root) if root else None
        self.mem: dict[tuple[str, int], Snapshot] = {} if keep_in_memory else None  # type: ignore[assignment]

    def _path(self, episode_id: str, turn: int) -> Path:
        return self.root / episode_id / f"{turn:04d}.json.gz"  # type: ignore[operator]

    def save(self, snap: Snapshot) -> None:
        if self.mem is not None:
            self.mem[(snap.episode_id, snap.turn)] = snap
        if self.root:
            p = self._path(snap.episode_id, snap.turn)
            p.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(p, "wt") as f:
                f.write(snap.model_dump_json())

    def load(self, episode_id: str, turn: int) -> Snapshot:
        if self.mem is not None and (episode_id, turn) in self.mem:
            return self.mem[(episode_id, turn)]
        with gzip.open(self._path(episode_id, turn), "rt") as f:
            return Snapshot.model_validate_json(f.read())

    def turns(self, episode_id: str) -> list[int]:
        if self.root and (self.root / episode_id).exists():
            return sorted(int(p.name.split(".")[0]) for p in (self.root / episode_id).glob("*.json.gz"))
        return sorted(t for (e, t) in (self.mem or {}) if e == episode_id)
