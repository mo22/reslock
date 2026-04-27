from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid4().hex[:12]


class Lease(BaseModel):
    id: str = Field(default_factory=_new_id)
    pid: int
    host_pid: int | None = None
    resources: dict[str, int]
    priority: int = 0
    acquired_at: datetime = Field(default_factory=_utcnow)
    queued_at: datetime | None = None
    wait_sec: float | None = None
    estimated_seconds: int | None = None
    reclaimable: bool = False
    reclaim_requested: bool = False
    label: str | None = None
    pids: list[int] = Field(default_factory=list[int])
    actual_resources: dict[str, int] = Field(default_factory=dict)
    cpu_seconds: float | None = None
    progress: float | None = None

    model_config = {"extra": "forbid"}


class QueueEntry(BaseModel):
    id: str = Field(default_factory=_new_id)
    pid: int
    host_pid: int | None = None
    resources: dict[str, int]
    priority: int = 0
    queued_at: datetime = Field(default_factory=_utcnow)
    label: str | None = None

    model_config = {"extra": "forbid"}


SCHEMA_VERSION = 2
"""Current state-file schema version.

Version 2 switched GPU VRAM keys from ``gpu{index}_vram_mb`` (nvidia-smi index)
to ``gpu_{uuid}_vram_mb`` (host-stable GPU UUID). Index-based keys broke
coordination across containers with partial GPU mappings, where the NVIDIA
container runtime renumbers visible devices starting at 0. On reading a file
with a lower ``version``, ``reslock.state`` drops ``resources``, ``leases``,
and ``queue`` so consumers repopulate with UUID-keyed entries.
"""


class State(BaseModel):
    version: int = SCHEMA_VERSION
    resources: dict[str, int] = Field(default_factory=dict)
    leases: list[Lease] = Field(default_factory=list[Lease])
    queue: list[QueueEntry] = Field(default_factory=list[QueueEntry])

    model_config = {"extra": "forbid"}

    def available(self) -> dict[str, int]:
        used: dict[str, int] = {}
        for lease in self.leases:
            for key, val in lease.resources.items():
                used[key] = used.get(key, 0) + val
        return {key: total - used.get(key, 0) for key, total in self.resources.items()}

    def can_fit(self, resources: dict[str, int]) -> bool:
        avail = self.available()
        return all(avail.get(key, 0) >= val for key, val in resources.items())

    def reclaimable_for(self, resources: dict[str, int]) -> list[Lease]:
        """Return reclaimable leases that would need to be reclaimed to fit the request."""
        avail = self.available()
        shortfall: dict[str, int] = {}
        for key, val in resources.items():
            deficit = val - avail.get(key, 0)
            if deficit > 0:
                shortfall[key] = deficit
        if not shortfall:
            return []
        candidates = [ls for ls in self.leases if ls.reclaimable and not ls.reclaim_requested]
        candidates.sort(key=lambda ls: ls.priority)
        result: list[Lease] = []
        remaining = dict(shortfall)
        for lease in candidates:
            if not remaining:
                break
            helps = False
            for key in list(remaining):
                contrib = lease.resources.get(key, 0)
                if contrib > 0:
                    helps = True
                    remaining[key] -= contrib
                    if remaining[key] <= 0:
                        del remaining[key]
            if helps:
                result.append(lease)
        if remaining:
            return []
        return result


class PoolStatus(BaseModel):
    resources: dict[str, int]
    available: dict[str, int]
    leases: list[Lease]
    queue: list[QueueEntry]
