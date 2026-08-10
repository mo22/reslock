from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

from reslock.detect import gpu_vram_key, parse_disk_mb_key, parse_gpu_vram_key


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid4().hex[:12]


def normalize_gpu_asks(
    *, vram_mb: list[int] | None, vram_mb_each: int | None, num_gpus: int
) -> list[int]:
    """Validate the two GPU request shapes and return asks largest-first."""
    if num_gpus < 0:
        raise ValueError(f"num_gpus must be non-negative, got {num_gpus}")
    if vram_mb is not None:
        if vram_mb_each is not None or num_gpus != 0:
            raise ValueError(
                "vram_mb is mutually exclusive with vram_mb_each and num_gpus "
                f"(got vram_mb_each={vram_mb_each!r}, num_gpus={num_gpus})"
            )
        if not vram_mb:
            raise ValueError("vram_mb must contain at least one per-slot request")
        if any(amount <= 0 for amount in vram_mb):
            raise ValueError(f"vram_mb values must be positive ints, got {vram_mb!r}")
        return sorted(vram_mb, reverse=True)
    if num_gpus > 0 and (vram_mb_each is None or vram_mb_each <= 0):
        raise ValueError(
            "vram_mb_each must be a positive int when num_gpus > 0 "
            f"(got vram_mb_each={vram_mb_each!r}, num_gpus={num_gpus})"
        )
    if num_gpus == 0 and vram_mb_each is not None:
        raise ValueError(
            f"vram_mb_each={vram_mb_each!r} requires num_gpus > 0; "
            "set num_gpus to a positive number or drop vram_mb_each"
        )
    if vram_mb_each is None:
        return []
    return [vram_mb_each] * num_gpus


class Lease(BaseModel):
    """A resource reservation.

    A ``Lease`` represents capacity that is currently held — it does NOT carry
    work-tracking metadata like remaining ETA or progress. Those live on the
    ``QueueEntry`` (see ``Lease.id`` referenced via ``QueueEntry.lease_id``).
    Reasoning: long-lived reclaimable leases (e.g. an aiserver model-load lease
    that lives across many requests so weights stay in VRAM) need to express
    "I'm holding VRAM" separately from "I'm currently running a 30s request".
    Peers reading status can compute remaining time from the attached entry's
    ``estimated_seconds`` + ``progress`` without being misled by an
    ``acquired_at`` that's an hour stale.
    """

    id: str = Field(default_factory=_new_id)
    pid: int
    host_pid: int | None = None
    resources: dict[str, int]
    priority: int = 0
    acquired_at: datetime = Field(default_factory=_utcnow)
    queued_at: datetime | None = None
    wait_sec: float | None = None
    reclaimable: bool = False
    reclaim_requested: bool = False
    label: str | None = None
    pids: list[int] = Field(default_factory=list[int])
    actual_resources: dict[str, int] = Field(default_factory=dict)
    cpu_seconds: float | None = None

    model_config = {"extra": "forbid"}


class QueueEntry(BaseModel):
    """A unit of work — queued waiting for a Lease, or actively running attached to one.

    Lifecycle:

    * ``queued_at`` is set on enqueue, ``started_at`` is unset, ``lease_id`` is None.
    * On promotion the scheduler creates a ``Lease``, sets ``lease_id`` and
      ``started_at``. The entry stays in ``state.queue`` for the lifetime of
      the work — peers reading ``pool.status().queue`` see both pending and
      running work. ``is_active`` distinguishes the two.
    * Per-request work attached to an existing lease (e.g. one inference call
      on aiserver's persistent model-load lease) is created via
      ``LeaseHandle.start_work(...)``. Such an entry has empty resource demand
      (``num_gpus == 0`` and ``resources == {}``) and skips the scheduler:
      ``lease_id`` and ``started_at`` are set immediately at enqueue.
    * On ``EntryHandle.complete()`` the entry is dropped. ``LeaseHandle.release()``
      also drops any still-attached entries (defensive cleanup).

    The work-tracking fields (``estimated_seconds``, ``progress``) live here
    rather than on ``Lease`` so peers can compute remaining time without
    consulting ``Lease.acquired_at``, which is meaningless on long-lived
    reclaimable leases.
    """

    id: str = Field(default_factory=_new_id)
    pid: int
    host_pid: int | None = None

    resources: dict[str, int] = Field(default_factory=dict)
    vram_mb: list[int] | None = None
    vram_mb_each: int | None = None
    num_gpus: int = 0
    reclaimable_intent: bool = False

    priority: int = 0
    label: str | None = None

    queued_at: datetime = Field(default_factory=_utcnow)
    started_at: datetime | None = None
    lease_id: str | None = None

    estimated_seconds: int | None = None
    progress: float | None = None

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_gpu_request(self) -> QueueEntry:
        normalize_gpu_asks(
            vram_mb=self.vram_mb,
            vram_mb_each=self.vram_mb_each,
            num_gpus=self.num_gpus,
        )
        return self

    @property
    def is_active(self) -> bool:
        """True when the entry has been promoted (or attached) to a Lease."""
        return self.lease_id is not None


SCHEMA_VERSION = 4
"""Current state-file schema version.

Version 4 (v0.11.0):

* ``QueueEntry`` gained ``vram_mb`` for non-uniform per-slot GPU VRAM asks.
  The existing ``vram_mb_each`` + ``num_gpus`` request shape remains valid.
* GPU requests are paired largest-first with GPUs sorted by effective free
  VRAM descending. The two request shapes are mutually exclusive.
* Mixed-version operation is unsafe: stop all consumers sharing the state
  file, delete ``state.json``, and restart them on v0.11.0 or newer.

Version 3 (v0.8.0):

* ``Lease`` lost ``estimated_seconds`` and ``progress`` — work tracking moved
  to ``QueueEntry`` so long-lived reclaimable leases don't carry stale ETAs.
* ``QueueEntry`` gained ``vram_mb_each``, ``num_gpus``, ``reclaimable_intent``,
  ``started_at``, ``lease_id``, ``estimated_seconds``, ``progress``. The
  request shape for GPUs changed from caller-pinned ``gpu_<uuid>_vram_mb``
  keys to ``vram_mb_each`` + ``num_gpus``; the scheduler resolves to specific
  UUIDs at promotion time using a spread-placement policy (most-free first).
* Entries persist across active work (not dropped on promotion) so peers can
  walk ``state.queue`` and see both queued and running work.
* Reclaim skips leases that have any active ``QueueEntry`` attached.

Version 2 switched GPU VRAM keys from ``gpu{index}_vram_mb`` to
``gpu_{uuid}_vram_mb``. Version 1 used index-based keys.

On reading a file with a lower ``version``, ``reslock.state`` drops
``resources``, ``leases``, and ``queue`` so consumers repopulate via
``set_resources()`` and re-acquire under the current schema. Coordinated
upgrade across consumers is required.
"""


class State(BaseModel):
    version: int = SCHEMA_VERSION
    resources: dict[str, int] = Field(default_factory=dict)
    leases: list[Lease] = Field(default_factory=list[Lease])
    queue: list[QueueEntry] = Field(default_factory=list[QueueEntry])

    model_config = {"extra": "forbid"}

    def used_per_key(self) -> dict[str, int]:
        used: dict[str, int] = {}
        for lease in self.leases:
            for key, val in lease.resources.items():
                used[key] = used.get(key, 0) + val
        return used

    def available(self) -> dict[str, int]:
        used = self.used_per_key()
        return {key: total - used.get(key, 0) for key, total in self.resources.items()}

    def per_gpu_free(self) -> dict[str, int]:
        """Free VRAM per GPU UUID, derived from ``resources`` capacities and lease holdings.

        Returns ``{gpu_uuid: free_mb}`` for every GPU registered via
        ``set_resources()``. Non-GPU keys are ignored.
        """
        used = self.used_per_key()
        out: dict[str, int] = {}
        for key, total in self.resources.items():
            uuid_str = parse_gpu_vram_key(key)
            if uuid_str is None:
                continue
            out[uuid_str] = total - used.get(key, 0)
        return out

    def try_resolve_request(
        self,
        *,
        vram_mb: list[int] | None = None,
        vram_mb_each: int | None,
        num_gpus: int,
        non_gpu: dict[str, int],
        nvml_free: dict[str, int] | None = None,
        disk_free: dict[str, int] | None = None,
    ) -> dict[str, int] | None:
        """Resolve an abstract resource request into a concrete bindings dict.

        Args:
            vram_mb: Non-uniform per-slot GPU VRAM asks. Mutually exclusive
                with ``vram_mb_each`` and ``num_gpus``.
            vram_mb_each: Per-GPU VRAM ask. Required when ``num_gpus > 0``.
                Setting this when ``num_gpus == 0`` is a programming error and
                raises — silently dropping the VRAM ask would let
                ``pool.acquire(vram_mb_each=8000)`` (a forgotten ``num_gpus=``)
                grant a non-GPU lease with no GPU reservation.
            num_gpus: Number of GPUs needed (any). 0 for non-GPU requests.
            non_gpu: Non-GPU keys (e.g. ``{"ram_mb": 8000}``).
            nvml_free: Optional per-UUID NVML-reported free VRAM. When
                supplied, placement uses ``min(internal_free, nvml_free)`` per
                GPU so that an internally-empty GPU that's actually held by an
                external (unaccounted) process is excluded from candidate
                picks. UUIDs missing from *nvml_free* are treated as 0 free
                (conservative: a GPU we registered but the driver didn't
                report counts as fully held externally).
            disk_free: Actual free disk space (statvfs) per ``disk_mb@<path>``
                key. Disk keys use free-space admission instead of registered
                capacity: grant only while ``request + sum(active disk leases
                on the key) <= actual free``. A disk key missing from
                *disk_free* refuses (no ground truth, conservative) — the pool
                supplies it via ``get_disk_free_mb()`` before every attempt.

        Returns:
            A ``{key: amount}`` dict suitable for storing on a ``Lease`` —
            with ``gpu_<uuid>_vram_mb`` keys filled in for the chosen GPUs —
            or ``None`` if the request can't currently be satisfied.

        Placement policy: asks are sorted descending, GPUs are sorted by
        effective free VRAM descending (ties broken by UUID), and the two
        lists are paired. If any ask exceeds its paired GPU's free VRAM, no
        assignment exists. This spreads multi-GPU work across the emptiest
        cards, matching the access pattern of multi-GPU inference (NCCL /
        tensor parallel). Folding NVML into placement lets the scheduler skip
        an internally-free GPU held by an external process.
        """
        gpu_asks = normalize_gpu_asks(
            vram_mb=vram_mb,
            vram_mb_each=vram_mb_each,
            num_gpus=num_gpus,
        )

        avail_non_gpu = self.available()
        used = self.used_per_key()
        for key, val in non_gpu.items():
            if val < 0:
                raise ValueError(f"resource {key!r} amount must be non-negative, got {val}")
            if parse_disk_mb_key(key) is not None:
                # Free-space admission: leases reserve headroom against the
                # live statvfs value, not a registered capacity.
                free = (disk_free or {}).get(key)
                if free is None or val > free - used.get(key, 0):
                    return None
            elif val > avail_non_gpu.get(key, 0):
                return None

        bindings: dict[str, int] = dict(non_gpu)

        if not gpu_asks:
            return bindings

        per_gpu = self.per_gpu_free()
        if nvml_free is not None:
            per_gpu = {u: min(free, nvml_free.get(u, 0)) for u, free in per_gpu.items()}
        # Sort by free desc, ties broken by UUID asc for determinism.
        candidates = sorted(per_gpu.items(), key=lambda item: (-item[1], item[0]))
        if len(candidates) < len(gpu_asks):
            return None
        for ask, (uuid_str, free) in zip(gpu_asks, candidates[: len(gpu_asks)], strict=True):
            if free < ask:
                return None
            bindings[gpu_vram_key(uuid_str)] = ask
        return bindings

    def can_fit(self, resources: dict[str, int]) -> bool:
        """True if the literal per-key reservation in *resources* fits available capacity.

        Used for non-GPU pre-checks and for the priority gate's "would this
        higher-priority entry be promotable on its own next tick?" question.
        For abstract GPU requests use :meth:`try_resolve_request`.
        """
        avail = self.available()
        return all(avail.get(key, 0) >= val for key, val in resources.items())

    def active_lease_ids(self) -> set[str]:
        """Lease IDs that have at least one active ``QueueEntry`` attached."""
        return {e.lease_id for e in self.queue if e.lease_id is not None}

    def reclaimable_for_shortfall(
        self, shortfall: dict[str, int], *, partial: bool = False
    ) -> list[Lease]:
        """Pick reclaimable leases that cover a precomputed shortfall.

        Walks reclaimable, not-yet-reclaim-requested leases in ascending
        priority order, accumulating those that contribute to any shortfall
        key. Leases with an active ``QueueEntry`` attached are skipped — the
        consumer is mid-request and would orphan in-flight work.

        When ``partial=False`` (the default), returns the selected leases only
        if they together fully cover the shortfall — otherwise returns an
        empty list, meaning "no point evicting, the request can't be
        satisfied even by reclaiming everything reclaimable". This matches
        the strict accounting case where partial eviction would be pointless
        churn since ``state.can_fit`` is deterministic from internal
        accounting alone.

        When ``partial=True``, returns every reclaimable that contributes,
        even if the union doesn't cover the shortfall. This is what the NVML
        pre-flight wants: the gap may be partly an unaccounted external
        process that reslock can never reclaim, but freeing what we *can*
        free is still useful — the lease may grant on a later poll once the
        external process exits, or its outer caller (e.g. scriba's 600s
        budget) decides to give up.
        """
        if not shortfall:
            return []
        active_ids = self.active_lease_ids()
        candidates = [
            ls
            for ls in self.leases
            if ls.reclaimable and not ls.reclaim_requested and ls.id not in active_ids
        ]
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
        if remaining and not partial:
            return []
        return result


class PoolStatus(BaseModel):
    resources: dict[str, int]
    available: dict[str, int]
    leases: list[Lease]
    queue: list[QueueEntry]
