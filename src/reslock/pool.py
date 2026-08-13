from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from reslock.detect import (
    CPU_CORES_KEY,
    RAM_MB_KEY,
    disk_mb_key,
    get_disk_free_mb,
    get_host_pid,
    get_self_actual_resources,
    get_self_cpu_seconds,
    gpu_vram_key,
    parse_disk_mb_key,
    parse_gpu_vram_key,
)

if TYPE_CHECKING:
    from reslock.audit import OrphanReport
from reslock.models import Lease, PoolStatus, QueueEntry, State, normalize_gpu_asks
from reslock.nvml import (
    NvmlUnavailableError,
    nvml_free_vram_mb,
    nvml_total_vram_mb,
)
from reslock.state import (
    DEFAULT_STATE_PATH,
    SchemaVersionMismatch,
    ensure_state_file,
    read_state,
    read_state_clean,
    transact,
)

logger = logging.getLogger(__name__)

GPU_CAPACITY_MIN_RATIO = 0.95
"""How far below the NVML physical total a registered GPU capacity may sit silently.

The CUDA driver reports less VRAM than the card physically has, and every
detection path that goes through CUDA (torch's ``total_memory``, our
``detect_gpu_vram_mb_cuda_driver``) registers *that* number — so an exact
comparison against the NVML total warns on every correctly configured host.

Measured 2026-08-12, both against ``nvidia-smi``'s ``memory.total``:

* kirk, RTX 3090, Linux driver 570.133.20 — ``cuDeviceTotalMem`` 24135 MB vs
  24576 MB physical, i.e. **1.8 %** low.
* spock, RTX 3090, Windows 11 / WDDM — 24575 MB vs 24576 MB, i.e. **0.004 %**.

The reserve is neither constant nor proportional across driver models, which
is why this is a ratio rather than a fixed MB allowance. The incident the
tripwire exists for sat at ~18000 of 24576 MB — **27 %** low, an order of
magnitude past any driver reserve, so 5 % separates the two cases with room
to spare in both directions.
"""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _read_nvml_for_request(num_gpus: int) -> dict[str, int] | None:
    """Read NVML free VRAM only when the request includes GPUs.

    Returns ``{gpu_uuid: free_mb}`` for the driver's view, or ``None`` when no
    GPU is requested. Raises ``NvmlUnavailableError`` when GPUs are requested
    but pynvml is missing or ``nvmlInit()`` fails — by design, since a
    CUDA-capable host with a broken NVML install would otherwise silently
    fall back to internal-accounting-only and reintroduce the
    accounting-vs-driver drift the pre-flight is meant to catch.
    """
    if num_gpus <= 0:
        return None
    return nvml_free_vram_mb()


def _read_disk_free_for_request(non_gpu: dict[str, int]) -> dict[str, int] | None:
    """Read statvfs ground truth for every ``disk_mb@<path>`` key in the request.

    Returns ``{key: actual_free_mb}``, or ``None`` when the request holds no
    disk keys. Raises ``ValueError`` when a mount path can't be statted — a
    typo'd path should fail the acquire loudly, not queue forever against a
    key the scheduler will always refuse.
    """
    disk_free: dict[str, int] = {}
    for key in non_gpu:
        path = parse_disk_mb_key(key)
        if path is None:
            continue
        free = get_disk_free_mb(path)
        if free is None:
            raise ValueError(f"cannot stat free disk space for {key!r} (path {path!r})")
        disk_free[key] = free
    return disk_free or None


def _gate_disk_free(non_gpu: dict[str, int]) -> dict[str, int] | None:
    """Tolerant variant of :func:`_read_disk_free_for_request` for gate entries.

    A peer's queue entry may reference a path we can't stat; treating it as
    0 free (entry can't fit → doesn't block us) beats raising and killing
    our own acquire.
    """
    disk_free: dict[str, int] = {}
    for key in non_gpu:
        path = parse_disk_mb_key(key)
        if path is None:
            continue
        disk_free[key] = get_disk_free_mb(path) or 0
    return disk_free or None


def _validate_non_gpu(non_gpu: dict[str, int]) -> None:
    """Reject ``gpu_<uuid>_vram_mb`` keys passed via the legacy v2 acquire shape.

    Since the v3 hard-cut callers use abstract asks and the scheduler binds
    specific UUIDs at promotion time. Raising loudly here is nicer than
    silently failing to schedule a GPU lease.
    """
    for key in non_gpu:
        if parse_gpu_vram_key(key) is not None:
            raise TypeError(
                f"GPU keys are not accepted in acquire() — got {key!r}. "
                "Use vram_mb_each + num_gpus or vram_mb=[...] instead; the scheduler picks "
                "GPUs automatically (spread placement, most-free first). "
                "Capacity registration via set_resources() still uses "
                "gpu_<uuid>_vram_mb keys."
            )


def _split_vram_mb_argument(
    vram_mb: list[int] | int | None, non_gpu: dict[str, int]
) -> tuple[list[int] | None, dict[str, int]]:
    """Separate the new slot-list API from the legacy free-form counter.

    Before per-slot asks existed, ``vram_mb=4000`` flowed through
    ``**non_gpu_resources`` as an ordinary custom counter. Keep that API
    intact: only a list selects the new abstract GPU scheduler path.
    """
    if isinstance(vram_mb, int):
        merged = dict(non_gpu)
        merged["vram_mb"] = vram_mb
        return None, merged
    return vram_mb, non_gpu


def _fold_first_class(
    non_gpu: dict[str, int],
    cpu_cores: int | None,
    ram_mb: int | None,
    disk_mb: int | None = None,
    disk_path: str | None = None,
) -> dict[str, int]:
    """Fold the first-class ``cpu_cores`` / ``ram_mb`` / ``disk_mb`` kwargs into
    the non-GPU demand dict.

    ``cpu_cores`` / ``ram_mb`` are ordinary counter resources under the
    standard keys (``CPU_CORES_KEY`` / ``RAM_MB_KEY``) — the explicit kwargs
    exist so consumers converge on one spelling instead of inventing
    ``mem_mb`` / ``host_ram`` variants. ``disk_mb`` becomes a free-space
    lease key ``disk_mb@<disk_path>`` (default mount ``/``) with statvfs-based
    admission. Values must be positive: a zero ask is a no-op that almost
    certainly means a bug at the call site. ``disk_path`` without ``disk_mb``
    raises — silently ignoring it would grant a lease with no disk
    reservation (same reasoning as ``vram_mb_each`` without ``num_gpus``).
    """
    merged = dict(non_gpu)
    for key, val in ((CPU_CORES_KEY, cpu_cores), (RAM_MB_KEY, ram_mb)):
        if val is None:
            continue
        if val <= 0:
            raise ValueError(f"{key} must be a positive int, got {val}")
        merged[key] = val
    if disk_mb is not None:
        if disk_mb <= 0:
            raise ValueError(f"disk_mb must be a positive int, got {disk_mb}")
        merged[disk_mb_key(disk_path or "/")] = disk_mb
    elif disk_path is not None:
        raise ValueError(
            f"disk_path={disk_path!r} requires disk_mb; "
            "set disk_mb to the free space to reserve or drop disk_path"
        )
    return merged


def _detect_host_pid() -> int | None:
    """Return host_pid if it differs from os.getpid(), else None."""
    host = get_host_pid()
    return host if host != os.getpid() else None


class EntryHandle:
    """Handle for an active QueueEntry attached to a Lease.

    Entries are work trackers — peers reading ``pool.status().queue`` see them
    as in-flight work with optional ``estimated_seconds`` / ``progress``.
    Reclaim is blocked on a lease while any of its entries are alive: the
    consumer is mid-work and would orphan in-flight requests if evicted.

    Created via:

    * ``LeaseHandle.start_work(estimated_seconds=N)`` — explicit, for per-request
      tracking on a long-lived lease (e.g. one inference call on aiserver's
      persistent model-load lease).
    * ``pool.acquire(estimated_seconds=N)`` shorthand — the queue entry that
      was used to enqueue stays attached to the new lease as the auto-tracked
      entry, exposed via ``LeaseHandle.entry``. Auto-completed on lease release.
    """

    def __init__(self, entry: QueueEntry, pool: ResourcePool) -> None:
        self._entry = entry
        self._pool = pool
        self._completed = False

    @property
    def id(self) -> str:
        return self._entry.id

    @property
    def lease_id(self) -> str | None:
        return self._entry.lease_id

    @property
    def estimated_seconds(self) -> int | None:
        return self._entry.estimated_seconds

    @property
    def progress(self) -> float | None:
        return self._entry.progress

    @property
    def label(self) -> str | None:
        return self._entry.label

    def update(
        self,
        *,
        estimated_seconds: int | None = None,
        progress: float | None = None,
        label: str | None = None,
    ) -> None:
        """Update work-tracking fields on this entry.

        Only the fields you pass are touched. Setting ``estimated_seconds``
        without resetting ``progress`` is fine — peers that prefer
        ``estimated_seconds * (1 - progress)`` will get the right answer if
        the consumer also resets ``progress=0.0`` at the start of new work.
        """
        if self._completed:
            return

        def _update(state: State) -> None:
            for entry in state.queue:
                if entry.id == self._entry.id:
                    if estimated_seconds is not None:
                        entry.estimated_seconds = estimated_seconds
                    if progress is not None:
                        entry.progress = progress
                    if label is not None:
                        entry.label = label
                    self._entry = entry
                    return

        transact(self._pool._path, _update)  # pyright: ignore[reportPrivateUsage]

    def complete(self) -> None:
        """Drop the entry from the queue. The Lease is unaffected.

        Raises whatever ``transact()`` raises (lock timeout, OS error,
        validation error). Callers MUST NOT swallow these silently — a
        failed ``complete()`` leaves the entry attached to the lease, which
        blocks reclaim and starves peer consumers. Retrying ``complete()``
        is safe and idempotent: the handle is only marked completed once
        the state write succeeds, so a re-call after a transient failure
        will actually re-attempt the write rather than silently no-op.

        See ``tests/test_complete_failure_semantics.py`` for the contract.
        """
        if self._completed:
            return

        entry_id = self._entry.id

        def _complete(state: State) -> None:
            state.queue = [e for e in state.queue if e.id != entry_id]

        transact(self._pool._path, _complete)  # pyright: ignore[reportPrivateUsage]
        self._completed = True

    def __enter__(self) -> EntryHandle:
        return self

    def __exit__(self, *args: object) -> None:
        self.complete()


class LeaseHandle:
    """Handle for an acquired Lease, used to update or release it."""

    def __init__(
        self,
        lease: Lease,
        pool: ResourcePool,
        tracking_entry: QueueEntry | None = None,
    ) -> None:
        self._lease = lease
        self._pool = pool
        self._released = False
        self._entry: EntryHandle | None = (
            EntryHandle(tracking_entry, pool) if tracking_entry is not None else None
        )

    @property
    def id(self) -> str:
        return self._lease.id

    @property
    def wait_sec(self) -> float | None:
        """Seconds spent in the queue before this lease was promoted.

        ``0.0`` for leases acquired via ``try_acquire`` (no queue path).
        ``None`` only for leases recovered from a state file written by an
        older reslock version that didn't record this field.
        """
        return self._lease.wait_sec

    @property
    def gpu_uuids(self) -> list[str]:
        """Host-stable GPU UUIDs the lease reserves VRAM on.

        Parsed from the lease's resolved ``gpu_{uuid}_vram_mb`` bindings.
        Empty when the lease holds no per-GPU VRAM (non-GPU lease).
        """
        return [u for u in (parse_gpu_vram_key(k) for k in self._lease.resources) if u]

    @property
    def gpu_torch_indices(self) -> list[int]:
        """Container-local torch device indices, derived from ``gpu_uuids``.

        Inside a container with partial GPU mapping, torch numbers visible
        devices from 0; this property returns those local indices for the
        UUIDs the lease holds. Returns an empty list when CUDA isn't
        available, no UUIDs resolve, or torch isn't installed.
        """
        try:
            import torch  # pyright: ignore[reportMissingImports]
        except ImportError:
            return []
        try:
            if not torch.cuda.is_available():  # pyright: ignore[reportUnknownMemberType]
                return []
            from reslock.detect import gpu_uuid_for_torch_index

            uuid_to_index: dict[str, int] = {}
            for i in range(torch.cuda.device_count()):  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                uid = gpu_uuid_for_torch_index(i)
                if uid:
                    uuid_to_index[uid] = i
            return sorted({uuid_to_index[u] for u in self.gpu_uuids if u in uuid_to_index})
        except Exception:
            return []

    @property
    def resources(self) -> dict[str, int]:
        """The lease's current resource reservation (a copy).

        Keyed by resource name — for GPU VRAM the resolved
        ``gpu_<uuid>_vram_mb`` form. Reflects the cached lease, kept in sync
        by :meth:`update` / :meth:`shrink`. Returns a copy so callers can't
        mutate the lease's reservation in place.
        """
        return dict(self._lease.resources)

    @property
    def reclaim_requested(self) -> bool:
        """Check the state file for whether reclaim has been requested."""
        state = read_state(self._pool._path)  # pyright: ignore[reportPrivateUsage]
        for lease in state.leases:
            if lease.id == self._lease.id:
                self._lease = lease
                return lease.reclaim_requested
        return True  # lease gone = treat as reclaimed

    @property
    def entry(self) -> EntryHandle | None:
        """The auto-tracked entry from ``pool.acquire(estimated_seconds=...)``.

        ``None`` when the lease was acquired without ``estimated_seconds``
        (no auto-tracking). Entries created via :meth:`start_work` are
        independent — the caller manages those handles directly.
        """
        return self._entry

    def wait_for_reclaim(self, poll_interval: float = 0.5) -> None:
        """Block until reclaim is requested for this lease."""
        while not self.reclaim_requested:
            time.sleep(poll_interval)

    async def wait_for_reclaim_async(self, poll_interval: float = 0.5) -> None:
        """Async wait until reclaim is requested for this lease."""
        while not self.reclaim_requested:
            await asyncio.sleep(poll_interval)

    def update(
        self,
        actual_resources: dict[str, int] | None = None,
        cpu_seconds: float | None = None,
        pids: list[int] | None = None,
        resources: dict[str, int] | None = None,
        auto_detect: bool = False,
        **kwargs: object,
    ) -> None:
        """Update lease metadata.

        Args:
            actual_resources: Actual resource usage (e.g., ``{"vram_mb": 4000, "ram_mb": 1200}``).
            cpu_seconds: CPU time consumed so far.
            pids: Additional PIDs to monitor (e.g., child processes).
            resources: New resource reservation for the lease (e.g. shrinking an
                over-conservative VRAM estimate down to the measured footprint).
                **Shrink-only and subset-only:** keys must be a subset of the
                lease's current ``resources`` keys (no new GPUs/resources), every
                value must be ``> 0``, and ``sum(new) <= sum(current)`` (a grow is
                rejected). Raises ``ValueError`` otherwise. The live NVML
                cross-check in placement is the OOM backstop, so a shrink can only
                remove over-conservatism — it can never place a lease into VRAM
                NVML reports as really used.
            auto_detect: If True, automatically detect actual_resources and cpu_seconds
                using OS APIs and torch (if loaded).

        For work tracking (``estimated_seconds``, ``progress``) use the
        ``EntryHandle`` returned by :meth:`start_work` or accessed via
        :attr:`entry` for the auto-tracked entry from
        ``pool.acquire(estimated_seconds=...)``. Passing those kwargs here
        raises ``TypeError`` — v3 moved them to the QueueEntry.
        """
        if "estimated_seconds" in kwargs or "progress" in kwargs:
            raise TypeError(
                "LeaseHandle.update() no longer accepts estimated_seconds or progress "
                "(moved to QueueEntry in v3). Use lease.entry.update(...) for the "
                "auto-tracked entry from acquire(estimated_seconds=...), or "
                "lease.start_work(...) to create a new tracking entry."
            )
        if kwargs:
            raise TypeError(f"unexpected keyword arguments: {sorted(kwargs)}")

        if auto_detect:
            detected = get_self_actual_resources()
            if detected:
                actual_resources = {**(actual_resources or {}), **detected}
            cpu = get_self_cpu_seconds()
            if cpu is not None and cpu_seconds is None:
                cpu_seconds = cpu

        old_total: int | None = None
        new_total: int | None = None
        if resources is not None:
            current = self._lease.resources
            unknown = set(resources) - set(current)
            if unknown:
                raise ValueError(
                    f"resources update introduces keys not held by the lease: "
                    f"{sorted(unknown)} (has: {sorted(current)})"
                )
            nonpositive = {k: v for k, v in resources.items() if v <= 0}
            if nonpositive:
                raise ValueError(f"resource values must be > 0, got {nonpositive}")
            old_total = sum(current.values())
            new_total = sum(resources.values())
            if new_total > old_total:
                raise ValueError(
                    f"resources update must be shrink-only: "
                    f"sum(new)={new_total} > sum(current)={old_total}"
                )

        def _update(state: State) -> None:
            for lease in state.leases:
                if lease.id == self._lease.id:
                    if actual_resources is not None:
                        lease.actual_resources = actual_resources
                    if cpu_seconds is not None:
                        lease.cpu_seconds = cpu_seconds
                    if pids is not None:
                        lease.pids = pids
                    if resources is not None:
                        lease.resources = dict(resources)
                    break

        transact(self._pool._path, _update)  # pyright: ignore[reportPrivateUsage]

        if resources is not None:
            # Keep the cached lease in sync so handle.resources / gpu_uuids /
            # gpu_torch_indices reflect the shrunk reservation.
            self._lease.resources = dict(resources)
            logger.info(
                "lease %s resources shrunk %dMB -> %dMB",
                self._lease.id,
                old_total,
                new_total,
            )

    def start_work(
        self,
        *,
        estimated_seconds: int | None = None,
        progress: float | None = None,
        label: str | None = None,
    ) -> EntryHandle:
        """Track a unit of work running on this lease.

        Creates a new ``QueueEntry`` attached to this lease (no resource
        demand — the lease already holds the resources). While any entry is
        alive, the reclaim path skips this lease, so peer consumers won't
        preempt in-flight work even if the lease is ``reclaimable=True``.

        Use this for per-request work on a persistent reclaimable lease.
        For one-shot work where the lease IS the work, prefer
        ``pool.acquire(estimated_seconds=...)`` and access the auto-tracked
        entry via :attr:`entry`.
        """
        pid = os.getpid()
        host_pid = self._pool._host_pid  # pyright: ignore[reportPrivateUsage]
        now = _utcnow()
        new_entry = QueueEntry(
            pid=pid,
            host_pid=host_pid,
            resources={},
            vram_mb_each=None,
            num_gpus=0,
            reclaimable_intent=False,
            priority=self._lease.priority,
            label=label,
            queued_at=now,
            started_at=now,
            lease_id=self._lease.id,
            estimated_seconds=estimated_seconds,
            progress=progress,
        )

        def _attach(state: State) -> None:
            state.queue.append(new_entry)

        transact(self._pool._path, _attach)  # pyright: ignore[reportPrivateUsage]
        return EntryHandle(new_entry, self._pool)

    def release(self) -> None:
        """Drop the lease (and any still-attached entries) from the state file.

        Raises whatever ``transact()`` raises (lock timeout, OS error,
        :class:`~reslock.state.SchemaVersionMismatch`). Callers MUST NOT
        swallow these silently — a failed ``release()`` leaves the lease in
        the file, holding capacity that peers are waiting for until the
        process dies and dead-PID cleanup removes it. Retrying ``release()``
        is safe and idempotent: the handle is only marked released once the
        state write succeeds, so a re-call after a transient failure will
        actually re-attempt the write rather than silently no-op.

        See ``tests/test_pool.py::test_release_retries_after_transient_transact_failure``
        for the contract (mirrors :meth:`EntryHandle.complete`).
        """
        if self._released:
            return
        lease_id = self._lease.id

        def _release(state: State) -> None:
            state.leases = [ls for ls in state.leases if ls.id != lease_id]
            # Defensive cleanup: drop any still-attached entries (auto-tracked
            # or start_work-created). EntryHandles held by callers become
            # no-op on subsequent complete().
            state.queue = [e for e in state.queue if e.lease_id != lease_id]

        transact(self._pool._path, _release)  # pyright: ignore[reportPrivateUsage]
        self._released = True
        if self._entry is not None:
            self._entry._completed = True  # pyright: ignore[reportPrivateUsage]

    def shrink(self, **resources: int) -> None:
        """Atomically decrement reserved resources on this lease.

        Frees capacity for other waiters without the release-and-reacquire race
        window. Waiters polling ``acquire()`` pick up the freed capacity on
        their next poll tick (same as after ``release()``).

        Args:
            **resources: Amount to decrement per resource key
                (e.g. ``ram_mb=500`` subtracts 500 from the current reservation).
                Values must be non-negative. For GPU VRAM the key is the
                resolved ``gpu_<uuid>_vram_mb`` form; for disk the
                ``disk_mb@<path>`` form (pass via ``**{...}``) — shrinking a
                disk lease as bytes land avoids double-counting the written
                bytes against the remaining reservation. Non-GPU shrinks are
                the typical use.

        Raises:
            ValueError: If any value is negative, references a key the lease
                does not hold, or would reduce a key below zero.

        Semantics:
            - Keys that reach zero are dropped from the lease.
            - If every remaining key reaches zero, the lease is released
              (also auto-completing any attached entries).
            - No-op on an already-released lease (matches ``release()``).
            - ``actual_resources`` is not modified — use ``update()`` for that.
            - Like ``release()``, nothing is latched on a failed ``transact()``:
              the handle stays usable. If the write that empties the lease
              succeeds but the follow-up ``release()`` raises, retry
              ``release()`` (not ``shrink()`` — the keys are already gone).
        """
        if self._released:
            return

        for key, delta in resources.items():
            if delta < 0:
                raise ValueError(f"shrink amount for {key!r} must be non-negative, got {delta}")

        should_release: list[bool] = []
        new_resources: list[dict[str, int]] = []

        def _shrink(state: State) -> None:
            for lease in state.leases:
                if lease.id != self._lease.id:
                    continue
                for key, delta in resources.items():
                    current = lease.resources.get(key)
                    if current is None:
                        raise ValueError(
                            f"lease does not hold resource {key!r} (has: {sorted(lease.resources)})"
                        )
                    new_val = current - delta
                    if new_val < 0:
                        raise ValueError(
                            f"shrink would reduce {key!r} below zero "
                            f"({current} - {delta} = {new_val})"
                        )
                    if new_val == 0:
                        del lease.resources[key]
                    else:
                        lease.resources[key] = new_val
                if not lease.resources:
                    should_release.append(True)
                new_resources.append(dict(lease.resources))
                break

        transact(self._pool._path, _shrink)  # pyright: ignore[reportPrivateUsage]

        # Keep the cached lease in sync so handle.resources / gpu_uuids reflect
        # the post-shrink reservation (mirrors update(resources=)).
        if new_resources:
            self._lease.resources = new_resources[0]

        if should_release:
            self.release()


class ResourcePool:
    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path else DEFAULT_STATE_PATH
        self._host_pid = _detect_host_pid()
        ensure_state_file(self._path)

    @contextmanager
    def acquire(
        self,
        *,
        vram_mb: list[int] | int | None = None,
        vram_mb_each: int | None = None,
        num_gpus: int = 0,
        cpu_cores: int | None = None,
        ram_mb: int | None = None,
        disk_mb: int | None = None,
        disk_path: str | None = None,
        priority: int = 0,
        reclaimable: bool = False,
        estimated_seconds: int | None = None,
        label: str | None = None,
        poll_interval: float = 0.25,
        **non_gpu_resources: int,
    ) -> Generator[LeaseHandle, None, None]:
        """Acquire resources, blocking until granted.

        Args:
            vram_mb: A list means per-slot GPU VRAM asks, mutually exclusive
                with ``vram_mb_each`` and ``num_gpus``. The scheduler pairs
                asks largest-first with the GPUs having the most free VRAM.
                A single int retains the legacy free-form ``vram_mb`` counter.
            vram_mb_each: Per-GPU VRAM ask (required when ``num_gpus > 0``).
            num_gpus: Number of GPUs needed (any). The scheduler picks UUIDs
                at promotion time using spread placement (most-free first,
                ties by UUID).
            cpu_cores: CPU cores to reserve (host-global counter under the
                standard ``cpu_cores`` key). Register capacity via
                ``set_resources(detect_cpu_cores())``.
            ram_mb: System RAM in MB to reserve (host-global counter under
                the standard ``ram_mb`` key). Register capacity via
                ``set_resources(detect_ram_mb(reserve_mb=...))``.
            disk_mb: Free disk space in MB to reserve on ``disk_path``
                (key ``disk_mb@<path>``). No capacity registration —
                admission is against live statvfs: granted only while
                ``request + sum(active disk leases on the mount) <= actual
                free``. Reserve before writing (a download, scratch space),
                write, release; ``shrink()`` as bytes land to avoid
                double-counting against peers.
            disk_path: Mount path for ``disk_mb`` (default ``/``). Requires
                ``disk_mb``.
            priority: Higher-priority waiters jump ahead in the queue.
            reclaimable: Allow this lease to be evicted by higher-priority
                requests when resources are short. Reclaim is blocked while
                any ``QueueEntry`` is attached (see :meth:`LeaseHandle.start_work`).
            estimated_seconds: If set, the queue entry used to acquire stays
                attached to the new lease as a tracking entry, exposed via
                ``lease.entry``. Auto-completed on release.
            label: Human-readable label for diagnostics.
            poll_interval: Seconds between scheduler polls.
            **non_gpu_resources: Additional non-GPU resource demands under
                free-form keys. ``gpu_<uuid>_vram_mb`` keys are rejected —
                use ``vram_mb_each`` + ``num_gpus`` instead.
        """
        vram_slots, non_gpu_resources = _split_vram_mb_argument(vram_mb, non_gpu_resources)
        _validate_non_gpu(non_gpu_resources)
        normalize_gpu_asks(
            vram_mb=vram_slots,
            vram_mb_each=vram_mb_each,
            num_gpus=num_gpus,
        )
        handle = self._acquire_blocking(
            vram_mb=list(vram_slots) if vram_slots is not None else None,
            vram_mb_each=vram_mb_each,
            num_gpus=num_gpus,
            non_gpu=_fold_first_class(non_gpu_resources, cpu_cores, ram_mb, disk_mb, disk_path),
            priority=priority,
            reclaimable=reclaimable,
            estimated_seconds=estimated_seconds,
            label=label,
            poll_interval=poll_interval,
        )
        try:
            yield handle
        finally:
            handle.release()

    async def acquire_async(
        self,
        *,
        vram_mb: list[int] | int | None = None,
        vram_mb_each: int | None = None,
        num_gpus: int = 0,
        cpu_cores: int | None = None,
        ram_mb: int | None = None,
        disk_mb: int | None = None,
        disk_path: str | None = None,
        priority: int = 0,
        reclaimable: bool = False,
        estimated_seconds: int | None = None,
        label: str | None = None,
        poll_interval: float = 0.25,
        **non_gpu_resources: int,
    ) -> LeaseHandle:
        """Async equivalent of :meth:`acquire`. Caller is responsible for ``release()``."""
        vram_slots, non_gpu_resources = _split_vram_mb_argument(vram_mb, non_gpu_resources)
        _validate_non_gpu(non_gpu_resources)
        normalize_gpu_asks(
            vram_mb=vram_slots,
            vram_mb_each=vram_mb_each,
            num_gpus=num_gpus,
        )
        new_entry = self._enqueue(
            vram_mb=list(vram_slots) if vram_slots is not None else None,
            vram_mb_each=vram_mb_each,
            num_gpus=num_gpus,
            non_gpu=_fold_first_class(non_gpu_resources, cpu_cores, ram_mb, disk_mb, disk_path),
            priority=priority,
            reclaimable=reclaimable,
            label=label,
            estimated_seconds=estimated_seconds,
        )
        try:
            while True:
                handle = self._try_promote(new_entry, estimated_seconds, reclaimable)
                if handle is not None:
                    return handle
                await asyncio.sleep(poll_interval)
        except BaseException:
            # Best-effort dequeue: this cleanup itself goes through transact(),
            # so if the reason we're unwinding is that the state file changed
            # schema underneath us, it would raise again and mask the original
            # failure with a less informative copy of itself.
            with contextlib.suppress(SchemaVersionMismatch):
                self._remove_from_queue(new_entry.id)
            raise

    def try_acquire(
        self,
        *,
        vram_mb: list[int] | int | None = None,
        vram_mb_each: int | None = None,
        num_gpus: int = 0,
        cpu_cores: int | None = None,
        ram_mb: int | None = None,
        disk_mb: int | None = None,
        disk_path: str | None = None,
        priority: int = 0,
        reclaimable: bool = False,
        estimated_seconds: int | None = None,
        label: str | None = None,
        **non_gpu_resources: int,
    ) -> LeaseHandle | None:
        """Try to acquire resources without queueing. Returns ``None`` if the request can't fit."""
        vram_slots, non_gpu_resources = _split_vram_mb_argument(vram_mb, non_gpu_resources)
        _validate_non_gpu(non_gpu_resources)
        gpu_asks = normalize_gpu_asks(
            vram_mb=vram_slots,
            vram_mb_each=vram_mb_each,
            num_gpus=num_gpus,
        )
        vram_slots = list(vram_slots) if vram_slots is not None else None
        non_gpu_resources = _fold_first_class(
            non_gpu_resources, cpu_cores, ram_mb, disk_mb, disk_path
        )
        pid = os.getpid()
        host_pid = self._host_pid

        # NVML / statvfs pre-flight: read driver-side free VRAM and actual
        # free disk before we take the file lock so the transact closure
        # stays fast. Raises if pynvml is unavailable and a GPU is requested
        # (intentional hard fail), or a disk mount path can't be statted.
        nvml_free = _read_nvml_for_request(len(gpu_asks))
        disk_free = _read_disk_free_for_request(non_gpu_resources)

        result: list[LeaseHandle] = []

        def _try(state: State) -> None:
            resolved = state.try_resolve_request(
                vram_mb=vram_slots,
                vram_mb_each=vram_mb_each,
                num_gpus=num_gpus,
                non_gpu=non_gpu_resources,
                nvml_free=nvml_free,
                disk_free=disk_free,
            )
            if resolved is None:
                # Either internal accounting is short, or NVML is short on
                # every internally-eligible GPU. ``try_acquire`` is non-blocking
                # by contract — refuse instead of signalling reclaim.
                return
            now = _utcnow()
            lease = Lease(
                pid=pid,
                host_pid=host_pid,
                resources=resolved,
                priority=priority,
                acquired_at=now,
                queued_at=now,
                wait_sec=0.0,
                reclaimable=reclaimable,
                label=label,
            )
            state.leases.append(lease)
            tracked: QueueEntry | None = None
            if estimated_seconds is not None:
                tracked = QueueEntry(
                    pid=pid,
                    host_pid=host_pid,
                    resources=dict(non_gpu_resources),
                    vram_mb=vram_slots,
                    vram_mb_each=vram_mb_each,
                    num_gpus=num_gpus,
                    reclaimable_intent=reclaimable,
                    priority=priority,
                    label=label,
                    queued_at=now,
                    started_at=now,
                    lease_id=lease.id,
                    estimated_seconds=estimated_seconds,
                )
                state.queue.append(tracked)
            result.append(LeaseHandle(lease, self, tracking_entry=tracked))

        transact(self._path, _try)
        return result[0] if result else None

    def set_resources(self, resources: dict[str, int]) -> None:
        """Register resource capacities.

        Each consumer should call this on startup before acquiring leases,
        declaring the resources it knows about. This is the recommended way
        to populate resource capacities.

        Existing keys are overwritten; keys not present in *resources* are
        left unchanged (so multiple consumers can register different resource
        types independently).

        Capacity means the *physical total* of the resource — for GPU VRAM
        keys, the card's ``memory.total`` as reported by NVML. External or
        transient VRAM usage is already handled by the NVML pre-flight at
        placement time, so consumers must NOT write derived free-based
        snapshots here: they go stale the moment the VRAM frees and can block
        promotion on a physically idle host. Registering less than total is
        legitimate only as a deliberate, static headroom reserve (cf.
        ``detect_ram_mb(reserve_mb=...)``); a GPU value below the NVML
        physical total logs a warning either way.

        Args:
            resources: Mapping of resource name to total capacity,
                e.g. ``{"gpu_GPU-1a2b3c4d-..._vram_mb": 24000, "cpu_cores": 16}``.
                GPU capacities still use UUID-keyed entries — the v3
                request-shape change only affects ``acquire()`` callers.
        """
        self._warn_gpu_capacity_mismatch(resources)

        def _set(state: State) -> None:
            state.resources.update(resources)

        transact(self._path, _set)

    @staticmethod
    def _warn_gpu_capacity_mismatch(resources: dict[str, int]) -> None:
        """Warn when a submitted GPU VRAM capacity differs from the NVML total.

        Tripwire for the 2026-07-20 kirk incident: a consumer persisted
        free-based snapshots as capacity via ``set_resources``, which went
        stale and blocked promotion on an idle host. Log-only — the value is
        registered regardless. Fails soft when NVML is unavailable;
        ``nvml_total_vram_mb`` caches totals per process, so frequent
        re-registration doesn't hammer the driver.

        Values slightly below the NVML total are tolerated, because the CUDA
        driver legitimately reports less than the physical total and that is
        what every torch- or driver-API-based detection registers — see
        ``GPU_CAPACITY_MIN_RATIO`` for the measured numbers. Above the total
        there is no tolerance: no detection method can produce it, and
        phantom capacity is the direction that actually mis-schedules.
        """
        gpu_keys = {k: u for k in resources if (u := parse_gpu_vram_key(k)) is not None}
        if not gpu_keys:
            return
        try:
            totals = nvml_total_vram_mb()
        except NvmlUnavailableError:
            return
        except Exception:
            logger.debug("reslock: NVML total lookup failed in set_resources", exc_info=True)
            return
        for key, uuid_str in gpu_keys.items():
            total = totals.get(uuid_str)
            if total is None:
                continue
            value = resources[key]
            if value < total:
                if value >= total * GPU_CAPACITY_MIN_RATIO:
                    logger.debug(
                        "reslock: set_resources registering %s=%d MB, %d MB (%.1f%%) below "
                        "the NVML physical total of %d MB — within the CUDA driver reserve, "
                        "not warning.",
                        key,
                        value,
                        total - value,
                        (1 - value / total) * 100,
                        total,
                    )
                    continue
                logger.warning(
                    "reslock: set_resources registering %s=%d MB, more than %.0f%% below the "
                    "NVML physical total of %d MB (caller pid %d). Capacity must be the "
                    "physical total — transient VRAM usage is handled by the NVML pre-flight "
                    "at placement time. Do not register derived free-based values; a stale "
                    "snapshot blocks promotion on an idle host.",
                    key,
                    value,
                    (1 - GPU_CAPACITY_MIN_RATIO) * 100,
                    total,
                    os.getpid(),
                )
            elif value > total:
                logger.warning(
                    "reslock: set_resources registering %s=%d MB, above the NVML physical "
                    "total of %d MB (caller pid %d). Leases granted against phantom "
                    "capacity will fail the NVML pre-flight at placement time.",
                    key,
                    value,
                    total,
                    os.getpid(),
                )

    def status(self) -> PoolStatus:
        state = read_state_clean(self._path)
        return PoolStatus(
            resources=state.resources,
            available=state.available(),
            leases=state.leases,
            queue=state.queue,
        )

    def gpu_orphans(self) -> list[OrphanReport]:
        """Return PIDs holding GPU VRAM that aren't registered with reslock.

        Reads ``nvidia-smi --query-compute-apps`` and diffs against the PIDs
        recorded in active leases (``lease.pid`` + ``lease.host_pid`` +
        ``lease.pids``). PIDs holding VRAM that don't belong to any lease are
        returned as orphans — typically: a process that crashed before
        reslock cleanup ran, a manually-spawned llama-server, or a child
        process that escaped its parent's lease registration.

        Reslock itself never terminates orphans — that policy belongs in the
        consumer (e.g. aiserver's gpu_audit). This is a diagnostic primitive.
        Returns an empty list when nvidia-smi is unavailable.
        """
        from reslock.audit import gpu_orphans

        state = read_state(self._path)
        return gpu_orphans(state)

    def _enqueue(
        self,
        *,
        vram_mb: list[int] | None,
        vram_mb_each: int | None,
        num_gpus: int,
        non_gpu: dict[str, int],
        priority: int,
        reclaimable: bool,
        label: str | None,
        estimated_seconds: int | None,
    ) -> QueueEntry:
        new_entry = QueueEntry(
            pid=os.getpid(),
            host_pid=self._host_pid,
            resources=dict(non_gpu),
            vram_mb=vram_mb,
            vram_mb_each=vram_mb_each,
            num_gpus=num_gpus,
            reclaimable_intent=reclaimable,
            priority=priority,
            label=label,
            estimated_seconds=estimated_seconds,
        )

        def _enqueue_fn(state: State) -> None:
            state.queue.append(new_entry)

        transact(self._path, _enqueue_fn)
        return new_entry

    def _acquire_blocking(
        self,
        *,
        vram_mb: list[int] | None = None,
        vram_mb_each: int | None,
        num_gpus: int,
        non_gpu: dict[str, int],
        priority: int,
        reclaimable: bool,
        estimated_seconds: int | None,
        label: str | None,
        poll_interval: float,
    ) -> LeaseHandle:
        new_entry = self._enqueue(
            vram_mb=vram_mb,
            vram_mb_each=vram_mb_each,
            num_gpus=num_gpus,
            non_gpu=non_gpu,
            priority=priority,
            reclaimable=reclaimable,
            label=label,
            estimated_seconds=estimated_seconds,
        )
        try:
            while True:
                handle = self._try_promote(new_entry, estimated_seconds, reclaimable)
                if handle is not None:
                    return handle
                time.sleep(poll_interval)
        except BaseException:
            # Best-effort dequeue: this cleanup itself goes through transact(),
            # so if the reason we're unwinding is that the state file changed
            # schema underneath us, it would raise again and mask the original
            # failure with a less informative copy of itself.
            with contextlib.suppress(SchemaVersionMismatch):
                self._remove_from_queue(new_entry.id)
            raise

    def _try_promote(
        self,
        own_entry_snapshot: QueueEntry,
        estimated_seconds: int | None,
        reclaimable: bool,
    ) -> LeaseHandle | None:
        pid = os.getpid()
        host_pid = self._host_pid
        entry_id = own_entry_snapshot.id
        priority = own_entry_snapshot.priority
        vram_mb = own_entry_snapshot.vram_mb
        vram_mb_each = own_entry_snapshot.vram_mb_each
        num_gpus = own_entry_snapshot.num_gpus
        non_gpu = dict(own_entry_snapshot.resources)
        label = own_entry_snapshot.label

        # NVML / statvfs pre-flight: read driver-side free VRAM and actual
        # free disk before we take the file lock. Raises if pynvml is
        # unavailable and a GPU is requested, or a disk path can't be
        # statted. Re-read every poll tick — external writes move the disk
        # ground truth just like external processes move NVML's.
        gpu_asks = normalize_gpu_asks(
            vram_mb=vram_mb,
            vram_mb_each=vram_mb_each,
            num_gpus=num_gpus,
        )
        nvml_free = _read_nvml_for_request(len(gpu_asks))
        disk_free = _read_disk_free_for_request(non_gpu)
        nvml_for_gate: dict[str, int] | None = nvml_free
        nvml_gate_unreachable = False

        result: list[LeaseHandle] = []

        def _promote(state: State) -> None:
            nonlocal nvml_for_gate, nvml_gate_unreachable

            own_entry: QueueEntry | None = None
            for e in state.queue:
                if e.id == entry_id and e.lease_id is None:
                    own_entry = e
                    break
            if own_entry is None:
                return

            # Higher-priority gate. A higher-pri pending entry only blocks us
            # if it could *actually* be promoted on its own next tick — i.e.
            # ``try_resolve_request`` succeeds for its shape under the same
            # spread placement we'd use, including NVML drift folded in.
            # Otherwise an NVML-short GPU waiter would silently block
            # lower-priority non-GPU waiters forever.
            for entry in state.queue:
                if entry.id == entry_id:
                    break
                if entry.lease_id is not None:
                    # Already-active entries don't gate; they're not waiting.
                    continue
                if entry.priority <= priority:
                    continue
                gate_nvml: dict[str, int] | None = None
                entry_gpu_asks = normalize_gpu_asks(
                    vram_mb=entry.vram_mb,
                    vram_mb_each=entry.vram_mb_each,
                    num_gpus=entry.num_gpus,
                )
                if entry_gpu_asks:
                    if nvml_for_gate is None and not nvml_gate_unreachable:
                        try:
                            nvml_for_gate = nvml_free_vram_mb()
                        except NvmlUnavailableError:
                            nvml_gate_unreachable = True
                    if nvml_gate_unreachable:
                        # NVML unreachable for the gate; be conservative and
                        # let the higher-pri entry block us (legacy behavior).
                        return
                    gate_nvml = nvml_for_gate
                cand_resolved = state.try_resolve_request(
                    vram_mb=entry.vram_mb,
                    vram_mb_each=entry.vram_mb_each,
                    num_gpus=entry.num_gpus,
                    non_gpu=entry.resources,
                    nvml_free=gate_nvml,
                    disk_free=_gate_disk_free(entry.resources),
                )
                if cand_resolved is None:
                    continue
                return  # higher-priority waiter can fit, let them go first

            # Try to resolve our request. NVML-short GPUs are excluded from
            # placement at this layer, so a returned ``resolved`` is already
            # NVML-clean.
            resolved = state.try_resolve_request(
                vram_mb=vram_mb,
                vram_mb_each=vram_mb_each,
                num_gpus=num_gpus,
                non_gpu=non_gpu,
                nvml_free=nvml_free,
                disk_free=disk_free,
            )

            if resolved is not None:
                acquired_at = _utcnow()
                wait_sec = (acquired_at - own_entry.queued_at).total_seconds()
                lease = Lease(
                    pid=pid,
                    host_pid=host_pid,
                    resources=resolved,
                    priority=priority,
                    acquired_at=acquired_at,
                    queued_at=own_entry.queued_at,
                    wait_sec=wait_sec,
                    reclaimable=reclaimable,
                    label=label,
                )
                state.leases.append(lease)
                tracked: QueueEntry | None = None
                if estimated_seconds is not None:
                    own_entry.lease_id = lease.id
                    own_entry.started_at = acquired_at
                    tracked = own_entry
                else:
                    state.queue = [e for e in state.queue if e.id != entry_id]
                result.append(LeaseHandle(lease, self, tracking_entry=tracked))
                return

            # Resolution failed — either internal accounting is short, or NVML
            # excluded enough GPUs to drop us below ``num_gpus``. Try to find
            # reclaimables whose eviction would let us resolve.
            _request_reclaim_to_resolve(
                state,
                vram_mb=vram_mb,
                vram_mb_each=vram_mb_each,
                num_gpus=num_gpus,
                non_gpu=non_gpu,
                nvml_free=nvml_free,
                disk_free=disk_free,
            )

        transact(self._path, _promote)
        return result[0] if result else None

    def _remove_from_queue(self, queue_id: str) -> None:
        def _remove(state: State) -> None:
            state.queue = [e for e in state.queue if e.id != queue_id]

        transact(self._path, _remove)


def _request_reclaim_to_resolve(
    state: State,
    *,
    vram_mb: list[int] | None,
    vram_mb_each: int | None,
    num_gpus: int,
    non_gpu: dict[str, int],
    nvml_free: dict[str, int] | None,
    disk_free: dict[str, int] | None = None,
) -> None:
    """Walk *relevant* reclaimable leases in priority order; mark each for
    reclaim until the request becomes resolvable.

    Relevance filter: only leases holding capacity on a key that contributes
    to the actual shortfall are considered. Specifically:

    * Non-GPU keys: any key the request demands more of than is currently
      available.
    * GPU keys: any GPU UUID whose effective free (``min(internal, NVML)``
      when NVML data is present) is below ``vram_mb_each`` — eviction of a
      lease holding capacity there could plausibly raise that GPU above the
      threshold.

    Reclaimables on GPUs that already qualify are skipped: evicting them
    can only reduce eligibility. Reclaimables that hold none of the
    shortfall keys (e.g. RAM-only leases when the gap is GPU VRAM) are
    skipped: their eviction can't close the gap. This avoids spurious
    cascading reclaim across unrelated workloads when the gap is
    irrecoverable (external process holding VRAM with no reslock lease).

    Skips leases with an active ``QueueEntry`` attached — evicting mid-work
    would orphan in-flight requests. With NVML data, marks reclaim even when
    the union of evictions can't fully cover the gap (partial=True semantics):
    the gap may include external (unaccounted) consumers that reslock can
    never reclaim, but freeing what we *can* free is still useful — the lease
    may grant on a later poll once the external process exits.

    Without NVML data the gap is purely internal accounting, which is
    deterministic — partial eviction would be pointless churn, so we only
    mark reclaim when the union of evictions fully covers.
    """
    relevant_keys = _shortfall_keys(
        state,
        vram_mb=vram_mb,
        vram_mb_each=vram_mb_each,
        num_gpus=num_gpus,
        non_gpu=non_gpu,
        nvml_free=nvml_free,
        disk_free=disk_free,
    )
    if not relevant_keys:
        return

    active_ids = state.active_lease_ids()
    candidates = sorted(
        [
            ls
            for ls in state.leases
            if ls.reclaimable
            and not ls.reclaim_requested
            and ls.id not in active_ids
            and any(k in ls.resources for k in relevant_keys)
        ],
        key=lambda ls: ls.priority,
    )
    if not candidates:
        return

    def _resolves_with(evicted: set[str]) -> bool:
        tmp = State(
            version=state.version,
            resources=dict(state.resources),
            leases=[ls for ls in state.leases if ls.id not in evicted],
        )
        return (
            tmp.try_resolve_request(
                vram_mb=vram_mb,
                vram_mb_each=vram_mb_each,
                num_gpus=num_gpus,
                non_gpu=non_gpu,
                nvml_free=nvml_free,
                disk_free=disk_free,
            )
            is not None
        )

    evicted: set[str] = set()
    found = False
    for lease in candidates:
        evicted.add(lease.id)
        if _resolves_with(evicted):
            found = True
            break

    if not found and nvml_free is None:
        # Pure internal-accounting shortfall and we couldn't cover even by
        # evicting everything reclaimable — eviction won't help, no churn.
        return

    for lease in state.leases:
        if lease.id in evicted:
            lease.reclaim_requested = True
            # Stamps the False→True transition only, because `candidates`
            # above filters out leases that already have reclaim_requested.
            # That filter is load-bearing for this field, not just an
            # optimisation: this function runs on every poll tick of every
            # waiter, so re-stamping would keep a nine-day-old pending reclaim
            # reading as milliseconds old — destroying exactly the signal the
            # timestamp exists to carry.
            lease.reclaim_requested_at = datetime.now(timezone.utc)


def _shortfall_keys(
    state: State,
    *,
    vram_mb: list[int] | None,
    vram_mb_each: int | None,
    num_gpus: int,
    non_gpu: dict[str, int],
    nvml_free: dict[str, int] | None,
    disk_free: dict[str, int] | None = None,
) -> set[str]:
    """Resource keys whose lease holders could plausibly help close the gap.

    Combines internal-accounting deficits with NVML drift on GPUs the
    scheduler would otherwise consider eligible. Used to keep the reclaim
    cascade from evicting leases on unrelated GPUs / non-GPU resources when
    the actual shortfall is elsewhere. Disk keys use the free-space check
    (``request > actual_free - leased``) — evicting another disk reclaimable
    frees its *reservation*, which can admit us even though it deletes no
    bytes.
    """
    keys: set[str] = set()
    avail = state.available()
    used = state.used_per_key()
    for key, val in non_gpu.items():
        if parse_disk_mb_key(key) is not None:
            if val > (disk_free or {}).get(key, 0) - used.get(key, 0):
                keys.add(key)
        elif val > avail.get(key, 0):
            keys.add(key)
    gpu_asks = normalize_gpu_asks(
        vram_mb=vram_mb,
        vram_mb_each=vram_mb_each,
        num_gpus=num_gpus,
    )
    if gpu_asks:
        per_gpu = state.per_gpu_free()
        if nvml_free is not None:
            per_gpu = {u: min(free, nvml_free.get(u, 0)) for u, free in per_gpu.items()}
        for uuid_str, free in per_gpu.items():
            if free < gpu_asks[0]:
                keys.add(gpu_vram_key(uuid_str))
    return keys
