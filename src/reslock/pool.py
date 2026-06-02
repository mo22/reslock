from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from reslock.detect import (
    get_host_pid,
    get_self_actual_resources,
    get_self_cpu_seconds,
    gpu_vram_key,
    parse_gpu_vram_key,
)

if TYPE_CHECKING:
    from reslock.audit import OrphanReport
from reslock.models import Lease, PoolStatus, QueueEntry, State
from reslock.nvml import (
    NvmlUnavailableError,
    nvml_free_vram_mb,
)
from reslock.state import (
    DEFAULT_STATE_PATH,
    ensure_state_file,
    read_state,
    read_state_clean,
    transact,
)


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


def _validate_non_gpu(non_gpu: dict[str, int]) -> None:
    """Reject ``gpu_<uuid>_vram_mb`` keys passed via the legacy v2 acquire shape.

    v3 hard-cut: callers declare ``vram_mb_each`` + ``num_gpus`` and the
    scheduler binds specific UUIDs at promotion time. Raising loudly here is
    nicer than silently failing to schedule a GPU lease.
    """
    for key in non_gpu:
        if parse_gpu_vram_key(key) is not None:
            raise TypeError(
                f"v3: GPU keys are not accepted in acquire() — got {key!r}. "
                "Use vram_mb_each + num_gpus instead; the scheduler picks "
                "GPUs automatically (spread placement, most-free first). "
                "Capacity registration via set_resources() still uses "
                "gpu_<uuid>_vram_mb keys."
            )


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
        auto_detect: bool = False,
        **kwargs: object,
    ) -> None:
        """Update lease metadata.

        Args:
            actual_resources: Actual resource usage (e.g., ``{"vram_mb": 4000, "ram_mb": 1200}``).
            cpu_seconds: CPU time consumed so far.
            pids: Additional PIDs to monitor (e.g., child processes).
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

        def _update(state: State) -> None:
            for lease in state.leases:
                if lease.id == self._lease.id:
                    if actual_resources is not None:
                        lease.actual_resources = actual_resources
                    if cpu_seconds is not None:
                        lease.cpu_seconds = cpu_seconds
                    if pids is not None:
                        lease.pids = pids
                    break

        transact(self._pool._path, _update)  # pyright: ignore[reportPrivateUsage]

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
        if self._released:
            return
        self._released = True
        lease_id = self._lease.id

        def _release(state: State) -> None:
            state.leases = [ls for ls in state.leases if ls.id != lease_id]
            # Defensive cleanup: drop any still-attached entries (auto-tracked
            # or start_work-created). EntryHandles held by callers become
            # no-op on subsequent complete().
            state.queue = [e for e in state.queue if e.lease_id != lease_id]

        transact(self._pool._path, _release)  # pyright: ignore[reportPrivateUsage]
        if self._entry is not None:
            self._entry._completed = True  # pyright: ignore[reportPrivateUsage]

    def shrink(self, **resources: int) -> None:
        """Atomically decrement reserved resources on this lease.

        Frees capacity for other waiters without the release-and-reacquire race
        window. Waiters polling ``acquire()`` pick up the freed capacity on
        their next poll tick (same as after ``release()``).

        Args:
            **resources: Amount to decrement per resource key
                (e.g. ``disk_mb=500`` subtracts 500 from the current reservation).
                Values must be non-negative. For GPU VRAM the key is the
                resolved ``gpu_<uuid>_vram_mb`` form — non-GPU shrinks are
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
        """
        if self._released:
            return

        for key, delta in resources.items():
            if delta < 0:
                raise ValueError(f"shrink amount for {key!r} must be non-negative, got {delta}")

        should_release: list[bool] = []

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
                break

        transact(self._pool._path, _shrink)  # pyright: ignore[reportPrivateUsage]

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
        vram_mb_each: int | None = None,
        num_gpus: int = 0,
        priority: int = 0,
        reclaimable: bool = False,
        estimated_seconds: int | None = None,
        label: str | None = None,
        poll_interval: float = 0.25,
        **non_gpu_resources: int,
    ) -> Generator[LeaseHandle, None, None]:
        """Acquire resources, blocking until granted.

        Args:
            vram_mb_each: Per-GPU VRAM ask (required when ``num_gpus > 0``).
            num_gpus: Number of GPUs needed (any). The scheduler picks UUIDs
                at promotion time using spread placement (most-free first,
                ties by UUID).
            priority: Higher-priority waiters jump ahead in the queue.
            reclaimable: Allow this lease to be evicted by higher-priority
                requests when resources are short. Reclaim is blocked while
                any ``QueueEntry`` is attached (see :meth:`LeaseHandle.start_work`).
            estimated_seconds: If set, the queue entry used to acquire stays
                attached to the new lease as a tracking entry, exposed via
                ``lease.entry``. Auto-completed on release.
            label: Human-readable label for diagnostics.
            poll_interval: Seconds between scheduler polls.
            **non_gpu_resources: Non-GPU resource demands (e.g. ``ram_mb=8000``).
                ``gpu_<uuid>_vram_mb`` keys are rejected — use ``vram_mb_each``
                + ``num_gpus`` instead.
        """
        _validate_non_gpu(non_gpu_resources)
        handle = self._acquire_blocking(
            vram_mb_each=vram_mb_each,
            num_gpus=num_gpus,
            non_gpu=non_gpu_resources,
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
        vram_mb_each: int | None = None,
        num_gpus: int = 0,
        priority: int = 0,
        reclaimable: bool = False,
        estimated_seconds: int | None = None,
        label: str | None = None,
        poll_interval: float = 0.25,
        **non_gpu_resources: int,
    ) -> LeaseHandle:
        """Async equivalent of :meth:`acquire`. Caller is responsible for ``release()``."""
        _validate_non_gpu(non_gpu_resources)
        new_entry = self._enqueue(
            vram_mb_each=vram_mb_each,
            num_gpus=num_gpus,
            non_gpu=non_gpu_resources,
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
            self._remove_from_queue(new_entry.id)
            raise

    def try_acquire(
        self,
        *,
        vram_mb_each: int | None = None,
        num_gpus: int = 0,
        priority: int = 0,
        reclaimable: bool = False,
        estimated_seconds: int | None = None,
        label: str | None = None,
        **non_gpu_resources: int,
    ) -> LeaseHandle | None:
        """Try to acquire resources without queueing. Returns ``None`` if the request can't fit."""
        _validate_non_gpu(non_gpu_resources)
        pid = os.getpid()
        host_pid = self._host_pid

        # NVML pre-flight: read driver-side free VRAM before we take the file
        # lock so the transact closure stays fast. Raises if pynvml is
        # unavailable and a GPU is requested (intentional hard fail).
        nvml_free = _read_nvml_for_request(num_gpus)

        result: list[LeaseHandle] = []

        def _try(state: State) -> None:
            resolved = state.try_resolve_request(
                vram_mb_each=vram_mb_each,
                num_gpus=num_gpus,
                non_gpu=non_gpu_resources,
                nvml_free=nvml_free,
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

        Args:
            resources: Mapping of resource name to total capacity,
                e.g. ``{"gpu_GPU-1a2b3c4d-..._vram_mb": 24000, "cpu_cores": 16}``.
                GPU capacities still use UUID-keyed entries — the v3
                request-shape change only affects ``acquire()`` callers.
        """

        def _set(state: State) -> None:
            state.resources.update(resources)

        transact(self._path, _set)

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
        vram_mb_each = own_entry_snapshot.vram_mb_each
        num_gpus = own_entry_snapshot.num_gpus
        non_gpu = dict(own_entry_snapshot.resources)
        label = own_entry_snapshot.label

        # NVML pre-flight: read driver-side free VRAM before we take the file
        # lock. Raises if pynvml is unavailable and a GPU is requested.
        nvml_free = _read_nvml_for_request(num_gpus)
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
                if entry.num_gpus > 0:
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
                    vram_mb_each=entry.vram_mb_each,
                    num_gpus=entry.num_gpus,
                    non_gpu=entry.resources,
                    nvml_free=gate_nvml,
                )
                if cand_resolved is None:
                    continue
                return  # higher-priority waiter can fit, let them go first

            # Try to resolve our request. NVML-short GPUs are excluded from
            # placement at this layer, so a returned ``resolved`` is already
            # NVML-clean.
            resolved = state.try_resolve_request(
                vram_mb_each=vram_mb_each,
                num_gpus=num_gpus,
                non_gpu=non_gpu,
                nvml_free=nvml_free,
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
                vram_mb_each=vram_mb_each,
                num_gpus=num_gpus,
                non_gpu=non_gpu,
                nvml_free=nvml_free,
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
    vram_mb_each: int | None,
    num_gpus: int,
    non_gpu: dict[str, int],
    nvml_free: dict[str, int] | None,
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
        state, vram_mb_each=vram_mb_each, num_gpus=num_gpus, non_gpu=non_gpu, nvml_free=nvml_free
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
                vram_mb_each=vram_mb_each,
                num_gpus=num_gpus,
                non_gpu=non_gpu,
                nvml_free=nvml_free,
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


def _shortfall_keys(
    state: State,
    *,
    vram_mb_each: int | None,
    num_gpus: int,
    non_gpu: dict[str, int],
    nvml_free: dict[str, int] | None,
) -> set[str]:
    """Resource keys whose lease holders could plausibly help close the gap.

    Combines internal-accounting deficits with NVML drift on GPUs the
    scheduler would otherwise consider eligible. Used to keep the reclaim
    cascade from evicting leases on unrelated GPUs / non-GPU resources when
    the actual shortfall is elsewhere.
    """
    keys: set[str] = set()
    avail = state.available()
    for key, val in non_gpu.items():
        if val > avail.get(key, 0):
            keys.add(key)
    if num_gpus > 0 and vram_mb_each is not None:
        per_gpu = state.per_gpu_free()
        if nvml_free is not None:
            per_gpu = {u: min(free, nvml_free.get(u, 0)) for u, free in per_gpu.items()}
        for uuid_str, free in per_gpu.items():
            if free < vram_mb_each:
                keys.add(gpu_vram_key(uuid_str))
    return keys
