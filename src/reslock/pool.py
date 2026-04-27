from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from reslock.detect import get_host_pid, get_self_actual_resources, get_self_cpu_seconds
from reslock.models import Lease, PoolStatus, QueueEntry, State
from reslock.state import (
    DEFAULT_STATE_PATH,
    ensure_state_file,
    read_state,
    read_state_clean,
    transact,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class LeaseHandle:
    """Handle for an acquired lease, used to update or release it."""

    def __init__(self, lease: Lease, pool: ResourcePool) -> None:
        self._lease = lease
        self._pool = pool
        self._released = False

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

        Parsed from the lease's ``gpu_{uuid}_vram_mb`` resource keys. Empty
        when the lease holds no per-GPU VRAM reservations.
        """
        from reslock.detect import parse_gpu_vram_key

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
        estimated_seconds: int | None = None,
        actual_resources: dict[str, int] | None = None,
        cpu_seconds: float | None = None,
        progress: float | None = None,
        pids: list[int] | None = None,
        auto_detect: bool = False,
    ) -> None:
        """Update lease metadata.

        Args:
            estimated_seconds: Estimated remaining seconds for this lease.
            actual_resources: Actual resource usage (e.g., {"vram_mb": 4000, "ram_mb": 1200}).
            cpu_seconds: CPU time consumed so far.
            progress: Progress indicator (0.0 to 1.0).
            pids: Additional PIDs to monitor (e.g., child processes).
            auto_detect: If True, automatically detect actual_resources and cpu_seconds
                using OS APIs and torch (if loaded).
        """
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
                    if estimated_seconds is not None:
                        lease.estimated_seconds = estimated_seconds
                    if actual_resources is not None:
                        lease.actual_resources = actual_resources
                    if cpu_seconds is not None:
                        lease.cpu_seconds = cpu_seconds
                    if progress is not None:
                        lease.progress = progress
                    if pids is not None:
                        lease.pids = pids
                    break

        transact(self._pool._path, _update)  # pyright: ignore[reportPrivateUsage]

    def release(self) -> None:
        if self._released:
            return
        self._released = True

        def _release(state: State) -> None:
            state.leases = [ls for ls in state.leases if ls.id != self._lease.id]

        transact(self._pool._path, _release)  # pyright: ignore[reportPrivateUsage]

    def shrink(self, **resources: int) -> None:
        """Atomically decrement reserved resources on this lease.

        Frees capacity for other waiters without the release-and-reacquire race
        window. Waiters polling ``acquire()`` pick up the freed capacity on
        their next poll tick (same as after ``release()``).

        Args:
            **resources: Amount to decrement per resource key
                (e.g. ``disk_mb=500`` subtracts 500 from the current reservation).
                Values must be non-negative.

        Raises:
            ValueError: If any value is negative, references a key the lease
                does not hold, or would reduce a key below zero.

        Semantics:
            - Keys that reach zero are dropped from the lease.
            - If every remaining key reaches zero, the lease is released.
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
            self._released = True

            def _drop(state: State) -> None:
                state.leases = [ls for ls in state.leases if ls.id != self._lease.id]

            transact(self._pool._path, _drop)  # pyright: ignore[reportPrivateUsage]


def _detect_host_pid() -> int | None:
    """Return host_pid if it differs from os.getpid(), else None."""
    host = get_host_pid()
    return host if host != os.getpid() else None


class ResourcePool:
    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path else DEFAULT_STATE_PATH
        self._host_pid = _detect_host_pid()
        ensure_state_file(self._path)

    @contextmanager
    def acquire(
        self,
        *,
        priority: int = 0,
        estimated_seconds: int | None = None,
        reclaimable: bool = False,
        label: str | None = None,
        poll_interval: float = 0.25,
        **resources: int,
    ) -> Generator[LeaseHandle, None, None]:
        handle = self._acquire_blocking(
            resources=resources,
            priority=priority,
            estimated_seconds=estimated_seconds,
            reclaimable=reclaimable,
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
        priority: int = 0,
        estimated_seconds: int | None = None,
        reclaimable: bool = False,
        label: str | None = None,
        poll_interval: float = 0.25,
        **resources: int,
    ) -> LeaseHandle:
        queue_id: str | None = None
        pid = os.getpid()
        host_pid = self._host_pid

        def _enqueue(state: State) -> str:
            entry = QueueEntry(
                pid=pid, host_pid=host_pid, resources=resources, priority=priority, label=label
            )
            state.queue.append(entry)
            return entry.id

        queue_id = transact(self._path, _enqueue)

        try:
            while True:
                handle = self._try_promote(
                    queue_id, resources, priority, estimated_seconds, reclaimable, label
                )
                if handle is not None:
                    return handle
                await asyncio.sleep(poll_interval)
        except BaseException:
            self._remove_from_queue(queue_id)
            raise

    def try_acquire(
        self,
        *,
        priority: int = 0,
        estimated_seconds: int | None = None,
        reclaimable: bool = False,
        label: str | None = None,
        **resources: int,
    ) -> LeaseHandle | None:
        pid = os.getpid()
        host_pid = self._host_pid
        result: list[LeaseHandle] = []

        def _try(state: State) -> None:
            if not state.can_fit(resources):
                return
            now = _utcnow()
            lease = Lease(
                pid=pid,
                host_pid=host_pid,
                resources=resources,
                priority=priority,
                acquired_at=now,
                queued_at=now,
                wait_sec=0.0,
                estimated_seconds=estimated_seconds,
                reclaimable=reclaimable,
                label=label,
            )
            state.leases.append(lease)
            result.append(LeaseHandle(lease, self))

        transact(self._path, _try)
        return result[0] if result else None

    def set_resources(self, resources: dict[str, int]) -> None:
        """Register resource capacities.

        Each consumer should call this on startup before acquiring leases,
        declaring the resources it knows about. This is the recommended way
        to populate resource capacities — it keeps reslock resource-agnostic
        and removes the need for ``reslock init``.

        Existing keys are overwritten; keys not present in *resources* are
        left unchanged (so multiple consumers can register different resource
        types independently).

        Args:
            resources: Mapping of resource name to total capacity,
                e.g. ``{"gpu_GPU-1a2b3c4d-..._vram_mb": 24000, "cpu_cores": 16}``.
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

    def _acquire_blocking(
        self,
        resources: dict[str, int],
        priority: int,
        estimated_seconds: int | None,
        reclaimable: bool,
        label: str | None,
        poll_interval: float,
    ) -> LeaseHandle:
        pid = os.getpid()
        host_pid = self._host_pid

        def _enqueue(state: State) -> str:
            entry = QueueEntry(
                pid=pid, host_pid=host_pid, resources=resources, priority=priority, label=label
            )
            state.queue.append(entry)
            return entry.id

        queue_id = transact(self._path, _enqueue)

        try:
            while True:
                handle = self._try_promote(
                    queue_id, resources, priority, estimated_seconds, reclaimable, label
                )
                if handle is not None:
                    return handle
                time.sleep(poll_interval)
        except BaseException:
            self._remove_from_queue(queue_id)
            raise

    def _try_promote(
        self,
        queue_id: str,
        resources: dict[str, int],
        priority: int,
        estimated_seconds: int | None,
        reclaimable: bool,
        label: str | None,
    ) -> LeaseHandle | None:
        pid = os.getpid()
        host_pid = self._host_pid
        result: list[LeaseHandle] = []

        def _promote(state: State) -> None:
            # Check if we're still in queue
            own_entry: QueueEntry | None = None
            for e in state.queue:
                if e.id == queue_id:
                    own_entry = e
                    break
            if own_entry is None:
                return

            # Check if higher-priority waiters should go first
            for entry in state.queue:
                if entry.id == queue_id:
                    break
                if entry.priority > priority and state.can_fit(entry.resources):
                    return  # higher priority waiter can fit, let them go first

            if state.can_fit(resources):
                acquired_at = _utcnow()
                wait_sec = (acquired_at - own_entry.queued_at).total_seconds()
                lease = Lease(
                    pid=pid,
                    host_pid=host_pid,
                    resources=resources,
                    priority=priority,
                    acquired_at=acquired_at,
                    queued_at=own_entry.queued_at,
                    wait_sec=wait_sec,
                    estimated_seconds=estimated_seconds,
                    reclaimable=reclaimable,
                    label=label,
                )
                state.leases.append(lease)
                state.queue = [e for e in state.queue if e.id != queue_id]
                result.append(LeaseHandle(lease, self))
            else:
                # Try reclaiming
                to_reclaim = state.reclaimable_for(resources)
                if to_reclaim:
                    for lease in to_reclaim:
                        lease.reclaim_requested = True

        transact(self._path, _promote)
        return result[0] if result else None

    def _remove_from_queue(self, queue_id: str) -> None:
        def _remove(state: State) -> None:
            state.queue = [e for e in state.queue if e.id != queue_id]

        transact(self._path, _remove)
