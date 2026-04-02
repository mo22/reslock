from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Generator
from contextlib import contextmanager
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
    def reclaim_requested(self) -> bool:
        """Check the state file for whether reclaim has been requested."""
        state = read_state(self._pool._path)
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

        transact(self._pool._path, _update)

    def release(self) -> None:
        if self._released:
            return
        self._released = True

        def _release(state: State) -> None:
            state.leases = [ls for ls in state.leases if ls.id != self._lease.id]

        transact(self._pool._path, _release)


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
            lease = Lease(
                pid=pid,
                host_pid=host_pid,
                resources=resources,
                priority=priority,
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
                e.g. ``{"gpu0_vram_mb": 24000, "cpu_cores": 16}``.
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
            in_queue = any(e.id == queue_id for e in state.queue)
            if not in_queue:
                return

            # Check if higher-priority waiters should go first
            for entry in state.queue:
                if entry.id == queue_id:
                    break
                if entry.priority > priority and state.can_fit(entry.resources):
                    return  # higher priority waiter can fit, let them go first

            if state.can_fit(resources):
                lease = Lease(
                    pid=pid,
                    host_pid=host_pid,
                    resources=resources,
                    priority=priority,
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
