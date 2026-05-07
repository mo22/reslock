"""GPU orphan diagnosis — find PIDs holding VRAM that aren't registered with reslock.

Reslock's NVML pre-flight catches the *fact* of a driver-vs-accounting drift
("internal says 19 GB free, driver says 4 GB free"). This module catches the
*who*: the PIDs that hold the missing VRAM but aren't covered by any active
lease's PID set.

Typical orphans:

* A process that crashed before reslock's dead-PID cleanup ran and another
  process re-used its PID.
* A manually-spawned ``llama-server`` / inference subprocess that wasn't
  registered via ``handle.update(pids=[...])``.
* A child process the consumer thought it released but that's still alive.

Reslock itself never terminates orphans — that policy belongs in the consumer.
This module is a diagnostic primitive that aiserver / scriba's GPU auditor
can consume in place of re-implementing nvidia-smi parsing.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from reslock.models import State


class OrphanReport(BaseModel):
    """A PID holding GPU VRAM that's not registered with any active reslock lease."""

    pid: int
    vram_mb: int = Field(..., description="Total VRAM held across all GPUs, in MiB.")
    gpus: dict[str, int] = Field(
        default_factory=dict,
        description="Per-GPU VRAM held, keyed by GPU UUID, value in MiB.",
    )
    cmdline: str | None = None

    model_config = {"extra": "forbid"}


def _registered_pids(state: State) -> set[int]:
    """Collect every PID known to be tied to an active lease.

    Includes lease owner ``pid``, ``host_pid`` (when running in a container),
    and tracked child ``pids`` (registered via ``handle.update(pids=[...])``).
    """
    out: set[int] = set()
    for lease in state.leases:
        out.add(lease.pid)
        if lease.host_pid is not None:
            out.add(lease.host_pid)
        out.update(lease.pids)
    return out


def _read_cmdline(pid: int) -> str | None:
    """Best-effort short cmdline for diagnostics. Linux-only via /proc."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            raw = f.read()
        if not raw:
            return None
        # /proc cmdlines are NUL-separated; first 200 chars is plenty for diagnostics.
        text = raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
        return text[:200] or None
    except OSError:
        return None


def _query_nvidia_smi() -> dict[int, dict[str, int]] | None:
    """Run ``nvidia-smi --query-compute-apps`` and return ``{pid: {gpu_uuid: mb}}``.

    Returns ``None`` if nvidia-smi isn't on PATH or the call fails — this
    signals "no diagnosis available" rather than "no orphans".
    """
    if not shutil.which("nvidia-smi"):
        return None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    usage: dict[int, dict[str, int]] = {}
    for line in result.stdout.strip().splitlines():
        parts = line.split(",")
        if len(parts) != 3:
            continue
        try:
            pid = int(parts[0].strip())
            gpu_uuid = parts[1].strip()
            mb = int(parts[2].strip())
        except ValueError:
            continue
        if not gpu_uuid:
            continue
        per_pid = usage.setdefault(pid, {})
        per_pid[gpu_uuid] = per_pid.get(gpu_uuid, 0) + mb
    return usage


def gpu_orphans(state: State) -> list[OrphanReport]:
    """Return PIDs holding GPU VRAM that aren't registered with any active lease.

    Args:
        state: Current pool state. Use :meth:`reslock.state.read_state` to
            get a snapshot, or :meth:`ResourcePool.gpu_orphans` to wrap this.

    Returns:
        One :class:`OrphanReport` per orphan PID, sorted by VRAM held
        descending (worst offenders first). Returns an empty list when
        nvidia-smi is unavailable or there are no compute-apps running.

    The reslock-side reclaim path (in ``pool._request_reclaim_to_resolve``)
    handles cascading evictions of *reslock-owned* reclaimable leases when
    the driver reports a shortfall. Orphans, by definition, are not reslock's
    to evict — this function exists so consumers can identify them and apply
    their own remediation policy (logging, paging, terminating their own
    children, etc.).
    """
    usage = _query_nvidia_smi()
    if not usage:
        return []
    registered = _registered_pids(state)
    orphans: list[OrphanReport] = []
    for pid, per_gpu in usage.items():
        if pid in registered:
            continue
        total = sum(per_gpu.values())
        if total <= 0:
            continue
        orphans.append(
            OrphanReport(
                pid=pid,
                vram_mb=total,
                gpus=dict(per_gpu),
                cmdline=_read_cmdline(pid),
            )
        )
    orphans.sort(key=lambda o: o.vram_mb, reverse=True)
    return orphans
