"""CLI for reslock."""

from __future__ import annotations

import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from reslock.detect import (
    detect_gpu_vram_mb,
    get_all_pid_vram_mb,
    get_all_pid_vram_per_gpu_mb,
    get_pid_cpu_seconds,
    get_pid_rss_mb,
    parse_gpu_vram_key,
)
from reslock.models import QueueEntry, State
from reslock.pool import ResourcePool
from reslock.state import DEFAULT_STATE_PATH, ensure_state_file, transact

console = Console()


def _parse_size(value: str) -> int:
    """Parse a size string like '4G', '500M', or plain number (MB)."""
    m = re.match(r"^(\d+(?:\.\d+)?)\s*([gGmM]?)$", value)
    if not m:
        raise click.BadParameter(f"Invalid size: {value}")
    num = float(m.group(1))
    unit = m.group(2).upper()
    if unit == "G":
        return int(num * 1024)
    return int(num)


@click.group()
@click.version_option(package_name="reslock")
def main() -> None:
    """Resource lock manager for coordinating shared system resources."""


@main.command(deprecated=True)
@click.option("--state", "-s", type=click.Path(path_type=Path), default=None)
def init(state: Path | None) -> None:
    """Initialize reslock state file, auto-detecting per-GPU VRAM.

    Deprecated: prefer pool.set_resources() with the built-in detection
    functions (e.g. detect_gpu_vram_mb()) — each consumer registers the
    resources it knows about on startup.
    """
    path = state or DEFAULT_STATE_PATH
    ensure_state_file(path)

    gpu = detect_gpu_vram_mb()

    def _init(st: State) -> None:
        for key, val in gpu.items():
            if key not in st.resources:
                st.resources[key] = val

    transact(path, _init)

    if gpu:
        for key, val in sorted(gpu.items()):
            console.print(f"[green]Detected:[/green] {key}={val}")
    console.print(f"[green]State file:[/green] {path}")
    console.print(
        "[yellow]Note:[/yellow] 'reslock init' is deprecated. "
        "Prefer pool.set_resources() with detect_gpu_vram_mb() in your application startup."
    )


@main.command(name="set")
@click.argument("resource")
@click.argument("value", type=int)
@click.option("--state", "-s", type=click.Path(path_type=Path), default=None)
def set_resource(resource: str, value: int, state: Path | None) -> None:
    """Set a resource capacity (e.g., reslock set vram_mb 24000)."""
    path = state or DEFAULT_STATE_PATH
    ensure_state_file(path)

    def _set(st: State) -> None:
        st.resources[resource] = value

    transact(path, _set)
    console.print(f"[green]Set[/green] {resource} = {value}")


def _shorten_resource_key(key: str) -> str:
    """Trim the UUID portion of a ``gpu_{uuid}_vram_mb`` key to its last 8 chars."""
    uuid_str = parse_gpu_vram_key(key)
    if uuid_str is None or len(uuid_str) <= 8:
        return key
    return f"gpu_…{uuid_str[-8:]}_vram_mb"


@main.command()
@click.option("--state", "-s", type=click.Path(path_type=Path), default=None)
@click.option("--short", is_flag=True, help="Abbreviate GPU UUIDs to the last 8 chars")
def status(state: Path | None, short: bool) -> None:
    """Show current resource status, leases, and queue."""
    path = state or DEFAULT_STATE_PATH
    ensure_state_file(path)
    pool = ResourcePool(path)
    st = pool.status()

    def _fmt_key(key: str) -> str:
        return _shorten_resource_key(key) if short else key

    def _fmt_resources(res: dict[str, int]) -> str:
        return ", ".join(f"{_fmt_key(k)}={v}" for k, v in res.items())

    if st.resources:
        table = Table(title="Resources")
        table.add_column("Resource", style="cyan")
        table.add_column("Total", justify="right")
        table.add_column("Used", justify="right")
        table.add_column("Free", justify="right")
        for key, total in st.resources.items():
            free = st.available.get(key, 0)
            used = total - free
            table.add_row(_fmt_key(key), str(total), str(used), str(free))
        console.print(table)
    else:
        console.print(
            "[dim]No resources configured. Use pool.set_resources() or 'reslock set'.[/dim]"
        )

    if st.leases:
        table = Table(title=f"Leases ({len(st.leases)} active)")
        table.add_column("ID", style="cyan")
        table.add_column("PID")
        table.add_column("Resources")
        table.add_column("Priority", justify="right")
        table.add_column("Label")
        table.add_column("Flags")
        table.add_column("Age")
        now = datetime.now(timezone.utc)
        for lease in st.leases:
            age = now - lease.acquired_at
            mins = int(age.total_seconds()) // 60
            age_str = f"{mins}m ago" if mins > 0 else f"{int(age.total_seconds())}s ago"
            res_str = _fmt_resources(lease.resources)
            flags: list[str] = []
            if lease.reclaimable:
                flags.append("reclaimable")
            if lease.reclaim_requested:
                flags.append("reclaim_requested")
            table.add_row(
                lease.id,
                str(lease.pid),
                res_str,
                str(lease.priority),
                lease.label or "",
                " ".join(flags),
                age_str,
            )
        console.print(table)

    if st.queue:
        active = [e for e in st.queue if e.is_active]
        waiting = [e for e in st.queue if not e.is_active]
        now = datetime.now(timezone.utc)

        def _fmt_request(e: QueueEntry) -> str:
            parts: list[str] = []
            if e.num_gpus > 0 and e.vram_mb_each is not None:
                parts.append(f"{e.num_gpus}x{e.vram_mb_each}MB GPU")
            if e.resources:
                parts.append(_fmt_resources(e.resources))
            return ", ".join(parts) or "-"

        if waiting:
            table = Table(title=f"Queue ({len(waiting)} waiting)")
            table.add_column("ID", style="cyan")
            table.add_column("PID")
            table.add_column("Request")
            table.add_column("Priority", justify="right")
            table.add_column("Label")
            table.add_column("Queued")
            for e in waiting:
                age = now - e.queued_at
                secs = int(age.total_seconds())
                queued_str = f"{secs // 60}m ago" if secs >= 60 else f"{secs}s ago"
                table.add_row(
                    e.id, str(e.pid), _fmt_request(e), str(e.priority), e.label or "", queued_str
                )
            console.print(table)

        if active:
            table = Table(title=f"Active work ({len(active)} entries)")
            table.add_column("ID", style="cyan")
            table.add_column("PID")
            table.add_column("Lease")
            table.add_column("ETA")
            table.add_column("Progress", justify="right")
            table.add_column("Label")
            for e in active:
                eta = f"{e.estimated_seconds}s" if e.estimated_seconds is not None else "-"
                progress = f"{e.progress:.0%}" if e.progress is not None else "-"
                table.add_row(
                    e.id,
                    str(e.pid),
                    e.lease_id or "-",
                    eta,
                    progress,
                    e.label or "",
                )
            console.print(table)

    if not st.leases and not st.queue:
        console.print("[dim]No active leases or queued requests.[/dim]")


@main.command()
@click.option("--state", "-s", type=click.Path(path_type=Path), default=None)
def list_(state: Path | None) -> None:
    """List active leases."""
    path = state or DEFAULT_STATE_PATH
    ensure_state_file(path)
    pool = ResourcePool(path)
    st = pool.status()
    if not st.leases:
        console.print("[dim]No active leases.[/dim]")
        return
    for lease in st.leases:
        res_str = ", ".join(f"{k}={v}" for k, v in lease.resources.items())
        lbl = f"  label={lease.label}" if lease.label else ""
        console.print(f"  {lease.id}  pid={lease.pid}  {res_str}  prio={lease.priority}{lbl}")


# Register "list" as the CLI command name
list_.__name__ = "list"  # pyright: ignore[reportAttributeAccessIssue]
main.add_command(list_, "list")


@main.command()
@click.argument("lease_id", required=False)
@click.option("--label", "-l", default=None, help="Release by label")
@click.option("--state", "-s", type=click.Path(path_type=Path), default=None)
def release(lease_id: str | None, label: str | None, state: Path | None) -> None:
    """Release a lease by ID or label."""
    if not lease_id and not label:
        raise click.UsageError("Provide a lease ID or --label")
    path = state or DEFAULT_STATE_PATH
    ensure_state_file(path)
    released: list[str] = []

    def _release(st: State) -> None:
        before = len(st.leases)
        removed: set[str] = set()
        if lease_id:
            removed = {ls.id for ls in st.leases if ls.id == lease_id}
            st.leases = [ls for ls in st.leases if ls.id != lease_id]
        elif label:
            removed = {ls.id for ls in st.leases if ls.label == label}
            st.leases = [ls for ls in st.leases if ls.label != label]
        # Drop any active QueueEntries attached to the released leases — v3
        # entries persist across active state, so a CLI release that doesn't
        # also clean them up would leave orphans pointing at a missing lease,
        # which keeps peer ETA calculations reading stale work.
        if removed:
            st.queue = [e for e in st.queue if e.lease_id not in removed]
        released.extend(["x"] * (before - len(st.leases)))

    transact(path, _release)
    if released:
        console.print(f"[green]Released {len(released)} lease(s).[/green]")
    else:
        console.print("[yellow]No matching lease found.[/yellow]")


@main.command()
@click.option("--state", "-s", type=click.Path(path_type=Path), default=None)
def reset(state: Path | None) -> None:
    """Clear all state (leases, queue)."""
    path = state or DEFAULT_STATE_PATH
    ensure_state_file(path)

    def _reset(st: State) -> None:
        st.leases.clear()
        st.queue.clear()

    transact(path, _reset)
    console.print("[green]State cleared.[/green]")


@main.command()
def schema() -> None:
    """Print the JSON schema for the state file."""
    import json

    click.echo(json.dumps(State.model_json_schema(), indent=2))


@main.command()
@click.option("--interval", "-n", type=float, default=2.0, help="Refresh interval in seconds")
@click.option(
    "--count", "-c", type=int, default=None, help="Number of iterations (default: unlimited)"
)
@click.option("--state", "-s", type=click.Path(path_type=Path), default=None)
def top(interval: float, count: int | None, state: Path | None) -> None:
    """Watch resource status with live-measured process usage."""
    from rich.live import Live

    path = state or DEFAULT_STATE_PATH
    ensure_state_file(path)
    pool = ResourcePool(path)
    iterations = 0

    def _render() -> Table:
        st = pool.status()
        # Gather live measurements — sum across main PID + registered child PIDs
        pid_vram = get_all_pid_vram_mb()
        pid_vram_per_gpu = get_all_pid_vram_per_gpu_mb()
        # Map lease_id → most recent attached entry, for ETA/progress display.
        attached_entry: dict[str, QueueEntry] = {}
        for e in st.queue:
            if e.lease_id is not None:
                attached_entry[e.lease_id] = e
        lease_rss: dict[str, int] = {}
        lease_cpu: dict[str, float] = {}
        lease_vram: dict[str, int] = {}
        lease_vram_per_gpu: dict[str, dict[str, int]] = {}
        for lease in st.leases:
            all_pids = [lease.pid, *lease.pids]
            rss_total = 0
            cpu_total = 0.0
            vram_total = 0
            per_gpu: dict[str, int] = {}
            for pid in all_pids:
                rss = get_pid_rss_mb(pid)
                if rss:
                    rss_total += rss
                cpu = get_pid_cpu_seconds(pid)
                if cpu:
                    cpu_total += cpu
                vram_total += pid_vram.get(pid, 0)
                for gpu_uuid, mb in pid_vram_per_gpu.get(pid, {}).items():
                    per_gpu[gpu_uuid] = per_gpu.get(gpu_uuid, 0) + mb
            if rss_total:
                lease_rss[lease.id] = rss_total
            if cpu_total:
                lease_cpu[lease.id] = cpu_total
            if vram_total:
                lease_vram[lease.id] = vram_total
            if per_gpu:
                lease_vram_per_gpu[lease.id] = per_gpu

        grid = Table(title="reslock top", expand=True)
        grid.add_column("", style="bold")
        grid.add_column("")

        # Resource summary
        if st.resources:
            res_parts: list[str] = []
            for key, total in st.resources.items():
                free = st.available.get(key, 0)
                used = total - free
                res_parts.append(f"{key}: {used}/{total}")
            grid.add_row("Resources", "  ".join(res_parts))
        grid.add_row("Leases", str(len(st.leases)))
        grid.add_row("Queue", str(len(st.queue)))

        if st.leases:
            grid.add_section()
            grid.add_row("", "")

            lease_table = Table(show_header=True, show_edge=False, pad_edge=False)
            lease_table.add_column("PID", style="cyan", justify="right")
            lease_table.add_column("Label")
            lease_table.add_column("Allocated", justify="right")
            lease_table.add_column("Actual", justify="right")
            lease_table.add_column("RSS", justify="right")
            lease_table.add_column("CPU", justify="right")
            lease_table.add_column("Age", justify="right")
            lease_table.add_column("Progress", justify="right")
            lease_table.add_column("Flags")

            now = datetime.now(timezone.utc)
            for lease in st.leases:
                age = now - lease.acquired_at
                secs = int(age.total_seconds())
                if secs >= 3600:
                    age_str = f"{secs // 3600}h{(secs % 3600) // 60}m"
                elif secs >= 60:
                    age_str = f"{secs // 60}m{secs % 60}s"
                else:
                    age_str = f"{secs}s"

                alloc_str = ", ".join(
                    f"{_shorten_resource_key(k)}={v}" for k, v in lease.resources.items()
                )

                # Actual resources: prefer self-reported, fall back to measured
                actual_parts: list[str] = []
                if lease.actual_resources:
                    actual_parts = [
                        f"{_shorten_resource_key(k)}={v}" for k, v in lease.actual_resources.items()
                    ]
                else:
                    per_gpu = lease_vram_per_gpu.get(lease.id, {})
                    if per_gpu:
                        for gpu_uuid in sorted(per_gpu):
                            short_uuid = gpu_uuid[-8:] if len(gpu_uuid) > 8 else gpu_uuid
                            actual_parts.append(f"gpu…{short_uuid}={per_gpu[gpu_uuid]}")
                    elif lease_vram.get(lease.id):
                        actual_parts.append(f"vram={lease_vram[lease.id]}")
                actual_str = ", ".join(actual_parts) if actual_parts else "-"

                # Check for overuse
                for key, val in lease.resources.items():
                    actual_val = lease.actual_resources.get(key)
                    if actual_val is None:
                        uuid_from_key = parse_gpu_vram_key(key)
                        if uuid_from_key is not None:
                            actual_val = lease_vram_per_gpu.get(lease.id, {}).get(uuid_from_key)
                    if actual_val is not None and actual_val > val:
                        actual_str = f"[red]{actual_str}[/red]"
                        break

                pid_str = str(lease.pid)
                if lease.pids:
                    pid_str += f"+{len(lease.pids)}"

                rss = lease_rss.get(lease.id)
                rss_str = f"{rss}M" if rss else "-"

                cpu = lease.cpu_seconds or lease_cpu.get(lease.id)
                if cpu is not None:
                    if cpu >= 3600:
                        cpu_str = f"{cpu / 3600:.1f}h"
                    elif cpu >= 60:
                        cpu_str = f"{cpu / 60:.1f}m"
                    else:
                        cpu_str = f"{cpu:.1f}s"
                else:
                    cpu_str = "-"

                attached = attached_entry.get(lease.id)
                progress_str = (
                    f"{attached.progress:.0%}"
                    if attached is not None and attached.progress is not None
                    else "-"
                )

                flags: list[str] = []
                if lease.reclaimable:
                    flags.append("R")
                if lease.reclaim_requested:
                    flags.append("[red]RECLAIM[/red]")

                lease_table.add_row(
                    pid_str,
                    lease.label or "",
                    alloc_str,
                    actual_str,
                    rss_str,
                    cpu_str,
                    age_str,
                    progress_str,
                    " ".join(flags),
                )

            console.print(lease_table, end="")

        waiting = [e for e in st.queue if not e.is_active]
        if waiting:
            grid.add_section()
            for e in waiting:
                parts: list[str] = []
                if e.num_gpus > 0 and e.vram_mb_each is not None:
                    parts.append(f"{e.num_gpus}x{e.vram_mb_each}MB GPU")
                if e.resources:
                    parts.append(", ".join(f"{k}={v}" for k, v in e.resources.items()))
                req = ", ".join(parts) or "-"
                grid.add_row(f"Queued [{e.id}]", f"pid={e.pid}  {req}  prio={e.priority}")

        return grid

    try:
        with Live(console=console, refresh_per_second=1, screen=False) as live:
            while True:
                live.update(_render())
                iterations += 1
                if count is not None and iterations >= count:
                    break
                time.sleep(interval)
    except KeyboardInterrupt:
        pass


@main.command()
@click.option(
    "--vram-mb-each",
    default=None,
    help="Per-GPU VRAM to reserve (e.g., 4G, 500M). Required when --num-gpus > 0.",
)
@click.option(
    "--num-gpus", type=int, default=0, help="Number of GPUs to reserve (scheduler picks any)"
)
@click.option("--ram", default=None, help="RAM to reserve (e.g., 16G)")
@click.option("--cpu", type=int, default=None, help="CPU cores to reserve")
@click.option("--priority", "-p", type=int, default=0, help="Priority (higher = more urgent)")
@click.option("--label", "-l", default=None, help="Label for this lease")
@click.option(
    "--reclaimable", is_flag=True, help="Allow lease to be reclaimed by higher-priority requests"
)
@click.option(
    "--estimated-seconds",
    type=int,
    default=None,
    help="Predicted wall-clock seconds for the work; auto-creates a tracking entry",
)
@click.option(
    "--reclaim-signal",
    default="SIGTERM",
    help="Signal to send to the child process when reclaim is requested (default: SIGTERM)",
)
@click.option("--state", "-s", type=click.Path(path_type=Path), default=None)
@click.argument("command", nargs=-1, required=True)
def run(
    vram_mb_each: str | None,
    num_gpus: int,
    ram: str | None,
    cpu: int | None,
    priority: int,
    label: str | None,
    reclaimable: bool,
    estimated_seconds: int | None,
    reclaim_signal: str,
    state: Path | None,
    command: tuple[str, ...],
) -> None:
    """Reserve resources and run a command.

    GPU placement: if --num-gpus > 0, the scheduler picks --num-gpus GPUs
    with the most free VRAM at promotion time (spread placement). Use
    --reclaimable to allow eviction by higher-priority requests; on reclaim
    the specified signal is sent to the child process.
    """
    vram_each_mb: int | None = None
    if vram_mb_each is not None:
        vram_each_mb = _parse_size(vram_mb_each)
    if num_gpus > 0 and vram_each_mb is None:
        raise click.UsageError("--vram-mb-each is required when --num-gpus > 0")
    if vram_each_mb is not None and num_gpus == 0:
        raise click.UsageError("--num-gpus must be > 0 when --vram-mb-each is set")

    non_gpu: dict[str, int] = {}
    if ram:
        non_gpu["ram_mb"] = _parse_size(ram)
    if cpu:
        non_gpu["cpu_cores"] = cpu

    if num_gpus == 0 and not non_gpu:
        raise click.UsageError(
            "Specify at least one resource (--vram-mb-each + --num-gpus, --ram, --cpu)"
        )

    sig = getattr(signal, reclaim_signal, None)
    if sig is None:
        raise click.BadParameter(f"Unknown signal: {reclaim_signal}")

    path = state or DEFAULT_STATE_PATH
    ensure_state_file(path)
    pool = ResourcePool(path)

    parts: list[str] = []
    if num_gpus > 0:
        parts.append(f"{num_gpus}x{vram_each_mb}MB GPU")
    if non_gpu:
        parts.extend(f"{k}={v}" for k, v in non_gpu.items())
    console.print(f"[dim]Waiting for resources: {', '.join(parts)}...[/dim]")
    with pool.acquire(
        vram_mb_each=vram_each_mb,
        num_gpus=num_gpus,
        priority=priority,
        label=label,
        reclaimable=reclaimable,
        estimated_seconds=estimated_seconds,
        **non_gpu,
    ) as lease:
        console.print(f"[green]Acquired lease {lease.id}[/green]")
        # Pin the child to the granted GPUs. CUDA accepts UUID-form
        # CUDA_VISIBLE_DEVICES, so we don't need to resolve to torch indices.
        # Without this, a child that defaults to GPU 0 could allocate on an
        # unreserved card and trample another lease.
        env: dict[str, str] | None = None
        if lease.gpu_uuids:
            import os as _os

            env = {**_os.environ, "CUDA_VISIBLE_DEVICES": ",".join(lease.gpu_uuids)}
        proc = subprocess.Popen(list(command), env=env)
        lease.update(pids=[proc.pid])

        def _sighandler(signum: int, _frame: object) -> None:
            proc.send_signal(signum)

        signal.signal(signal.SIGTERM, _sighandler)
        signal.signal(signal.SIGINT, _sighandler)

        if reclaimable:

            def _watch_reclaim() -> None:
                while proc.poll() is None:
                    if lease.reclaim_requested:
                        console.print(
                            f"[yellow]Reclaim requested, sending {reclaim_signal} to child[/yellow]"
                        )
                        proc.send_signal(sig)
                        return
                    time.sleep(0.5)

            threading.Thread(target=_watch_reclaim, daemon=True).start()

        returncode = proc.wait()

    console.print(f"[dim]Lease released. Process exited with code {returncode}.[/dim]")
    sys.exit(returncode)


if __name__ == "__main__":
    main()
