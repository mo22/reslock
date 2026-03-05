# Docker container PID namespace support

## Problem

reslock's dead process cleanup uses `os.kill(pid, 0)` to check if a process is alive.
This doesn't work across PID namespaces — a container's internal PID (e.g., PID 1)
differs from its host PID (e.g., PID 12345).

When a container acquires a reslock lease using `os.getpid()`, the host can't verify
the PID is alive, so leases from crashed containers won't auto-cleanup.

## Requested features

### 1. Explicit host PID parameter

Allow passing the host-visible PID when acquiring a lease:

```python
# Inside a Docker container, pass the host PID explicitly
pool.acquire(vram_mb=4000, host_pid=get_host_pid())
```

The container can discover its host PID via `/proc/1/sched` or `/proc/self/status`
(NSpid field on Linux). reslock could provide a helper:

```python
def get_host_pid() -> int:
    """Get the host-visible PID of this process (for containers)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("NSpid:"):
                    pids = line.split()[1:]
                    return int(pids[0])  # first PID is in the host namespace
    except (FileNotFoundError, IndexError):
        pass
    return os.getpid()  # fallback: not in a container
```

### 2. Lease cleanup on container stop

When a container stops, its leases should be cleaned up. Options:
- The container runs a shutdown hook that calls `lease.release()`
- reslock's PID check detects the host PID is gone (works if host PID was stored)
- An external `reslock release --label aiserver:*` command runs on container stop

### Context

This is needed for the Scriba AI Server, which runs in Docker with `--gpus all`
and coordinates GPU memory with other processes on the host via reslock.

The recommended workaround is `--pid=host`, but explicit host PID support would
be cleaner and work without privileged PID namespace access.
