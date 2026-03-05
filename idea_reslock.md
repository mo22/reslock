# reslock — Resource Lock Manager

A small Python library + CLI for managing shared system resources (GPU VRAM, RAM, CPU cores, or arbitrary named resources) across multiple processes on a single machine.

## Problem

Multiple GPU-consuming processes (llama.cpp, whisper, vLLM, training jobs) compete for limited resources — especially VRAM. Without coordination, they OOM or degrade each other. Current workaround: manual scheduling or hoping for the best.

## Core Concepts

### State File

All coordination happens through a single JSON state file (e.g., `/var/run/reslock/state.json` or `~/.reslock/state.json`). No daemon required — processes coordinate via file locking.

**File locking:** Use `fcntl.flock` on Linux/macOS, `msvcrt.locking` on Windows (or `portalocker` as a cross-platform dependency — small, well-maintained). The lock is held only during state file reads/writes, not for the duration of a lease.

### Resources

Resources are named quantities with a total capacity:

```json
{
  "resources": {
    "vram_mb": 24000,
    "ram_mb": 65536,
    "cpu_cores": 20
  }
}
```

VRAM is auto-detected via `nvidia-smi` on init. Other resources are configured manually. Resource names are arbitrary strings — users can define whatever they need (e.g., `"gpu_slots": 2`).

### Leases

A lease reserves a set of resources. Leases are identified by a UUID and tracked by PID for dead-process cleanup.

```json
{
  "id": "abc123",
  "pid": 1234,
  "resources": {"vram_mb": 4000},
  "priority": 5,
  "acquired_at": "2026-03-05T10:00:00Z",
  "estimated_seconds": 120,
  "reclaimable": false
}
```

### Priority Queue

Waiters register in a queue. When resources free up, the highest-priority waiter (highest number = highest priority) that fits gets the lease. Ties broken by queue arrival time (FIFO).

```json
{
  "id": "def456",
  "pid": 5678,
  "resources": {"vram_mb": 8000},
  "priority": 10,
  "queued_at": "2026-03-05T10:01:00Z"
}
```

### Reclaimable Leases

A process can hold resources but mark them as **reclaimable** — meaning "I'm using this, but I can give it up if someone else needs it." This is the key mechanism for keeping models loaded in memory while allowing higher-priority work to preempt them.

**How it works:**

1. Process A loads a model and marks its lease as reclaimable
2. Process B requests resources that would require reclaiming A's lease
3. reslock marks A's lease as `"reclaim_requested": true`
4. Process A (polling or watching the state file) sees the request, unloads the model, and releases the lease
5. Process B's waiting acquire completes

Since reslock is primarily used as a Python library, the reclaimable lease integrates naturally:

```python
async with pool.acquire(vram_mb=4000, reclaimable=True) as lease:
    load_model()
    while True:
        await lease.wait_for_work_or_reclaim()
        if lease.reclaim_requested:
            unload_model()
            break
        handle_inference_request()
```

The library handles polling/watching the state file internally. The caller just needs to check `reclaim_requested` and cooperate.

**Timeout:** If a reclaimable lease doesn't release within a configurable timeout (default: 30s), the waiting process can choose to proceed anyway (force-reclaim) or keep waiting. The lease holder's PID can also be sent a signal as escalation.

## State File Structure

```json
{
  "version": 1,
  "resources": {
    "vram_mb": 24000,
    "ram_mb": 65536
  },
  "leases": [
    {
      "id": "abc-123",
      "pid": 1234,
      "resources": {"vram_mb": 4000},
      "priority": 5,
      "acquired_at": "2026-03-05T10:00:00Z",
      "estimated_seconds": 120,
      "reclaimable": false,
      "reclaim_requested": false,
      "label": "whisper-large-v3"
    }
  ],
  "queue": [
    {
      "id": "def-456",
      "pid": 5678,
      "resources": {"vram_mb": 8000},
      "priority": 10,
      "queued_at": "2026-03-05T10:01:00Z",
      "label": "llama-70b"
    }
  ]
}
```

## Dead Process Cleanup

On every state file access, reslock checks all PIDs in leases and queue entries. Dead processes (PID no longer exists) are removed automatically. This handles crashes, killed processes, etc. without requiring a cleanup daemon.

On Windows: `os.kill(pid, 0)` or `ctypes.windll.kernel32.OpenProcess`. On Unix: `os.kill(pid, 0)`.

## Python API

```python
from reslock import ResourcePool

# Connect to (or initialize) a pool
pool = ResourcePool()  # uses default path ~/.reslock/state.json
pool = ResourcePool("/var/run/reslock/state.json")

# --- Basic acquire/release ---

# Blocking acquire — waits in priority queue
with pool.acquire(vram_mb=4000, priority=5, estimated_seconds=120, label="whisper") as lease:
    run_whisper(audio_file)
    lease.update(estimated_seconds=30)  # revise estimate
# auto-released

# Async version
async with pool.acquire_async(vram_mb=4000, priority=5) as lease:
    await run_inference()

# Non-blocking try
lease = pool.try_acquire(vram_mb=4000)
if lease:
    try:
        do_work()
    finally:
        lease.release()

# --- Reclaimable leases ---

lease = pool.acquire(vram_mb=4000, reclaimable=True)
load_model()
# ... later, when reclaim is requested:
if lease.reclaim_requested:
    unload_model()
    lease.release()

# --- Status ---

status = pool.status()
print(status.resources)       # total capacity
print(status.available)       # currently free
print(status.leases)          # active leases
print(status.queue)           # waiting requests
```

## CLI

### Initialize / Configure

```bash
# Auto-detect GPU and initialize
reslock init
# -> Detected: vram_mb=24576 (NVIDIA RTX 3090)
# -> Created ~/.reslock/state.json

# Set/update resources manually
reslock set vram_mb 24000
reslock set gpu_slots 2
reslock set ram_mb 65536

# Show current state
reslock status
# RESOURCES        TOTAL    USED    FREE
# vram_mb         24000    4000   20000
# ram_mb          65536    8192   57344
#
# LEASES (1 active)
# abc-123  pid=1234  vram_mb=4000  prio=5  label=whisper  reclaimable  2m ago
#
# QUEUE (1 waiting)
# def-456  pid=5678  vram_mb=8000  prio=10  label=llama-70b  queued 30s ago
```

### Run Commands with Resource Reservation

```bash
# Reserve VRAM and run a command — lease held for duration of the subprocess
reslock run --vram 4G llama-cli --model llama-70b.gguf ...

# With priority and label
reslock run --vram 8G --priority 10 --label "llama-70b" llama-cli ...

# Multiple resource types
reslock run --vram 4G --ram 16G --cpu 4 python train.py

# Shorthand for common units
reslock run --vram 4G    # 4096 MB
reslock run --vram 500M  # 500 MB
```

The `run` command:
1. Enters the priority queue
2. Waits for resources (prints status while waiting)
3. Acquires the lease
4. Spawns the subprocess
5. Releases the lease when the subprocess exits

### Other Commands

```bash
# List active leases
reslock list

# Release a specific lease (by ID or label)
reslock release abc-123
reslock release --label whisper

# Reset / clear all state
reslock reset
```

## Implementation Notes

### Cross-Platform File Locking

Python's `fcntl` is Unix-only. Options:

1. **`portalocker`** — pip package, wraps platform-specific locking. Small, well-maintained, no compiled extensions. Preferred.
2. **`filelock`** — another option, uses lock files rather than `fcntl`/`msvcrt`. Works everywhere but slightly different semantics.
3. **Manual:** `fcntl.flock` on Unix, `msvcrt.locking` on Windows. More code but zero dependencies.

Recommendation: use `portalocker` or `filelock` — both are pure Python and widely used.

### Polling vs. Watching

Waiters poll the state file. Default interval: 250ms. This is simple, cross-platform, and the overhead of reading a small JSON file is negligible. File-watching (inotify/kqueue/ReadDirectoryChanges) would be more efficient but adds complexity for minimal gain.

### Atomicity

State file updates follow read-lock-modify-write-unlock pattern:
1. Open and lock the file (shared lock for reads, exclusive for writes)
2. Read current state
3. Modify state
4. Write entire file (write to temp file + rename for atomicity)
5. Release lock

### Package Structure

```
reslock/
  pyproject.toml
  src/reslock/
    __init__.py       # public API re-exports
    pool.py           # ResourcePool class
    state.py          # state file read/write/locking
    models.py         # dataclasses for Lease, QueueEntry, State
    cleanup.py        # dead process detection
    cli.py            # click/typer CLI
    detect.py         # auto-detect GPU/system resources
  tests/
    test_pool.py
    test_state.py
    test_cli.py
    test_cleanup.py
```

### Dependencies

- **Runtime:** `portalocker` (or `filelock`), `click` (CLI)
- **Dev:** `pytest`, `ruff`
- Python 3.10+

## Future / Out of Scope (for now)

- **Multi-machine coordination:** A higher-level service that reads state files from multiple machines (via SSH, NFS, or a simple HTTP API). The single-machine library stays unchanged.
- **GPU device selection:** Track per-GPU resources when multiple GPUs are present (e.g., `gpu0_vram_mb`, `gpu1_vram_mb`).
- **Usage statistics / history:** Log lease durations, wait times, etc. for capacity planning.
- **Integration with cgroups / GPU isolation:** Actually enforce resource limits, not just coordinate.
