# reslock

Resource lock manager for coordinating shared system resources (GPU VRAM, RAM, CPU cores) across multiple processes on a single machine.

## Problem

Multiple GPU-consuming processes (llama.cpp, whisper, vLLM, training jobs) compete for limited resources — especially VRAM. Without coordination, they OOM or degrade each other.

## How it works

- All coordination happens through a single JSON state file — no daemon required
- Processes coordinate via file locking (held only during reads/writes, not for lease duration)
- Dead processes are automatically cleaned up via PID checking
- Priority queue determines which waiter gets resources next
- Reclaimable leases allow loaded models to be preempted by higher-priority work

## Install

```bash
pip install reslock
```

## Python API

```python
from reslock import ResourcePool

pool = ResourcePool()  # uses ~/.reslock/state.json

# Context manager — blocks until resources are available
with pool.acquire(vram_mb=4000, priority=5, label="whisper") as lease:
    run_whisper(audio_file)

# Non-blocking
lease = pool.try_acquire(vram_mb=4000)
if lease:
    try:
        do_work()
    finally:
        lease.release()

# Async
async with pool.acquire_async(vram_mb=4000) as lease:
    await run_inference()

# Reclaimable lease — can be preempted
lease = pool.acquire(vram_mb=4000, reclaimable=True)
load_model()
# ... later:
if lease.reclaim_requested:
    unload_model()
    lease.release()

# Check status
status = pool.status()
print(status.available)  # free resources
```

## CLI

```bash
# Initialize (auto-detects GPU)
reslock init

# Set resources manually
reslock set vram_mb 24000
reslock set gpu_slots 2

# Show status
reslock status

# Run a command with reserved resources
reslock run --vram 4G llama-cli --model model.gguf
reslock run --vram 8G --priority 10 --label "llama-70b" llama-cli ...
reslock run --vram 4G --ram 16G --cpu 4 python train.py

# Manage leases
reslock list
reslock release abc-123
reslock release --label whisper
reslock reset
```

## How resources work

Resources are named quantities with a total capacity. Resource names are arbitrary strings — define whatever you need:

```bash
reslock set vram_mb 24000
reslock set ram_mb 65536
reslock set gpu_slots 2
```

Leases reserve amounts from these pools. When a lease is released (or its process dies), the resources become available again.

## Priority queue

When resources aren't immediately available, requests enter a priority queue. Higher priority number = more urgent. Ties are broken by arrival time (FIFO).

## Reclaimable leases

A process can mark its lease as **reclaimable** — "I'm using this, but can give it up if needed." When a higher-priority request needs those resources, `reclaim_requested` is set to `True`. The lease holder cooperates by releasing.

## Development

```bash
uv venv && uv pip install -e ".[dev]"
pytest
ruff check src/ tests/
```
