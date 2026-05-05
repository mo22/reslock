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
pip install reslock           # core
pip install reslock[cuda]     # adds nvidia-ml-py for the GPU pre-flight check
```

The `cuda` extra is required on machines that issue GPU VRAM leases (any
`gpu_<uuid>_vram_mb` resource key). reslock cross-checks its internal lease
accounting against the NVIDIA driver before granting a GPU lease, and hard-
fails if `pynvml` / `nvmlInit()` are unavailable on a CUDA host — silent
fallback to state-file-only accounting would defeat the purpose.

## Python API

```python
from reslock import ResourcePool

pool = ResourcePool()  # uses /var/lib/reslock/state.json (falls back to ~/.reslock if /var/lib is not writable)

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

## Registering resources

reslock is resource-agnostic — it tracks arbitrary named quantities without knowing what they represent. Each consumer registers the resources it knows about on startup, before acquiring any leases:

```python
from reslock import ResourcePool, detect_gpu_vram_mb

pool = ResourcePool()
pool.set_resources(detect_gpu_vram_mb())
# → {"gpu_GPU-1a2b3c4d-..._vram_mb": 24576, "gpu_GPU-5e6f7890-..._vram_mb": 24576}
```

GPU VRAM resources are keyed by the host-stable **GPU UUID** (as reported by `nvidia-smi` or `torch.cuda.get_device_properties(i).uuid`), not the nvidia-smi index. This keeps coordination correct across containers that get partial GPU mappings from the NVIDIA container runtime — each container sees only its mapped cards renumbered from 0, but UUIDs are invariant.

Consumers that hold a local torch device index should build the resource key via `gpu_resource_key(torch_index)`:

```python
from reslock import gpu_resource_key

key = gpu_resource_key(0)  # → "gpu_GPU-1a2b3c4d-..._vram_mb"
with pool.acquire(**{key: 4000}):
    run_model()
```

Multiple consumers can register different resource types independently — keys that aren't mentioned are left unchanged. This means an AI server can register GPU VRAM while a separate build system registers CPU cores, and they share the same state file.

### Built-in detection functions

reslock ships detection functions for common resource types. Dependencies like torch are imported lazily inside each function — safe to import even when they're not installed.

| Function | Resources | Method |
|----------|-----------|--------|
| `detect_gpu_vram_mb()` | `gpu_{uuid}_vram_mb`, ... | torch, then nvidia-smi fallback |
| `detect_gpu_vram_mb_torch()` | `gpu_{uuid}_vram_mb`, ... | torch CUDA runtime only (torch ≥ 2.0) |
| `detect_gpu_vram_mb_nvidia_smi()` | `gpu_{uuid}_vram_mb`, ... | nvidia-smi CLI only |
| `gpu_resource_key(torch_index)` | `gpu_{uuid}_vram_mb` | maps local torch index → UUID key |
| `detect_cpu_cores()` | `cpu_cores` | `os.sched_getaffinity` / `os.cpu_count` |
| `detect_disk_mb(["/", "/data"])` | `disk_root_mb`, ... | `shutil.disk_usage` |
| `detect_network_bandwidth()` | `net_eth0_mbps`, ... | sysfs (Linux) / networksetup (macOS) |

Example startup:

```python
from reslock import ResourcePool, detect_gpu_vram_mb, detect_cpu_cores

pool = ResourcePool()
pool.set_resources(detect_gpu_vram_mb())
pool.set_resources(detect_cpu_cores())
```

## CLI

```bash
# Set resources manually (use --short on status to fit UUID keys on screen)
reslock set cpu_cores 16

# Show status
reslock status
reslock status --short   # abbreviate GPU UUIDs to last 8 chars

# Run a command with reserved resources — indexes are resolved to UUIDs
reslock run --vram 4G llama-cli --model model.gguf
reslock run --gpu-vram 0:4G --gpu-vram 1:8G python train.py
reslock run --vram 8G --priority 10 --label "llama-70b" llama-cli ...
reslock run --vram 4G --ram 16G --cpu 4 python train.py

# Manage leases
reslock list
reslock release abc-123
reslock release --label whisper
reslock reset
```

## How resources work

Resources are named quantities with a total capacity. Resource names are arbitrary strings — define whatever you need. GPU VRAM keys specifically follow the `gpu_{uuid}_vram_mb` convention so that partial GPU mappings across containers coordinate correctly:

```python
pool.set_resources({"gpu_GPU-1a2b3c4d-...._vram_mb": 24000, "ram_mb": 65536, "gpu_slots": 2})
```

Or via CLI:

```bash
reslock set ram_mb 65536
```

Leases reserve amounts from these pools. When a lease is released (or its process dies), the resources become available again.

## Priority queue

When resources aren't immediately available, requests enter a priority queue. Higher priority number = more urgent. Ties are broken by arrival time (FIFO).

## Reclaimable leases

A process can mark its lease as **reclaimable** — "I'm using this, but can give it up if needed." When a higher-priority request needs those resources, `reclaim_requested` is set to `True`. The lease holder cooperates by releasing.

## Docker

Containers need access to the shared state file. Reslock's default state directory is `/var/lib/reslock` (mode `1777`, world-writable + sticky bit like `/tmp`); mount that path 1:1 from the host:

```bash
docker run --pid=host \
  -v /var/lib/reslock:/var/lib/reslock \
  my-gpu-app
```

- **`--pid=host`** — Required so the host can check container PIDs for dead-process cleanup. Without it, container PIDs are invisible to the host and leases won't be cleaned up when containers exit.
- **`-v /var/lib/reslock:/var/lib/reslock`** — Mounts the canonical state file directory. All containers and the host share the same `state.json`. The path inside the container must match where reslock will look — using the same path on host and container is the simplest setup.

**Multi-user:** The state directory is created with mode `1777` (world-writable + sticky bit, like `/tmp`) and the state file with mode `666`, so multiple containers running as different UIDs can share it without permission issues.

If `/var/lib/reslock` isn't writable in your environment (e.g. read-only host filesystem, hosts you don't own), reslock falls back to `~/.reslock`. Mount that path instead — adjusting the in-container side to match the user reslock runs as:

```bash
docker run --pid=host \
  -v ~/.reslock:/root/.reslock \
  my-gpu-app
```

Or override the path explicitly via `RESLOCK_DIR` or `state_path=`:

```python
# Both host and container code use the same explicit path
pool = ResourcePool(state_path="/shared/reslock/state.json")
```

```bash
docker run --pid=host \
  -v /shared/reslock:/shared/reslock \
  my-gpu-app
```

## Development

```bash
uv venv && uv pip install -e ".[dev]"
pytest
ruff check src/ tests/
```
