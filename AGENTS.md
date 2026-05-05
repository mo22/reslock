# reslock

Resource lock manager for coordinating shared system resources (GPU VRAM, RAM, CPU) across processes on a single machine. No daemon — all coordination via a JSON state file with file locking.

## Project Structure

- `src/reslock/` — library source
  - `pool.py` — `ResourcePool` and `LeaseHandle` (main API)
  - `models.py` — Pydantic models (`State`, `Lease`, `QueueEntry`, `PoolStatus`)
  - `state.py` — file-locked state read/write
  - `detect.py` — system resource detection (GPU VRAM via nvidia-smi/torch, CPU, disk)
  - `nvml.py` — pynvml pre-flight (driver-side free VRAM ground truth, hard-fails on CUDA hosts without pynvml)
  - `resources.py` — public detection API (re-exports from detect.py)
  - `cleanup.py` — dead-PID lease cleanup
  - `cli.py` — Click CLI (`reslock status`, `reslock acquire`, etc.)
- `tests/` — pytest tests
- `examples/` — usage examples

## Development

```bash
uv sync --all-extras        # install deps
uv run pytest               # run tests
uvx ruff format src/ tests/ # format
uvx ruff check --fix src/ tests/  # lint
```

## Publishing

Publishing is handled by GitHub Actions (`.github/workflows/publish.yml`). To release:

1. Bump version in `pyproject.toml`
2. Commit and push to main
3. Create a GitHub release with tag `vX.Y.Z` — this triggers trusted publishing to PyPI

Do NOT publish manually with `uv publish` — the project uses PyPI trusted publishing via OIDC.

## Key Design Decisions

- State file default: `/var/lib/reslock/state.json` (system-wide, sticky-bit 1777). Falls back to `~/.reslock/state.json` with a warning if `/var/lib` is not writable. Override with `RESLOCK_DIR` env var. Path selection (in `_default_state_path()`) is side-effect-free; directory creation happens in `ensure_state_file()`.
- File locks are held only during reads/writes, not for lease duration
- `read_state()` uses shared lock (`portalocker "r"`), `transact()` uses exclusive lock (`"r+"`) — read-heavy workloads don't block each other
- Dead processes cleaned up automatically via PID checking; a lease is alive if owner PID OR any child PID in `lease.pids` is alive
- Per-GPU VRAM tracking for multi-GPU systems. Keys are UUID-based: `gpu_{UUID}_vram_mb` (e.g. `gpu_GPU-1a2b3c4d-..._vram_mb`). Keying by the host-stable GPU UUID (not nvidia-smi index) keeps coordination correct across containers with partial GPU mappings — the NVIDIA container runtime renumbers visible devices from 0, but UUIDs are invariant. Consumers that hold a local torch index should use `reslock.gpu_resource_key(i)` to build the key.
- State-file schema version: `2` (bumped from `1` in v0.5.0 when GPU keys switched to UUIDs). On read, files with a different `version` are reset — `resources`, `leases`, `queue` are all cleared so consumers repopulate with UUID keys. Upgrade procedure: stop all reslock consumers on a host, delete `state.json`, redeploy with matching versions.
- Priority queue determines which waiter gets resources next
- Reclaimable leases allow preemption by higher-priority work
- NVML pre-flight (v0.7.0+, `nvml.py`): on every GPU VRAM acquire, reslock reads driver-reported free VRAM via pynvml and compares against the request. If the driver disagrees with internal lease accounting (typically because a process holds VRAM without a registered lease — see SCRIBA-325), reslock either refuses (`try_acquire`) or signals reclaim on its own opportunistic leases (`acquire`). Hard-fails on a CUDA host without pynvml — silent fallback would defeat the cross-check. Cache window 1s. Non-GPU acquires never touch pynvml (the `cuda` extra is optional). Uses `State.reclaimable_for_shortfall(..., partial=True)` so we evict what we can even when an external (unaccounted) consumer will keep us short anyway — the lease is granted on a later poll once the external process exits, or the caller's outer timeout kicks in.
- `LeaseHandle.shrink()` doesn't need its own scheduler hook — waiters in `_acquire_blocking` / `acquire_async` already poll `_try_promote()` every `poll_interval`, so freed capacity is picked up naturally (same as `release()`).
- `Lease.queued_at` / `Lease.wait_sec` (added in v0.6.0) are stamped at promotion time so consumers can split "queued behind reslock" from "running on GPU" without timing the acquire themselves. `try_acquire` records `wait_sec=0.0` (no queue path); `_try_promote` computes `acquired_at - queued_at`. `LeaseHandle.gpu_uuids` and `LeaseHandle.gpu_torch_indices` derive GPU device lists from the lease's `gpu_{uuid}_vram_mb` keys (the latter is empty when CUDA/torch isn't available). Schema version stays at 2 because the new fields are additive `Optional[...] = None`, but `Lease.model_config = {"extra": "forbid"}` means a 0.5.x reader will reject a state file written by 0.6.0 — upgrade all consumers on a host together.

## Platform compatibility

- `resource` (POSIX stdlib) is imported with `try/except ImportError` in `detect.py` — Windows doesn't ship it. `get_self_rss_mb()` and `get_self_cpu_seconds()` return `None` when it's absent; downstream `get_self_actual_resources()` already handles `None` RSS. Regression covered in `tests/test_windows_compat.py` by mocking `builtins.__import__` + `importlib.reload(reslock.detect)` — runs on Linux/macOS CI without needing a Windows runner.
- Windows consumers (scriba on-prem installs — SCRIBA-308) require `reslock>=0.4.1`.

## CI/CD Notes

- `astral-sh/setup-uv`: pin to exact version (e.g. `v8.0.0`) — rolling major tags may lag behind releases
- PyPI trusted publishing environment (`pypi`) only works with `release` events, not `workflow_dispatch`
- Release tags are immutable snapshots — if you fix the workflow after tagging, force-update the tag and recreate the release
