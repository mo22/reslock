# reslock

Resource lock manager for coordinating shared system resources (GPU VRAM, RAM, CPU) across processes on a single machine. No daemon — all coordination via a JSON state file with file locking.

## Project Structure

- `src/reslock/` — library source
  - `pool.py` — `ResourcePool` and `LeaseHandle` (main API)
  - `models.py` — Pydantic models (`State`, `Lease`, `QueueEntry`, `PoolStatus`)
  - `state.py` — file-locked state read/write
  - `detect.py` — system resource detection (GPU VRAM via nvidia-smi/torch, CPU, disk)
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
- Per-GPU VRAM tracking for multi-GPU systems (keyed by `gpu0_vram_mb`, `gpu1_vram_mb`, etc.)
- Priority queue determines which waiter gets resources next
- Reclaimable leases allow preemption by higher-priority work

## CI/CD Notes

- `astral-sh/setup-uv`: pin to exact version (e.g. `v8.0.0`) — rolling major tags may lag behind releases
- PyPI trusted publishing environment (`pypi`) only works with `release` events, not `workflow_dispatch`
- Release tags are immutable snapshots — if you fix the workflow after tagging, force-update the tag and recreate the release
