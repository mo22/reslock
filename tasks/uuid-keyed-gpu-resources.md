# UUID-keyed GPU resources (partial-mapping correctness)

## Status
Open — filed 2026-04-22 from scriba_bugfixer. Blocks the matching tasks in `~/workspace/straiqr/scriba` (`mo_dev`) and `~/workspace/straiqr/aiserver/tasks/uuid-keyed-gpu-reslock.md`.

## Problem

`detect_gpu_vram_mb()` (both `src/reslock/detect.py` and `src/reslock/resources.py`) builds resource keys from the nvidia-smi `index` column:

```python
result = subprocess.run(
    ["nvidia-smi", "--query-gpu=index,memory.total", "--format=csv,noheader,nounits"],
    ...
)
# → {"gpu0_vram_mb": 24000, "gpu1_vram_mb": 24000}
```

Inside a container launched with partial GPU mapping (`docker run --gpus device=2`, `NVIDIA_VISIBLE_DEVICES=GPU-uuid`, or `CUDA_VISIBLE_DEVICES=2,3`), the NVIDIA container runtime **renumbers visible GPUs starting at 0**. So two containers on the same host, sharing a bind-mounted `/var/lib/reslock/state.json`, can each detect their single mapped card as `gpu0` and:

1. **Disjoint mapping (A has host GPU 0, B has host GPU 1):** both lock `gpu0_vram_mb` → unnecessary serialization on a shared key that refers to different physical GPUs.
2. **Overlapping mapping (A has {0,1,2}, B has {1,2,3}):** A's `gpu1` = host GPU 1, B's `gpu0` = host GPU 1 → **no mutual exclusion on the shared physical card → concurrent model loads can OOM.** This is the silent-correctness bug.
3. **Capacity races on `set_resources`:** each container's startup overwrites the shared `gpu0_vram_mb` total with the local card's capacity. If the two cards differ in size (e.g. 3090 vs. A100), whichever container initialized last wins — the other container now sees a wrong total.

The fix is to key resources by the host-stable **GPU UUID** (from `nvidia-smi --query-gpu=uuid`), which is identical inside every container regardless of index renumbering.

## Current consumers (what this breaks)

Both consumers hard-code the `gpu{int}_vram_mb` format:

- `scriba` (on `mo_dev`, `backend/scriba/simplemodel_tools.py`): `_reslock_acquire_single`, `_reslock_acquire_multi`, `_refresh_reslock_gpu_totals`, `_warn_vram_headroom`, `_inline_reclaim_idle_cached_models`. Also `simplemodel_llamacpp.py` auto-tensor-split: `{f"gpu{i}_vram_mb": ... for i in range(torch.cuda.device_count())}`.
- `aiserver` (`src/aiserver/gpu.py`): `GpuAllocation.reslock_resources`, `reslock_vram_resources()`. See sister task file.

Neither caller can unilaterally switch without breaking coordination with the other.

## Proposed reslock API

### 1. Change key format (breaking)

`detect_gpu_vram_mb()` returns UUID-keyed entries:

```python
# nvidia-smi --query-gpu=uuid,memory.total --format=csv,noheader,nounits
# GPU-1a2b3c4d-..., 24576
# GPU-5e6f7890-..., 24576
# →
{"gpu_GPU-1a2b3c4d-..._vram_mb": 24576, "gpu_GPU-5e6f7890-..._vram_mb": 24576}
```

Full UUID in the key (not a prefix) — 40-ish chars is fine, reslock keys are plain dict strings and the state file is human-readable. No ambiguity, no collision.

Update `detect_gpu_vram_mb_nvidia_smi()` and `detect_gpu_vram_mb_torch()` identically. For the torch path, use `torch.cuda.get_device_properties(i).uuid` (CUDA 11+, torch 2.0+ — the project already requires `torch>=2.6` per aiserver/pyproject.toml).

### 2. Add mapping helpers

Consumers hold a local torch device index (e.g. from `torch.cuda.mem_get_info(i)`) and need to build the corresponding resource key. Two shapes:

```python
def gpu_resource_key(torch_index: int) -> Optional[str]:
    """Return the reslock resource key ``gpu_{uuid}_vram_mb`` for a local torch
    device index, or None if UUID cannot be determined.

    Inside a container with partial GPU mapping, torch_index is the
    container-local index but the returned key uses the host-stable UUID, so
    leases coordinate correctly across containers sharing the reslock state."""

def gpu_uuid_for_torch_index(torch_index: int) -> Optional[str]:
    """Lower-level helper — returns just the UUID string."""
```

Implementation: prefer `torch.cuda.get_device_properties(idx).uuid`; fall back to `nvidia-smi --query-gpu=index,uuid` (the existing `_nvidia_smi_gpu_uuid_to_index` already does this, just invert the dict).

Cache the index→UUID map per-process. Invalidation is not a concern — GPUs don't hot-swap.

### 3. State-file migration

The JSON state file (`state.json`) contains both `resources` (capacity) and `leases[].resources`. On upgrade, old index-keyed entries (`gpu0_vram_mb`) become unrecognized — lease release is fine (they just age out), but stale capacity entries would confuse `can_fit`.

Simplest approach: **bump the state file schema version**. Add a top-level `"schema_version": 2` field. When reslock reads a file with missing/older `schema_version`, it drops the `resources` dict and lets the first `set_resources()` call repopulate. Leases from an older version are either released (best) or converted to UUID keys via `_nvidia_smi_gpu_uuid_to_index` on read (nice-to-have but brittle — the host UUID→index mapping is only stable if the old keys were written by a container with the same GPU topology as the current reader).

Document in the release notes: "on upgrade, stop all reslock consumers, delete `/var/lib/reslock/state.json`, then restart." That's the safe path.

### 4. CLI surface

`reslock status` already prints keys verbatim — no change needed, just the output gets longer. Consider a `--short` flag that truncates UUIDs to the last 8 chars for display.

## Non-goals

- **Heterogeneous-GPU labeling.** Don't try to smuggle the GPU model name (RTX 4090 vs A100) into the key. Callers that need per-model selection should read `nvidia-smi --query-gpu=uuid,name` separately. Keep reslock's key format mechanical.
- **Automatic state-file migration across schema versions.** Not worth the complexity — a restart-with-clean-state step in the release notes is fine.

## Out-of-band concern: pid: host

Independent of this task: reslock's dead-PID cleanup relies on `pid: host` in compose files (already documented in `aiserver/DESIGN.md`). The UUID switch doesn't change that — just noting for the reader so it doesn't get conflated.

## Release plan

1. Land reslock changes + version bump (minor version, since it's a breaking state-file schema change but a pure dep bump for consumers).
2. scriba and aiserver update their `reslock>=...` constraint and switch every `f"gpu{i}_vram_mb"` call site to `reslock.gpu_resource_key(i)`.
3. Coordinated deploy: bring down all scriba / aiserver / kirk-rpcserver containers on a given host, wipe `state.json`, redeploy with matching versions.

## References

- scriba analysis (why this bug exists): see `scriba_bugfixer` session 2026-04-22 (reslock partial-mapping audit).
- scriba-side consumers: `~/workspace/straiqr/scriba/backend/scriba/simplemodel_tools.py:501,514,241,549`, `simplemodel_llamacpp.py:1168,1264`.
- aiserver-side consumers: `~/workspace/straiqr/aiserver/src/aiserver/gpu.py:142,197,263,301`.
- reslock existing UUID helper (for piggyback): `src/reslock/detect.py:136` (`_nvidia_smi_gpu_uuid_to_index`).
