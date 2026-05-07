"""Scheduler-level tests for the v3 abstract GPU request shape.

These exercise ``State.try_resolve_request`` (placement policy) and the
ResourcePool wiring that uses it. No NVML — pure internal accounting; the
NVML cross-check has its own coverage in ``test_pool_nvml.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from reslock import ResourcePool
from reslock.detect import gpu_vram_key
from reslock.models import State

UUID_A = "GPU-aaaaaaaa-1111-2222-3333-444444444444"
UUID_B = "GPU-bbbbbbbb-1111-2222-3333-444444444444"
UUID_C = "GPU-cccccccc-1111-2222-3333-444444444444"


def _make_pool(tmp_path: Path, **resources: int) -> ResourcePool:
    pool = ResourcePool(tmp_path / "state.json")
    pool.set_resources(resources)
    return pool


def test_resolve_picks_emptiest_gpus_first(tmp_path: Path) -> None:
    """Spread placement: 3 GPUs at 24 / 16 / 8 free, request 10 MB on 2 GPUs.

    The two emptiest GPUs (24 and 16) are picked, not the 8-GB one.
    """
    pool = _make_pool(
        tmp_path,
        **{
            gpu_vram_key(UUID_A): 24,
            gpu_vram_key(UUID_B): 16,
            gpu_vram_key(UUID_C): 8,
        },
    )
    h = pool.try_acquire(vram_mb_each=10, num_gpus=2)
    assert h is not None
    picked = sorted(h.gpu_uuids)
    assert picked == sorted([UUID_A, UUID_B])
    h.release()


def test_resolve_returns_none_when_per_gpu_too_small(tmp_path: Path) -> None:
    """4 GPUs at 12 free each, ask 24 each — no single GPU can satisfy."""
    pool = _make_pool(
        tmp_path,
        **{
            gpu_vram_key(UUID_A): 12,
            gpu_vram_key(UUID_B): 12,
            gpu_vram_key(UUID_C): 12,
        },
    )
    h = pool.try_acquire(vram_mb_each=24, num_gpus=1)
    assert h is None


def test_resolve_breaks_ties_by_uuid(tmp_path: Path) -> None:
    """Two GPUs equally free → deterministic pick by UUID ascending."""
    pool = _make_pool(
        tmp_path,
        **{
            gpu_vram_key(UUID_A): 16,
            gpu_vram_key(UUID_B): 16,
        },
    )
    h = pool.try_acquire(vram_mb_each=10, num_gpus=1)
    assert h is not None
    # UUID_A < UUID_B lexicographically; A is picked first.
    assert h.gpu_uuids == [UUID_A]
    h.release()


def test_resolve_request_with_zero_gpus_returns_non_gpu_only(tmp_path: Path) -> None:
    """num_gpus=0 paths through ``try_resolve_request`` return only non-GPU bindings."""
    pool = _make_pool(tmp_path, ram_mb=4000)
    h = pool.try_acquire(num_gpus=0, ram_mb=1000)
    assert h is not None
    assert h.gpu_uuids == []
    h.release()


def test_acquire_rejects_legacy_gpu_uuid_keys(tmp_path: Path) -> None:
    """v3 hard-cut: gpu_<uuid>_vram_mb keys are not accepted by acquire/try_acquire."""
    pool = _make_pool(tmp_path, **{gpu_vram_key(UUID_A): 24000})
    with pytest.raises(TypeError, match="GPU keys are not accepted"):
        pool.try_acquire(**{gpu_vram_key(UUID_A): 8000})  # pyright: ignore[reportArgumentType]


def test_resolve_validates_request_shape() -> None:
    """num_gpus > 0 without a positive vram_mb_each raises immediately."""
    state = State(resources={gpu_vram_key(UUID_A): 24000})
    with pytest.raises(ValueError, match="vram_mb_each"):
        state.try_resolve_request(vram_mb_each=None, num_gpus=1, non_gpu={})
    with pytest.raises(ValueError, match="vram_mb_each"):
        state.try_resolve_request(vram_mb_each=0, num_gpus=1, non_gpu={})


def test_resolve_rejects_vram_each_without_num_gpus(tmp_path: Path) -> None:
    """vram_mb_each set but num_gpus=0 raises (regression for codex review P2).

    Without this guard, ``pool.acquire(vram_mb_each=8000)`` (a forgotten
    ``num_gpus=``) would silently grant a non-GPU lease, dropping the VRAM
    ask on the floor.
    """
    state = State(resources={gpu_vram_key(UUID_A): 24000})
    with pytest.raises(ValueError, match="num_gpus > 0"):
        state.try_resolve_request(vram_mb_each=8000, num_gpus=0, non_gpu={})

    pool = _make_pool(tmp_path, **{gpu_vram_key(UUID_A): 24000})
    with pytest.raises(ValueError, match="num_gpus > 0"):
        pool.try_acquire(vram_mb_each=8000)  # forgot num_gpus=


def test_resolve_picks_alternative_gpu_when_picked_one_is_nvml_short(
    tmp_path: Path,
) -> None:
    """Regression for codex review P1: when the internally-emptiest GPU is
    NVML-short but another registered GPU has enough driver-reported free VRAM,
    spread placement should skip the short one and pick the alternative.
    """
    pool = _make_pool(
        tmp_path,
        **{
            gpu_vram_key(UUID_A): 24000,
            gpu_vram_key(UUID_B): 20000,
        },
    )
    state = State(
        resources={
            gpu_vram_key(UUID_A): 24000,
            gpu_vram_key(UUID_B): 20000,
        }
    )
    # NVML reports A as held by an external process (4 GB free) but B as fully free.
    nvml_free = {UUID_A: 4000, UUID_B: 20000}
    resolved = state.try_resolve_request(
        vram_mb_each=16000, num_gpus=1, non_gpu={}, nvml_free=nvml_free
    )
    assert resolved is not None
    # Spread placement would have picked A (most internally free) without NVML;
    # with NVML folded in, effective free is min(24000, 4000)=4000 for A vs
    # min(20000, 20000)=20000 for B → B is picked.
    assert resolved == {gpu_vram_key(UUID_B): 16000}
    # Sanity: pool-level try_acquire works the same way under the conftest
    # fake (which reports plenty of NVML headroom for every UUID).
    h = pool.try_acquire(vram_mb_each=16000, num_gpus=1)
    assert h is not None
    h.release()


def test_resolve_request_picks_correct_count(tmp_path: Path) -> None:
    """Asking for fewer GPUs than available picks exactly num_gpus."""
    pool = _make_pool(
        tmp_path,
        **{
            gpu_vram_key(UUID_A): 24,
            gpu_vram_key(UUID_B): 24,
            gpu_vram_key(UUID_C): 24,
        },
    )
    h = pool.try_acquire(vram_mb_each=8, num_gpus=2)
    assert h is not None
    assert len(h.gpu_uuids) == 2
    h.release()
