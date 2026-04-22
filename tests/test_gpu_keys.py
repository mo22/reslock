"""Tests for the UUID-based GPU resource key helpers."""

from __future__ import annotations

from reslock.detect import gpu_vram_key, parse_gpu_vram_key


def test_gpu_vram_key_round_trip() -> None:
    uuid = "GPU-1a2b3c4d-5e6f-7890-abcd-ef1234567890"
    key = gpu_vram_key(uuid)
    assert key == f"gpu_{uuid}_vram_mb"
    assert parse_gpu_vram_key(key) == uuid


def test_parse_gpu_vram_key_rejects_other_keys() -> None:
    assert parse_gpu_vram_key("ram_mb") is None
    assert parse_gpu_vram_key("cpu_cores") is None
    assert parse_gpu_vram_key("gpu0_vram_mb") is None  # old index-based format
