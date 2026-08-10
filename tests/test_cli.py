from __future__ import annotations

import sys
from pathlib import Path

from click.testing import CliRunner

from reslock.cli import main
from reslock.detect import gpu_vram_key
from reslock.models import State

UUID_A = "GPU-aaaaaaaa-1111-2222-3333-444444444444"
UUID_B = "GPU-bbbbbbbb-1111-2222-3333-444444444444"


def test_init(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    runner = CliRunner()
    result = runner.invoke(main, ["init", "--state", str(path)])
    assert result.exit_code == 0
    assert path.exists()


def test_set_and_status(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    runner = CliRunner()
    runner.invoke(main, ["init", "--state", str(path)])
    result = runner.invoke(main, ["set", "vram_mb", "24000", "--state", str(path)])
    assert result.exit_code == 0
    result = runner.invoke(main, ["status", "--state", str(path)])
    assert result.exit_code == 0
    assert "24000" in result.output


def test_reset(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    runner = CliRunner()
    runner.invoke(main, ["init", "--state", str(path)])
    runner.invoke(main, ["set", "vram_mb", "8000", "--state", str(path)])
    result = runner.invoke(main, ["reset", "--state", str(path)])
    assert result.exit_code == 0
    state = State.model_validate_json(path.read_text())
    assert len(state.leases) == 0


def test_release_drops_attached_entries(tmp_path: Path) -> None:
    """Regression for codex follow-up P2: CLI release must drop active
    QueueEntries attached to the released lease(s), or peers see stale work
    pointing at a missing lease.
    """
    from reslock import ResourcePool

    path = tmp_path / "state.json"
    pool = ResourcePool(path)
    pool.set_resources({"ram_mb": 4000})
    lease = pool.try_acquire(ram_mb=1000, estimated_seconds=30, label="trackme")
    assert lease is not None
    assert pool.status().queue, "auto-tracked entry should exist before release"

    runner = CliRunner()
    result = runner.invoke(main, ["release", lease.id, "--state", str(path)])
    assert result.exit_code == 0

    state = State.model_validate_json(path.read_text())
    assert state.leases == []
    assert state.queue == [], "attached entry must be dropped together with the lease"


def test_release_by_label_drops_attached_entries(tmp_path: Path) -> None:
    from reslock import ResourcePool

    path = tmp_path / "state.json"
    pool = ResourcePool(path)
    pool.set_resources({"ram_mb": 4000})
    lease = pool.try_acquire(ram_mb=1000, estimated_seconds=30, label="trackme")
    assert lease is not None

    runner = CliRunner()
    result = runner.invoke(main, ["release", "--label", "trackme", "--state", str(path)])
    assert result.exit_code == 0

    state = State.model_validate_json(path.read_text())
    assert state.leases == []
    assert state.queue == []


def test_status_shows_per_slot_gpu_request(tmp_path: Path) -> None:
    from reslock import ResourcePool

    path = tmp_path / "state.json"
    pool = ResourcePool(path)
    pool.set_resources({gpu_vram_key(UUID_A): 22, gpu_vram_key(UUID_B): 19})
    lease = pool.try_acquire(vram_mb=[22, 19], estimated_seconds=30)
    assert lease is not None

    result = CliRunner().invoke(main, ["status", "--state", str(path)])

    assert result.exit_code == 0
    assert "22+19MB GPU" in result.output
    lease.release()


def test_run_accepts_repeated_vram_slots(tmp_path: Path) -> None:
    from reslock import ResourcePool

    path = tmp_path / "state.json"
    pool = ResourcePool(path)
    pool.set_resources({gpu_vram_key(UUID_A): 22, gpu_vram_key(UUID_B): 19})

    result = CliRunner().invoke(
        main,
        [
            "run",
            "--vram",
            "22",
            "--vram",
            "19",
            "--state",
            str(path),
            "--",
            sys.executable,
            "-c",
            "pass",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "22+19MB GPU" in result.output
    state = State.model_validate_json(path.read_text())
    assert state.leases == []


def test_run_rejects_mixed_gpu_request_shapes(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        main,
        [
            "run",
            "--vram",
            "22",
            "--vram-mb-each",
            "19",
            "--num-gpus",
            "1",
            "--state",
            str(tmp_path / "state.json"),
            "--",
            sys.executable,
            "-c",
            "pass",
        ],
    )

    assert result.exit_code == 2
    assert "mutually exclusive" in result.output
