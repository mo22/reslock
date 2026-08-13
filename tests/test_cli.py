from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from click.testing import CliRunner

from reslock.cli import main
from reslock.detect import gpu_vram_key
from reslock.models import SCHEMA_VERSION, Lease, State
from reslock.state import ensure_state_file, transact

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


def _write_foreign_schema(path: Path) -> None:
    path.write_text(
        '{"version": 2, "resources": {"gpu_GPU-abc_vram_mb": 24000}, '
        '"leases": [{"id": "abc123abcdef", "pid": 99999, '
        '"resources": {"gpu_GPU-abc_vram_mb": 8000}}], "queue": []}'
    )


def test_schema_mismatch_reports_both_versions_and_exits_3(tmp_path: Path) -> None:
    """A foreign state file is an operational condition, not a traceback.

    The message has to carry both versions — knowing only that something is
    wrong doesn't tell an operator which side is the laggard.
    """
    path = tmp_path / "state.json"
    _write_foreign_schema(path)

    result = CliRunner().invoke(main, ["status", "--state", str(path)])

    assert result.exit_code == 3
    assert "v2" in result.output
    assert f"v{SCHEMA_VERSION}" in result.output
    assert "reslock reset --force" in result.output
    assert "Traceback" not in result.output


def test_schema_mismatch_leaves_the_file_alone(tmp_path: Path) -> None:
    """Every command refuses; none of them rewrites the file."""
    path = tmp_path / "state.json"
    _write_foreign_schema(path)
    original = path.read_text()

    for argv in (
        ["status"],
        ["set", "vram_mb", "1000"],
        ["release", "abc123abcdef"],
        ["reset"],
    ):
        result = CliRunner().invoke(main, [*argv, "--state", str(path)])
        assert result.exit_code == 3, f"{argv} should have refused: {result.output}"
        assert path.read_text() == original, f"{argv} modified the state file"


def test_reset_force_recovers_from_schema_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    _write_foreign_schema(path)

    result = CliRunner().invoke(main, ["reset", "--force", "--state", str(path)])
    assert result.exit_code == 0

    # And the file is usable again afterwards.
    result = CliRunner().invoke(main, ["status", "--state", str(path)])
    assert result.exit_code == 0
    state = State.model_validate_json(path.read_text())
    assert state.version == SCHEMA_VERSION
    assert state.leases == []


def test_status_shows_how_long_a_reclaim_has_been_pending(tmp_path: Path) -> None:
    """The age is the point of the field — 'reclaim_requested' alone was the
    display up to v0.11.1, and it looked identical after ten seconds and after
    nine days (the 2026-08-01..10 incident).
    """
    path = tmp_path / "state.json"
    ensure_state_file(path)

    def _seed(st: State) -> None:
        st.resources["vram_mb"] = 24000
        st.leases.append(
            Lease(
                # A live PID: status() cleans dead-process leases before rendering.
                pid=os.getpid(),
                resources={"vram_mb": 8000},
                reclaimable=True,
                reclaim_requested=True,
                reclaim_requested_at=datetime.now(timezone.utc) - timedelta(days=9),
            )
        )

    transact(path, _seed)

    # Wide terminal: rich truncates the Flags column at the default 80 chars.
    result = CliRunner().invoke(main, ["status", "--state", str(path)], env={"COLUMNS": "200"})

    assert result.exit_code == 0
    assert "reclaim_requested (9d)" in result.output


def test_status_omits_the_age_when_the_timestamp_is_absent(tmp_path: Path) -> None:
    """A lease flipped before the upgrade has no timestamp; 'unknown' must not
    render as '0s', which would read as a healthy just-now reclaim.
    """
    path = tmp_path / "state.json"
    ensure_state_file(path)

    def _seed(st: State) -> None:
        st.leases.append(
            Lease(pid=os.getpid(), resources={"vram_mb": 8000}, reclaim_requested=True)
        )

    transact(path, _seed)

    result = CliRunner().invoke(main, ["status", "--state", str(path)], env={"COLUMNS": "200"})

    assert result.exit_code == 0
    assert "reclaim_requested" in result.output
    assert "reclaim_requested (" not in result.output
