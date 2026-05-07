from __future__ import annotations

from pathlib import Path

import pytest

from reslock.models import SCHEMA_VERSION, State
from reslock.state import ensure_state_file, read_state, transact


def test_ensure_state_file(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "state.json"
    ensure_state_file(path)
    assert path.exists()
    state = State.model_validate_json(path.read_text())
    assert state.version == SCHEMA_VERSION


def test_transact(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    ensure_state_file(path)

    def _set(st: State) -> str:
        st.resources["vram_mb"] = 8000
        return "done"

    result = transact(path, _set)
    assert result == "done"

    state = State.model_validate_json(path.read_text())
    assert state.resources["vram_mb"] == 8000


def test_old_v1_schema_is_reset_on_read(tmp_path: Path) -> None:
    """A v1 state file should be migrated to an empty v3 state on read."""
    path = tmp_path / "state.json"
    path.write_text(
        '{"version": 1, "resources": {"gpu0_vram_mb": 24000}, "leases": [], "queue": []}'
    )
    state = read_state(path)
    assert state.version == SCHEMA_VERSION
    assert state.resources == {}
    assert state.leases == []
    assert state.queue == []


def test_empty_state_file_raises_instead_of_silent_reset(tmp_path: Path) -> None:
    """A truncated/empty state file (e.g. crash mid ``transact()`` rewrite)
    must fail-closed via pydantic ValidationError, not silently reset to a
    fresh empty State that would let new acquires proceed while in-flight
    leases are forgotten. Regression for codex follow-up review P2.
    """
    import pydantic

    path = tmp_path / "state.json"
    path.write_text("")
    with pytest.raises(pydantic.ValidationError):
        read_state(path)

    path.write_text("   \n  \n")  # whitespace-only counts the same
    with pytest.raises(pydantic.ValidationError):
        read_state(path)


def test_v07x_state_with_extra_lease_fields_resets_on_read(tmp_path: Path) -> None:
    """A v0.7.x state file written before the v3 split carries
    ``estimated_seconds`` / ``progress`` directly on Lease. v3 marks Lease as
    ``extra="forbid"``, so without the version-peek-first migration this would
    crash on read instead of resetting. Regression for codex review P1.
    """
    path = tmp_path / "state.json"
    path.write_text(
        '{"version": 2, "resources": {"gpu_GPU-abc_vram_mb": 24000}, '
        '"leases": [{"id": "abc123abcdef", "pid": 99999, '
        '"resources": {"gpu_GPU-abc_vram_mb": 8000}, '
        '"estimated_seconds": 30, "progress": 0.5}], '
        '"queue": []}'
    )
    state = read_state(path)
    assert state.version == SCHEMA_VERSION
    assert state.leases == []


def test_old_v2_schema_is_reset_on_read(tmp_path: Path) -> None:
    """A v2 state file (UUID-keyed but old Lease/QueueEntry shape) is reset on read.

    v3 split work-tracking fields off Lease into QueueEntry and added the
    abstract GPU request shape; coordinated upgrade across consumers is
    required, so the state file resets and consumers re-register.
    """
    path = tmp_path / "state.json"
    path.write_text(
        '{"version": 2, "resources": {"gpu_GPU-abc_vram_mb": 24000}, "leases": [], "queue": []}'
    )
    state = read_state(path)
    assert state.version == SCHEMA_VERSION
    assert state.resources == {}
    assert state.leases == []
    assert state.queue == []


def test_old_schema_reset_persisted_on_transact(tmp_path: Path) -> None:
    """After transact() on a v1 file, the file is rewritten at the current schema."""
    path = tmp_path / "state.json"
    path.write_text(
        '{"version": 1, "resources": {"gpu0_vram_mb": 24000}, "leases": [], "queue": []}'
    )

    def _noop(_: State) -> None:
        return None

    transact(path, _noop)

    state = State.model_validate_json(path.read_text())
    assert state.version == SCHEMA_VERSION
    assert state.resources == {}
