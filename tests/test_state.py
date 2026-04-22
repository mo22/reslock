from __future__ import annotations

from pathlib import Path

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


def test_old_schema_is_reset_on_read(tmp_path: Path) -> None:
    """A v1 state file should be migrated to an empty v2 state on read."""
    path = tmp_path / "state.json"
    path.write_text(
        '{"version": 1, "resources": {"gpu0_vram_mb": 24000}, "leases": [], "queue": []}'
    )
    state = read_state(path)
    assert state.version == SCHEMA_VERSION
    assert state.resources == {}
    assert state.leases == []
    assert state.queue == []


def test_old_schema_reset_persisted_on_transact(tmp_path: Path) -> None:
    """After transact() on a v1 file, the file is rewritten at schema v2."""
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
