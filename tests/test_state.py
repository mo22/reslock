from __future__ import annotations

from pathlib import Path

from reslock.models import State
from reslock.state import ensure_state_file, transact


def test_ensure_state_file(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "state.json"
    ensure_state_file(path)
    assert path.exists()
    state = State.model_validate_json(path.read_text())
    assert state.version == 1


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
