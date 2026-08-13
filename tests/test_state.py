from __future__ import annotations

from pathlib import Path

import pytest

from reslock.models import SCHEMA_VERSION, State
from reslock.state import (
    SchemaVersionMismatch,
    ensure_state_file,
    force_reset_state,
    peek_state_version,
    read_state,
    transact,
)


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


def test_old_v1_schema_raises_on_read(tmp_path: Path) -> None:
    """A foreign schema is refused, not reset (v0.12.0 behaviour change)."""
    path = tmp_path / "state.json"
    path.write_text(
        '{"version": 1, "resources": {"gpu0_vram_mb": 24000}, "leases": [], "queue": []}'
    )
    with pytest.raises(SchemaVersionMismatch) as excinfo:
        read_state(path)
    assert excinfo.value.found == 1
    assert excinfo.value.expected == SCHEMA_VERSION
    assert excinfo.value.path == path


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


def test_foreign_schema_is_reported_as_mismatch_not_as_field_error(tmp_path: Path) -> None:
    """The version peek must happen before strict field validation.

    A v0.7.x file carries ``estimated_seconds`` / ``progress`` on Lease, which
    the current ``extra="forbid"`` Lease rejects. Without the peek the operator
    would get a wall of pydantic field errors instead of the one fact that
    explains all of them. Regression for codex review P1, kept across the
    v0.12.0 reset→raise change.
    """
    path = tmp_path / "state.json"
    path.write_text(
        '{"version": 2, "resources": {"gpu_GPU-abc_vram_mb": 24000}, '
        '"leases": [{"id": "abc123abcdef", "pid": 99999, '
        '"resources": {"gpu_GPU-abc_vram_mb": 8000}, '
        '"estimated_seconds": 30, "progress": 0.5}], '
        '"queue": []}'
    )
    with pytest.raises(SchemaVersionMismatch):
        read_state(path)


def test_v4_state_raises_on_read(tmp_path: Path) -> None:
    """The immediately-previous schema gets no special treatment."""
    path = tmp_path / "state.json"
    path.write_text(
        '{"version": 4, "resources": {"gpu_GPU-abc_vram_mb": 24000}, '
        '"leases": [], "queue": [{"pid": 123, "vram_mb_each": 8000, "num_gpus": 1}]}'
    )
    with pytest.raises(SchemaVersionMismatch):
        read_state(path)


def test_newer_schema_also_raises(tmp_path: Path) -> None:
    """Mismatch is refused in both directions.

    A file from a *newer* reslock is just as unreadable, and writing our older
    shape over it would corrupt the newer consumers' bookkeeping — the exact
    damage this change exists to prevent, only with the roles swapped.
    """
    path = tmp_path / "state.json"
    path.write_text(
        f'{{"version": {SCHEMA_VERSION + 1}, "resources": {{}}, "leases": [], "queue": []}}'
    )
    with pytest.raises(SchemaVersionMismatch) as excinfo:
        read_state(path)
    assert excinfo.value.found == SCHEMA_VERSION + 1


def test_transact_on_foreign_schema_leaves_file_untouched(tmp_path: Path) -> None:
    """The core regression: a stale consumer must not erase the lease table.

    Up to v0.11.1 this rewrote the file with an empty state — one acquire from
    an installation with an old schema wiped every other consumer's leases,
    with a WARNING as the only trace. The file must now come out byte-identical.
    """
    path = tmp_path / "state.json"
    original = (
        '{"version": 1, "resources": {"gpu0_vram_mb": 24000}, '
        '"leases": [{"id": "abc123abcdef", "pid": 99999, '
        '"resources": {"gpu0_vram_mb": 8000}}], "queue": []}'
    )
    path.write_text(original)

    called = False

    def _noop(_: State) -> None:
        nonlocal called
        called = True

    with pytest.raises(SchemaVersionMismatch):
        transact(path, _noop)

    assert path.read_text() == original
    assert not called, "fn must not run against a state we refused to parse"


def test_force_reset_state_recovers_a_foreign_file(tmp_path: Path) -> None:
    """``force_reset_state`` is the documented way out of a mismatch."""
    path = tmp_path / "state.json"
    path.write_text(
        '{"version": 1, "resources": {"gpu0_vram_mb": 24000}, '
        '"leases": [{"id": "abc123abcdef", "pid": 99999, '
        '"resources": {"gpu0_vram_mb": 8000}}], "queue": []}'
    )
    with pytest.raises(SchemaVersionMismatch):
        read_state(path)

    force_reset_state(path)

    state = read_state(path)
    assert state.version == SCHEMA_VERSION
    assert state.resources == {}
    assert state.leases == []
    assert state.queue == []


def test_force_reset_state_creates_a_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "state.json"
    force_reset_state(path)
    assert read_state(path).version == SCHEMA_VERSION


def test_force_reset_state_truncates_a_longer_file(tmp_path: Path) -> None:
    """Rewrite-in-place must not leave a tail of the old, longer content."""
    path = tmp_path / "state.json"
    path.write_text(
        '{"version": 1, "resources": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, '
        '"f": 6, "g": 7, "h": 8, "i": 9, "j": 10, "k": 11, "l": 12}, '
        '"leases": [], "queue": []}'
    )
    force_reset_state(path)
    # Parses at all → no trailing garbage after the JSON object.
    assert read_state(path).resources == {}


def test_peek_state_version_reads_a_file_we_refuse_to_parse(tmp_path: Path) -> None:
    """Reporting *about* a foreign file stays possible; only using it doesn't."""
    path = tmp_path / "state.json"
    path.write_text('{"version": 2, "resources": {}, "leases": [], "queue": []}')
    assert peek_state_version(path) == 2

    ensure_state_file(tmp_path / "fresh.json")
    assert peek_state_version(tmp_path / "fresh.json") == SCHEMA_VERSION


@pytest.mark.parametrize(
    "content",
    ["", "not json at all", "[1, 2, 3]", '{"resources": {}}', '{"version": "four"}'],
)
def test_peek_state_version_returns_none_when_it_cannot_tell(tmp_path: Path, content: str) -> None:
    path = tmp_path / "state.json"
    path.write_text(content)
    assert peek_state_version(path) is None


def test_peek_state_version_returns_none_for_missing_file(tmp_path: Path) -> None:
    assert peek_state_version(tmp_path / "nope.json") is None
