from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from reslock.cli import main
from reslock.models import State


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
