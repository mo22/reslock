"""Regression tests for Windows compatibility.

The POSIX ``resource`` stdlib module doesn't exist on Windows. These tests
simulate that by removing it from ``sys.modules`` and reloading
``reslock.detect``, then verify that the affected APIs degrade gracefully
(return ``None`` / omit the missing keys) rather than raising ``ImportError``.
"""

from __future__ import annotations

import builtins
import importlib
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


@pytest.fixture
def detect_without_resource(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    """Reload ``reslock.detect`` with the ``resource`` module hidden."""
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "resource":
            raise ImportError("No module named 'resource'")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "resource", raising=False)
    monkeypatch.delitem(sys.modules, "reslock.detect", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    import reslock.detect as detect

    importlib.reload(detect)
    try:
        yield detect
    finally:
        monkeypatch.setattr(builtins, "__import__", real_import)
        importlib.reload(detect)


def test_detect_imports_without_resource_module(detect_without_resource: ModuleType) -> None:
    assert detect_without_resource.resource is None


def test_get_self_rss_mb_returns_none_without_resource(
    detect_without_resource: ModuleType,
) -> None:
    assert detect_without_resource.get_self_rss_mb() is None


def test_get_self_cpu_seconds_returns_none_without_resource(
    detect_without_resource: ModuleType,
) -> None:
    assert detect_without_resource.get_self_cpu_seconds() is None


def test_get_self_actual_resources_omits_ram_without_resource(
    detect_without_resource: ModuleType,
) -> None:
    result: dict[str, int] = detect_without_resource.get_self_actual_resources()
    assert isinstance(result, dict)
    assert "ram_mb" not in result


def test_resource_pool_importable_without_resource_module(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """End-to-end: reslock package should import and ResourcePool() should
    instantiate even when the POSIX ``resource`` module is unavailable."""
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "resource":
            raise ImportError("No module named 'resource'")
        return real_import(name, *args, **kwargs)

    for mod in ("resource", "reslock", "reslock.detect", "reslock.pool", "reslock.resources"):
        monkeypatch.delitem(sys.modules, mod, raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    import reslock

    importlib.reload(reslock)
    pool = reslock.ResourcePool(tmp_path / "state.json")
    assert pool is not None
