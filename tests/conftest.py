"""Pytest hooks: ensure minimal TmrlData pickles exist before importing ``tmrl.config``."""

from __future__ import annotations

from pathlib import Path

from tests.tmrl_test_min_assets import ensure_min_tmrl_test_pickles_if_missing


def pytest_configure(config) -> None:
    """Runs before test collection so modules that import ``tmrl.config`` see required assets."""
    del config  # unused
    base = Path.home() / "TmrlData"
    if not base.is_dir():
        return
    ensure_min_tmrl_test_pickles_if_missing(base)
