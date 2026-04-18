"""Pytest hooks: ensure minimal TmrlData pickles exist before importing ``tmrl.config``."""

from __future__ import annotations

from pathlib import Path

from tests.tmrl_test_min_assets import write_min_tmrl_test_pickles


def pytest_configure(config) -> None:
    """Runs before test collection so modules that import ``tmrl.config`` see required assets.

    Creates ``~/TmrlData`` and minimal pickle files when the directory does not exist yet
    (e.g. fresh CI machines, clean dev environments).
    """
    del config  # unused
    base = Path.home() / "TmrlData"
    if not base.is_dir():
        base.mkdir(parents=True, exist_ok=True)
        (base / "config").mkdir(parents=True, exist_ok=True)
    write_min_tmrl_test_pickles(base)
