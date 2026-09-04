from __future__ import annotations

from pathlib import Path

import pytest
import torch


def pytest_configure() -> None:
    (Path(__file__).resolve().parents[1] / ".pytest-cache").mkdir(exist_ok=True)


@pytest.fixture(autouse=True)
def _isolate_test_hardware(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(
        "trackmaniarl.algorithms.execution.visible_accelerators",
        lambda: set(),
    )
