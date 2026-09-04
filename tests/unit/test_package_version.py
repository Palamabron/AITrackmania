from __future__ import annotations

import importlib.metadata

import pytest

import trackmaniarl


def test_source_version_falls_back_without_distribution_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_distribution(_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(trackmaniarl, "version", missing_distribution)

    assert trackmaniarl._resolve_version() == "0+unknown"
