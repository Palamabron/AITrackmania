from __future__ import annotations

import importlib.metadata

import pytest

from trackmaniarl import _version as package_version


def test_source_version_falls_back_without_distribution_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_distribution(_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(package_version, "version", missing_distribution)

    assert package_version._resolve_version() == "0+unknown"
