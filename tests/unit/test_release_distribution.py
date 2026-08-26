from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path

import pytest
import yaml

from scripts import check_distribution

REPOSITORY = Path(__file__).resolve().parents[2]


def test_release_workflow_gates_publish_on_full_quality_suite() -> None:
    workflow = yaml.safe_load(
        (REPOSITORY / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    )
    gate = workflow["jobs"]["gate"]
    publish = workflow["jobs"]["publish"]
    gate_commands = "\n".join(str(step.get("run", "")) for step in gate["steps"])
    publish_commands = "\n".join(str(step.get("run", "")) for step in publish["steps"])

    assert publish["needs"] == "gate"
    assert "uv run ruff format --check ." in gate_commands
    assert "uv run ruff check ." in gate_commands
    assert "uv run mypy --strict trackmaniarl" in gate_commands
    assert "uv run pytest" in gate_commands
    assert "uv build" in publish_commands
    assert 'scripts/check_distribution.py --tag "$GITHUB_REF_NAME"' in publish_commands
    assert "--with ./dist/*.whl trackmaniarl --help" in publish_commands
    assert "trackmaniarl init" in publish_commands
    assert "uv lock" in publish_commands
    assert "uv publish" in publish_commands


def test_wheel_validation_rejects_post_build_source_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    source = Path("trackmaniarl") / "module.py"
    source.parent.mkdir()
    source.write_bytes(b"value = 'built'\n")
    archive = tmp_path / "package.whl"
    with zipfile.ZipFile(archive, "w") as package:
        package.write(source, "trackmaniarl/module.py")
    with zipfile.ZipFile(archive) as package:
        check_distribution._validate_wheel_checkout_files(package, archive)

    source.write_bytes(b"value = 'mutated'\n")

    with (
        zipfile.ZipFile(archive) as package,
        pytest.raises(RuntimeError, match="differs from the checkout"),
    ):
        check_distribution._validate_wheel_checkout_files(package, archive)


def test_sdist_validation_rejects_post_build_source_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    source = Path("scripts") / "tool.py"
    source.parent.mkdir()
    source.write_bytes(b"value = 'built'\n")
    root = "trackmaniarl-2.0.0"
    archive = tmp_path / "package.tar.gz"
    with tarfile.open(archive, "w:gz") as package:
        package.add(source, f"{root}/scripts/tool.py")
    with tarfile.open(archive) as package:
        check_distribution._validate_sdist_checkout_files(package, archive, root)

    source.write_bytes(b"value = 'mutated'\n")

    with (
        tarfile.open(archive) as package,
        pytest.raises(RuntimeError, match="differs from the checkout"),
    ):
        check_distribution._validate_sdist_checkout_files(package, archive, root)
