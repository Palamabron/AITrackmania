from __future__ import annotations

import json
import re
import tarfile
import zipfile
from pathlib import Path
from typing import cast

import pytest
import yaml

from scripts import check_distribution

REPOSITORY = Path(__file__).resolve().parents[2]
PINNED_ACTION = re.compile(r"[^@]+@[0-9a-f]{40}")
CPU_TORCH_ENV = {
    "TORCH_CPU_INDEX": "https://download.pytorch.org/whl/cpu",
    "TORCH_CPU_VERSION": "2.11.0+cpu",
    "VGAMEPAD_SKIP_VIGEMBUS_INSTALL": "true",
}


def _workflow() -> dict[str, object]:
    return yaml.safe_load(
        (REPOSITORY / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    )


def _commands(job: dict[str, object]) -> str:
    steps = job["steps"]
    assert isinstance(steps, list)
    return "\n".join(str(step.get("run", "")) for step in steps)


def _jobs() -> dict[str, dict[str, object]]:
    return cast(dict[str, dict[str, object]], _workflow()["jobs"])


def _step_named(job: dict[str, object], name: str) -> dict[str, object]:
    steps = cast(list[dict[str, object]], job["steps"])
    return next(step for step in steps if step.get("name") == name)


def test_python_files_respect_the_modularity_budget() -> None:
    roots = (
        REPOSITORY / "trackmaniarl",
        REPOSITORY / "tests",
        REPOSITORY / "scripts",
        REPOSITORY / "docs" / "diagrams",
    )
    oversized = {
        path.relative_to(REPOSITORY): line_count
        for root in roots
        for path in root.rglob("*.py")
        if (line_count := len(path.read_text(encoding="utf-8").splitlines())) >= 400
    }
    assert oversized == {}


def test_release_workflow_uses_canonical_job_order_and_matrices() -> None:
    workflow = _workflow()
    jobs = _jobs()
    source_gate = jobs["source-gate"]
    build = jobs["build"]
    verify = jobs["verify-dist"]
    publish = jobs["attest-publish"]
    assert workflow["permissions"] == {}
    assert workflow["env"] == {"UV_VERSION": "0.12.5"}
    expected_matrix = {"os": ["ubuntu-latest", "windows-latest"]}
    assert source_gate["strategy"]["matrix"] == expected_matrix
    assert verify["strategy"]["matrix"] == expected_matrix
    assert build["needs"] == "source-gate"
    assert verify["needs"] == "build"
    assert publish["needs"] == "verify-dist"


def _assert_release_source_gate_runs_the_full_quality_suite() -> None:
    workflow = _workflow()
    source_gate = _jobs()["source-gate"]
    gate_commands = _commands(source_gate)
    assert "uv lock --check" in gate_commands
    assert "uv run --no-sync ruff format --check ." in gate_commands
    assert "uv run --no-sync ruff check ." in gate_commands
    assert "uv run --no-sync mypy --strict trackmaniarl" in gate_commands
    assert "uv run --no-sync pytest" in gate_commands
    assert "UV_INDEX" not in workflow["env"]


def _assert_windows_job_installs_cpu_torch(job_name: str) -> None:
    job = _jobs()[job_name]
    cpu_step = next(
        step
        for step in cast(list[dict[str, object]], job["steps"])
        if "CPU Torch on Windows" in str(step.get("name", ""))
    )
    assert cpu_step["if"] == "runner.os == 'Windows'"
    assert cpu_step["env"] == CPU_TORCH_ENV
    command = str(cpu_step["run"])
    assert "uv sync --locked --group dev --no-install-package torch" in command
    assert "uv run --no-sync" in command
    assert '--with "torch==$env:TORCH_CPU_VERSION"' in command
    assert '--index "$env:TORCH_CPU_INDEX"' in command
    assert "torch.version.cuda is None" in command


def _assert_release_workflow_never_uses_uv_pip() -> None:
    commands = "\n".join(_commands(job) for job in _jobs().values())
    assert "uv pip" not in commands


def _assert_windows_wheel_validation_requires_cpu_torch() -> None:
    workflow = _workflow()
    verify = _jobs()["verify-dist"]
    step = _step_named(verify, "Verify wheel CLI and generated project on Windows")
    command = str(step["run"])
    assert step["if"] == "runner.os == 'Windows'"
    assert cast(dict[str, str], step["env"])["UV_INDEX"].endswith("/whl/cpu")
    assert str(workflow).count("'UV_INDEX':") == 1
    assert "generated Windows project resolved a CUDA Torch wheel" in command
    assert "generated Windows project did not resolve CPU Torch" in command


def test_release_workflow_uses_uv_quality_gate() -> None:
    _assert_release_source_gate_runs_the_full_quality_suite()
    _assert_release_workflow_never_uses_uv_pip()
    _assert_release_workflow_gates_publish_on_full_quality_suite()


def test_release_windows_jobs_use_cpu_torch() -> None:
    for job_name in ("source-gate", "verify-dist"):
        _assert_windows_job_installs_cpu_torch(job_name)
    _assert_windows_wheel_validation_requires_cpu_torch()


def test_release_builds_once_and_verifies_canonical_bytes() -> None:
    jobs = _jobs()
    build_commands = _commands(jobs["build"])
    verify_commands = _commands(jobs["verify-dist"])
    all_commands = "\n".join(_commands(job) for job in jobs.values())
    assert all_commands.count("uv build") == 1
    assert "--write-checksums" in build_commands
    assert "--verify-checksums" in verify_commands
    assert "trackmaniarl --help" in verify_commands
    assert "import trackmaniarl" in verify_commands
    assert "trackmaniarl init" in verify_commands
    assert "uv lock" in verify_commands


def _assert_distribution_contains_current_sources() -> None:
    scaffold_modules = {
        "trackmaniarl/project/scaffold.py",
        "trackmaniarl/project/scaffold_run_templates.py",
        "trackmaniarl/project/scaffold_templates.py",
    }
    wheel_members = check_distribution._wheel_required_members("2.0.0")
    sdist_members = check_distribution._sdist_required_members("trackmaniarl-2.0.0")
    assert scaffold_modules <= wheel_members
    assert {f"trackmaniarl-2.0.0/{name}" for name in scaffold_modules} <= sdist_members
    sources = (
        REPOSITORY / "trackmaniarl" / "__init__.py",
        REPOSITORY / "trackmaniarl" / "project" / "scaffold_templates.py",
    )
    assert all("PackageNotFoundError" not in path.read_text(encoding="utf-8") for path in sources)


def test_distribution_contains_current_packaging_sources() -> None:
    _assert_distribution_contains_current_sources()


def _assert_release_workflow_uploads_archives_and_spdx_sbom() -> None:
    jobs = _jobs()
    build = jobs["build"]
    build_steps = cast(list[dict[str, object]], build["steps"])
    artifact = next(
        step for step in build_steps if "actions/upload-artifact@" in str(step.get("uses", ""))
    )
    sbom = next(step for step in build_steps if "anchore/sbom-action@" in str(step.get("uses", "")))
    assert artifact["with"]["name"] == "release-dist"
    assert artifact["with"]["path"] == "dist/"
    assert sbom["with"]["path"] == "sbom-root"
    assert sbom["with"]["format"] == "spdx-json"
    assert sbom["with"]["output-file"] == "dist/trackmaniarl-release.spdx.json"


def _assert_release_workflow_attests_archives_and_sbom() -> None:
    publish = _jobs()["attest-publish"]
    publish_steps = cast(list[dict[str, object]], publish["steps"])
    attestations = [
        step for step in publish_steps if str(step.get("uses", "")).startswith("actions/attest@")
    ]
    pep740 = next(
        step
        for step in publish_steps
        if str(step.get("uses", "")).startswith("astral-sh/attest-action@")
    )
    assert len(attestations) == 2
    assert any("sbom-path" in step["with"] for step in attestations)
    assert "dist/*.whl" in pep740["with"]["paths"]


def _assert_release_workflow_publishes_without_rebuilding() -> None:
    publish = _jobs()["attest-publish"]
    commands = _commands(publish)
    assert "uv build" not in commands
    assert "uv publish dist/*.whl dist/*.tar.gz" in commands
    assert publish["environment"] == "pypi"
    assert publish["permissions"] == {
        "contents": "read",
        "id-token": "write",
        "attestations": "write",
        "artifact-metadata": "write",
    }
    assert "gh attestation verify" in commands
    assert "--predicate-type https://spdx.dev/Document/v2.3" in commands
    assert "pypi-attestations==0.0.30" in commands
    assert "pypi-attestations verify attestation" in commands


def _assert_release_workflow_pins_every_action_and_disables_checkout_credentials() -> None:
    workflow = _workflow()
    jobs = workflow["jobs"]
    for job in jobs.values():
        for step in job["steps"]:
            action = step.get("uses")
            if action is None:
                continue
            assert PINNED_ACTION.fullmatch(action), action
            if action.startswith("actions/checkout@"):
                assert step["with"]["persist-credentials"] is False
            if action.startswith("astral-sh/setup-uv@"):
                assert step["with"]["version"] == "${{ env.UV_VERSION }}"


def _assert_release_workflow_gates_publish_on_full_quality_suite() -> None:
    workflow = yaml.safe_load(
        (REPOSITORY / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    )
    gate = workflow["jobs"]["source-gate"]
    publish = workflow["jobs"]["attest-publish"]
    gate_commands = _commands(gate)
    publish_commands = _commands(publish)

    assert publish["needs"] == "verify-dist"
    assert "uv run --no-sync ruff format --check ." in gate_commands
    assert "uv run --no-sync ruff check ." in gate_commands
    assert "uv run --no-sync mypy --strict trackmaniarl" in gate_commands
    assert "uv run --no-sync pytest" in gate_commands
    assert "uv build" not in publish_commands
    assert "uv publish dist/*.whl dist/*.tar.gz" in publish_commands


def test_release_workflow_secures_and_publishes_canonical_artifacts() -> None:
    _assert_release_workflow_uploads_archives_and_spdx_sbom()
    _assert_release_workflow_attests_archives_and_sbom()
    _assert_release_workflow_publishes_without_rebuilding()
    _assert_release_workflow_pins_every_action_and_disables_checkout_credentials()


def _assert_wheel_validation_rejects_post_build_source_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    source = Path("trackmaniarl") / "module.py"
    source.parent.mkdir()
    source.write_bytes(b"value = 'built'\n")
    archive = tmp_path / "package.whl"
    _write_wheel_source(archive, source)
    with zipfile.ZipFile(archive) as package:
        check_distribution._validate_wheel_checkout_files(package, archive)
    source.write_bytes(b"value = 'mutated'\n")

    with (
        zipfile.ZipFile(archive) as package,
        pytest.raises(RuntimeError, match="differs from the checkout"),
    ):
        check_distribution._validate_wheel_checkout_files(package, archive)


def _write_wheel_source(archive: Path, source: Path) -> None:
    with zipfile.ZipFile(archive, "w") as package:
        package.write(source, "trackmaniarl/module.py")


def _assert_sdist_validation_rejects_post_build_source_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    source = Path("scripts") / "tool.py"
    source.parent.mkdir()
    source.write_bytes(b"value = 'built'\n")
    archive, root = tmp_path / "package.tar.gz", "trackmaniarl-2.0.0"
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


def test_distribution_validation_rejects_post_build_source_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _assert_wheel_validation_rejects_post_build_source_mutation(tmp_path, monkeypatch)
    _assert_sdist_validation_rejects_post_build_source_mutation(tmp_path, monkeypatch)


def _assert_release_checksums_cover_exactly_archives_and_sbom(tmp_path: Path) -> None:
    subjects = tuple(tmp_path / name for name in ("package.whl", "package.tar.gz", "sbom.json"))
    for index, subject in enumerate(subjects):
        subject.write_bytes(f"subject-{index}".encode())
    checksums = tmp_path / "SHA256SUMS"
    check_distribution._write_checksums(subjects, checksums)
    check_distribution._verify_checksums(subjects, checksums)

    subjects[0].write_bytes(b"mutated")
    with pytest.raises(RuntimeError, match=r"Checksum mismatch: package\.whl"):
        check_distribution._verify_checksums(subjects, checksums)


def _assert_release_checksums_reject_unlisted_subjects(tmp_path: Path) -> None:
    subjects = (tmp_path / "package.whl",)
    subjects[0].write_bytes(b"wheel")
    checksums = tmp_path / "SHA256SUMS"
    check_distribution._write_checksums(subjects, checksums)
    checksums.write_text(
        checksums.read_text(encoding="utf-8") + f"{'0' * 64}  unexpected.txt\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="exactly the release subjects"):
        check_distribution._verify_checksums(subjects, checksums)


def test_release_checksums_bind_the_exact_subject_set(tmp_path: Path) -> None:
    _assert_release_checksums_cover_exactly_archives_and_sbom(tmp_path)
    _assert_release_checksums_reject_unlisted_subjects(tmp_path)


def _valid_sbom() -> dict[str, object]:
    dependency_names = sorted(check_distribution.CORE_DEPENDENCIES)
    packages = [{"name": "trackmaniarl", "versionInfo": "2.0.0", "SPDXID": "SPDXRef-trackmaniarl"}]
    packages.extend(
        {"name": name, "versionInfo": "1", "SPDXID": f"SPDXRef-{name}"} for name in dependency_names
    )
    relationships = [
        {
            "spdxElementId": f"SPDXRef-{name}",
            "relatedSpdxElement": "SPDXRef-trackmaniarl",
            "relationshipType": "DEPENDENCY_OF",
        }
        for name in dependency_names
    ]
    return {
        "spdxVersion": "SPDX-2.3",
        "packages": packages,
        "relationships": relationships,
    }


def test_release_sbom_requires_trackmaniarl_and_dependency_relationships(tmp_path: Path) -> None:
    sbom = tmp_path / "release.spdx.json"
    sbom.write_text(json.dumps(_valid_sbom()))
    check_distribution._validate_sbom(sbom, "2.0.0")

    invalid = _valid_sbom()
    invalid["relationships"] = []
    sbom.write_text(json.dumps(invalid))
    with pytest.raises(RuntimeError, match="dependency relationships"):
        check_distribution._validate_sbom(sbom, "2.0.0")

    sbom.write_text(json.dumps({"spdxVersion": "SPDX-2.2", "packages": []}))
    with pytest.raises(RuntimeError, match=r"not an SPDX 2\.3 JSON document"):
        check_distribution._validate_sbom(sbom, "2.0.0")


def test_release_archive_set_rejects_extra_publishable_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    directory = tmp_path / "dist"
    directory.mkdir()
    wheel = directory / "trackmaniarl-2.0.0-py3-none-any.whl"
    sdist = directory / "trackmaniarl-2.0.0.tar.gz"
    wheel.touch()
    sdist.touch()
    check_distribution._validate_archive_set(Path("dist") / wheel.name, Path("dist") / sdist.name)

    (directory / "unexpected.whl").touch()
    with pytest.raises(RuntimeError, match="archive set differs"):
        check_distribution._validate_archive_set(
            Path("dist") / wheel.name, Path("dist") / sdist.name
        )
