"""Validate the release archives produced from this checkout."""

from __future__ import annotations

import argparse
import email.parser
import hashlib
import json
import re
import tarfile
import tomllib
import zipfile
from email.message import Message
from pathlib import Path, PurePosixPath
from typing import cast

SBOM_NAME = "trackmaniarl-release.spdx.json"
CHECKSUMS_NAME = "SHA256SUMS"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
CORE_DEPENDENCIES = frozenset(
    {"gymnasium", "numpy", "pydantic", "pyyaml", "tensordict", "torch", "zstandard"}
)
DIAGRAM_STEMS = (
    "checkpoint-resume",
    "demonstration-timing",
    "distributed-security",
    "imitation-learning",
    "model-composition",
    "replay-sequence",
    "reward-decomposition",
    "runtime-architecture",
    "trackmania-integration",
)
DIAGRAM_SUFFIXES = (
    ".spec.json",
    ".excalidraw",
    "-preview.png",
    "-preview.svg",
    "-preview.html",
)
SDIST_REQUIRED_PATHS = frozenset(
    {
        "CHANGELOG.md",
        "CONTRIBUTING.md",
        "docs/assets/trackmaniarl-logo.png",
        "docs/diagrams/render.py",
        "LICENSE",
        "NOTICE",
        "README.md",
        "SECURITY.md",
        "pyproject.toml",
        "readme/development.md",
        "readme/trackmania.md",
        "scripts/fetch_analysis.py",
        "scripts/iteration_report.py",
        "scripts/verify_soak.py",
        "tests/unit/test_release_distribution.py",
        "tests/unit/test_verify_soak.py",
        "tests/unit/core/test_run_spec_serialization.py",
        "trackmaniarl/py.typed",
        "trackmaniarl/project/scaffold.py",
        "trackmaniarl/project/scaffold_run_templates.py",
        "trackmaniarl/project/scaffold_templates.py",
    }
)


def _project_version() -> str:
    with Path("pyproject.toml").open("rb") as file:
        return str(tomllib.load(file)["project"]["version"])


def _require_members(names: set[str], required: set[str], archive: Path) -> None:
    missing = sorted(required - names)
    if missing:
        raise RuntimeError(f"{archive} is missing: {', '.join(missing)}")


def _checkout_bytes(relative: PurePosixPath, archive: Path) -> bytes:
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise RuntimeError(f"{archive} has unsafe source member {relative.as_posix()!r}")
    source = Path(*relative.parts)
    if not source.is_file():
        raise RuntimeError(f"{archive} member {relative.as_posix()!r} has no checkout file")
    return source.read_bytes()


def _normalize_checkout_line_endings(content: bytes) -> bytes:
    """Match Git's Windows checkout conversion without touching binary assets."""
    try:
        content.decode("utf-8")
    except UnicodeDecodeError:
        return content
    return content.replace(b"\r\n", b"\n")


def _validate_wheel_checkout_files(package: zipfile.ZipFile, archive: Path) -> None:
    for member in package.infolist():
        relative = PurePosixPath(member.filename.replace("\\", "/"))
        if member.is_dir() or not relative.parts or relative.parts[0] != "trackmaniarl":
            continue
        packaged = package.read(member)
        checkout = _normalize_checkout_line_endings(_checkout_bytes(relative, archive))
        if packaged != checkout:
            raise RuntimeError(
                f"{archive} member {relative.as_posix()!r} differs from the checkout"
            )


def _sdist_checkout_path(member_name: str, root: str) -> PurePosixPath | None:
    packaged = PurePosixPath(member_name.replace("\\", "/"))
    if len(packaged.parts) < 2 or packaged.parts[0] != root:
        return None
    relative = PurePosixPath(*packaged.parts[1:])
    if relative.as_posix() in {"PKG-INFO", "setup.cfg"}:
        return None
    if relative.parts[0].lower().endswith(".egg-info"):
        return None
    return relative


def _validate_sdist_checkout_files(package: tarfile.TarFile, archive: Path, root: str) -> None:
    for member in package.getmembers():
        relative = _sdist_checkout_path(member.name, root)
        if relative is None or not member.isfile():
            continue
        packaged = package.extractfile(member)
        if packaged is None:
            raise RuntimeError(f"{archive} cannot read member {member.name!r}")
        checkout = _normalize_checkout_line_endings(_checkout_bytes(relative, archive))
        if packaged.read() != checkout:
            raise RuntimeError(f"{archive} member {member.name!r} differs from the checkout")


def _wheel_required_members(version: str) -> set[str]:
    dist_info = f"trackmaniarl-{version}.dist-info"
    sources = {path.as_posix() for path in Path("trackmaniarl").rglob("*.py")}
    return sources | {
        "trackmaniarl/py.typed",
        "trackmaniarl/project/scaffold.py",
        "trackmaniarl/project/scaffold_run_templates.py",
        "trackmaniarl/project/scaffold_templates.py",
        "trackmaniarl/project/openplanet/README.md",
        "trackmaniarl/project/openplanet/TrackmaniaRL_Connect.as",
        "trackmaniarl/project/openplanet/info.toml",
        f"{dist_info}/entry_points.txt",
        f"{dist_info}/licenses/LICENSE",
        f"{dist_info}/licenses/NOTICE",
        f"{dist_info}/METADATA",
    }


def _read_wheel_metadata(archive: Path, version: str) -> Message:
    dist_info = f"trackmaniarl-{version}.dist-info"
    with zipfile.ZipFile(archive) as package:
        _require_members(set(package.namelist()), _wheel_required_members(version), archive)
        _validate_wheel_checkout_files(package, archive)
        return email.parser.BytesParser().parsebytes(package.read(f"{dist_info}/METADATA"))


def _validate_wheel_metadata(metadata: Message, archive: Path, version: str) -> None:
    expected = {
        "Name": "TrackmaniaRL",
        "Version": version,
    }
    invalid = [
        f"{key}={metadata[key]!r}" for key, value in expected.items() if metadata[key] != value
    ]
    requires_python = metadata["Requires-Python"]
    constraints = (
        {part.strip() for part in requires_python.split(",")} if requires_python else set()
    )
    if constraints != {">=3.12", "<3.13"}:
        invalid.append(f"Requires-Python={requires_python!r}")
    if invalid:
        raise RuntimeError(f"{archive} has invalid metadata: {', '.join(invalid)}")


def _validate_wheel(archive: Path, version: str) -> None:
    _validate_wheel_metadata(_read_wheel_metadata(archive, version), archive, version)


def _sdist_required_members(root: str) -> set[str]:
    diagrams = {
        f"{root}/docs/diagrams/{stem}{suffix}"
        for stem in DIAGRAM_STEMS
        for suffix in DIAGRAM_SUFFIXES
    }
    return diagrams | {f"{root}/{path}" for path in SDIST_REQUIRED_PATHS}


def _validate_sdist(archive: Path, version: str) -> None:
    root = f"trackmaniarl-{version}"
    with tarfile.open(archive) as package:
        names = {member.name.replace("\\", "/") for member in package.getmembers()}
        _validate_sdist_checkout_files(package, archive, root)
    _require_members(names, _sdist_required_members(root), archive)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_package_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _load_sbom(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise RuntimeError(f"Missing release SBOM: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{path} is not valid JSON") from error
    if not isinstance(document, dict) or document.get("spdxVersion") != "SPDX-2.3":
        raise RuntimeError(f"{path} is not an SPDX 2.3 JSON document")
    return cast(dict[str, object], document)


def _package_index(document: dict[str, object], path: Path) -> dict[str, tuple[str, str]]:
    packages = document.get("packages")
    if not isinstance(packages, list) or not packages:
        raise RuntimeError(f"{path} does not describe any packages")
    package_index: dict[str, tuple[str, str]] = {}
    for package_value in packages:
        if not isinstance(package_value, dict):
            continue
        package = cast(dict[str, object], package_value)
        identifier = package.get("SPDXID")
        name = package.get("name")
        package_version = package.get("versionInfo")
        if isinstance(identifier, str) and isinstance(name, str):
            package_index[identifier] = (
                _normalize_package_name(name),
                package_version if isinstance(package_version, str) else "",
            )
    return package_index


def _trackmania_ids(
    package_index: dict[str, tuple[str, str]], path: Path, version: str
) -> set[str]:
    trackmania_ids = {
        identifier
        for identifier, (name, package_version) in package_index.items()
        if name == "trackmaniarl" and package_version == version
    }
    if not trackmania_ids:
        raise RuntimeError(f"{path} does not describe TrackmaniaRL {version}")
    return trackmania_ids


def _relationship_dependencies(value: object, trackmania_ids: set[str]) -> set[str]:
    if not isinstance(value, dict):
        return set()
    relationship = cast(dict[str, object], value)
    source = relationship.get("spdxElementId")
    target = relationship.get("relatedSpdxElement")
    kind = relationship.get("relationshipType")
    if kind == "DEPENDENCY_OF" and target in trackmania_ids and isinstance(source, str):
        return {source}
    if kind == "DEPENDS_ON" and source in trackmania_ids and isinstance(target, str):
        return {target}
    return set()


def _dependency_ids(document: dict[str, object], trackmania_ids: set[str]) -> set[str]:
    relationships = document.get("relationships")
    dependency_ids: set[str] = set()
    if isinstance(relationships, list):
        for value in relationships:
            dependency_ids.update(_relationship_dependencies(value, trackmania_ids))
    return dependency_ids


def _validate_dependency_names(
    package_index: dict[str, tuple[str, str]], dependency_ids: set[str], path: Path
) -> None:
    dependency_names = {
        package_index[identifier][0] for identifier in dependency_ids if identifier in package_index
    }
    missing_dependencies = sorted(CORE_DEPENDENCIES - dependency_names)
    if missing_dependencies:
        raise RuntimeError(
            f"{path} has no TrackmaniaRL dependency relationships for: "
            f"{', '.join(missing_dependencies)}"
        )


def _validate_sbom(path: Path, version: str) -> None:
    document = _load_sbom(path)
    package_index = _package_index(document, path)
    trackmania_ids = _trackmania_ids(package_index, path, version)
    dependency_ids = _dependency_ids(document, trackmania_ids)
    _validate_dependency_names(package_index, dependency_ids, path)


def _release_subjects(version: str) -> tuple[Path, Path, Path]:
    directory = Path("dist")
    return (
        directory / f"trackmaniarl-{version}-py3-none-any.whl",
        directory / f"trackmaniarl-{version}.tar.gz",
        directory / SBOM_NAME,
    )


def _validate_archive_set(wheel: Path, sdist: Path) -> None:
    found = set(Path("dist").glob("*.whl")) | set(Path("dist").glob("*.tar.gz"))
    expected = {wheel, sdist}
    if found != expected:
        names = ", ".join(sorted(path.name for path in found ^ expected))
        raise RuntimeError(f"Release archive set differs from the expected pair: {names}")


def _write_checksums(subjects: tuple[Path, ...], target: Path) -> None:
    lines = [f"{_sha256(path)}  {path.name}" for path in sorted(subjects)]
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(target)


def _read_checksums(target: Path) -> dict[str, str]:
    if not target.is_file():
        raise RuntimeError(f"Missing release checksums: {target}")
    checksums: dict[str, str] = {}
    for line in target.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if separator != "  " or SHA256_PATTERN.fullmatch(digest) is None:
            raise RuntimeError(f"{target} has an invalid checksum entry")
        if Path(name).name != name or "/" in name or "\\" in name or name in checksums:
            raise RuntimeError(f"{target} has an unsafe or duplicate subject {name!r}")
        checksums[name] = digest
    return checksums


def _verify_checksums(subjects: tuple[Path, ...], target: Path) -> None:
    expected_names = {path.name for path in subjects}
    checksums = _read_checksums(target)
    if set(checksums) != expected_names:
        raise RuntimeError(f"{target} does not list exactly the release subjects")
    mismatches = [path.name for path in subjects if checksums[path.name] != _sha256(path)]
    if mismatches:
        raise RuntimeError(f"Checksum mismatch: {', '.join(sorted(mismatches))}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", help="Release tag, expected in the form v<project-version>")
    checksum_mode = parser.add_mutually_exclusive_group()
    checksum_mode.add_argument("--write-checksums", action="store_true")
    checksum_mode.add_argument("--verify-checksums", action="store_true")
    return parser.parse_args()


def _validate_release_tag(tag: str | None, version: str) -> None:
    if tag is not None and tag != f"v{version}":
        raise RuntimeError(f"Release tag {tag!r} does not match project version {version!r}")


def main() -> None:
    args = _parse_args()
    version = _project_version()
    _validate_release_tag(args.tag, version)
    wheel, sdist, sbom = _release_subjects(version)
    missing = [str(archive) for archive in (wheel, sdist) if not archive.is_file()]
    if missing:
        raise RuntimeError(f"Missing release archives: {', '.join(missing)}")
    _validate_archive_set(wheel, sdist)
    _validate_wheel(wheel, version)
    _validate_sdist(sdist, version)
    if args.write_checksums or args.verify_checksums:
        _validate_sbom(sbom, version)
        subjects = (wheel, sdist, sbom)
        checksums = Path("dist") / CHECKSUMS_NAME
        if args.write_checksums:
            _write_checksums(subjects, checksums)
        else:
            _verify_checksums(subjects, checksums)


if __name__ == "__main__":
    main()
