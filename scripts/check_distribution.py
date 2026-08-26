"""Validate the release archives produced from this checkout."""

from __future__ import annotations

import argparse
import email.parser
import tarfile
import tomllib
import zipfile
from pathlib import Path, PurePosixPath


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


def _validate_wheel_checkout_files(package: zipfile.ZipFile, archive: Path) -> None:
    for member in package.infolist():
        relative = PurePosixPath(member.filename.replace("\\", "/"))
        if member.is_dir() or not relative.parts or relative.parts[0] != "trackmaniarl":
            continue
        if package.read(member) != _checkout_bytes(relative, archive):
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
        if packaged.read() != _checkout_bytes(relative, archive):
            raise RuntimeError(f"{archive} member {member.name!r} differs from the checkout")


def _validate_wheel(archive: Path, version: str) -> None:
    dist_info = f"trackmaniarl-{version}.dist-info"
    with zipfile.ZipFile(archive) as package:
        names = set(package.namelist())
        _require_members(
            names,
            {
                "trackmaniarl/py.typed",
                "trackmaniarl/project/openplanet/README.md",
                "trackmaniarl/project/openplanet/TrackmaniaRL_GrabData_IQN.as",
                "trackmaniarl/project/openplanet/info.toml",
                f"{dist_info}/entry_points.txt",
                f"{dist_info}/licenses/LICENSE",
                f"{dist_info}/licenses/NOTICE",
                f"{dist_info}/METADATA",
            },
            archive,
        )
        _validate_wheel_checkout_files(package, archive)
        metadata = email.parser.BytesParser().parsebytes(package.read(f"{dist_info}/METADATA"))
        package_init = package.read("trackmaniarl/__init__.py").decode()
        scaffold = package.read("trackmaniarl/project/scaffold.py").decode()
    expected = {"Name": "TrackmaniaRL", "Version": version, "Requires-Python": ">=3.12"}
    invalid = [
        f"{key}={metadata[key]!r}" for key, value in expected.items() if metadata[key] != value
    ]
    if invalid:
        raise RuntimeError(f"{archive} has invalid metadata: {', '.join(invalid)}")
    if f'__version__ = "{version}"' not in package_init:
        raise RuntimeError(f"{archive} has a stale source-checkout version fallback")
    if f'installed_version = "{version}"' not in scaffold:
        raise RuntimeError(f"{archive} has a stale generated-project version fallback")


def _validate_sdist(archive: Path, version: str) -> None:
    root = f"trackmaniarl-{version}"
    with tarfile.open(archive) as package:
        names = {member.name.replace("\\", "/") for member in package.getmembers()}
        _validate_sdist_checkout_files(package, archive, root)
    diagram_stems = (
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
    diagram_members = {
        f"{root}/docs/diagrams/{stem}{suffix}"
        for stem in diagram_stems
        for suffix in (
            ".spec.json",
            ".excalidraw",
            "-preview.png",
            "-preview.svg",
            "-preview.html",
        )
    }
    _require_members(
        names,
        diagram_members
        | {
            f"{root}/CHANGELOG.md",
            f"{root}/CONTRIBUTING.md",
            f"{root}/docs/diagrams/render.py",
            f"{root}/LICENSE",
            f"{root}/NOTICE",
            f"{root}/README.md",
            f"{root}/SECURITY.md",
            f"{root}/pyproject.toml",
            f"{root}/scripts/fetch_analysis.py",
            f"{root}/scripts/iteration_report.py",
            f"{root}/tests/unit/core/test_run_spec_serialization.py",
            f"{root}/trackmaniarl/py.typed",
        },
        archive,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", help="Release tag, expected in the form v<project-version>")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    version = _project_version()
    if args.tag is not None and args.tag != f"v{version}":
        raise RuntimeError(f"Release tag {args.tag!r} does not match project version {version!r}")
    wheel = Path("dist") / f"trackmaniarl-{version}-py3-none-any.whl"
    sdist = Path("dist") / f"trackmaniarl-{version}.tar.gz"
    missing = [str(archive) for archive in (wheel, sdist) if not archive.is_file()]
    if missing:
        raise RuntimeError(f"Missing release archives: {', '.join(missing)}")
    _validate_wheel(wheel, version)
    _validate_sdist(sdist, version)


if __name__ == "__main__":
    main()
