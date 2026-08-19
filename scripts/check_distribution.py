"""Validate the release archives produced from this checkout."""

from __future__ import annotations

import argparse
import email.parser
import tarfile
import tomllib
import zipfile
from pathlib import Path


def _project_version() -> str:
    with Path("pyproject.toml").open("rb") as file:
        return str(tomllib.load(file)["project"]["version"])


def _require_members(names: set[str], required: set[str], archive: Path) -> None:
    missing = sorted(required - names)
    if missing:
        raise RuntimeError(f"{archive} is missing: {', '.join(missing)}")


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
        metadata = email.parser.BytesParser().parsebytes(package.read(f"{dist_info}/METADATA"))
    expected = {"Name": "TrackmaniaRL", "Version": version, "Requires-Python": ">=3.12"}
    invalid = [
        f"{key}={metadata[key]!r}" for key, value in expected.items() if metadata[key] != value
    ]
    if invalid:
        raise RuntimeError(f"{archive} has invalid metadata: {', '.join(invalid)}")


def _validate_sdist(archive: Path, version: str) -> None:
    root = f"trackmaniarl-{version}"
    with tarfile.open(archive) as package:
        names = {member.name.replace("\\", "/") for member in package.getmembers()}
    _require_members(
        names,
        {
            f"{root}/CHANGELOG.md",
            f"{root}/CONTRIBUTING.md",
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
