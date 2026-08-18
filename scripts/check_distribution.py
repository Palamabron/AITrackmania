"""Fail CI when a built source or wheel distribution omits its license."""

from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path


def _contains_license(archive: Path) -> bool:
    if archive.suffix == ".whl":
        with zipfile.ZipFile(archive) as package:
            return any(name.endswith("/licenses/LICENSE") for name in package.namelist())
    with tarfile.open(archive) as package:
        return any(Path(member.name).name == "LICENSE" for member in package.getmembers())


def main() -> None:
    archives = sorted(Path("dist").glob("*.tar.gz")) + sorted(Path("dist").glob("*.whl"))
    if not archives:
        raise RuntimeError("No sdist or wheel was built in dist/")
    missing = [str(archive) for archive in archives if not _contains_license(archive)]
    if missing:
        raise RuntimeError(f"Built distributions missing LICENSE: {', '.join(missing)}")


if __name__ == "__main__":
    main()
