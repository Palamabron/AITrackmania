"""Download ``resources.zip`` (TmrlData assets) from GitHub releases.

Tries the release tag matching the installed package version first, then a known
stable fallback so fresh installs still work before the matching release asset exists.
"""

from __future__ import annotations

import socket
import urllib.error
import urllib.request
from pathlib import Path

_RELEASE_BASE = "https://github.com/trackmania-rl/tmrl/releases/download"
# Oldest tag known to ship a compatible resources.zip; used if v{package} is missing.
_FALLBACK_TAG = "v0.6.0"


def resources_zip_urls() -> tuple[str, ...]:
    """Candidate URLs, most specific first."""
    urls: list[str] = []
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            ver = version("tmrl").split("+", 1)[0].strip()
        except PackageNotFoundError:
            ver = ""
        if ver:
            urls.append(f"{_RELEASE_BASE}/v{ver}/resources.zip")
    except Exception:
        pass
    fb = f"{_RELEASE_BASE}/{_FALLBACK_TAG}/resources.zip"
    if fb not in urls:
        urls.append(fb)
    return tuple(urls)


def download_resources_zip(dest: Path) -> str:
    """Download ``resources.zip`` to ``dest`` (file path). Return the URL that succeeded."""
    dest = Path(dest).expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for url in resources_zip_urls():
        try:
            urllib.request.urlretrieve(url, str(dest))
            return url
        except urllib.error.HTTPError as e:
            if e.code == 404:
                errors.append(f"{url} (HTTP 404)")
                continue
            raise ConnectionError(f"could not download {url} (HTTP {e.code})") from e
        except (socket.gaierror, urllib.error.URLError) as err:
            errors.append(f"{url} ({err!s})")
            continue
    raise ConnectionError(
        "Could not download TMRL resources.zip from any release URL. "
        "Publish `resources.zip` on the matching GitHub release, or use the fallback. "
        f"Tried: {'; '.join(errors)}"
    ) from None
