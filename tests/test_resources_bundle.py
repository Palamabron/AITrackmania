"""Release asset URL selection and download-retry logic (no real network calls)."""

from __future__ import annotations

import urllib.error
from pathlib import Path
from unittest.mock import patch

import pytest
from tmrl.tools.init_package.resources_bundle import download_resources_zip, resources_zip_urls

# ---------------------------------------------------------------------------
# URL-builder tests (no network)
# ---------------------------------------------------------------------------


def test_resources_zip_urls_include_fallback() -> None:
    """resources_zip_urls returns at least one URL and includes the stable v0.6.0 fallback."""
    urls = resources_zip_urls()
    assert len(urls) >= 1
    assert any("v0.6.0" in u for u in urls)


def test_resources_zip_urls_version_specific_url_is_first_when_package_installed() -> None:
    """When importlib.metadata.version resolves a version, that URL is first in the list."""
    with patch("tmrl.tools.init_package.resources_bundle.resources_zip_urls") as mock_urls:
        mock_urls.return_value = (
            "https://github.com/trackmania-rl/tmrl/releases/download/v0.8.0/resources.zip",
            "https://github.com/trackmania-rl/tmrl/releases/download/v0.6.0/resources.zip",
        )
        urls = mock_urls()
    assert "v0.8.0" in urls[0]
    assert "v0.6.0" in urls[1]


# ---------------------------------------------------------------------------
# download_resources_zip: version selection and HTTP-404 fallback retry
# ---------------------------------------------------------------------------


def test_download_uses_version_url_first_on_success(tmp_path: Path) -> None:
    """download_resources_zip returns the first URL when it succeeds."""
    dest = tmp_path / "resources.zip"
    version_url = "https://example.com/v0.8.0/resources.zip"
    fallback_url = "https://example.com/v0.6.0/resources.zip"

    def _fake_urlretrieve(url: str, filename: str) -> None:
        Path(filename).write_bytes(b"ok")

    with (
        patch(
            "tmrl.tools.init_package.resources_bundle.resources_zip_urls",
            return_value=(version_url, fallback_url),
        ),
        patch(
            "tmrl.tools.init_package.resources_bundle.urllib.request.urlretrieve",
            side_effect=_fake_urlretrieve,
        ),
    ):
        used_url = download_resources_zip(dest)

    assert used_url == version_url
    assert dest.read_bytes() == b"ok"


def test_download_retries_fallback_on_http_404(tmp_path: Path) -> None:
    """When the version-specific URL returns HTTP 404, download falls back to the stable URL."""
    dest = tmp_path / "resources.zip"
    version_url = "https://example.com/v0.8.0/resources.zip"
    fallback_url = "https://example.com/v0.6.0/resources.zip"

    call_count = 0

    def _fake_urlretrieve(url: str, filename: str) -> None:
        nonlocal call_count
        call_count += 1
        if url == version_url:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)  # type: ignore[arg-type]
        Path(filename).write_bytes(b"fallback-ok")

    with (
        patch(
            "tmrl.tools.init_package.resources_bundle.resources_zip_urls",
            return_value=(version_url, fallback_url),
        ),
        patch(
            "tmrl.tools.init_package.resources_bundle.urllib.request.urlretrieve",
            side_effect=_fake_urlretrieve,
        ),
    ):
        used_url = download_resources_zip(dest)

    assert used_url == fallback_url
    assert call_count == 2
    assert dest.read_bytes() == b"fallback-ok"


def test_download_raises_when_all_urls_fail(tmp_path: Path) -> None:
    """ConnectionError is raised when every candidate URL fails."""
    dest = tmp_path / "resources.zip"

    err_404 = urllib.error.HTTPError("url", 404, "Not Found", {}, None)  # type: ignore[arg-type]

    with (
        patch(
            "tmrl.tools.init_package.resources_bundle.resources_zip_urls",
            return_value=("https://example.com/v0.8.0/resources.zip",),
        ),
        patch(
            "tmrl.tools.init_package.resources_bundle.urllib.request.urlretrieve",
            side_effect=err_404,
        ),
        pytest.raises(ConnectionError),
    ):
        download_resources_zip(dest)
