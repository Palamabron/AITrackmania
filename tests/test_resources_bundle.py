"""Release asset URL selection (no network)."""

from tmrl.tools.init_package.resources_bundle import resources_zip_urls


def test_resources_zip_urls_include_fallback() -> None:
    urls = resources_zip_urls()
    assert len(urls) >= 1
    assert any("v0.6.0" in u for u in urls)
