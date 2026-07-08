"""Entry-point plugin discovery for tmrl — see CONTRIBUTING.md for the authoring guide."""

from __future__ import annotations

import logging
import threading

from tmrl.registry import ALGORITHMS, INTERFACES, MEMORIES, MODELS, Registry

_log = logging.getLogger(__name__)

_lock = threading.Lock()
_loaded = False

_GROUP_MAP: dict[str, Registry] = {
    "tmrl.algorithms": ALGORITHMS,
    "tmrl.models": MODELS,
    "tmrl.interfaces": INTERFACES,
    "tmrl.memories": MEMORIES,
}


def load_plugins() -> dict[str, list[str]]:
    """Discover and register all installed TMRL entry-point plugins (idempotent, thread-safe)."""
    global _loaded
    results: dict[str, list[str]] = {group: [] for group in _GROUP_MAP}
    with _lock:
        if _loaded:
            return results
        _loaded = True
    for group, registry in _GROUP_MAP.items():
        newly = registry.discover_entry_points(group)
        results[group] = newly
        if newly:
            _log.info("Loaded %d plugin(s) for group %r: %s", len(newly), group, ", ".join(newly))
        else:
            _log.debug("No new plugins for group %r.", group)
    return results


def discover_plugins() -> dict[str, list[str]]:
    """List advertised entry-point names without loading them."""
    from importlib.metadata import entry_points

    return {group: [ep.name for ep in entry_points(group=group)] for group in _GROUP_MAP}
