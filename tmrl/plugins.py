"""Plugin discovery and loading for TMRL via Python entry points.

External packages can register new algorithms, models, interfaces, and memories
without modifying any TMRL source code.  Declare entry points in your package's
``pyproject.toml`` under the appropriate group::

    [project.entry-points."tmrl.algorithms"]
    MyAlgo = "mypackage.module:MyAlgoClass"

    [project.entry-points."tmrl.models"]
    MyModel = "mypackage.models:MyModelClass"

    [project.entry-points."tmrl.interfaces"]
    MyInterface = "mypackage.interfaces:MyInterfaceClass"

    [project.entry-points."tmrl.memories"]
    MyMemory = "mypackage.memories:MyMemoryClass"

After installing your package (``pip install mypackage``), TMRL will
automatically discover and register its components on ``import tmrl``.
"""

from __future__ import annotations

import logging

from tmrl.registry import ALGORITHMS, INTERFACES, MEMORIES, MODELS, Registry

_log = logging.getLogger(__name__)

_LOADED: bool = False

_GROUP_MAP: dict[str, Registry] = {
    "tmrl.algorithms": ALGORITHMS,
    "tmrl.models": MODELS,
    "tmrl.interfaces": INTERFACES,
    "tmrl.memories": MEMORIES,
}


def load_plugins() -> dict[str, list[str]]:
    """Discover and load all installed TMRL plugins (idempotent).

    Iterates the four standard entry-point groups (``tmrl.algorithms``,
    ``tmrl.models``, ``tmrl.interfaces``, ``tmrl.memories``), loads each
    advertised class into the corresponding registry, and returns a mapping
    of group name to the list of newly-registered names.

    Subsequent calls are no-ops and return empty lists for all groups (the
    module-level ``_LOADED`` flag prevents redundant scanning).

    Returns:
        A dict mapping each group name to the list of names registered in
        this call.  Values are empty lists on repeated calls.
    """
    global _LOADED
    results: dict[str, list[str]] = {group: [] for group in _GROUP_MAP}
    if _LOADED:
        return results
    _LOADED = True

    for group, registry in _GROUP_MAP.items():
        newly = registry.discover_entry_points(group)
        results[group] = newly
        if newly:
            _log.info(
                "Loaded %d plugin(s) for group %r: %s",
                len(newly),
                group,
                ", ".join(newly),
            )
        else:
            _log.debug("No new plugins for group %r.", group)

    return results


def discover_plugins() -> dict[str, list[str]]:
    """List advertised entry-point names for all TMRL plugin groups.

    Unlike :func:`load_plugins`, this function does **not** import or register
    anything — it only reads the entry-point metadata.  Useful for inspection
    without side effects.

    Returns:
        A dict mapping each group name to the list of advertised entry-point
        names (regardless of whether they are already registered).
    """
    from importlib.metadata import entry_points

    return {group: [ep.name for ep in entry_points(group=group)] for group in _GROUP_MAP}
