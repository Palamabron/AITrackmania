"""Lazy public access to bundled replay-memory implementations."""

from __future__ import annotations

import importlib
from typing import Any, cast

_REPLAY = {
    "InMemoryReplayStore": "tmrl.core.replay:InMemoryReplayStore",
    "UniformSampler": "tmrl.core.replay:UniformSampler",
    "PrioritizedSampler": "tmrl.core.replay:PrioritizedSampler",
    "SequenceSampler": "tmrl.core.replay:SequenceSampler",
    "DemoMixSampler": "tmrl.core.replay:DemoMixSampler",
}


def replay_class(name: str) -> type[Any]:
    """Resolve a built-in replay implementation without importing every variant."""

    try:
        path = _REPLAY[name]
    except KeyError as exc:
        raise AttributeError(f"Unknown bundled replay {name!r}: {sorted(_REPLAY)}") from exc
    module_name, _, class_name = path.partition(":")
    return cast(type[Any], getattr(importlib.import_module(module_name), class_name))


def __getattr__(name: str) -> Any:
    if name in _REPLAY:
        return replay_class(name)
    raise AttributeError(name)


__all__ = ["replay_class", *_REPLAY]
