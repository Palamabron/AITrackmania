"""Central registry for dynamically-resolved TMRL components.

Usage::

    from tmrl.registry import ALGORITHMS

    @ALGORITHMS.register("SAC")
    class SpinupSacAgent(TrainingAgent):
        ...

    agent_cls = ALGORITHMS.get("SAC")
"""

from __future__ import annotations

from collections.abc import KeysView
from typing import TypeVar

T = TypeVar("T")


class Registry[T]:
    """String-keyed registry for component classes (algorithms, memories, interfaces).

    Provides a ``register`` class decorator and a ``get`` lookup.  Keys are
    case-sensitive and must be unique within a registry instance.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._entries: dict[str, type[T]] = {}

    def register(self, key: str):
        """Class decorator that registers *cls* under *key*."""

        def decorator(cls: type[T]) -> type[T]:
            if key in self._entries:
                existing = self._entries[key]
                raise ValueError(
                    f"Duplicate {self._name} registration for {key!r}: "
                    f"{cls!r} conflicts with {existing!r}"
                )
            self._entries[key] = cls
            return cls

        return decorator

    def get(self, key: str) -> type[T]:
        """Return the class registered under *key*, or raise ``KeyError``."""
        try:
            return self._entries[key]
        except KeyError:
            available = ", ".join(sorted(self._entries)) or "(none registered)"
            raise KeyError(f"Unknown {self._name} {key!r}. Registered: {available}") from None

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def keys(self) -> KeysView[str]:
        """All registered keys."""
        return self._entries.keys()

    def __repr__(self) -> str:
        return f"Registry({self._name!r}, keys={sorted(self._entries)})"


ALGORITHMS: Registry = Registry("algorithm")
MEMORIES: Registry = Registry("memory")
INTERFACES: Registry = Registry("interface")
MODELS: Registry = Registry("model")
