"""Asynchronous actor/learner runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from trackmaniarl.distributed.actor import ActorRuntime
    from trackmaniarl.distributed.coordinator import Coordinator

__all__ = ["ActorRuntime", "Coordinator"]


def __getattr__(name: str) -> Any:
    if name == "ActorRuntime":
        from trackmaniarl.distributed.actor import ActorRuntime

        return ActorRuntime
    if name == "Coordinator":
        from trackmaniarl.distributed.coordinator import Coordinator

        return Coordinator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
