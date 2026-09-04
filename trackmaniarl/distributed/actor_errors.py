"""Typed failures raised by distributed actors."""

from __future__ import annotations


class ActorRuntimeError(RuntimeError):
    """Base class for failures that must terminate an actor process."""


class ActorEnvironmentError(ActorRuntimeError):
    """The actor could not restore its environment connection."""


class ActorBackgroundError(ActorRuntimeError):
    """A required actor background worker failed."""

    def __init__(self, stage: str, cause: BaseException) -> None:
        super().__init__(f"{stage} failed: {type(cause).__name__}: {cause}")
        self.cause = cause
