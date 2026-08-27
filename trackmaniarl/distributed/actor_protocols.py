"""Structural contracts shared by actor services."""

from __future__ import annotations

from collections.abc import Mapping
from threading import Event
from typing import Any, Protocol

from trackmaniarl.core.contracts import ReplicablePolicy
from trackmaniarl.core.data import Transition
from trackmaniarl.core.spec import RunSpec
from trackmaniarl.distributed.actor_requests import (
    EnvironmentReset,
    EvaluationEpisodeRequest,
    SpoolRequest,
    TelemetryFailure,
)
from trackmaniarl.distributed.codec import WireCodec


class CollectionRuntime(Protocol):
    spec: RunSpec
    actor_id: str
    session_id: str
    stop: Event
    stop_reason: str
    evaluate: Event
    codec: WireCodec
    _evaluation_index: int
    _evaluation_request: tuple[bytes, int] | None
    _evaluation_request_lock: Any

    def _actor_seed(self) -> int: ...

    def _policy(self) -> tuple[ReplicablePolicy, float, int]: ...

    def _new_policy(self) -> ReplicablePolicy: ...

    def _evaluate(self, environment: Any, pipeline: Any) -> None: ...

    def _evaluate_episode(self, request: EvaluationEpisodeRequest) -> dict[str, Any]: ...

    def _evaluation_policy(self) -> tuple[ReplicablePolicy, int]: ...

    def _evaluation_telemetry_failure(self, failure: TelemetryFailure) -> dict[str, Any]: ...

    def _reset_environment(self, request: EnvironmentReset) -> Any: ...

    def _should_flush(self, transitions: list[Transition], started: float) -> bool: ...

    def _spool(self, request: SpoolRequest) -> None: ...

    def _summary(
        self, reward: float, info: Mapping[str, Any], transitions: int
    ) -> dict[str, Any]: ...
