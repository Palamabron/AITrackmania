"""Coordinator policy publication and stop-state decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from time import monotonic, perf_counter
from typing import TYPE_CHECKING

from trackmaniarl.core.contracts import ReplicablePolicy

if TYPE_CHECKING:
    from trackmaniarl.distributed.coordinator import Coordinator


class PolicyPublicationMode(StrEnum):
    SCHEDULED = "scheduled"
    FORCED = "forced"


@dataclass(frozen=True, slots=True)
class _PolicyPublication:
    coordinator: Coordinator
    now: float
    mode: PolicyPublicationMode


def has_active_actor(coordinator: Coordinator) -> bool:
    with coordinator._lock:
        return bool(set(coordinator._last_heartbeats) - coordinator._timed_out_actors)


def can_update(coordinator: Coordinator) -> bool:
    return not coordinator._external_stop_requested() and (
        coordinator._has_active_actor()
        or coordinator.counters.transitions >= coordinator.run.spec.training.total_transitions
    )


def publish_policy(
    coordinator: Coordinator,
    mode: PolicyPublicationMode = PolicyPublicationMode.SCHEDULED,
) -> None:
    publication = _PolicyPublication(coordinator, monotonic(), mode)
    if not _publication_due(publication):
        return
    started = perf_counter()
    payload = _policy_payload(coordinator)
    with coordinator._lock:
        coordinator._policy_payload = payload
        coordinator.counters.policy_version = coordinator.counters.updates
    _complete_publication(publication, started)


def _publication_due(publication: _PolicyPublication) -> bool:
    if publication.mode is PolicyPublicationMode.FORCED:
        return True
    coordinator = publication.coordinator
    if coordinator.counters.updates == coordinator._last_policy_update:
        return False
    elapsed = publication.now - coordinator._last_policy_publish
    return elapsed >= coordinator.run.spec.distributed.policy_refresh_s


def _policy_payload(coordinator: Coordinator) -> bytes:
    policy = coordinator.run.learner.policy()
    if not isinstance(policy, ReplicablePolicy):
        raise TypeError("distributed training requires learner.policy() to be ReplicablePolicy")
    return coordinator.codec.encode(dict(policy.export_state()))


def _complete_publication(publication: _PolicyPublication, started: float) -> None:
    coordinator = publication.coordinator
    coordinator._last_policy_update = coordinator.counters.updates
    coordinator._last_policy_publish = publication.now
    coordinator.run.logger.log(
        "distributed/policy_published",
        {
            "policy_version": coordinator.counters.policy_version,
            "timing/policy_publish_s": perf_counter() - started,
        },
        step=coordinator.counters.updates,
    )


def should_stop(coordinator: Coordinator) -> bool:
    return (
        coordinator.counters.transitions >= coordinator.run.spec.training.total_transitions
        or getattr(coordinator, "_evaluation_stop_reason", None) is not None
        or coordinator._external_stop_requested()
    )


def external_stop_requested(coordinator: Coordinator) -> bool:
    return bool(coordinator.external_stop is not None and coordinator.external_stop.is_set())


def log_execution(coordinator: Coordinator) -> None:
    execution = getattr(coordinator.run.learner, "execution_manifest", None)
    if callable(execution):
        coordinator.run.logger.log(
            "train/execution", dict(execution()), step=coordinator.counters.updates
        )
