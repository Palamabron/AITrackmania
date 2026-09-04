from __future__ import annotations

from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING or __package__:
    from scripts.soak_report_types import (
        ActorRegistration,
        ArtifactFacts,
        IdentityCollector,
        IdentityFacts,
        ResumeBoundary,
        ResumeContext,
        ResumeFacts,
        SegmentIdentityState,
        WalRecoveryFacts,
    )
    from scripts.soak_types import (
        FATAL_EVENTS,
        SHA256_PATTERN,
        TELEMETRY_FAILURE_KEYS,
        Checkpoint,
        Event,
        ResumeEvidence,
        VerificationInputError,
        integer,
        sha256,
        string,
    )
else:
    from soak_report_types import (
        ActorRegistration,
        ArtifactFacts,
        IdentityCollector,
        IdentityFacts,
        ResumeBoundary,
        ResumeContext,
        ResumeFacts,
        SegmentIdentityState,
        WalRecoveryFacts,
    )
    from soak_types import (
        FATAL_EVENTS,
        SHA256_PATTERN,
        TELEMETRY_FAILURE_KEYS,
        Checkpoint,
        Event,
        ResumeEvidence,
        VerificationInputError,
        integer,
        sha256,
        string,
    )


def segments(events: list[Event]) -> tuple[list[str], dict[str, list[Event]]]:
    order: list[str] = []
    grouped: dict[str, list[Event]] = defaultdict(list)
    for event in events:
        if event.segment_id not in grouped:
            order.append(event.segment_id)
        grouped[event.segment_id].append(event)
    return order, dict(grouped)


def segment_runtime(segment_events: list[Event]) -> float:
    return max(event.elapsed_s for event in segment_events)


def _payload_failure(value: object, key: str = "") -> bool:
    if isinstance(value, dict):
        items = cast(dict[object, object], value)
        return any(_payload_failure(item, str(name)) for name, item in items.items())
    if key == "termination" and value == "telemetry_error":
        return True
    if key not in TELEMETRY_FAILURE_KEYS or value is None or value is False:
        return False
    if isinstance(value, (int, float)):
        return float(value) > 0.0
    return bool(value)


def failure_events(events: list[Event]) -> list[Event]:
    return [
        event for event in events if event.name in FATAL_EVENTS or _payload_failure(event.payload)
    ]


def checkpoint_file(run_dir: Path, reported_path: str) -> Path:
    normalized = reported_path.replace("\\", "/")
    name = normalized.rsplit("/", maxsplit=1)[-1]
    if not name or name in {".", ".."}:
        raise VerificationInputError("checkpoint event contains an invalid path")
    return run_dir / "checkpoints" / name


def checkpoints(run_dir: Path, events: list[Event]) -> list[Checkpoint]:
    return [
        _completed_checkpoint(run_dir, event)
        for event in events
        if event.name == "train/checkpoint_completed"
    ]


def _completed_checkpoint(run_dir: Path, event: Event) -> Checkpoint:
    if event.step is None or event.step < 0:
        raise VerificationInputError("checkpoint completion has no non-negative step")
    frontier = integer(
        event.payload.get("journal_applied_frontier"),
        "checkpoint journal_applied_frontier",
    )
    reported = string(event.payload.get("path"), "checkpoint path")
    return Checkpoint(
        event.index,
        event.timestamp,
        event.segment_id,
        event.step,
        frontier,
        reported,
        checkpoint_file(run_dir, reported),
    )


def actor_identities(segment_order: list[str], grouped: dict[str, list[Event]]) -> IdentityFacts:
    collector = IdentityCollector()
    actor_sets: list[set[str]] = []
    sessions_valid = True
    for segment in segment_order:
        actors, seen = _segment_actor_identities(segment, grouped[segment], collector)
        actor_sets.append(actors)
        sessions_valid = sessions_valid and seen
    stable = set.intersection(*actor_sets) if actor_sets and all(actor_sets) else set()
    fingerprints = collector.fingerprints
    return IdentityFacts(
        collector.registrations,
        stable,
        fingerprints,
        sessions_valid,
        len(fingerprints) == 1
        and all(SHA256_PATTERN.fullmatch(value) is not None for value in fingerprints),
    )


def _segment_actor_identities(
    segment: str, events: list[Event], collector: IdentityCollector
) -> tuple[set[str], bool]:
    state = SegmentIdentityState(segment, collector)
    actors: set[str] = set()
    valid = True
    for event in events:
        if event.name != "actor/registered":
            continue
        registration = _actor_registration(event)
        actors.add(registration.actor_id)
        collector.fingerprints.add(registration.fingerprint)
        valid = _session_is_valid(state, registration) and valid
        _record_registration(state, registration)
    return actors, valid


def _actor_registration(event: Event) -> ActorRegistration:
    return ActorRegistration(
        string(event.payload.get("actor_id"), "actor registration actor_id"),
        string(event.payload.get("session_id"), "actor registration session_id"),
        string(event.payload.get("run_fingerprint"), "actor registration run_fingerprint"),
    )


def _session_is_valid(state: SegmentIdentityState, registration: ActorRegistration) -> bool:
    identity = (state.segment, registration.actor_id)
    previous = state.collector.sessions.get(registration.session_id)
    state.collector.sessions[registration.session_id] = identity
    return previous is None or previous == identity


def _record_registration(state: SegmentIdentityState, registration: ActorRegistration) -> None:
    key = (registration.actor_id, registration.session_id)
    if key in state.registrations:
        return
    state.collector.registrations.append(_registration_json(state.segment, registration))
    state.registrations.add(key)


def _registration_json(segment: str, registration: ActorRegistration) -> dict[str, str]:
    return {
        "segment_id": segment,
        "actor_id": registration.actor_id,
        "session_id": registration.session_id,
        "run_fingerprint": registration.fingerprint,
    }


def _policy_version(event: Event) -> int:
    if event.step is None:
        raise VerificationInputError("policy publication has no step")
    version = integer(event.payload.get("policy_version"), "published policy_version")
    if version != event.step:
        raise VerificationInputError("published policy_version does not match its event step")
    return version


def resume_evidence(
    segment_order: list[str], grouped: dict[str, list[Event]], completed: list[Checkpoint]
) -> ResumeFacts:
    context = ResumeContext(grouped, completed)
    evidence: list[ResumeEvidence] = []
    failures: list[str] = []
    for previous, current in pairwise(segment_order):
        result = _resume_boundary(previous, current, context)
        evidence.extend(result.evidence)
        failures.extend(result.failures)
    return ResumeFacts(evidence, failures)


def _resume_boundary(previous: str, current: str, context: ResumeContext) -> ResumeFacts:
    publications = [
        event for event in context.grouped[current] if event.name == "distributed/policy_published"
    ]
    if not publications:
        return ResumeFacts([], [f"{current}: no policy publication"])
    publication = publications[0]
    boundary = ResumeBoundary(
        previous,
        current,
        publication,
        _policy_version(publication),
        context.completed,
    )
    return _matched_resume(boundary)


def _matched_resume(boundary: ResumeBoundary) -> ResumeFacts:
    candidates = _resume_sources(boundary)
    if boundary.version <= 0:
        message = f"{boundary.current}: resumed policy version is not positive"
    elif not candidates:
        message = f"{boundary.current}: no completed checkpoint matches policy {boundary.version}"
    elif not _post_resume_checkpoints(boundary):
        message = f"{boundary.current}: no newer checkpoint completed after resume"
    else:
        evidence = ResumeEvidence(
            boundary.previous,
            boundary.current,
            boundary.version,
            candidates[-1],
        )
        return ResumeFacts([evidence], [])
    return ResumeFacts([], [message])


def _resume_sources(boundary: ResumeBoundary) -> list[Checkpoint]:
    return [
        checkpoint
        for checkpoint in boundary.completed
        if checkpoint.event_index < boundary.publication.index
        and checkpoint.segment_id == boundary.previous
        and checkpoint.step == boundary.version
    ]


def _post_resume_checkpoints(boundary: ResumeBoundary) -> list[Checkpoint]:
    return [
        checkpoint
        for checkpoint in boundary.completed
        if checkpoint.event_index > boundary.publication.index
        and checkpoint.segment_id == boundary.current
        and checkpoint.step > boundary.version
    ]


def checkpoint_artifact(checkpoint: Checkpoint, run_dir: Path) -> dict[str, object]:
    return {
        "path": checkpoint.file.relative_to(run_dir).as_posix(),
        "sha256": sha256(checkpoint.file),
        "size_bytes": checkpoint.file.stat().st_size,
        "update": checkpoint.step,
        "journal_applied_frontier": checkpoint.frontier,
    }


def _latest_post_resume(
    evidence: list[ResumeEvidence], completed: list[Checkpoint]
) -> Checkpoint | None:
    if not evidence:
        return None
    last = evidence[-1]
    newer = [
        checkpoint
        for checkpoint in completed
        if checkpoint.segment_id == last.to_segment
        and checkpoint.step > last.resumed_policy_version
    ]
    return newer[-1] if newer else None


def verified_artifacts(
    evidence: list[ResumeEvidence], completed: list[Checkpoint], run_dir: Path
) -> ArtifactFacts:
    selected = _selected_artifacts(evidence, completed)
    missing = _missing_artifacts(selected, run_dir)
    artifacts = [] if missing else _artifact_records(selected, run_dir)
    return ArtifactFacts(artifacts, missing, _latest_post_resume(evidence, completed))


def _selected_artifacts(
    evidence: list[ResumeEvidence], completed: list[Checkpoint]
) -> dict[Path, tuple[Checkpoint, set[str]]]:
    selected: dict[Path, tuple[Checkpoint, set[str]]] = {}
    for item in evidence:
        checkpoint, roles = selected.get(item.source.file, (item.source, set()))
        roles.add(f"resume-source:{item.to_segment}")
        selected[item.source.file] = (checkpoint, roles)
    latest = _latest_post_resume(evidence, completed)
    if latest is not None:
        checkpoint, roles = selected.get(latest.file, (latest, set()))
        roles.add("latest-post-resume")
        selected[latest.file] = (checkpoint, roles)
    return selected


def _missing_artifacts(
    selected: dict[Path, tuple[Checkpoint, set[str]]], run_dir: Path
) -> list[str]:
    checkpoint_root = (run_dir / "checkpoints").resolve()
    return [
        path.name
        for path in selected
        if not path.is_file() or not path.resolve().is_relative_to(checkpoint_root)
    ]


def _artifact_records(
    selected: dict[Path, tuple[Checkpoint, set[str]]], run_dir: Path
) -> list[dict[str, object]]:
    return [
        {**checkpoint_artifact(checkpoint, run_dir), "roles": sorted(roles)}
        for checkpoint, roles in selected.values()
    ]


def transitions(events: list[Event]) -> list[int]:
    return [
        integer(event.payload.get("transitions"), "ingest transitions")
        for event in events
        if event.name == "distributed/ingest"
    ]


def wal_recoveries(events: list[Event]) -> WalRecoveryFacts:
    records: list[dict[str, int]] = []
    valid = True
    for event in events:
        if event.name != "distributed/wal_recovery":
            continue
        record = _wal_recovery(event)
        valid = _valid_wal_recovery(record) and valid
        records.append(record)
    return WalRecoveryFacts(records, valid)


def _wal_recovery(event: Event) -> dict[str, int]:
    return {
        "from_frontier": integer(event.payload.get("from_frontier"), "WAL recovery from_frontier"),
        "to_frontier": integer(event.payload.get("to_frontier"), "WAL recovery to_frontier"),
        "rows": integer(event.payload.get("rows"), "WAL recovery rows"),
        "transitions": integer(event.payload.get("transitions"), "WAL recovery transitions"),
    }


def _valid_wal_recovery(record: dict[str, int]) -> bool:
    return (
        0 <= record["from_frontier"] <= record["to_frontier"]
        and record["rows"] > 0
        and record["transitions"] > 0
    )
