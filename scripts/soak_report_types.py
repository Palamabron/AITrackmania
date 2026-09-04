from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING or __package__:
    from scripts.soak_types import (
        REPORT_SCHEMA,
        Check,
        Checkpoint,
        Event,
        ResumeEvidence,
        sha256,
        string,
    )
else:
    from soak_types import (
        REPORT_SCHEMA,
        Check,
        Checkpoint,
        Event,
        ResumeEvidence,
        sha256,
        string,
    )


@dataclass(frozen=True, slots=True)
class ActorRegistration:
    actor_id: str
    session_id: str
    fingerprint: str


@dataclass(slots=True)
class IdentityCollector:
    sessions: dict[str, tuple[str, str]] = field(default_factory=dict)
    fingerprints: set[str] = field(default_factory=set)
    registrations: list[dict[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class SegmentIdentityState:
    segment: str
    collector: IdentityCollector
    registrations: set[tuple[str, str]] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class ResumeContext:
    grouped: dict[str, list[Event]]
    completed: list[Checkpoint]


@dataclass(frozen=True, slots=True)
class ResumeBoundary:
    previous: str
    current: str
    publication: Event
    version: int
    completed: list[Checkpoint]


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    run_dir: Path
    manifest: dict[str, object]
    attempts: list[dict[str, object]]
    events: list[Event]
    evaluation: dict[str, object] | None
    assets: list[dict[str, object]]
    segment_order: list[str]
    grouped_events: dict[str, list[Event]]


@dataclass(frozen=True, slots=True)
class AttemptFacts:
    platforms: list[str]
    revisions: list[str]


@dataclass(frozen=True, slots=True)
class IdentityFacts:
    registrations: list[dict[str, str]]
    stable_actors: set[str]
    fingerprints: set[str]
    sessions_valid: bool
    fingerprint_valid: bool


@dataclass(frozen=True, slots=True)
class ResumeFacts:
    evidence: list[ResumeEvidence]
    failures: list[str]


@dataclass(frozen=True, slots=True)
class ArtifactFacts:
    artifacts: list[dict[str, object]]
    missing: list[str]
    final_checkpoint: Checkpoint | None


@dataclass(frozen=True, slots=True)
class DurabilityMeasurements:
    frontiers: list[int]
    recoveries: list[dict[str, int]]
    transition_counts: list[int]


@dataclass(frozen=True, slots=True)
class WalRecoveryFacts:
    records: list[dict[str, int]]
    valid: bool


@dataclass(frozen=True, slots=True)
class DurabilityFacts:
    completed: list[Checkpoint]
    frontiers: list[int]
    recoveries: list[dict[str, int]]
    transition_counts: list[int]
    artifact_facts: ArtifactFacts


@dataclass(frozen=True, slots=True)
class RuntimeFacts:
    observed_s: float
    minimum_s: float
    attempts: AttemptFacts


@dataclass(frozen=True, slots=True)
class RecoveryFacts:
    resume: ResumeFacts
    benchmark: dict[str, object]
    durability: DurabilityFacts


@dataclass(frozen=True, slots=True)
class ReportFacts:
    bundle: EvidenceBundle
    checks: list[Check]
    runtime: RuntimeFacts
    identities: IdentityFacts
    recovery: RecoveryFacts
    failures: list[Event]


def build_report(facts: ReportFacts) -> dict[str, object]:
    return {
        "schema_version": REPORT_SCHEMA,
        "status": "passed" if all(check.passed for check in facts.checks) else "failed",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_files": _source_files(facts.bundle.run_dir),
        "run": _run_report(facts),
        "identities": _identity_report(facts),
        "resume": _resume_report(facts),
        "benchmark": facts.recovery.benchmark,
        "durability": _durability_report(facts),
        "failure_events": _failure_report(facts),
        "checks": [check.as_json() for check in facts.checks],
    }


def _source_files(run_dir: Path) -> dict[str, dict[str, object]]:
    sources: dict[str, dict[str, object]] = {}
    names = ("manifest.json", "manifest-attempts.jsonl", "events.jsonl", "evaluation.json")
    for name in names:
        path = run_dir / name
        sources[name] = _source_file(path)
    return sources


def _source_file(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {"present": False}
    return {"present": True, "sha256": sha256(path), "size_bytes": path.stat().st_size}


def _resume_report(facts: ReportFacts) -> list[dict[str, object]]:
    artifacts = facts.recovery.durability.artifact_facts.artifacts
    checkpoint_by_path = {item["path"]: item for item in artifacts}
    return [
        {
            "from_segment": item.from_segment,
            "to_segment": item.to_segment,
            "resumed_policy_version": item.resumed_policy_version,
            "matched_checkpoint": checkpoint_by_path.get(
                item.source.file.relative_to(facts.bundle.run_dir).as_posix(),
                {"path": item.source.file.relative_to(facts.bundle.run_dir).as_posix()},
            ),
        }
        for item in facts.recovery.resume.evidence
    ]


def _identity_report(facts: ReportFacts) -> dict[str, object]:
    identities = facts.identities
    fingerprint = next(iter(identities.fingerprints)) if identities.fingerprint_valid else None
    return {
        "segment_ids": facts.bundle.segment_order,
        "stable_actor_ids": sorted(identities.stable_actors),
        "run_fingerprint": fingerprint,
        "actor_registrations": identities.registrations,
    }


def _durability_report(facts: ReportFacts) -> dict[str, object]:
    durability = facts.recovery.durability
    transitions = durability.transition_counts
    return {
        "completed_checkpoint_count": len(durability.completed),
        "checkpoint_frontiers": durability.frontiers,
        "verified_checkpoint_artifacts": durability.artifact_facts.artifacts,
        "wal_recoveries": durability.recoveries,
        "last_transition_count": transitions[-1] if transitions else 0,
    }


def _failure_report(facts: ReportFacts) -> list[dict[str, object]]:
    return [
        {
            "event_index": event.index,
            "event": event.name,
            "timestamp_utc": event.timestamp.isoformat(),
            "segment_id": event.segment_id,
        }
        for event in facts.failures
    ]


def _run_report(facts: ReportFacts) -> dict[str, object]:
    bundle = facts.bundle
    attempts = facts.runtime.attempts
    return {
        "run_id": string(bundle.manifest.get("run_id"), "manifest.run_id"),
        "api_version": bundle.manifest.get("api_version"),
        "observed_runtime_s": facts.runtime.observed_s,
        "observed_runtime_hours": facts.runtime.observed_s / 3600.0,
        "minimum_runtime_s": facts.runtime.minimum_s,
        "attempt_count": len(bundle.attempts),
        "segment_count": len(bundle.segment_order),
        "platforms": sorted(set(attempts.platforms)),
        "git_revisions": sorted(set(attempts.revisions)),
        "evaluation_assets": bundle.assets,
    }
