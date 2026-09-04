from __future__ import annotations

import json
import math
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING or __package__:
    from scripts.soak_benchmark import BenchmarkContext, benchmark_evidence
    from scripts.soak_evidence import (
        actor_identities,
        checkpoints,
        failure_events,
        resume_evidence,
        segment_runtime,
        segments,
        transitions,
        verified_artifacts,
        wal_recoveries,
    )
    from scripts.soak_report_types import (
        ArtifactFacts,
        AttemptFacts,
        DurabilityFacts,
        DurabilityMeasurements,
        EvidenceBundle,
        IdentityFacts,
        RecoveryFacts,
        ReportFacts,
        ResumeFacts,
        RuntimeFacts,
        WalRecoveryFacts,
        build_report,
    )
    from scripts.soak_types import (
        GIT_REVISION_PATTERN,
        LIVE_ENVIRONMENT,
        MINIMUM_HOURS,
        RUN_API_VERSION,
        Check,
        Checkpoint,
        Event,
        VerificationInputError,
        add_check,
        attempt_environment,
        evaluation_assets,
        events,
        load_json,
        load_jsonl,
        manifest_environment,
        string,
        valid_evaluation_asset,
    )
else:
    from soak_benchmark import BenchmarkContext, benchmark_evidence
    from soak_evidence import (
        actor_identities,
        checkpoints,
        failure_events,
        resume_evidence,
        segment_runtime,
        segments,
        transitions,
        verified_artifacts,
        wal_recoveries,
    )
    from soak_report_types import (
        ArtifactFacts,
        AttemptFacts,
        DurabilityFacts,
        DurabilityMeasurements,
        EvidenceBundle,
        IdentityFacts,
        RecoveryFacts,
        ReportFacts,
        ResumeFacts,
        RuntimeFacts,
        WalRecoveryFacts,
        build_report,
    )
    from soak_types import (
        GIT_REVISION_PATTERN,
        LIVE_ENVIRONMENT,
        MINIMUM_HOURS,
        RUN_API_VERSION,
        Check,
        Checkpoint,
        Event,
        VerificationInputError,
        add_check,
        attempt_environment,
        evaluation_assets,
        events,
        load_json,
        load_jsonl,
        manifest_environment,
        string,
        valid_evaluation_asset,
    )


def write_report(report: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(output)


def _attempt_facts(attempts: list[dict[str, object]], checks: list[Check]) -> AttemptFacts:
    environments = [attempt_environment(attempt, index) for index, attempt in enumerate(attempts)]
    platforms = [string(item.get("platform"), "attempt platform") for item in environments]
    revisions = [string(item.get("git_revision"), "attempt git_revision") for item in environments]
    windows = all(platform.startswith("Windows-") for platform in platforms)
    one_revision = len(set(revisions)) == 1 and all(
        GIT_REVISION_PATTERN.fullmatch(item) for item in revisions
    )
    windows_check = Check("real_windows_attempts", windows, f"platforms={sorted(set(platforms))}")
    revision_check = Check(
        "same_git_revision",
        one_revision,
        f"unique_revisions={len(set(revisions))}",
    )
    add_check(checks, windows_check)
    add_check(checks, revision_check)
    return AttemptFacts(platforms, revisions)


def _basic_checks(bundle: EvidenceBundle, checks: list[Check]) -> None:
    for check in (
        _api_version_check(bundle.manifest),
        _run_identity_check(bundle.manifest, bundle.events),
        _environment_check(bundle.manifest),
        _asset_check(bundle.assets),
        _timestamp_check(bundle.events),
    ):
        add_check(checks, check)


def _api_version_check(manifest: dict[str, object]) -> Check:
    return Check("run_api_version", manifest.get("api_version") == RUN_API_VERSION, "expected 2.0")


def _run_identity_check(manifest: dict[str, object], run_events: list[Event]) -> Check:
    run_id = string(manifest.get("run_id"), "manifest.run_id")
    return Check(
        "run_identity", all(event.run_id == run_id for event in run_events), f"run_id={run_id}"
    )


def _environment_check(manifest: dict[str, object]) -> Check:
    live = manifest_environment(manifest) == LIVE_ENVIRONMENT
    return Check("live_openplanet_environment", live, f"expected {LIVE_ENVIRONMENT}")


def _asset_check(assets: list[dict[str, object]]) -> Check:
    valid = bool(assets) and all(valid_evaluation_asset(asset) for asset in assets)
    return Check("evaluation_assets", valid, f"validated_assets={len(assets)}")


def _timestamp_check(run_events: list[Event]) -> Check:
    timestamps = [event.timestamp for event in run_events]
    ordered = all(first <= second for first, second in pairwise(timestamps))
    return Check("event_timestamp_order", ordered, f"events={len(run_events)}")


def _segment_checks(bundle: EvidenceBundle, checks: list[Check]) -> float:
    missing = _missing_segment_events(bundle)
    elapsed_order = _segment_elapsed_order(bundle)
    runtime_s = sum(
        segment_runtime(bundle.grouped_events[segment]) for segment in bundle.segment_order
    )
    identity_detail = f"attempts={len(bundle.attempts)}, segments={len(bundle.segment_order)}"
    add_check(
        checks,
        Check(
            "attempt_segment_identity",
            len(bundle.attempts) == len(bundle.segment_order),
            identity_detail,
        ),
    )
    add_check(checks, Check("segment_event_coverage", not missing, f"missing={missing}"))
    detail = "elapsed_s is per process segment"
    add_check(checks, Check("segment_elapsed_order", elapsed_order, detail))
    return runtime_s


def _missing_segment_events(bundle: EvidenceBundle) -> dict[str, list[str]]:
    required = {
        "actor/registered",
        "distributed/ingest",
        "distributed/policy_published",
        "train/update",
    }
    return {
        segment: sorted(required - {event.name for event in bundle.grouped_events[segment]})
        for segment in bundle.segment_order
        if required - {event.name for event in bundle.grouped_events[segment]}
    }


def _segment_elapsed_order(bundle: EvidenceBundle) -> bool:
    return all(
        _events_have_ordered_elapsed(bundle.grouped_events[segment])
        for segment in bundle.segment_order
    )


def _events_have_ordered_elapsed(run_events: list[Event]) -> bool:
    ordered = all(first.elapsed_s <= second.elapsed_s for first, second in pairwise(run_events))
    return ordered and all(event.elapsed_s >= 0.0 for event in run_events)


def _durability_checks(
    run_events: list[Event], completed: list[Checkpoint], checks: list[Check]
) -> DurabilityMeasurements:
    transition_counts = transitions(run_events)
    frontiers = [checkpoint.frontier for checkpoint in completed]
    recovery = wal_recoveries(run_events)
    facts = DurabilityMeasurements(frontiers, recovery.records, transition_counts)
    for check in _durability_results(facts, completed, recovery):
        add_check(checks, check)
    return facts


def _durability_results(
    facts: DurabilityMeasurements,
    completed: list[Checkpoint],
    recovery: WalRecoveryFacts,
) -> tuple[Check, Check, Check]:
    last = facts.transition_counts[-1] if facts.transition_counts else 0
    transitions_valid = _monotonic_transitions(facts.transition_counts)
    checkpoints_valid = _monotonic_checkpoints(completed, facts.frontiers)
    return (
        Check("monotonic_ingest_transitions", transitions_valid, f"last={last}"),
        Check("checkpoint_frontiers", checkpoints_valid, f"completed={len(completed)}"),
        Check("wal_recovery_ranges", recovery.valid, f"recoveries={len(facts.recoveries)}"),
    )


def _monotonic_transitions(values: list[int]) -> bool:
    return (
        bool(values)
        and values[-1] > 0
        and all(first <= second for first, second in pairwise(values))
    )


def _monotonic_checkpoints(completed: list[Checkpoint], frontiers: list[int]) -> bool:
    steps = [checkpoint.step for checkpoint in completed]
    return (
        len(completed) >= 2
        and all(value >= 0 for value in frontiers)
        and all(first <= second for first, second in pairwise(steps))
        and all(first <= second for first, second in pairwise(frontiers))
    )


def verify_run(
    run_dir: Path, *, minimum_hours: float = MINIMUM_HOURS, output: Path | None = None
) -> dict[str, object]:
    minimum_runtime_s = _minimum_runtime(minimum_hours)
    bundle = _load_evidence(run_dir.resolve())
    checks: list[Check] = []
    _basic_checks(bundle, checks)
    runtime = _runtime_facts(bundle, checks, minimum_runtime_s)
    identities = _identity_facts(bundle, checks)
    failures = failure_events(bundle.events)
    add_check(checks, Check("no_runtime_failures", not failures, f"failure_events={len(failures)}"))
    recovery = _recovery_facts(bundle, checks)
    facts = ReportFacts(bundle, checks, runtime, identities, recovery, failures)
    report = build_report(facts)
    destination = output.resolve() if output is not None else bundle.run_dir / "soak-report.json"
    write_report(report, destination)
    return report


def _minimum_runtime(minimum_hours: float) -> float:
    if minimum_hours < MINIMUM_HOURS or not math.isfinite(minimum_hours):
        raise VerificationInputError("minimum_hours must be finite and at least 4.0")
    return minimum_hours * 3600.0


def _load_evidence(run_dir: Path) -> EvidenceBundle:
    manifest = load_json(run_dir / "manifest.json")
    attempts = load_jsonl(run_dir / "manifest-attempts.jsonl")
    run_events = events(load_jsonl(run_dir / "events.jsonl"))
    evaluation_path = run_dir / "evaluation.json"
    evaluation = load_json(evaluation_path) if evaluation_path.is_file() else None
    assets = evaluation_assets(manifest)
    segment_order, grouped = segments(run_events)
    return EvidenceBundle(
        run_dir,
        manifest,
        attempts,
        run_events,
        evaluation,
        assets,
        segment_order,
        grouped,
    )


def _runtime_facts(
    bundle: EvidenceBundle, checks: list[Check], minimum_runtime_s: float
) -> RuntimeFacts:
    attempt_facts = _attempt_facts(bundle.attempts, checks)
    runtime_s = _segment_checks(bundle, checks)
    detail = f"observed_s={runtime_s:.3f}, required_s={minimum_runtime_s:.3f}"
    check = Check("minimum_observed_runtime", runtime_s >= minimum_runtime_s, detail)
    add_check(checks, check)
    return RuntimeFacts(runtime_s, minimum_runtime_s, attempt_facts)


def _identity_facts(bundle: EvidenceBundle, checks: list[Check]) -> IdentityFacts:
    facts = actor_identities(bundle.segment_order, bundle.grouped_events)
    for check in _identity_checks(facts):
        add_check(checks, check)
    return facts


def _identity_checks(identities: IdentityFacts) -> tuple[Check, Check, Check]:
    stable = Check(
        "stable_actor_identity",
        bool(identities.stable_actors),
        f"stable_actor_ids={sorted(identities.stable_actors)}",
    )
    sessions = Check(
        "fresh_actor_sessions",
        identities.sessions_valid,
        f"registrations={len(identities.registrations)}",
    )
    fingerprints = Check(
        "run_fingerprint_identity",
        identities.fingerprint_valid,
        f"unique_fingerprints={len(identities.fingerprints)}",
    )
    return stable, sessions, fingerprints


def _recovery_facts(bundle: EvidenceBundle, checks: list[Check]) -> RecoveryFacts:
    completed = checkpoints(bundle.run_dir, bundle.events)
    measured = _durability_checks(bundle.events, completed, checks)
    resume = resume_evidence(bundle.segment_order, bundle.grouped_events, completed)
    _record_resume_check(resume, bundle.segment_order, checks)
    artifacts = verified_artifacts(resume.evidence, completed, bundle.run_dir)
    _record_artifact_check(artifacts, checks)
    benchmark = _benchmark_result(bundle, artifacts.final_checkpoint, checks)
    durability = DurabilityFacts(
        completed,
        measured.frontiers,
        measured.recoveries,
        measured.transition_counts,
        artifacts,
    )
    return RecoveryFacts(resume, benchmark, durability)


def _record_resume_check(
    resume: ResumeFacts, segment_order: list[str], checks: list[Check]
) -> None:
    expected = max(0, len(segment_order) - 1)
    passed = len(resume.evidence) == expected and expected >= 1 and not resume.failures
    detail = f"boundaries={len(resume.evidence)}/{expected}; failures={resume.failures}"
    add_check(checks, Check("checkpoint_resume", passed, detail))


def _record_artifact_check(artifacts: ArtifactFacts, checks: list[Check]) -> None:
    passed = bool(artifacts.artifacts) and not artifacts.missing
    detail = f"hashed={len(artifacts.artifacts)}, missing={artifacts.missing}"
    add_check(checks, Check("checkpoint_artifact_hashes", passed, detail))


def _benchmark_result(
    bundle: EvidenceBundle, final_checkpoint: Checkpoint | None, checks: list[Check]
) -> dict[str, object]:
    context = BenchmarkContext(
        bundle.evaluation,
        bundle.manifest,
        bundle.assets,
        final_checkpoint,
        bundle.run_dir,
        checks,
    )
    return benchmark_evidence(context)
