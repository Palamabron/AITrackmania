from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest

from scripts import verify_soak

RUN_FINGERPRINT = "c" * 64


@dataclass(frozen=True, slots=True)
class BundleOptions:
    first_duration_s: float = 7_200.0
    second_duration_s: float = 7_500.0
    platform: str = "Windows-11-10.0.26100-SP0"
    second_frontier: int = 20
    fatal_event: str | None = None


@dataclass(frozen=True, slots=True)
class SegmentSpec:
    started: datetime
    segment: str
    session: str
    initial_step: int
    final_step: int
    transitions: int
    frontier: int
    duration_s: float


@dataclass(frozen=True, slots=True)
class EventTemplate:
    elapsed_s: float
    name: str
    payload: dict[str, object]
    step: int


class EvidenceBundleBuilder:
    def __init__(self, tmp_path: Path, options: BundleOptions | None = None) -> None:
        self.run_dir = tmp_path / "soak-v1"
        self.options = options or BundleOptions()

    def build(self) -> Path:
        (self.run_dir / "checkpoints").mkdir(parents=True)
        self._write_json("manifest.json", self._manifest())
        self._write_attempts()
        self._write_events()
        self._write_checkpoints()
        self._write_json("evaluation.json", self._evaluation())
        return self.run_dir

    def _write_json(self, name: str, value: dict[str, object]) -> None:
        (self.run_dir / name).write_text(json.dumps(value), encoding="utf-8")

    def _write_jsonl(self, name: str, records: list[dict[str, object]]) -> None:
        content = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
        (self.run_dir / name).write_text(content, encoding="utf-8")

    def _manifest(self) -> dict[str, object]:
        environment = {
            "class_path": "trackmaniarl.trackmania.environment:OpenPlanetEnvironmentFactory"
        }
        config = {
            "components": {"environment": environment},
            "evaluation": self._evaluation_suite(),
        }
        return {
            "api_version": "2.0",
            "run_id": "soak-v1",
            "config": config,
            "evaluation_assets": [self._evaluation_asset()],
        }

    def _evaluation_suite(self) -> dict[str, object]:
        return {
            "name": "release-suite",
            "version": "1",
            "trials_per_map": 1,
            "min_finish_rate": 1.0,
            "target_median_s": 40.0,
        }

    def _evaluation_asset(self) -> dict[str, object]:
        return {
            "map_id": "stadium-a01",
            "map_uid": "map-uid-a01",
            "geometry_sha256": "a" * 64,
            "plugin_protocol_version": "2.4.0",
        }

    def _write_attempts(self) -> None:
        revision = "b" * 40
        attempts: list[dict[str, object]] = [
            {
                "timestamp_utc": "2026-08-26T10:00:00+00:00",
                "environment": {"platform": self.options.platform, "git_revision": revision},
            },
            {
                "timestamp_utc": "2026-08-26T13:00:00+00:00",
                "environment": {"platform": self.options.platform, "git_revision": revision},
            },
        ]
        self._write_jsonl("manifest-attempts.jsonl", attempts)

    def _write_events(self) -> None:
        events = self._segment_events(self._first_segment())
        events.extend(self._segment_events(self._second_segment()))
        events.extend(self._fatal_events())
        self._write_jsonl("events.jsonl", events)

    def _first_segment(self) -> SegmentSpec:
        return SegmentSpec(
            datetime(2026, 8, 26, 10, tzinfo=UTC),
            "segment-a",
            "session-a",
            0,
            100,
            100,
            10,
            self.options.first_duration_s,
        )

    def _second_segment(self) -> SegmentSpec:
        return SegmentSpec(
            datetime(2026, 8, 26, 13, tzinfo=UTC),
            "segment-b",
            "session-b",
            100,
            200,
            200,
            self.options.second_frontier,
            self.options.second_duration_s,
        )

    def _segment_events(self, spec: SegmentSpec) -> list[dict[str, object]]:
        templates = self._event_templates(spec)
        return [self._event(spec, index, item) for index, item in enumerate(templates)]

    def _event_templates(self, spec: SegmentSpec) -> tuple[EventTemplate, ...]:
        initial, final = spec.initial_step, spec.final_step
        policy: dict[str, object] = {"policy_version": initial}
        ingest: dict[str, object] = {"actor_id": "local-actor", "transitions": spec.transitions}
        return (
            EventTemplate(0.0, "distributed/policy_published", policy, initial),
            EventTemplate(1.0, "actor/registered", self._registration(spec), initial),
            EventTemplate(2.0, "distributed/ingest", ingest, final),
            EventTemplate(3.0, "train/update", {"loss/total": 0.5}, final),
            EventTemplate(
                spec.duration_s,
                "train/checkpoint_completed",
                self._checkpoint_payload(spec),
                final,
            ),
        )

    def _registration(self, spec: SegmentSpec) -> dict[str, object]:
        return {
            "actor_id": "local-actor",
            "session_id": spec.session,
            "run_fingerprint": RUN_FINGERPRINT,
        }

    def _checkpoint_payload(self, spec: SegmentSpec) -> dict[str, object]:
        return {
            "path": f"checkpoints/distributed-update-{spec.final_step:08d}.pt",
            "journal_applied_frontier": spec.frontier,
        }

    def _event(self, spec: SegmentSpec, index: int, template: EventTemplate) -> dict[str, object]:
        return {
            "schema_version": "1.0",
            "timestamp_utc": (spec.started + timedelta(seconds=index)).isoformat(),
            "elapsed_s": template.elapsed_s,
            "run_id": "soak-v1",
            "segment_id": spec.segment,
            "event": template.name,
            "payload": template.payload,
            "step": template.step,
        }

    def _fatal_events(self) -> list[dict[str, object]]:
        name = self.options.fatal_event
        if name is None:
            return []
        spec = self._second_segment()
        template = EventTemplate(
            self.options.second_duration_s + 1.0,
            name,
            {"exception_type": "RuntimeError"},
            200,
        )
        return [self._event(spec, 6, template)]

    def _write_checkpoints(self) -> None:
        checkpoints = self.run_dir / "checkpoints"
        (checkpoints / "distributed-update-00000100.pt").write_bytes(b"resume checkpoint")
        (checkpoints / "distributed-update-00000200.pt").write_bytes(b"post-resume checkpoint")

    def _evaluation(self) -> dict[str, object]:
        return {
            "schema_version": "1",
            "plugin_protocol_version": "2.4.0",
            "suite": {"name": "release-suite", "version": "1"},
            "checkpoint": "checkpoints/distributed-update-00000200.pt",
            "metrics": {"eval/finish_rate": 1.0, "eval/median_finish_time_s": 35.0},
            "trials": [self._benchmark_trial()],
        }

    def _benchmark_trial(self) -> dict[str, object]:
        return {
            "map_id": "stadium-a01",
            "map_uid": "map-uid-a01",
            "trial_index": 0,
            "finished": True,
            "finish_time_s": 35.0,
            "telemetry_error": None,
            "controller_error": None,
        }


class ReportView:
    def __init__(self, report: dict[str, object]) -> None:
        self.report = report

    def section(self, name: str) -> dict[str, object]:
        return _json_object(self.report[name])

    def objects(self, name: str) -> list[dict[str, object]]:
        return _json_objects(self.report[name])

    def check(self, name: str) -> bool:
        checks = self.objects("checks")
        match = next(item for item in checks if item["name"] == name)
        return bool(match["passed"])


def _bundle(tmp_path: Path, options: BundleOptions | None = None) -> Path:
    return EvidenceBundleBuilder(tmp_path, options).build()


def _json_object(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    assert all(isinstance(key, str) for key in value)
    return cast(dict[str, object], value)


def _json_objects(value: object) -> list[dict[str, object]]:
    assert isinstance(value, list)
    return [_json_object(item) for item in value]


def _load_json(path: Path) -> dict[str, object]:
    return _json_object(json.loads(path.read_text(encoding="utf-8")))


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [_json_object(json.loads(line)) for line in lines]


INVALID_OPTIONS = (
    (
        BundleOptions(first_duration_s=3_000.0, second_duration_s=3_000.0),
        "minimum_observed_runtime",
    ),
    (BundleOptions(platform="Linux-6.8-x86_64"), "real_windows_attempts"),
    (BundleOptions(second_frontier=5), "checkpoint_frontiers"),
    (BundleOptions(fatal_event="distributed/wal_error"), "no_runtime_failures"),
)
EVIDENCE_FILES = (
    ("checkpoints/distributed-update-00000100.pt", "checkpoint_artifact_hashes"),
    ("evaluation.json", "final_benchmark_artifact"),
)
BENCHMARK_ERROR_KEYS = ("telemetry_error", "controller_error")


def test_verify_soak_writes_passed_report_with_resume_and_hashes(tmp_path: Path) -> None:
    run_dir = _bundle(tmp_path)
    report = verify_soak.verify_run(run_dir)
    view = ReportView(report)

    assert report["status"] == "passed"
    _assert_run_identity(view)
    _assert_resume(view)
    digest = hashlib.sha256(b"post-resume checkpoint").hexdigest()
    assert view.section("benchmark")["checkpoint_sha256"] == digest
    assert view.check("benchmark_trial_health")
    assert view.check("benchmark_checkpoint_binding")
    assert view.section("durability")["checkpoint_frontiers"] == [10, 20]
    assert _load_json(run_dir / "soak-report.json")["status"] == "passed"


def _assert_run_identity(view: ReportView) -> None:
    assert view.section("run")["observed_runtime_s"] == 14_700.0
    identities = view.section("identities")
    assert identities["stable_actor_ids"] == ["local-actor"]
    assert identities["run_fingerprint"] == RUN_FINGERPRINT


def _assert_resume(view: ReportView) -> None:
    resume = view.objects("resume")[0]
    checkpoint = _json_object(resume["matched_checkpoint"])
    assert resume["resumed_policy_version"] == 100
    assert checkpoint["journal_applied_frontier"] == 10
    assert checkpoint["sha256"] == hashlib.sha256(b"resume checkpoint").hexdigest()


def test_verify_soak_rejects_fingerprint_change_across_resume(tmp_path: Path) -> None:
    run_dir = _bundle(tmp_path)
    path = run_dir / "events.jsonl"
    events = _load_jsonl(path)
    registrations = [event for event in events if event["event"] == "actor/registered"]
    _json_object(registrations[-1]["payload"])["run_fingerprint"] = "d" * 64
    EvidenceBundleBuilder(tmp_path)._write_jsonl("events.jsonl", events)

    report = verify_soak.verify_run(run_dir)
    view = ReportView(report)
    assert report["status"] == "failed"
    assert view.section("identities")["run_fingerprint"] is None
    assert not view.check("run_fingerprint_identity")


def _assert_invalid_run_fact(tmp_path: Path, options: BundleOptions, check_name: str) -> None:
    report = verify_soak.verify_run(_bundle(tmp_path, options))
    assert report["status"] == "failed"
    assert not ReportView(report).check(check_name)


def test_verify_soak_rejects_invalid_run_facts(tmp_path: Path) -> None:
    for options, check_name in INVALID_OPTIONS:
        _assert_invalid_run_fact(tmp_path / check_name, options, check_name)


def _assert_required_evidence_file(tmp_path: Path, relative_path: str, check_name: str) -> None:
    run_dir = _bundle(tmp_path)
    (run_dir / relative_path).unlink()
    report = verify_soak.verify_run(run_dir)

    assert report["status"] == "failed"
    assert not ReportView(report).check(check_name)


def test_verify_soak_requires_evidence_files(tmp_path: Path) -> None:
    for relative_path, check_name in EVIDENCE_FILES:
        _assert_required_evidence_file(tmp_path / check_name, relative_path, check_name)


def _assert_benchmark_error_rejected(tmp_path: Path, error_key: str) -> None:
    run_dir = _bundle(tmp_path)
    path = run_dir / "evaluation.json"
    evaluation = _load_json(path)
    _json_objects(evaluation["trials"])[0][error_key] = "RuntimeError"
    path.write_text(json.dumps(evaluation), encoding="utf-8")
    report = verify_soak.verify_run(run_dir)

    assert report["status"] == "failed"
    assert not ReportView(report).check("benchmark_trial_health")
    assert ReportView(report).section("benchmark")["error_trial_indices"] == [0]


def test_verify_soak_rejects_every_benchmark_error(tmp_path: Path) -> None:
    for error_key in BENCHMARK_ERROR_KEYS:
        _assert_benchmark_error_rejected(tmp_path / error_key, error_key)


def test_verify_soak_rejects_benchmark_bound_to_resume_source(tmp_path: Path) -> None:
    run_dir = _bundle(tmp_path)
    path = run_dir / "evaluation.json"
    evaluation = _load_json(path)
    evaluation["checkpoint"] = "checkpoints/distributed-update-00000100.pt"
    path.write_text(json.dumps(evaluation), encoding="utf-8")

    report = verify_soak.verify_run(run_dir)

    assert report["status"] == "failed"
    assert not ReportView(report).check("benchmark_checkpoint_binding")


def test_verify_soak_cli_returns_failure_and_preserves_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _bundle(tmp_path, BundleOptions(platform="Linux-6.8-x86_64"))
    monkeypatch.setattr(sys, "argv", ["verify_soak.py", str(run_dir)])

    with pytest.raises(SystemExit) as error:
        verify_soak.main()

    assert error.value.code == 1
    assert _load_json(run_dir / "soak-report.json")["status"] == "failed"
