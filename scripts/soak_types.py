from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

REPORT_SCHEMA = "trackmaniarl-soak-report-v1"
EVENT_SCHEMA = "1.0"
RUN_API_VERSION = "2.0"
MINIMUM_HOURS = 4.0
LIVE_ENVIRONMENT = "trackmaniarl.trackmania.environment:OpenPlanetEnvironmentFactory"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GIT_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
FATAL_EVENTS = frozenset(
    {
        "actor/timeout",
        "distributed/rollout_rejected",
        "distributed/wal_error",
        "run/failure",
        "train/checkpoint_failed",
    }
)
TELEMETRY_FAILURE_KEYS = frozenset(
    {
        "telemetry/error",
        "telemetry_error",
        "telemetry_error_rate",
        "termination/telemetry_error",
    }
)


class VerificationInputError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Event:
    index: int
    timestamp: datetime
    elapsed_s: float
    run_id: str
    segment_id: str
    name: str
    payload: dict[str, object]
    step: int | None


@dataclass(frozen=True, slots=True)
class Checkpoint:
    event_index: int
    timestamp: datetime
    segment_id: str
    step: int
    frontier: int
    reported_path: str
    file: Path


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    passed: bool
    detail: str

    def as_json(self) -> dict[str, object]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class ResumeEvidence:
    from_segment: str
    to_segment: str
    resumed_policy_version: int
    source: Checkpoint


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise VerificationInputError(f"{label} must be a JSON object with string keys")
    return cast(dict[str, object], value)


def string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise VerificationInputError(f"{label} must be a non-empty string")
    return value


def integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise VerificationInputError(f"{label} must be an integer")
    return value


def number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VerificationInputError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise VerificationInputError(f"{label} must be finite")
    return result


def timestamp(value: object, label: str) -> datetime:
    text = string(value, label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise VerificationInputError(f"{label} must be an ISO-8601 timestamp") from error
    if parsed.utcoffset() != timedelta(0):
        raise VerificationInputError(f"{label} must include a UTC offset")
    return parsed.astimezone(UTC)


def load_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise VerificationInputError(f"missing evidence file: {path.name}")
    try:
        value = cast(object, json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as error:
        raise VerificationInputError(f"{path.name} is not valid JSON") from error
    return mapping(value, path.name)


def load_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise VerificationInputError(f"missing evidence file: {path.name}")
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = cast(object, json.loads(line))
        except json.JSONDecodeError as error:
            raise VerificationInputError(f"{path.name}:{line_number} is not valid JSON") from error
        records.append(mapping(value, f"{path.name}:{line_number}"))
    if not records:
        raise VerificationInputError(f"{path.name} contains no records")
    return records


def parse_event(record: dict[str, object], index: int) -> Event:
    label = f"events.jsonl:{index + 1}"
    if record.get("schema_version") != EVENT_SCHEMA:
        raise VerificationInputError(f"{label} has an unsupported schema_version")
    step_value = record.get("step")
    step = None if step_value is None else integer(step_value, f"{label}.step")
    return Event(
        index=index,
        timestamp=timestamp(record.get("timestamp_utc"), f"{label}.timestamp_utc"),
        elapsed_s=number(record.get("elapsed_s"), f"{label}.elapsed_s"),
        run_id=string(record.get("run_id"), f"{label}.run_id"),
        segment_id=string(record.get("segment_id"), f"{label}.segment_id"),
        name=string(record.get("event"), f"{label}.event"),
        payload=mapping(record.get("payload"), f"{label}.payload"),
        step=step,
    )


def events(records: list[dict[str, object]]) -> list[Event]:
    return [parse_event(record, index) for index, record in enumerate(records)]


def manifest_environment(manifest: dict[str, object]) -> str:
    config = mapping(manifest.get("config"), "manifest.config")
    components = mapping(config.get("components"), "manifest.config.components")
    environment = mapping(components.get("environment"), "manifest environment component")
    return string(environment.get("class_path"), "manifest environment class_path")


def evaluation_assets(manifest: dict[str, object]) -> list[dict[str, object]]:
    value = manifest.get("evaluation_assets")
    if not isinstance(value, list):
        raise VerificationInputError("manifest.evaluation_assets must be a list")
    return [
        mapping(item, f"manifest.evaluation_assets[{index}]") for index, item in enumerate(value)
    ]


def valid_evaluation_asset(asset: dict[str, object]) -> bool:
    map_uid = asset.get("map_uid")
    geometry = asset.get("geometry_sha256")
    protocol = asset.get("plugin_protocol_version")
    return (
        isinstance(map_uid, str)
        and bool(map_uid)
        and isinstance(geometry, str)
        and SHA256_PATTERN.fullmatch(geometry) is not None
        and isinstance(protocol, str)
        and bool(protocol)
    )


def attempt_environment(record: dict[str, object], index: int) -> dict[str, object]:
    timestamp(record.get("timestamp_utc"), f"manifest-attempts.jsonl:{index + 1}.timestamp_utc")
    return mapping(record.get("environment"), f"manifest-attempts.jsonl:{index + 1}.environment")


def add_check(checks: list[Check], check: Check) -> None:
    checks.append(check)
