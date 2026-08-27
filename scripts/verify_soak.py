"""Verify the evidence bundle from a resumed real-game Windows soak run."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING or __name__ != "__main__":
    from scripts.soak_report import verify_run, write_report
    from scripts.soak_types import (
        MINIMUM_HOURS,
        REPORT_SCHEMA,
        Check,
        Checkpoint,
        Event,
        ResumeEvidence,
        VerificationInputError,
    )
else:
    from soak_report import verify_run, write_report
    from soak_types import (
        MINIMUM_HOURS,
        REPORT_SCHEMA,
        Check,
        Checkpoint,
        Event,
        ResumeEvidence,
        VerificationInputError,
    )

__all__ = [
    "Check",
    "Checkpoint",
    "Event",
    "ResumeEvidence",
    "VerificationInputError",
    "main",
    "verify_run",
]


@dataclass(frozen=True, slots=True)
class SoakArguments:
    run_dir: Path
    minimum_hours: float
    output: Path | None


def _parse_args() -> SoakArguments:
    parser = argparse.ArgumentParser(
        description="Verify a stopped, resumed TrackmaniaRL Windows soak evidence directory."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--minimum-hours", type=float, default=MINIMUM_HOURS)
    parser.add_argument("--output", type=Path)
    values = parser.parse_args()
    return SoakArguments(values.run_dir, values.minimum_hours, values.output)


def main() -> None:
    args = _parse_args()
    output = _output_path(args)
    try:
        report = verify_run(args.run_dir, minimum_hours=args.minimum_hours, output=output)
    except VerificationInputError as error:
        report = _input_error_report(error)
        write_report(report, output)
    if report["status"] != "passed":
        raise SystemExit(1)


def _output_path(args: SoakArguments) -> Path:
    if args.output is not None:
        return args.output.resolve()
    return args.run_dir.resolve() / "soak-report.json"


def _input_error_report(error: VerificationInputError) -> dict[str, object]:
    return {
        "schema_version": REPORT_SCHEMA,
        "status": "failed",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input_error": str(error),
        "checks": [],
    }


if __name__ == "__main__":
    main()
