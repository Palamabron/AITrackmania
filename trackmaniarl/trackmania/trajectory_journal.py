"""Append-only journal serialization for trajectory search trials."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from trackmaniarl.trackmania.trajectory_optimization import (
        TrajectorySearchOutcome,
        TrajectorySearchRecord,
    )


def append_trajectory_record(path: Path, record: TrajectorySearchRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_record_payload(record), sort_keys=True) + "\n")


def _record_payload(record: TrajectorySearchRecord) -> dict[str, Any]:
    return {
        "trial": record.trial,
        "kind": record.kind,
        "window": (
            [record.window.first_segment, record.window.stop_segment]
            if record.window is not None
            else None
        ),
        "side": record.side,
        "ticks": record.ticks,
        "accepted": record.accepted,
        "outcome": _outcome_payload(record.outcome),
    }


def _outcome_payload(outcome: TrajectorySearchOutcome) -> dict[str, Any]:
    return {
        "finished": outcome.finished,
        "finish_time_s": outcome.finish_time_s,
        "progress_pct": outcome.progress_pct,
        "error": outcome.error,
    }
