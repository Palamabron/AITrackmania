"""Progress reporting for offline expert diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from trackmaniarl.trackmania.diagnostics import aggregate_expert_actions


@dataclass(frozen=True, slots=True)
class ExpertProgressReporter:
    logger: Any
    demonstration_count: int
    started_at: float = field(default_factory=perf_counter)

    def report(self, reports: list[dict[str, Any]]) -> None:
        payload = self._payload(reports)
        self.logger.log("diagnose/expert_progress", payload, step=int(payload["count"]))
        print(self._message(payload), flush=True)

    def _payload(self, reports: list[dict[str, Any]]) -> dict[str, Any]:
        completed = len(reports)
        elapsed = perf_counter() - self.started_at
        rate = completed / max(elapsed, 1e-9)
        actions = aggregate_expert_actions(report["actions"] for report in reports)
        return {
            **actions,
            "demonstrations/completed": completed,
            "demonstrations/count": self.demonstration_count,
            "elapsed_s": elapsed,
            "eta_s": (self.demonstration_count - completed) / rate,
        }

    @staticmethod
    def _message(payload: dict[str, Any]) -> str:
        return (
            "Expert diagnostics: "
            f"demos={payload['demonstrations/completed']}/{payload['demonstrations/count']}, "
            f"transitions={int(payload['count'])}, "
            f"exact={payload['exact_action_accuracy']:.3f}, "
            f"steering={payload['steering_bin_accuracy']:.3f}, "
            f"switch={payload['expert_steering_switch_step_accuracy']:.3f}, "
            f"ETA={payload['eta_s']:.1f}s"
        )
