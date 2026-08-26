"""Optional remote tracker adapters; local JSONL remains the source of truth."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from queue import Full, Queue
from threading import Lock, Thread
from time import monotonic
from typing import Any
from uuid import uuid4

_UPDATE_METRICS = {
    "replay_size": "replay/size",
    "replay_fill_fraction": "replay/fill_fraction",
    "update_credit": "pipeline/update_credit",
    "rollout_queue_depth": "pipeline/rollout_queue_depth",
    "updates_per_s": "performance/updates_per_s",
    "transitions_per_s": "performance/transitions_per_s",
    "cumulative_transitions_per_s": "performance/cumulative_transitions_per_s",
    "target_updates_per_s": "performance/target_updates_per_s",
    "update_throughput_ratio": "performance/update_throughput_ratio",
    "update_backlog_s": "pipeline/update_backlog_s",
    "episodes": "outcome/train_episodes",
    "finish_rate": "outcome/train_finish_rate",
    "per_beta": "replay/per_beta",
    "accelerator_memory_bytes": "system/torch_cuda_memory_bytes",
}
_DEBUG_METRICS = {
    "action_batch_entropy",
    "action_batch_unique_fraction",
    "action_entropy",
    "action_unique_fraction",
    "bootstrap_discount_mean",
    "bootstrap_zero_fraction",
    "demonstration_action_accuracy",
    "gradient_clip_coefficient",
    "gradient_clipped_fraction",
    "gradient_norm",
    "gradient_norm_max",
    "importance_weight_mean",
    "importance_weight_min",
    "n_step_return_mean",
    "q_selected_abs_max",
    "q_selected_max",
    "q_selected_mean",
    "q_selected_std_mean",
    "q_target_max",
    "q_target_mean",
    "reward_abs_max",
    "reward_mean",
    "target_abs_max",
    "target_mean",
    "target_std_mean",
    "target_synced_fraction",
    "td_abs_max",
    "td_abs_mean",
}
_TIMING_METRICS = {
    "backward_s",
    "checkpoint_snapshot_s",
    "forward_s",
    "gradient_clip_s",
    "host_to_device_s",
    "learner_update_s",
    "logging_s",
    "optimizer_s",
    "policy_publish_s",
    "replay_sample_s",
    "replay_wait_s",
    "update_s",
}
_EPISODE_METRICS = {
    "collision/count",
    "collision/detected_count",
    "control/brake_fraction",
    "control/brake_tap_fraction",
    "control/gas_fraction",
    "control/steer_abs_mean",
    "duration_s",
    "exploration_epsilon",
    "finish_rate",
    "finish_time_s",
    "finished",
    "index",
    "policy_version",
    "pace/reference_time_s",
    "pace/time_debt_s",
    "potential/progress",
    "progress/accepted_delta_m",
    "progress_m",
    "progress/nearest_distance_m",
    "progress_pct",
    "progress/steps_since",
    "progress/window_m",
    "q_margin/mean",
    "q_margin/min",
    "q_margin/start_mean",
    "race_time_s",
    "return",
    "reward/collision",
    "reward/pace",
    "reward/pbrs",
    "reward/progress",
    "reward/projected_speed",
    "reward/projected_velocity",
    "reward/steering_delta",
    "reward/terminal",
    "reward/time",
    "reward/time_attack_terminal",
    "reward_per_transition",
    "steps",
    "telemetry/error",
    "termination/max_steps",
    "termination/no_progress",
    "termination/off_track",
    "termination/slow_progress",
    "termination/telemetry_error",
    "termination/time_limit",
    "timing/policy_inference_ms_max",
    "timing/policy_inference_ms_mean",
    "timing/step_race_ms_max",
    "timing/step_race_ms_mean",
    "timing/step_race_ms_p99",
    "velocity/projected_mps",
    "velocity/ratio",
    "velocity/ratio_mean",
    "velocity/ratio_max",
}
_EVALUATION_METRICS = {
    "collision_rate",
    "control_brake_fraction_mean",
    "control_brake_tap_fraction_mean",
    "control_gas_fraction_mean",
    "control_steer_abs_mean",
    "failure_progress_best_pct",
    "failure_progress_mean_pct",
    "failure_progress_median_pct",
    "finish_rate",
    "finish_time_best_s",
    "finish_time_mean_s",
    "finish_time_median_s",
    "finished_trials",
    "off_track_rate",
    "policy_version",
    "projected_velocity_ratio_mean",
    "q_margin_start_mean",
    "telemetry_error_rate",
    "trials",
}
_IMITATION_METRICS = {
    "accuracy",
    "balanced_accuracy",
    "best",
    "control_score",
    "intervention_accuracy",
    "learning_rate",
    "loss",
    "steering_accuracy",
    "steering_transition_accuracy",
    "student_disagreement_accuracy",
    "transition_accuracy",
    "weighted_accuracy",
}
_IGNORED_REMOTE_EVENTS = {
    "actor/heartbeat",
    "eval/progress_bin",
    "train/execution",
    "train/progress_bin",
}


def _flat_name(namespace: str, key: str) -> str:
    return f"{namespace}/{key.replace('/', '_')}"


def _selected(payload: Mapping[str, Any], keys: set[str], namespace: str) -> dict[str, Any]:
    return {_flat_name(namespace, key): payload[key] for key in keys if key in payload}


def _update_metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
    values = {name: payload[key] for key, name in _UPDATE_METRICS.items() if key in payload}
    for key, value in payload.items():
        prefix, separator, suffix = key.partition("/")
        if prefix in {"loss", "gradients"} and separator:
            values[_flat_name("learner", key)] = value
        elif prefix == "debug" and suffix in _DEBUG_METRICS:
            values[_flat_name("learner", suffix)] = value
        elif prefix == "timing" and suffix in _TIMING_METRICS:
            values[_flat_name("performance", suffix)] = value
    return values


def _event_metrics(event: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    if event == "train/update":
        return _update_metrics(payload)
    if event == "train/episode":
        return _selected(payload, _EPISODE_METRICS, "episode")
    if event == "eval/summary":
        return _selected(payload, _EVALUATION_METRICS, "evaluation")
    if event in {"bc/train", "bc/validation"}:
        phase = event.removeprefix("bc/")
        return _selected(payload, _IMITATION_METRICS, f"imitation_{phase}")
    if event == "distributed/policy_published":
        return {
            "pipeline/policy_version": payload.get("policy_version", 0),
            "performance/policy_publish_s": payload.get("timing/policy_publish_s", 0.0),
        }
    if event == "train/checkpoint":
        return {
            "performance/checkpoint_snapshot_s": payload.get("timing/checkpoint_snapshot_s", 0.0),
        }
    return {}


def _load_wandb_key_from_dotenv(run_dir: str | None) -> Path | None:
    """Load ``WANDB_API_KEY`` from the nearest project ``.env`` if necessary."""

    if os.environ.get("WANDB_API_KEY"):
        return None
    starts = [Path.cwd()]
    if run_dir:
        starts.append(Path(run_dir))
    checked: set[Path] = set()
    for start in starts:
        for directory in (start, *start.parents):
            candidate = directory / ".env"
            if candidate in checked:
                continue
            checked.add(candidate)
            value = _dotenv_value(candidate, "WANDB_API_KEY")
            if value:
                os.environ["WANDB_API_KEY"] = value
                return candidate
    return None


def _dotenv_value(path: Path, requested_key: str) -> str | None:
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if text.startswith("export "):
            text = text[7:].lstrip()
        key, separator, value = text.partition("=")
        if separator and key.strip() == requested_key:
            return value.strip().strip('"').strip("'") or None
    return None


class WandbTracker:
    """Bounded asynchronous W&B projection of the neutral event stream."""

    def __init__(
        self,
        project: str,
        entity: str | None = None,
        run_dir: str | None = None,
        run_id: str | None = None,
        config: Mapping[str, Any] | None = None,
        queue_size: int = 10_000,
        attempt_id: str | None = None,
        resumed_from: str | None = None,
    ) -> None:
        if queue_size < 1:
            raise ValueError("W&B queue_size must be positive")
        _load_wandb_key_from_dotenv(run_dir)
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("Install trackmaniarl[wandb] to configure WandbTracker") from exc
        self._wandb: Any = wandb
        self._started_at = monotonic()
        self._attempt_id = attempt_id or uuid4().hex
        self._run = self._init_run(project, entity, run_dir, run_id, config, resumed_from)
        self._define_metrics()
        self._events: Queue[dict[str, Any] | None] = Queue(maxsize=queue_size)
        self._lock = Lock()
        self._enabled = True
        self._failed = False
        self._dropped_events = 0
        self._worker_errors = 0
        self._evaluation_batches = 0
        self._episode_count = 0
        self._latest_ingest: dict[str, Any] = {}
        self._heartbeats: dict[str, tuple[float, int]] = {}
        self._event_counts: dict[str, int] = {}
        self._worker = Thread(target=self._send_events, name="trackmaniarl-wandb", daemon=True)
        self._worker.start()

    @property
    def dropped_events(self) -> int:
        with self._lock:
            return self._dropped_events

    def log(self, event: str, payload: Mapping[str, Any], *, step: int | None = None) -> None:
        with self._lock:
            values = self._project(event, payload, step)
            enabled = self._enabled
        if not enabled or not values:
            return
        try:
            self._events.put_nowait(values)
        except Full:
            self._record_drop()

    def close(self) -> None:
        try:
            self._events.put(None, timeout=5.0)
        except Full:
            self._record_drop()
        else:
            self._worker.join(timeout=10.0)
        if self._worker.is_alive():
            self._failed = True
            print("W&B worker did not stop within 10 seconds.", flush=True)
        self._run.finish(exit_code=int(self._failed))

    def _init_run(
        self,
        project: str,
        entity: str | None,
        run_dir: str | None,
        run_id: str | None,
        config: Mapping[str, Any] | None,
        resumed_from: str | None,
    ) -> Any:
        run_config = dict(config) if config is not None else {}
        run_config["observability/attempt_id"] = self._attempt_id
        if resumed_from is not None:
            run_config["observability/resumed_from"] = resumed_from
        run = self._wandb.init(
            project=project,
            entity=entity,
            dir=run_dir,
            name=run_id,
            group=run_id,
            job_type="training",
            config=run_config,
            reinit="finish_previous",
            settings=self._wandb.Settings(
                console="wrap", x_graphql_timeout_seconds=10, x_service_wait=5
            ),
        )
        if run is None:
            raise RuntimeError("W&B did not create a run")
        if getattr(run, "url", None):
            print(f"Weights & Biases run: {run.url}", flush=True)
        return run

    def _define_metrics(self) -> None:
        for axis in (
            "trainer/update",
            "env/transitions",
            "env/episode",
            "eval/batch",
            "system/elapsed_s",
        ):
            self._run.define_metric(axis)
        patterns = {
            "trainer/update": (
                "imitation_train/*",
                "imitation_validation/*",
                "learner/*",
                "outcome/*",
                "performance/*",
                "replay/*",
            ),
            "env/transitions": ("pipeline/*",),
            "env/episode": ("episode/*",),
            "eval/batch": ("evaluation/*",),
            "system/elapsed_s": ("health/*", "system/*"),
        }
        for axis, names in patterns.items():
            for name in names:
                self._run.define_metric(name, step_metric=axis)

    def _project(self, event: str, payload: Mapping[str, Any], step: int | None) -> dict[str, Any]:
        if event == "distributed/ingest":
            self._remember_ingest(payload)
            return {}
        if event == "actor/heartbeat":
            self._remember_heartbeat(payload)
            return {}
        if event in _IGNORED_REMOTE_EVENTS:
            return {}
        if event == "run/failure":
            self._failed = True
        values = _event_metrics(event, payload)
        values.update(self._health_metrics(event, payload))
        self._add_axes(values, event, payload, step)
        return values

    def _remember_ingest(self, payload: Mapping[str, Any]) -> None:
        for key in ("transitions", "utd"):
            if key in payload:
                self._latest_ingest[key] = payload[key]
        for key in ("policy_lag_updates", "queue_delay_s", "rollout_queue_depth"):
            if key in payload:
                self._latest_ingest[key] = max(
                    float(payload[key]), float(self._latest_ingest.get(key, 0.0))
                )

    def _remember_heartbeat(self, payload: Mapping[str, Any]) -> None:
        actor_id = str(payload.get("actor_id", ""))
        if actor_id:
            self._heartbeats[actor_id] = (monotonic(), int(payload.get("spool_bytes", 0)))

    def _health_metrics(self, event: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        values: dict[str, Any] = {}
        if event == "train/update":
            values.update(self._pipeline_snapshot())
            values["health/tracker_dropped_events"] = self._dropped_events
            values["health/tracker_worker_errors"] = self._worker_errors
        counter_name = {
            "actor/timeout": "actor_timeouts",
            "distributed/rollout_rejected": "rollouts_rejected",
            "distributed/wal_error": "wal_errors",
            "distributed/wal_recovery": "wal_recoveries",
            "run/failure": "run_failures",
            "train/checkpoint": "checkpoints_queued",
            "train/checkpoint_completed": "checkpoints_completed",
            "train/checkpoint_failed": "checkpoint_failures",
        }.get(event)
        if counter_name is not None:
            self._event_counts[event] = self._event_counts.get(event, 0) + 1
            values[f"health/{counter_name}"] = self._event_counts[event]
        if event == "actor/timeout" and "silence_s" in payload:
            values["health/max_heartbeat_age_s"] = payload["silence_s"]
        return values

    def _pipeline_snapshot(self) -> dict[str, Any]:
        values = {_flat_name("pipeline", key): value for key, value in self._latest_ingest.items()}
        cutoff = monotonic() - 30.0
        self._heartbeats = {
            actor: item for actor, item in self._heartbeats.items() if item[0] >= cutoff
        }
        values["health/active_actors"] = len(self._heartbeats)
        values["health/spool_bytes"] = sum(item[1] for item in self._heartbeats.values())
        for key in ("policy_lag_updates", "queue_delay_s", "rollout_queue_depth"):
            self._latest_ingest.pop(key, None)
        return values

    def _add_axes(
        self,
        values: dict[str, Any],
        event: str,
        payload: Mapping[str, Any],
        step: int | None,
    ) -> None:
        if not values:
            return
        values["system/elapsed_s"] = monotonic() - self._started_at
        if event == "train/episode":
            self._episode_count = max(self._episode_count + 1, int(payload.get("index", 0)))
            values["env/episode"] = self._episode_count
        elif event == "eval/summary":
            self._evaluation_batches += 1
            values["eval/batch"] = self._evaluation_batches
        else:
            values["trainer/update"] = int(step or 0)
        transitions = payload.get("transitions", self._latest_ingest.get("transitions"))
        if transitions is not None:
            values["env/transitions"] = int(transitions)

    def _record_drop(self) -> None:
        with self._lock:
            self._dropped_events += 1
            first = self._dropped_events == 1
        if first:
            print("W&B event queue is full; remote metrics are being dropped.", flush=True)

    def _send_events(self) -> None:
        while True:
            values = self._events.get()
            try:
                if values is None:
                    return
                with self._lock:
                    enabled = self._enabled
                if not enabled:
                    continue
                self._run.log(values)
            except Exception as exc:
                with self._lock:
                    self._enabled = False
                    self._failed = True
                    self._worker_errors += 1
                print(f"W&B logging disabled after {type(exc).__name__}: {exc}", flush=True)
            finally:
                self._events.task_done()
