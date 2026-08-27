"""Single-process trainer for local on-policy and off-policy runs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from trackmaniarl.core.data import PriorityUpdate
from trackmaniarl.core.runtime import ResolvedRun
from trackmaniarl.core.training_loop import run_training
from trackmaniarl.core.training_support import (
    TrainingCounters,
    _load_state_dict,
    _state_dict,
)
from trackmaniarl.core.training_support import (
    TrainingResult as TrainingResult,
)
from trackmaniarl.core.training_support import episode_metrics as _extract_episode_metrics

_CHECKPOINT_SCHEMA_VERSION = "1.0"
_CHECKPOINT_KEYS = frozenset({"schema_version", "learner", "replay_store", "sampler", "counters"})
_COUNTER_KEYS = frozenset(
    {"transitions", "updates", "episodes", "fractional_updates", "next_episode_index"}
)


class Trainer:
    """Collect local episodes, update the learner, and persist training state."""

    def __init__(self, run: ResolvedRun, *, resume_checkpoint: str | Path | None = None) -> None:
        if run.environment_factory is None:
            raise ValueError("trackmaniarl train requires components.environment")
        self.run = run
        self.environment_factory = run.environment_factory
        self.resume_checkpoint = Path(resume_checkpoint) if resume_checkpoint is not None else None
        self.on_policy = bool(getattr(run.learner, "on_policy", False))
        self._validate_on_policy_run()

    def _validate_on_policy_run(self) -> None:
        if self.on_policy and not getattr(self.run.sampler, "on_policy_rollouts", False):
            raise ValueError("On-policy learners require OnPolicySequenceSampler")
        training = self.run.spec.training
        if self.on_policy and training.total_transitions % training.sequence_length:
            raise ValueError("On-policy total_transitions must be divisible by sequence_length")

    def train(self) -> TrainingResult:
        return run_training(self)

    def _update_priorities(self, update: PriorityUpdate) -> None:
        self.run.sampler.update_priorities(update)

    @staticmethod
    def _episode_metrics(result: Any) -> dict[str, float]:
        return _extract_episode_metrics(result)

    def _set_evaluation_checkpoint(self, checkpoints: list[Path]) -> None:
        if not checkpoints or self.run.evaluator is None:
            return
        setter = getattr(self.run.evaluator, "set_checkpoint", None)
        if callable(setter):
            setter(checkpoints[-1])

    def _checkpoint_for_evaluation(
        self, checkpoints: list[Path], counters: TrainingCounters
    ) -> None:
        expected = self._checkpoint_path(counters.updates)
        if not checkpoints or checkpoints[-1] != expected:
            checkpoints.append(self._write_checkpoint(counters))
        self._set_evaluation_checkpoint(checkpoints)

    def _write_checkpoint(self, counters: TrainingCounters) -> Path:
        path = self._checkpoint_path(counters.updates)
        state = self._checkpoint_state(counters)
        self._log("train/checkpoint", {"path": str(path)}, counters)
        try:
            self.run.checkpoint_codec.save(state, path)
        except BaseException as exc:
            self._log_checkpoint_failure(path, exc, counters)
            raise
        self._checkpoint_completed(path, counters)
        return path

    def _checkpoint_state(self, counters: TrainingCounters) -> dict[str, Any]:
        stored = TrainingCounters(
            counters.transitions,
            counters.updates,
            counters.episodes,
            counters.next_episode_index,
            counters.fractional_updates,
        )
        stored.next_episode_index = self._next_episode_index(counters.episodes)
        return {
            "schema_version": _CHECKPOINT_SCHEMA_VERSION,
            "learner": self.run.learner.state_dict(),
            "replay_store": None if self.on_policy else _state_dict(self.run.replay_store),
            "sampler": None if self.on_policy else _state_dict(self.run.sampler),
            "counters": stored.as_mapping(),
        }

    def _log_checkpoint_failure(
        self, path: Path, exc: BaseException, counters: TrainingCounters
    ) -> None:
        self._log(
            "train/checkpoint_failed",
            {"path": str(path), "exception_type": type(exc).__name__, "message": str(exc)},
            counters,
        )

    def _checkpoint_completed(self, path: Path, counters: TrainingCounters) -> None:
        self._log("train/checkpoint_completed", {"path": str(path)}, counters)
        print(f"Checkpoint saved: {path}", flush=True)

    def _checkpoint_path(self, update: int) -> Path:
        return self.run.run_dir / "checkpoints" / f"update-{update:08d}.pt"

    def _restore_checkpoint(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        learner_state, counters = self._validated_checkpoint(state)
        self.run.learner.load_state_dict(learner_state)
        if not self.on_policy:
            _load_state_dict(self.run.replay_store, state["replay_store"])
            _load_state_dict(self.run.sampler, state["sampler"])
        return counters

    def _validated_checkpoint(
        self, state: Mapping[str, Any]
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        self._validate_checkpoint_keys(state)
        if state["schema_version"] != _CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("Unsupported training checkpoint schema")
        learner_state = state["learner"]
        counters = state["counters"]
        if not isinstance(learner_state, Mapping):
            raise ValueError("Training checkpoint is missing learner state")
        if not isinstance(counters, Mapping):
            raise ValueError("Training checkpoint is missing counters")
        self._validate_replay_state(state)
        self._validate_checkpoint_counters(counters)
        return learner_state, counters

    @staticmethod
    def _validate_checkpoint_keys(state: Mapping[str, Any]) -> None:
        missing = _CHECKPOINT_KEYS - state.keys()
        unexpected = state.keys() - _CHECKPOINT_KEYS
        if missing or unexpected:
            raise ValueError(
                f"Training checkpoint keys differ: missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )

    def _validate_replay_state(self, state: Mapping[str, Any]) -> None:
        required = {"replay_store", "sampler"}
        missing = required - state.keys()
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"Training checkpoint is missing: {names}")
        if not self.on_policy and any(state[name] is None for name in required):
            raise ValueError("Off-policy checkpoint is missing replay or sampler state")

    def _validate_checkpoint_counters(self, counters: Mapping[str, Any]) -> None:
        missing = _COUNTER_KEYS - counters.keys()
        unexpected = counters.keys() - _COUNTER_KEYS
        if missing or unexpected:
            raise ValueError(
                f"Training checkpoint counter keys differ: missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )

    def _next_episode_index(self, completed_episodes: int) -> int:
        if not self.on_policy:
            return completed_episodes
        transition_ids = self.run.replay_store.available_ids()
        if not transition_ids:
            return completed_episodes
        latest = self.run.replay_store.get([transition_ids[-1]])[0]
        return completed_episodes + int(not latest.terminated and not latest.truncated)

    def _log(self, event: str, payload: Mapping[str, object], counters: TrainingCounters) -> None:
        values = {
            "transitions": counters.transitions,
            "updates": counters.updates,
            "episodes": counters.episodes,
        }
        self.run.logger.log(event, {**payload, "counters": values}, step=counters.updates)
