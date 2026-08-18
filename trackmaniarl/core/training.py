"""Single-process trainer for the TrackMania 1.0 runtime.

Workers and remote servers can use the same contracts later; this trainer owns
the local lifecycle so a project can train and debug without hidden processes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trackmaniarl.core.data import PriorityUpdate
from trackmaniarl.core.runtime import ResolvedRun, prepare_run
from trackmaniarl.observability.artifacts import AsyncEpisodeWriter
from trackmaniarl.trackmania.collector import TrackmaniaCollector


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Counters returned by a completed bounded training run."""

    episodes: int
    transitions: int
    updates: int
    checkpoints: tuple[Path, ...]
    evaluation: Mapping[str, float] | None


class Trainer:
    """Collect TrackMania episodes, update an off-policy learner, and persist state."""

    def __init__(self, run: ResolvedRun, *, resume_checkpoint: str | Path | None = None) -> None:
        if run.environment_factory is None:
            raise ValueError("trackmaniarl train requires components.environment")
        self.run = run
        self.environment_factory = run.environment_factory
        self.resume_checkpoint = Path(resume_checkpoint) if resume_checkpoint is not None else None

    def train(self) -> TrainingResult:
        spec = self.run.spec.training
        self.run.learner.setup(
            {
                "seed": self.run.spec.seed,
                "run_dir": self.run.run_dir,
                "model_factory": self.run.model_factory,
            }
        )
        prepare_run(self.run)
        print(
            f"Training started: run_id={self.run.spec.run_id}, "
            f"target_transitions={spec.total_transitions}, artifacts={self.run.run_dir}",
            flush=True,
        )
        writer = AsyncEpisodeWriter(
            self.run.run_dir / "episodes", max_artifacts=spec.max_episode_artifacts
        )
        checkpoints: list[Path] = []
        transitions = 0
        updates = 0
        episodes = 0
        fractional_updates = 0.0
        evaluation: Mapping[str, float] | None = None
        if self.resume_checkpoint is not None:
            state = self.run.checkpoint_codec.load(self.resume_checkpoint)
            counters = self._restore_checkpoint(state)
            transitions = int(counters["transitions"])
            updates = int(counters["updates"])
            episodes = int(counters["episodes"])
            fractional_updates = float(counters["fractional_updates"])
            self._log(
                "train/resumed",
                {"checkpoint": str(self.resume_checkpoint)},
                transitions,
                updates,
                episodes,
            )
        try:
            while transitions < spec.total_transitions:
                previous_transitions = transitions
                environment = self.environment_factory.create(seed=self.run.spec.seed + episodes)
                collector = TrackmaniaCollector(
                    self.run.replay_store, self.run.feature_pipeline, self.run.learner.policy()
                )
                try:
                    remaining = spec.total_transitions - transitions
                    result = collector.collect(
                        environment,
                        episode_id=f"episode-{episodes:08d}",
                        max_steps=min(spec.max_episode_steps, remaining),
                    )
                finally:
                    close = getattr(environment, "close", None)
                    if callable(close):
                        close()
                writer.submit(result.artifact)
                transitions += result.transitions
                episodes += 1
                if result.transitions == 0:
                    raise RuntimeError(
                        "Environment returned an empty episode; refusing to spin forever"
                    )
                self._log(
                    "train/episode",
                    {
                        "reward": result.total_reward,
                        "transitions": result.transitions,
                        "replay_size": len(self.run.replay_store),
                        "termination": result.artifact.metadata.get("termination", "unknown"),
                        **self._episode_metrics(result),
                    },
                    transitions,
                    updates,
                    episodes,
                )
                if episodes:
                    termination = result.artifact.metadata.get("termination", "unknown")
                    episode_metrics = self._episode_metrics(result)
                    print(
                        f"Episode {episodes}: progress={episode_metrics['progress_pct']:.1f}%, "
                        f"reward={result.total_reward:.3f}, "
                        f"time={episode_metrics['episode_elapsed_s']:.2f}s, "
                        f"race={episode_metrics['race_time_s']:.2f}s, termination={termination}; "
                        f"transitions={transitions}/{spec.total_transitions}, updates={updates}",
                        flush=True,
                    )
                sample_footprint = spec.batch_size * spec.sequence_length + spec.n_step - 1
                ready = max(spec.warmup_transitions, sample_footprint)
                # Warm-up gathers data only.  Do not build an update debt that
                # would turn the first trainable episode into a large burst.
                newly_trainable = max(0, transitions - ready) - max(0, previous_transitions - ready)
                fractional_updates += newly_trainable * spec.updates_per_transition
                while len(self.run.replay_store) >= ready and fractional_updates >= 1:
                    batch = self.run.sampler.sample(
                        self.run.replay_store,
                        spec.batch_request(beta=spec.replay_beta(transitions)),
                    )
                    update = self.run.learner.update(batch)
                    metrics, priorities = update if isinstance(update, tuple) else (update, None)
                    if priorities is not None:
                        self._update_priorities(priorities)
                    updates += 1
                    fractional_updates -= 1
                    self._log(
                        "train/update",
                        {**metrics, "replay_size": len(self.run.replay_store)},
                        transitions,
                        updates,
                        episodes,
                    )
                    if (
                        spec.checkpoint_interval_updates is not None
                        and updates % spec.checkpoint_interval_updates == 0
                    ):
                        checkpoints.append(
                            self._checkpoint(transitions, updates, episodes, fractional_updates)
                        )
                    if updates == 1 or updates % 100 == 0:
                        print(
                            f"Training progress: transitions={transitions}/"
                            f"{spec.total_transitions}, "
                            f"updates={updates}, loss={metrics.get('loss/iqn', 'n/a')}",
                            flush=True,
                        )
                if (
                    self.run.evaluator is not None
                    and spec.evaluate_every_episodes is not None
                    and episodes % spec.evaluate_every_episodes == 0
                    and transitions < spec.total_transitions
                ):
                    self._checkpoint_for_evaluation(
                        checkpoints, transitions, updates, episodes, fractional_updates
                    )
                    evaluation = self.run.evaluator.evaluate(self.run.learner.policy())
                    self._log("eval/suite", evaluation, transitions, updates, episodes)
            if not checkpoints or checkpoints[-1].name != f"update-{updates:08d}.pt":
                checkpoints.append(
                    self._checkpoint(transitions, updates, episodes, fractional_updates)
                )
            if self.run.evaluator is not None:
                # Release artifacts must name the checkpoint that produced them.
                # A periodic evaluation can otherwise leave evaluation.json pointing
                # to a stale policy after subsequent training updates.
                self._set_evaluation_checkpoint(checkpoints)
                evaluation = self.run.evaluator.evaluate(self.run.learner.policy())
                self._log("eval/suite", evaluation, transitions, updates, episodes)
            print(
                f"Training finished: transitions={transitions}, updates={updates}, "
                f"episodes={episodes}",
                flush=True,
            )
            return TrainingResult(episodes, transitions, updates, tuple(checkpoints), evaluation)
        except BaseException as exc:
            self._log(
                "train/failure",
                {"exception_type": type(exc).__name__, "message": str(exc)},
                transitions,
                updates,
                episodes,
            )
            raise
        finally:
            writer.close()

    def _update_priorities(self, update: PriorityUpdate) -> None:
        self.run.sampler.update_priorities(update)

    @staticmethod
    def _episode_metrics(result: Any) -> dict[str, float]:
        """Extract the final, user-facing telemetry summary from an episode."""

        if not result.artifact.telemetry:
            return {
                "progress_pct": 0.0,
                "progress_m": 0.0,
                "episode_elapsed_s": 0.0,
                "race_time_s": 0.0,
            }
        final = result.artifact.telemetry[-1]
        return {
            "progress_pct": float(final.get("progress_pct", 0.0)),
            "progress_m": float(final.get("progress_m", 0.0)),
            "episode_elapsed_s": float(final.get("episode_elapsed_s", 0.0)),
            "race_time_s": float(final.get("race_time_ms", 0.0)) / 1_000.0,
        }

    def _set_evaluation_checkpoint(self, checkpoints: list[Path]) -> None:
        if not checkpoints or self.run.evaluator is None:
            return
        setter = getattr(self.run.evaluator, "set_checkpoint", None)
        if callable(setter):
            setter(checkpoints[-1])

    def _checkpoint_for_evaluation(
        self,
        checkpoints: list[Path],
        transitions: int,
        updates: int,
        episodes: int,
        fractional_updates: float,
    ) -> None:
        """Persist the current learner before attaching it to an evaluation artifact."""

        expected = self.run.run_dir / "checkpoints" / f"update-{updates:08d}.pt"
        if not checkpoints or checkpoints[-1] != expected:
            checkpoints.append(self._checkpoint(transitions, updates, episodes, fractional_updates))
        self._set_evaluation_checkpoint(checkpoints)

    def _checkpoint(
        self, transitions: int, update: int, episodes: int, fractional_updates: float
    ) -> Path:
        path = self.run.run_dir / "checkpoints" / f"update-{update:08d}.pt"
        state = {
            "schema_version": "1.0",
            "learner": self.run.learner.state_dict(),
            "replay_store": _state_dict(self.run.replay_store),
            "sampler": _state_dict(self.run.sampler),
            "counters": {
                "transitions": transitions,
                "updates": update,
                "episodes": episodes,
                "fractional_updates": fractional_updates,
            },
        }
        self.run.checkpoint_codec.save(state, path)
        self._log("train/checkpoint", {"path": str(path)}, transitions, update, episodes)
        print(f"Checkpoint saved: {path}", flush=True)
        return path

    def _restore_checkpoint(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        if state.get("schema_version") != "1.0":
            raise ValueError("Unsupported training checkpoint schema")
        learner_state = state.get("learner")
        if not isinstance(learner_state, Mapping):
            raise ValueError("Training checkpoint is missing learner state")
        counters = state.get("counters")
        if not isinstance(counters, Mapping):
            raise ValueError("Training checkpoint is missing counters")
        required_counters = {"transitions", "updates", "episodes", "fractional_updates"}
        if required_counters - counters.keys():
            raise ValueError("Training checkpoint has incomplete counters")
        self.run.learner.load_state_dict(learner_state)
        _load_state_dict(self.run.replay_store, state.get("replay_store"))
        _load_state_dict(self.run.sampler, state.get("sampler"))
        return counters

    def _log(
        self,
        event: str,
        payload: Mapping[str, object],
        transitions: int,
        updates: int,
        episodes: int,
    ) -> None:
        self.run.logger.log(
            event,
            {
                **payload,
                "counters": {"transitions": transitions, "updates": updates, "episodes": episodes},
            },
            step=updates,
        )


def _state_dict(component: object) -> Mapping[str, object] | None:
    method = getattr(component, "state_dict", None)
    return method() if callable(method) else None


def _load_state_dict(component: object, state: object) -> None:
    if state is None:
        return
    method = getattr(component, "load_state_dict", None)
    if not callable(method):
        raise TypeError(
            f"{type(component).__name__} cannot resume because it has no load_state_dict()"
        )
    method(state)
