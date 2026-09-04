"""Shared deterministic runtime doubles and specifications."""

from __future__ import annotations

from pathlib import Path

import torch

from trackmaniarl.core.contracts import EvaluatorRuntimeRequest
from trackmaniarl.core.data import Transition, TransitionId
from trackmaniarl.core.spec import RunSpec


class FakeEnvironment:
    def __init__(self) -> None:
        self.step_index = 0
        self.closed = False

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, float], dict[str, object]]:
        del seed
        self.step_index = 0
        return {"speed": 0.0}, {}

    def step(self, action: object) -> tuple[dict[str, float], float, bool, bool, dict[str, object]]:
        del action
        self.step_index += 1
        return {"speed": float(self.step_index)}, 1.0, self.step_index == 2, False, {}

    def close(self) -> None:
        self.closed = True


class FakeEnvironmentFactory:
    def create(self, *, seed: int) -> FakeEnvironment:
        del seed
        return FakeEnvironment()


class FailingEnvironment(FakeEnvironment):
    def step(self, action: object) -> tuple[dict[str, float], float, bool, bool, dict[str, object]]:
        del action
        raise RuntimeError("simulated environment failure")


class FailingEnvironmentFactory:
    def create(self, *, seed: int) -> FailingEnvironment:
        del seed
        return FailingEnvironment()


class PpoFakeEnvironment:
    def __init__(self) -> None:
        self.step_index = 0

    def reset(self, *, seed: int | None = None) -> tuple[torch.Tensor, dict[str, object]]:
        del seed
        self.step_index = 0
        return torch.zeros(33), {}

    def step(self, action: object) -> tuple[torch.Tensor, float, bool, bool, dict[str, object]]:
        del action
        self.step_index += 1
        return torch.full((33,), float(self.step_index)), 1.0, self.step_index == 2, False, {}

    def close(self) -> None:
        return


class PpoFakeEnvironmentFactory:
    def create(self, *, seed: int) -> PpoFakeEnvironment:
        del seed
        return PpoFakeEnvironment()


class RecordingEvaluator:
    def __init__(self, request: EvaluatorRuntimeRequest) -> None:
        del request
        self.checkpoints: list[Path] = []

    def set_checkpoint(self, checkpoint: str | Path) -> None:
        self.checkpoints.append(Path(checkpoint))

    def evaluate(self, policy: object) -> dict[str, float]:
        del policy
        return {"eval/finish_rate": 1.0}


class CapturingLogger:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class BasicReplayStore:
    def __init__(self) -> None:
        self.transitions: list[Transition] = []

    def append(self, transition: Transition) -> TransitionId:
        self.transitions.append(transition)
        return len(self.transitions) - 1

    def get(self, transition_ids: list[TransitionId]) -> list[Transition]:
        return [self.transitions[index] for index in transition_ids]

    def available_ids(self) -> list[TransitionId]:
        return list(range(len(self.transitions)))

    def contains(self, transition_id: TransitionId) -> bool:
        return 0 <= transition_id < len(self.transitions)

    def __len__(self) -> int:
        return len(self.transitions)


def runtime_spec(tmp_path: Path) -> RunSpec:
    return RunSpec.model_validate(
        {
            "api_version": "2.0",
            "run_id": "smoke",
            "artifacts_dir": str(tmp_path / "artifacts"),
            "components": {
                "learner": {"class_path": "trackmaniarl.core.builtins:SmokeLearner"},
                "replay_store": {"class_path": "trackmaniarl.core.replay:InMemoryReplayStore"},
                "sampler": {"class_path": "trackmaniarl.core.replay:UniformSampler"},
                "feature_pipeline": {
                    "class_path": "trackmaniarl.core.builtins:IdentityFeaturePipeline"
                },
            },
        }
    )
