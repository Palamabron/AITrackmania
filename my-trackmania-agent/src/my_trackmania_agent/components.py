"""Editable components. Run `tmrl validate run.yaml` after each change."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from tmrl.core.data import SampleBatch, Transition


class StarterFeaturePipeline:
    """Keep observations as PyTrees; replace this with TrackMania feature extraction."""

    def transform_observation(self, observation: Any) -> Any:
        return observation

    def collate(self, transitions: list[Transition]) -> dict[str, Any]:
        return {
            "observations": [item.observation for item in transitions],
            "actions": [item.action for item in transitions],
            "rewards": [item.reward for item in transitions],
            "next_observations": [item.next_observation for item in transitions],
            "terminated": [item.terminated for item in transitions],
            "truncated": [item.truncated for item in transitions],
        }


class StarterEnvironment:
    """Deterministic stand-in for a real TrackMania adapter during local checks."""

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.step_index = 0

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, float], dict[str, Any]]:
        self.seed = self.seed if seed is None else seed
        self.step_index = 0
        return {"speed": 0.0}, {}

    def step(self, action: float) -> tuple[dict[str, float], float, bool, bool, dict[str, Any]]:
        self.step_index += 1
        speed = float(self.step_index)
        return {"speed": speed}, 1.0 - abs(float(action)), self.step_index >= 8, False, {}


class StarterEnvironmentFactory:
    """Replace with a factory that opens your local TrackMania adapter."""

    def create(self, *, seed: int) -> StarterEnvironment:
        return StarterEnvironment(seed)


class StarterMlpPolicy:
    """A small MLP policy; replace the scalar input with your real feature vector."""

    def __init__(self, network: nn.Module) -> None:
        self.network = network

    def act(self, observation: Any, *, deterministic: bool = False) -> float:
        del deterministic
        speed = float(observation.get("speed", 0.0))
        with torch.no_grad():
            return float(torch.tanh(self.network(torch.tensor([[speed]])).squeeze()).item())


class StarterMlpLearner:
    """Trainable contract example, not a complete racing algorithm."""

    def __init__(self) -> None:
        self.network = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 1))
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-3)
        self._policy = StarterMlpPolicy(self.network)

    def setup(self, context: Mapping[str, Any]) -> None:
        torch.manual_seed(int(context["seed"]))

    def update(self, batch: SampleBatch) -> Mapping[str, float]:
        speeds = torch.tensor(
            [[float(item.get("speed", 0.0))] for item in batch.data["observations"]]
        )
        targets = torch.tensor([[float(value)] for value in batch.data["rewards"]])
        loss = (self.network(speeds) - targets).square().mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss/starter": float(loss.item())}

    def policy(self) -> StarterMlpPolicy:
        return self._policy

    def state_dict(self) -> Mapping[str, Any]:
        return {"model": self.network.state_dict(), "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.network.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])


class TorchCheckpointCodec:
    def save(self, state: Mapping[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(state), path)

    def load(self, path: Path) -> Mapping[str, Any]:
        return cast(Mapping[str, Any], torch.load(path, map_location="cpu", weights_only=False))
