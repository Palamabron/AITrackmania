from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

_PYPROJECT_TEMPLATE = (
    '[build-system]\nrequires = ["setuptools==83.0.0"]\n'
    'build-backend = "setuptools.build_meta"\n\n'
    '[project]\nname = "{name}"\nversion = "0.1.0"\n'
    'requires-python = ">=3.12,<3.13"\ndependencies = ["{requirement}", "torch>=2.4"]\n\n'
    '[dependency-groups]\ndev = ["mypy>=1.8", "poethepoet>=0.36", '
    '"pytest>=7.0", "ruff>=0.4"]\n\n'
    '[tool.setuptools.packages.find]\nwhere = ["src"]\n\n'
    '[tool.ruff]\ntarget-version = "py312"\nline-length = 100\n'
    'src = ["src", "tests"]\n\n'
    '[tool.ruff.lint]\nselect = ["E", "F", "I", "N", "W", "UP", "B", '
    '"ANN", "SIM", "RUF", "C4", "PT", "TID", "FBT001", "FBT002", "PLR0913"]\n'
    'ignore = ["ANN401", "N999", "N812"]\n'
    'extend-safe-fixes = ["RUF022"]\n\n'
    "[tool.ruff.lint.pylint]\nmax-args = 3\n\n[tool.poe.tasks]\n"
    'fmt = [{{ cmd = "ruff format ." }}, {{ cmd = "ruff check --fix ." }}]\n'
    'types = "mypy --strict src tests"\ntest = "pytest"\n'
)


def _trackmaniarl_requirement(extras: str) -> str:
    source_root = Path(__file__).resolve().parents[2]
    if (source_root / "pyproject.toml").is_file():
        return f"trackmaniarl{extras}"
    installed_version = version("trackmaniarl")
    release = installed_version.split(".", maxsplit=2)
    major = int(release[0])
    minor = int(release[1])
    return f"trackmaniarl{extras}>={installed_version},<{major}.{minor + 1}"


def _pyproject(
    name: str,
    *,
    trackmaniarl_extras: str = "",
    template: str = "starter",
) -> str:
    requirement = _trackmaniarl_requirement(trackmaniarl_extras)
    project = _PYPROJECT_TEMPLATE.format(name=name, requirement=requirement)
    return project + (_trackmania_poe_tasks() if template == "trackmania" else "")


def _trackmania_poe_tasks() -> str:
    command = "trackmaniarl track"
    return (
        f'record-left = "{command} record-boundary left assets/trackmaniarl-test-left.npy"\n'
        f'record-right = "{command} record-boundary right assets/trackmaniarl-test-right.npy"\n'
        f'build-geometry = "{command} build-geometry assets/trackmaniarl-test.geometry.npz '
        "--left assets/trackmaniarl-test-left.npy "
        "--right assets/trackmaniarl-test-right.npy "
        "--map-uid REPLACE_WITH_TEST_3_UID --map-path maps/trackmaniarl-test.Map.Gbx" + '"\n'
    )


def _local_source() -> str:
    source_root = Path(__file__).resolve().parents[2]
    return (
        f'trackmaniarl = {{ path = "{source_root.as_posix()}", editable = true }}\n'
        if (source_root / "pyproject.toml").is_file()
        else ""
    )


def _vgamepad_source(template: str) -> str:
    return (
        'vgamepad = { git = "https://github.com/Palamabron/vgamepad", '
        'rev = "5f3435df3f8a0e658feb58b207d9137cdb5183cd" }\n'
        if template == "trackmania"
        else ""
    )


def _torch_source() -> str:
    return (
        'torch = [\n  { index = "pytorch-cuda", marker = "sys_platform == \'win32\' or '
        "sys_platform == 'linux'\" },\n"
        '  { index = "pytorch-cpu", marker = "sys_platform != \'win32\' and '
        "sys_platform != 'linux'\" },\n]\n"
    )


def _torch_indexes() -> str:
    return (
        "\n[[tool.uv.index]]\n"
        'name = "pytorch-cpu"\n'
        'url = "https://download.pytorch.org/whl/cpu"\n'
        "explicit = true\n"
        "\n[[tool.uv.index]]\n"
        'name = "pytorch-cuda"\n'
        'url = "https://download.pytorch.org/whl/cu128"\n'
        "explicit = true\n"
    )


def _uv_options(template: str) -> str:
    sources = _local_source() + _vgamepad_source(template) + _torch_source()
    return f"\n[tool.uv.sources]\n{sources}{_torch_indexes()}"


_PROJECT_README = (
    "# TrackmaniaRL project\n\n```powershell\nuv sync\n"
    "uv run trackmaniarl validate run.yaml\nuv run pytest\n```\n"
)

_TRACKMANIA_README = """
## Live Trackmania setup

1. In Openplanet's Plugin Manager, install the signed
   [**TrackmaniaRL Connect**](https://openplanet.dev/plugin/sac_getdata) plugin
   (`SAC_GetData`) version **2.4.0** and enable **School Mode**.
2. Replace every `REPLACE_WITH_TEST_3_UID` value in `run.yaml`, record both
   boundaries, and rebuild geometry with the configured local `.Map.Gbx`.
3. Enter that map with a visible vehicle, then run:

```powershell
uv run trackmaniarl track check --config run.yaml
uv run trackmaniarl smoke run.yaml --transitions 100
```

The `openplanet` directory is a developer-reference source snapshot, not a
second plugin installation path. Do not run its loose script alongside the
managed Plugin Manager installation.

## Optional Weights & Biases logging

The generated run writes local JSONL logs only. To opt in to W&B, add the extra
and configure an additional logger explicitly:

```powershell
uv add "trackmaniarl[trackmania,distributed,wandb]"
```

```yaml
components:
  additional_loggers:
    - class_path: trackmaniarl.observability.trackers:WandbTracker
      kwargs: {project: my-trackmania-agent}
```

Set `WANDB_API_KEY` in your private environment; never commit it.
"""


def _project_readme(template: str) -> str:
    if template == "starter":
        return _PROJECT_README
    return _PROJECT_README + _TRACKMANIA_README


COMPONENTS = '''"""Editable components. Run `trackmaniarl validate run.yaml` after each change."""

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from trackmaniarl.core.contracts import PolicyMode
from trackmaniarl.core.data import TrainingBatch, Transition


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
    def create(self, *, seed: int) -> StarterEnvironment:
        return StarterEnvironment(seed)


class StarterMlpPolicy:
    def __init__(self, network: nn.Module) -> None:
        self.network = network

    def act(self, observation: Any, mode: PolicyMode = PolicyMode.ONLINE) -> float:
        del mode
        speed = float(observation["speed"])
        with torch.no_grad():
            return float(torch.tanh(self.network(torch.tensor([[speed]])).squeeze()).item())

    def export_state(self) -> Mapping[str, Any]:
        return {"model": self.network.state_dict()}

    def load_state(self, state: Mapping[str, Any]) -> None:
        self.network.load_state_dict(state["model"])


class StarterMlpLearner:
    """Trainable contract example, not a complete racing algorithm."""

    def __init__(self) -> None:
        self.network = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 1))
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-3)
        self._policy = StarterMlpPolicy(self.network)

    def setup(self, context: Mapping[str, Any]) -> None:
        torch.manual_seed(int(context["seed"]))
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-3)

    def update(self, batch: TrainingBatch) -> Mapping[str, float]:
        speeds = torch.tensor(
            [[float(item["speed"])] for item in batch.data["observations"]]
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
'''
