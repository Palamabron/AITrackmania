"""Release contracts for TrackMania scaffold and evaluation."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import numpy as np
import pytest

from trackmaniarl.core.contracts import EvaluatorRuntimeRequest, PolicyMode
from trackmaniarl.core.spec import EvaluationMapSpec, EvaluationSuiteSpec, RunSpec
from trackmaniarl.project.scaffold import create_project
from trackmaniarl.trackmania.evaluation import TrackmaniaEvaluator


class _IdentityPipeline:
    def transform_observation(self, observation: object) -> object:
        return observation


class _EvaluationPolicy:
    def act(self, observation: object, mode: PolicyMode = PolicyMode.ONLINE) -> float:
        del observation
        assert mode is PolicyMode.EVALUATION
        return 0.0


class _SuccessfulEnvironment:
    def reset(self, *, seed: int | None = None) -> tuple[float, dict[str, object]]:
        del seed
        return 0.0, {}

    def step(self, action: object) -> tuple[float, float, bool, bool, dict[str, str | float]]:
        del action
        info: dict[str, str | float] = {
            "termination_reason": "finished",
            "race_time_ms": 12_345.0,
            "controller_apply_ms": 1.5,
            "telemetry_wait_ms": 7.5,
            "telemetry_skipped_frames": 2.0,
        }
        return 1.0, 2.0, True, False, info

    def close(self) -> None:
        pass


class _SuccessfulFactory:
    def create(self, *, seed: int, evaluation_map: object) -> _SuccessfulEnvironment:
        del seed, evaluation_map
        return _SuccessfulEnvironment()


class _RawEnvironment:
    def __init__(self, raw: np.ndarray) -> None:
        self.raw = raw

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, object]]:
        del seed
        return self.raw.copy(), {}

    def step(self, action: object) -> tuple[np.ndarray, float, bool, bool, dict[str, str]]:
        del action
        return self.raw.copy(), 0.0, True, False, {"termination_reason": "no_progress"}

    def close(self) -> None:
        pass


class _RawFactory:
    def __init__(self, raw: np.ndarray) -> None:
        self.raw = raw

    def create(self, *, seed: int, evaluation_map: object) -> _RawEnvironment:
        del seed, evaluation_map
        return _RawEnvironment(self.raw)


class _PreparedPipeline:
    def transform_observation(self, observation: object) -> str:
        del observation
        return "prepared"


class _RawPolicy:
    requires_raw_observation = True

    def __init__(self, raw: np.ndarray) -> None:
        self.raw = raw

    def act(self, observation: object, mode: PolicyMode = PolicyMode.ONLINE) -> float:
        assert mode is PolicyMode.EVALUATION
        assert np.array_equal(observation, self.raw)
        return 0.0


class _StatefulPolicy:
    def __init__(self) -> None:
        self.state = 0
        self.actions: list[int] = []

    def act(self, observation: object, mode: PolicyMode = PolicyMode.ONLINE) -> int:
        del observation
        assert mode is PolicyMode.EVALUATION
        action = self.state
        self.actions.append(action)
        self.state += 1
        return action

    def reset_episode(self) -> None:
        self.state = 0


class _RecordingEnvironment:
    def __init__(self) -> None:
        self.reset_count = 0
        self.actions: list[int] = []

    def reset(self, *, seed: int | None = None) -> tuple[float, dict[str, object]]:
        del seed
        self.reset_count += 1
        return 0.0, {}

    def step(self, action: object) -> tuple[float, float, bool, bool, dict[str, str]]:
        self.actions.append(int(action))
        return 0.0, 0.0, True, False, {"termination_reason": "no_progress"}

    def close(self) -> None:
        pass


class _RecordingFactory:
    def __init__(self) -> None:
        self.environment = _RecordingEnvironment()

    def create(self, *, seed: int, evaluation_map: object) -> _RecordingEnvironment:
        del seed, evaluation_map
        return self.environment


class _CollisionEnvironment:
    def __init__(self) -> None:
        self.steps = 0

    def reset(self, *, seed: int | None = None) -> tuple[float, dict[str, object]]:
        del seed
        self.steps = 0
        return 0.0, {}

    def step(self, action: object) -> tuple[float, float, bool, bool, dict[str, object]]:
        del action
        self.steps += 1
        if self.steps == 1:
            return (
                0.0,
                0.0,
                False,
                False,
                {
                    "termination_reason": "",
                    "collision_detected": True,
                },
            )
        return 0.0, 0.0, True, False, {"termination_reason": "no_progress"}

    def close(self) -> None:
        pass


class _CollisionFactory:
    def create(self, *, seed: int, evaluation_map: object) -> _CollisionEnvironment:
        del seed, evaluation_map
        return _CollisionEnvironment()


class _Geometry:
    def __init__(self, path: Path, expected_map_uid: str) -> None:
        del path, expected_map_uid

    def validate_map(self, path: Path) -> None:
        del path


def _evaluation_suite(tmp_path: Path, trials_per_map: int = 1) -> EvaluationSuiteSpec:
    map_spec = EvaluationMapSpec(
        id="test-map",
        map_path=tmp_path / "test.Map.Gbx",
        geometry_path=tmp_path / "test.geometry.npz",
        expected_map_uid="test-map-uid",
    )
    return EvaluationSuiteSpec(
        name="test", version="1", maps=(map_spec,), trials_per_map=trials_per_map
    )


def _patch_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("trackmaniarl.trackmania.evaluation.BoundaryGeometry", _Geometry)


def _assert_trackmania_config(target: Path) -> None:
    config = (target / "run.yaml").read_text(encoding="utf-8")
    assert "OpenPlanetEnvironmentFactory" in config
    assert "control_backend: gamepad" in config
    assert "action_repeat_frames: 1" in config
    assert "decision_interval_ms: 50.0" in config
    assert "demonstration_control_aggregation: true" in config
    assert "TrackmaniaEvaluator" in config
    assert "WandbTracker" not in config
    assert RunSpec.from_yaml(target / "run.yaml").evaluation is not None


def _assert_openplanet_plugin(target: Path) -> None:
    plugin = target / "openplanet" / "SAC_GetData-2.4.0-reference.as"
    assert plugin.is_file()
    plugin_source = plugin.read_text(encoding="utf-8")
    assert 'const string PROTOCOL_VERSION = "2"' in plugin_source
    assert "if (api is null || vis is null) { yield(); continue; }" in plugin_source
    assert "vis is null ?" not in plugin_source
    info = tomllib.loads((target / "openplanet" / "info.reference.toml").read_text())
    assert info["meta"]["name"] == "TrackmaniaRL Connect"
    assert info["meta"]["version"] == "2.4.0"
    assert info["meta"]["siteid"] == 421


def _assert_project_files(target: Path) -> None:
    assets = target / "assets"
    assert not (assets / "trajectory.csv").exists()
    with np.load(assets / "trackmaniarl-test.geometry.npz", allow_pickle=False) as geometry:
        assert "recorded_count" in geometry.files
    assert (target / "maps").is_dir()
    env_example = (target / ".env-example").read_text()
    assert "TRACKMANIARL_DISTRIBUTED_TOKEN=" in env_example
    assert "WANDB_API_KEY" not in env_example
    assert ".env" in (target / ".gitignore").read_text()
    assert not (target / "run.py").exists()


def _assert_project_metadata(target: Path) -> None:
    pyproject = tomllib.loads((target / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["tool"]["poe"]["tasks"]["record-left"]
    assert pyproject["tool"]["uv"]["sources"]["torch"]
    assert pyproject["tool"]["uv"]["sources"]["vgamepad"]
    assert "wandb" not in pyproject["project"]["dependencies"][0]
    readme = (target / "README.md").read_text(encoding="utf-8")
    assert "Plugin Manager" in readme
    assert "SAC_GetData" in readme
    assert 'uv add "trackmaniarl[trackmania,distributed,wandb]"' in readme


def test_trackmania_template_contains_first_party_components(tmp_path: Path) -> None:
    target = create_project(tmp_path / "agent", "agent", template="trackmania")
    _assert_trackmania_config(target)
    _assert_openplanet_plugin(target)
    _assert_project_files(target)
    _assert_project_metadata(target)


def _assert_success_metrics(metrics: dict[str, float]) -> None:
    assert metrics["eval/finish_rate"] == 1.0
    assert metrics["eval/reward"] == 2.0
    assert metrics["eval/finish_time_s"] == pytest.approx(12.345)
    assert metrics["eval/controller_apply_ms"] == pytest.approx(1.5)
    assert metrics["eval/telemetry_wait_ms"] == pytest.approx(7.5)
    assert metrics["eval/telemetry_skipped_frames_total"] == 8.0
    assert metrics["eval/telemetry_skipped_frames_mean"] == 2.0
    assert metrics["eval/telemetry_skipped_frames_max"] == 2.0
    assert metrics["eval/telemetry_steps_with_skipped_frames_fraction"] == 1.0


def test_trackmania_evaluator_runs_every_declared_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_geometry(monkeypatch)
    suite = _evaluation_suite(tmp_path, trials_per_map=4)
    request = EvaluatorRuntimeRequest(
        suite, _SuccessfulFactory(), _IdentityPipeline(), run_dir=tmp_path
    )
    metrics = TrackmaniaEvaluator(request).evaluate(_EvaluationPolicy())
    _assert_success_metrics(metrics)
    artifact = json.loads((tmp_path / "evaluation.json").read_text(encoding="utf-8"))
    assert [trial["steps"] for trial in artifact["trials"]] == [1, 1, 1, 1]


def test_trackmania_evaluator_passes_raw_observation_to_opt_in_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_geometry(monkeypatch)
    raw = np.arange(33, dtype=np.float32)
    suite = _evaluation_suite(tmp_path)
    request = EvaluatorRuntimeRequest(suite, _RawFactory(raw), _PreparedPipeline())
    metrics = TrackmaniaEvaluator(request).evaluate(_RawPolicy(raw))
    assert metrics["eval/finish_rate"] == 0.0


def test_trackmania_evaluator_prewarms_policy_before_a_fresh_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_geometry(monkeypatch)
    factory = _RecordingFactory()
    policy = _StatefulPolicy()
    request = EvaluatorRuntimeRequest(
        _evaluation_suite(tmp_path, trials_per_map=2), factory, _IdentityPipeline()
    )
    TrackmaniaEvaluator(request).evaluate(policy)
    assert factory.environment.reset_count == 3
    assert policy.actions == [0, 0, 0]
    assert factory.environment.actions == [0, 0]


def test_trackmania_evaluator_counts_detected_collision_as_a_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_geometry(monkeypatch)
    request = EvaluatorRuntimeRequest(
        _evaluation_suite(tmp_path), _CollisionFactory(), _IdentityPipeline()
    )
    metrics = TrackmaniaEvaluator(request).evaluate(_EvaluationPolicy())
    assert metrics["eval/crash_rate"] == 1.0
