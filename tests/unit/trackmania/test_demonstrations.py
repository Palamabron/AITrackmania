from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from tests.unit.trackmania._demonstration_fixtures import (
    _config,
    _demonstration,
    _frames,
    _geometry,
    _IdentityPipeline,
)
from trackmaniarl.commands.diagnostics import _expert_diagnostics, _ExpertContext
from trackmaniarl.models.composite import CompositeModules, CompositeValueModel
from trackmaniarl.models.contracts import RiskSpec
from trackmaniarl.models.heads import ImplicitQuantileHead, ImplicitQuantileHeadConfig
from trackmaniarl.models.strategies import RandomQuantileStrategy
from trackmaniarl.models.temporal import GruTemporalCore
from trackmaniarl.models.track_graphs import TrackNeighborGraph
from trackmaniarl.trackmania.actions import (
    BRAKE_TAP_SENTINEL,
    build_brake_tap_action_table,
    continuous_control_to_discrete_indices_batch,
)
from trackmaniarl.trackmania.demonstrations import (
    Demonstration,
    DemonstrationResamplingConfig,
    DemonstrationResamplingRequest,
    DemonstrationTransitionContext,
    _control,
    demonstration_transitions,
    load_demonstration,
    resample_demonstration,
    resample_demonstration_for_environment,
    save_demonstration,
)
from trackmaniarl.trackmania.diagnostics import (
    ExpertActionDiagnostics,
    ExpertDiagnosticRecord,
    aggregate_expert_actions,
)
from trackmaniarl.trackmania.environment import TrackmaniaEnvironmentConfig
from trackmaniarl.trackmania.geometry import BoundaryGeometry
from trackmaniarl.trackmania.telemetry import TelemetryFrame

DEFAULT_RESAMPLING_CONFIG = DemonstrationResamplingConfig()


class _DictTensorPipeline:
    def reset_episode(self) -> None:
        return

    def transform_observation(self, observation: Any) -> dict[str, torch.Tensor]:
        values = torch.as_tensor(np.asarray(observation, dtype=np.float32).copy())
        physics = torch.zeros(60, dtype=torch.float32)
        physics[: len(values)] = values
        return {"physics": physics, "track": torch.zeros(3, 88)}


class _GraphDictEncoder(torch.nn.Module):
    output_dim = 8

    def __init__(self) -> None:
        super().__init__()
        self.graph = TrackNeighborGraph(hidden_dim=8, layer_count=1)
        self.physics = torch.nn.Linear(60, 8)

    def forward(self, observation: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.graph(observation["track"]) + self.physics(observation["physics"])


@dataclass(frozen=True, slots=True)
class _ExpertDiagnosticCase:
    path: Path
    context: _ExpertContext
    source_count: int
    evaluated_count: int


def _resample(
    demonstration: Demonstration,
    decision_interval_ms: float | None,
    config: DemonstrationResamplingConfig = DEFAULT_RESAMPLING_CONFIG,
) -> tuple[np.ndarray, np.ndarray]:
    return resample_demonstration(
        DemonstrationResamplingRequest(demonstration, decision_interval_ms, config)
    )


def _short_demonstration(tmp_path: Path) -> tuple[Demonstration, np.ndarray]:
    demonstration = _demonstration(_geometry(tmp_path))
    frames = demonstration.frames[:6].copy()
    frames[:, 3] = np.arange(len(frames), dtype=np.float32) * 10.0
    frames[-1, 2] = 1.0
    return demonstration, frames


def _lead_demonstration(tmp_path: Path) -> tuple[Demonstration, np.ndarray]:
    demonstration, frames = _short_demonstration(tmp_path)
    actions = np.asarray([0, 1, 3, 39, 75], dtype=np.int64)
    _, table = build_brake_tap_action_table()
    updated = replace(
        demonstration,
        frames=frames,
        actions=actions,
        controls=np.asarray([table[action] for action in actions], dtype=np.float32),
        finish_time_s=0.05,
    )
    return updated, frames


def _controlled_demonstration(
    tmp_path: Path, controls: np.ndarray
) -> tuple[Demonstration, list[np.ndarray]]:
    demonstration, frames = _short_demonstration(tmp_path)
    _, table = build_brake_tap_action_table()
    updated = replace(
        demonstration,
        frames=frames,
        actions=continuous_control_to_discrete_indices_batch(controls, table),
        controls=controls,
        finish_time_s=0.05,
        control_alignment="frame_start",
    )
    return updated, table


def _pwm_controls() -> np.ndarray:
    return np.asarray(
        [
            [1.0, 0.0, -1.0],
            [1.0, 1.0, -1.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def test_recorded_control_preserves_openplanet_steering_direction() -> None:
    values = np.zeros(33, dtype=np.float32)
    values[30] = -1.0
    values[31] = 1.0

    control = _control(TelemetryFrame(values))

    np.testing.assert_array_equal(control, np.asarray([1.0, 0.0, -1.0], dtype=np.float32))


def test_demonstration_round_trip_and_transition_conversion(tmp_path: Path) -> None:
    geometry = _geometry(tmp_path)
    path = save_demonstration(tmp_path / "lap", _demonstration(geometry))

    loaded = load_demonstration(path)
    context = DemonstrationTransitionContext(_config(geometry), geometry)
    transitions = demonstration_transitions(path, _IdentityPipeline(), context)

    assert loaded.finish_time_s == 36.0
    assert loaded.decision_interval_ms is None
    assert len(transitions) == 60
    assert transitions[-1].terminated
    assert transitions[-1].info["is_demo"] is True
    assert transitions[-1].info["sampling/projected_lap_time_s"] == 36.0
    _assert_demonstration_switches(loaded, transitions)


def _assert_demonstration_switches(demonstration: Demonstration, transitions: list[Any]) -> None:
    switches = [bool(item.info["demonstration_steering_switch"]) for item in transitions]
    distances = [int(item.info["demonstration_steering_switch_distance"]) for item in transitions]
    expected = [False]
    expected.extend(
        left // 6 != right // 6
        for left, right in zip(demonstration.actions[:-1], demonstration.actions[1:], strict=True)
    )
    assert switches == expected
    switch_indices = [index for index, value in enumerate(expected) if value]
    expected_distances = [60] * 60
    if switch_indices:
        expected_distances = [
            min(abs(index - switch) for switch in switch_indices) for index in range(60)
        ]
    assert distances == expected_distances


def _archive_values(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


def test_demonstration_loader_rejects_an_outdated_archive(tmp_path: Path) -> None:
    geometry = _geometry(tmp_path)
    path = save_demonstration(tmp_path / "lap", _demonstration(geometry))
    values = _archive_values(path)
    values["format"] = np.asarray("trackmaniarl-trackmania-demo-v4")
    np.savez_compressed(path, **values)

    with pytest.raises(ValueError, match="unsupported"):
        load_demonstration(path)


def test_resample_demonstration_uses_the_online_decision_interval(tmp_path: Path) -> None:
    geometry = _geometry(tmp_path)
    demonstration = _demonstration(geometry)
    frames = demonstration.frames.copy()
    frames[:, 3] = np.arange(len(frames), dtype=np.float32) * 10.0
    demonstration = replace(demonstration, frames=frames, finish_time_s=0.6)

    selected_frames, selected_actions = _resample(demonstration, 20.0)

    assert np.array_equal(selected_frames[:, 3], frames[::2, 3])
    assert len(selected_actions) == len(selected_frames) - 1


def test_resample_demonstration_leads_actions_by_race_time(tmp_path: Path) -> None:
    demonstration, _ = _lead_demonstration(tmp_path)

    selected_frames, selected_actions = _resample(
        demonstration, 20.0, DemonstrationResamplingConfig(action_lead_ms=20.0)
    )

    assert selected_frames[:, 3].tolist() == [0.0, 20.0, 40.0, 50.0]
    assert selected_actions.tolist() == [3, 75, 75]


def test_resample_demonstration_aggregates_keyboard_pwm_into_analog_action(
    tmp_path: Path,
) -> None:
    demonstration, table = _controlled_demonstration(tmp_path, _pwm_controls())

    _, default_actions = _resample(demonstration, 50.0)
    _, aggregated_actions = _resample(
        demonstration, 50.0, DemonstrationResamplingConfig(aggregate_controls=True)
    )

    assert int(default_actions[0]) == int(demonstration.actions[0])
    gas, brake, steer = table[int(aggregated_actions[0])]
    assert gas == pytest.approx(1.0)
    assert brake == pytest.approx(BRAKE_TAP_SENTINEL)
    assert steer == pytest.approx(1.0 / 6.0)


def test_expert_action_diagnostics_report_switch_and_steady_accuracy() -> None:
    diagnostics = _expert_action_diagnostics()
    summary = diagnostics.action_summary()

    _assert_expert_action_summary(summary)
    aggregate = aggregate_expert_actions((summary, summary))
    assert aggregate["count"] == 10
    assert aggregate["steering_bin_accuracy"] == pytest.approx(0.8)


def _expert_action_diagnostics() -> ExpertActionDiagnostics:
    diagnostics = ExpertActionDiagnostics(bin_count=1)
    actions = ((6, 6), (7, 8), (13, 8), (13, 14), (13, 13))
    for expert, greedy in actions:
        diagnostics.record(
            ExpertDiagnosticRecord(
                50.0,
                1.0,
                2.0,
                2,
                expert,
                greedy,
                expert // 6,
                greedy // 6,
            )
        )
    return diagnostics


def _assert_expert_action_summary(summary: dict[str, float]) -> None:
    assert summary["count"] == 5
    assert summary["exact_action_accuracy"] == pytest.approx(0.4)
    assert summary["steering_bin_accuracy"] == pytest.approx(0.8)
    assert summary["expert_action_switch_count"] == 2
    assert summary["policy_action_switch_count"] == 3
    assert summary["action_switch_recall"] == pytest.approx(0.5)
    assert summary["expert_steering_switch_count"] == 1
    assert summary["policy_steering_switch_count"] == 1
    assert summary["steering_switch_recall"] == 0.0
    assert summary["expert_steering_switch_step_accuracy"] == 0.0
    assert summary["expert_steering_steady_step_accuracy"] == 1.0
    assert summary["expert_action_steady_step_exact_accuracy"] == pytest.approx(0.5)


def test_expert_diagnostics_use_composite_model_with_dict_observations(
    tmp_path: Path,
) -> None:
    case = _expert_diagnostic_case(tmp_path)

    _, diagnostics = _expert_diagnostics(case.path, case.context)
    _, repeated = _expert_diagnostics(case.path, case.context)

    summary = diagnostics.action_summary()
    assert case.source_count == 60
    assert summary["count"] == case.evaluated_count == 12
    assert 0.0 <= summary["exact_action_accuracy"] <= 1.0
    assert 0.0 <= summary["steering_bin_accuracy"] <= 1.0
    assert repeated.action_summary() == summary


def _expert_diagnostic_case(tmp_path: Path) -> _ExpertDiagnosticCase:
    geometry = _geometry(tmp_path)
    demonstration = _expert_contract_demonstration(geometry)
    path = save_demonstration(tmp_path / "expert", demonstration)
    config, evaluated_count = _expert_contract_config(geometry, demonstration)
    context = _expert_policy_context(config)
    return _ExpertDiagnosticCase(path, context, len(demonstration.actions), evaluated_count)


def _expert_contract_demonstration(geometry: BoundaryGeometry) -> Demonstration:
    demonstration = _demonstration(geometry)
    frames = _frames()
    frames[:, 3] = np.arange(len(frames), dtype=np.float32) * 10.0
    actions = np.resize(np.asarray([3, 39, 75], dtype=np.int64), len(frames) - 1)
    _, table = build_brake_tap_action_table()
    return replace(
        demonstration,
        action_repeat_frames=1,
        frames=frames,
        actions=actions,
        controls=np.asarray([table[action] for action in actions], dtype=np.float32),
        finish_time_s=0.6,
    )


def _expert_contract_config(
    geometry: BoundaryGeometry, demonstration: Demonstration
) -> tuple[TrackmaniaEnvironmentConfig, int]:
    config = _config(geometry, action_repeat_frames=1).model_copy(
        update={
            "decision_interval_ms": 50.0,
            "demonstration_action_lead_ms": 20.0,
            "demonstration_control_aggregation": True,
            "minimum_finish_steps": 1,
            "limit_progress_by_kinematics": False,
        }
    )
    _, selected_actions = resample_demonstration_for_environment(demonstration, config)
    compact_ids = tuple(sorted(int(action) for action in np.unique(selected_actions)))
    config = config.model_copy(update={"compact_action_ids": compact_ids})
    return config, len(selected_actions)


def _expert_policy_context(config: TrackmaniaEnvironmentConfig) -> _ExpertContext:
    compact_ids = config.compact_action_ids
    if compact_ids is None:
        raise RuntimeError("expert test requires compact actions")
    learner = SimpleNamespace(
        model=_expert_value_model(len(compact_ids)),
        device=torch.device("cpu"),
        evaluation_risk=RiskSpec(),
    )
    geometry = _geometry_from_config(config)
    return _ExpertContext(learner, _DictTensorPipeline(), config, geometry)


def _expert_value_model(action_count: int) -> CompositeValueModel:
    return CompositeValueModel(
        CompositeModules(
            _GraphDictEncoder(),
            GruTemporalCore(8, 8),
            ImplicitQuantileHead(ImplicitQuantileHeadConfig(8, action_count, 8, True)),
            RandomQuantileStrategy(4, 5, 6),
        )
    )


def _geometry_from_config(config: TrackmaniaEnvironmentConfig) -> BoundaryGeometry:
    return BoundaryGeometry(config.geometry_path, expected_map_uid=config.expected_map_uid)
