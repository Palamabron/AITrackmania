from __future__ import annotations

import threading
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from tests.integration.runtime.distributed_runtime_support import _Pipeline
from trackmaniarl.core.contracts import PolicyMode
from trackmaniarl.core.data import Transition
from trackmaniarl.distributed.actor import ActorRuntime
from trackmaniarl.distributed.actor_requests import SpoolRequest
from trackmaniarl.distributed.codec import WireCodec


@dataclass(slots=True)
class _EvaluationProbe:
    modes: list[PolicyMode] = field(default_factory=list)
    requests: list[SpoolRequest] = field(default_factory=list)

    def spool(self, request: SpoolRequest) -> None:
        self.requests.append(request)


class _EvaluationPolicy:
    def __init__(self, probe: _EvaluationProbe) -> None:
        self.probe = probe

    def act(self, observation: Any, mode: PolicyMode = PolicyMode.ONLINE) -> int:
        del observation
        self.probe.modes.append(mode)
        return 0

    def export_state(self) -> dict[str, Any]:
        return {}


class _EvaluationEnvironment:
    def __init__(self) -> None:
        self.steps = 0

    def reset(self, *, seed: int) -> tuple[int, dict[str, Any]]:
        assert seed == 1_000_007
        return 0, {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        assert action == 0
        self.steps += 1
        if self.steps == 1:
            return 1, 2.0, False, False, {"reward_time": -0.1, "reward_pbrs": 2.1}
        return 2, 3.0, True, False, _finished_evaluation_info()


def _finished_evaluation_info() -> dict[str, Any]:
    return {
        "termination_reason": "finished",
        "race_time_ms": 12_500.0,
        "reward_time": -0.2,
        "reward_pbrs": 3.2,
        "reward_terminal": 10.0,
    }


def _evaluation_actor(probe: _EvaluationProbe) -> ActorRuntime:
    actor = object.__new__(ActorRuntime)
    actor.spec = SimpleNamespace(
        evaluation=None,
        training=SimpleNamespace(max_episode_steps=3),
    )
    actor.codec = WireCodec(1024)
    actor.stop = threading.Event()
    actor._evaluation_request_lock = threading.Lock()
    actor._evaluation_request = None
    actor._evaluation_index = 0
    actor._actor_seed = lambda: 7
    actor._policy = lambda: (_EvaluationPolicy(probe), 0.5, 9)
    actor._spool = probe.spool
    return actor


def test_actor_evaluation_is_greedy_and_never_spooled_as_training_data() -> None:
    probe = _EvaluationProbe()

    _evaluation_actor(probe)._evaluate(_EvaluationEnvironment(), _Pipeline())

    request = probe.requests[0]
    summary = request.evaluations[0] if request.evaluations else {}
    assert probe.modes == [PolicyMode.EVALUATION] * 3
    assert (request.transitions, request.summaries, request.policy_version) == ([], [], 9)
    assert summary["finish_time_s"] == 12.5
    assert summary["reward/time"] == pytest.approx(-0.3)
    assert summary["reward/pbrs"] == pytest.approx(5.3)
    assert summary["reward/terminal"] == 10.0


class _MarginPolicy:
    def __init__(self, version: int) -> None:
        self.version = version
        self.margins = iter((3.0, 1.0, 2.0))
        self.last_q_margin: float | None = None
        self.calls = 0

    def act(self, observation: Any, mode: PolicyMode = PolicyMode.ONLINE) -> int:
        del observation, mode
        self.calls += 1
        self.last_q_margin = next(self.margins)
        return 0


@dataclass(slots=True)
class _TrainingProbe:
    policies: list[_MarginPolicy] = field(default_factory=list)
    requests: list[SpoolRequest] = field(default_factory=list)

    def next_policy(self) -> tuple[_MarginPolicy, float, int]:
        policy = _MarginPolicy(len(self.policies))
        self.policies.append(policy)
        return policy, 0.1, policy.version

    def spool(self, request: SpoolRequest) -> None:
        self.requests.append(request)


class _TrainingEnvironment:
    def __init__(self, stop: threading.Event) -> None:
        self.stop = stop
        self.episode_steps = 0
        self.total_steps = 0

    def reset(self, *, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
        del seed
        self.episode_steps = 0
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert action == 0
        self.episode_steps += 1
        self.total_steps += 1
        terminal = self.episode_steps == 3
        if self.total_steps == 6:
            self.stop.set()
        return np.zeros(1, dtype=np.float32), 1.0, terminal, False, self._info()

    def _info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "control_gas": 1.0,
            "control_brake": 0.0,
            "control_steer": 0.5,
            "step_race_time_ms": 66.0,
        }
        if self.episode_steps == 3:
            info.update({"termination_reason": "finished", "race_time_ms": 1_000.0})
        return info


def _training_actor(probe: _TrainingProbe) -> ActorRuntime:
    actor = object.__new__(ActorRuntime)
    actor.spec = SimpleNamespace(
        training=SimpleNamespace(max_episode_steps=10),
        distributed=SimpleNamespace(rollout_chunk_transitions=128, rollout_flush_s=60.0),
    )
    actor.actor_id = "actor"
    actor.session_id = "session"
    actor.stop = threading.Event()
    actor.evaluate = threading.Event()
    actor._actor_seed = lambda: 7
    actor._policy = probe.next_policy
    actor._spool = probe.spool
    return actor


def _assert_episode_versions(probe: _TrainingProbe) -> None:
    acting = [policy for policy in probe.policies if policy.calls > 1]
    assert [policy.calls for policy in acting] == [3, 3]
    versions = [
        {item.info["policy_version"] for item in request.transitions}
        for request in probe.requests
        if request.transitions
    ]
    assert versions == [{acting[0].version}, {acting[1].version}]


def _assert_summary(summary: dict[str, Any]) -> None:
    assert summary["q_margin/mean"] == pytest.approx(2.0)
    assert summary["q_margin/min"] == pytest.approx(1.0)
    assert summary["q_margin/start_mean"] == pytest.approx(2.0)
    assert summary["control/gas_fraction"] == pytest.approx(1.0)
    assert summary["control/brake_fraction"] == 0.0
    assert summary["control/steer_abs_mean"] == pytest.approx(0.5)
    assert summary["timing/step_race_ms_mean"] == pytest.approx(66.0)
    assert summary["timing/step_race_ms_p99"] == pytest.approx(66.0)
    assert summary["timing/step_race_ms_max"] == pytest.approx(66.0)
    assert summary["termination/time_limit"] == 0.0


def test_actor_training_episode_freezes_one_policy_and_reports_action_gaps() -> None:
    probe = _TrainingProbe()
    actor = _training_actor(probe)

    actor._collect(_TrainingEnvironment(actor.stop), _Pipeline())

    _assert_episode_versions(probe)
    summary = probe.requests[0].summaries[0]
    _assert_summary(summary)
    assert summary["episode_id"] == "actor/session/00000000"


def _assert_chunked_episode_requests(requests: list[SpoolRequest]) -> None:
    transition_requests = [request for request in requests if request.transitions]
    summary_requests = [request for request in requests if request.summaries]
    assert len(transition_requests) == 6
    assert len(summary_requests) == 2
    assert all(not request.transitions for request in summary_requests)
    transition_ids = {
        transition.episode_id
        for request in transition_requests
        for transition in request.transitions
    }
    summary_ids = {
        summary["episode_id"] for request in summary_requests for summary in request.summaries
    }
    expected = {"actor/session/00000000", "actor/session/00000001"}
    assert summary_ids == transition_ids == expected


def test_actor_sends_identified_summary_after_flushed_chunks() -> None:
    probe = _TrainingProbe()
    actor = _training_actor(probe)
    actor.spec.distributed.rollout_chunk_transitions = 1

    actor._collect(_TrainingEnvironment(actor.stop), _Pipeline())

    _assert_chunked_episode_requests(probe.requests)


class _PrewarmPolicy:
    def __init__(self) -> None:
        self.state = 0
        self.modes: list[PolicyMode] = []
        self.states: list[int] = []

    def act(self, observation: Any, mode: PolicyMode = PolicyMode.ONLINE) -> int:
        del observation
        self.modes.append(mode)
        self.states.append(self.state)
        self.state += 1
        return 0

    def reset_episode(self) -> None:
        self.state = 0


class _PrewarmPipeline(_Pipeline):
    def __init__(self) -> None:
        self.reset_count = 0

    def reset_episode(self) -> None:
        self.reset_count += 1


class _PrewarmEnvironment:
    def __init__(self, stop: threading.Event) -> None:
        self.stop = stop
        self.reset_count = 0
        self.step_count = 0

    def reset(self, *, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
        del seed
        self.reset_count += 1
        return np.asarray([self.reset_count], dtype=np.float32), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert action == 0
        self.step_count += 1
        self.stop.set()
        info = {"termination_reason": "finished", "race_time_ms": 1_000.0}
        return np.zeros(1, dtype=np.float32), 1.0, True, False, info


@dataclass(slots=True)
class _PrewarmProbe:
    policy: _PrewarmPolicy = field(default_factory=_PrewarmPolicy)
    requests: list[SpoolRequest] = field(default_factory=list)

    def current_policy(self) -> tuple[_PrewarmPolicy, float, int]:
        return self.policy, 0.1, 3

    def spool(self, request: SpoolRequest) -> None:
        self.requests.append(request)


def _prewarm_actor(probe: _PrewarmProbe) -> ActorRuntime:
    actor = _training_actor(_TrainingProbe())
    actor._policy = probe.current_policy
    actor._spool = probe.spool
    return actor


def test_actor_prewarms_once_before_collecting_the_first_transition() -> None:
    probe = _PrewarmProbe()
    actor = _prewarm_actor(probe)
    pipeline = _PrewarmPipeline()
    environment = _PrewarmEnvironment(actor.stop)
    actor._collect(environment, pipeline)
    transitions = [item for request in probe.requests for item in request.transitions]
    assert environment.reset_count == 2
    assert environment.step_count == 1
    assert probe.policy.modes == [PolicyMode.EVALUATION, PolicyMode.ONLINE]
    assert probe.policy.states == [0, 0]
    assert pipeline.reset_count == 3
    assert len(transitions) == 1


class _InterruptPolicy:
    action_count = 2

    def act(self, observation: Any, mode: PolicyMode = PolicyMode.ONLINE) -> int:
        del observation, mode
        return 0


class _InterruptedEnvironment:
    def __init__(self, stop: threading.Event) -> None:
        self.stop = stop
        self.steps = 0

    def reset(self, *, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
        del seed
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert action == 0
        self.steps += 1
        if self.steps == 1:
            return np.ones(1, dtype=np.float32), 1.0, False, False, {}
        self.stop.set()
        raise TimeoutError("telemetry packet missing")


@dataclass(slots=True)
class _InterruptionProbe:
    transitions: list[list[Transition]] = field(default_factory=list)

    def spool(self, request: SpoolRequest) -> None:
        self.transitions.append(list(request.transitions))


def _interruption_actor(probe: _InterruptionProbe) -> ActorRuntime:
    actor = object.__new__(ActorRuntime)
    actor.spec = SimpleNamespace(
        training=SimpleNamespace(max_episode_steps=3),
        distributed=SimpleNamespace(rollout_chunk_transitions=128, rollout_flush_s=60.0),
    )
    actor.actor_id = "actor"
    actor.session_id = "session"
    actor.stop = threading.Event()
    actor.evaluate = threading.Event()
    actor._actor_seed = lambda: 7
    actor._policy = lambda: (_InterruptPolicy(), 0.0, 1)
    actor._spool = probe.spool
    return actor


def test_actor_marks_mid_episode_telemetry_interruption_as_truncated() -> None:
    probe = _InterruptionProbe()
    actor = _interruption_actor(probe)

    actor._collect(_InterruptedEnvironment(actor.stop), _Pipeline())

    transition = next(batch[0] for batch in probe.transitions if batch)
    assert not transition.terminated
    assert transition.truncated
    assert transition.info["termination_reason"] == "telemetry_interruption"
