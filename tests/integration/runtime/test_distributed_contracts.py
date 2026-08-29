from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from tests.integration.runtime.distributed_runtime_support import (
    _DISTRIBUTED_TOKEN,
    _Context,
)
from trackmaniarl.core import fingerprint
from trackmaniarl.core.data import BatchRequest
from trackmaniarl.core.spec import RunSpec
from trackmaniarl.distributed.actor import (
    ActorRuntime,
)
from trackmaniarl.distributed.actor_requests import ActorProcessRequest
from trackmaniarl.distributed.coordinator import Coordinator
from trackmaniarl.distributed.coordinator_ingest import replay_info_for_transition
from trackmaniarl.distributed.coordinator_support import _BatchPrefetcher, _MetricAccumulator
from trackmaniarl.distributed.coordinator_types import CoordinatorConfig
from trackmaniarl.distributed.protocol import (
    authenticate,
    run_fingerprint,
)


class _CountingSampler:
    def __init__(self, batch: object) -> None:
        self.batch = batch
        self.calls = 0

    def sample(self, store: object, request: BatchRequest) -> object:
        del store, request
        self.calls += 1
        return self.batch


class _CountingLearner:
    def __init__(self) -> None:
        self.prepare_calls = 0

    def prepare_batch(self, value: Any) -> Any:
        self.prepare_calls += 1
        return value


def test_metric_accumulator_averages_each_metric_over_its_own_observations() -> None:
    metrics = _MetricAccumulator()

    metrics.add({"loss/total": 2.0})
    metrics.add({"loss/total": 4.0, "debug/q_selected_mean": 8.0})
    metrics.add({"debug/td_abs_max": 5.0})
    metrics.add({"debug/td_abs_max": 3.0})

    assert metrics.flush() == {
        "loss/total": 3.0,
        "debug/q_selected_mean": 8.0,
        "debug/td_abs_max": 5.0,
    }
    assert metrics.flush() == {}


def test_online_partial_progress_is_not_mislabeled_as_an_elite_lap() -> None:
    partial = replay_info_for_transition({"progress_pct": 75.0, "race_time_ms": 27_000.0})
    expert = replay_info_for_transition(
        {
            "is_demo": True,
            "progress_pct": 75.0,
            "race_time_ms": 27_000.0,
            "sampling/projected_lap_time_s": 36.035,
        }
    )

    assert "sampling/projected_lap_time_s" not in partial
    assert expert == {
        "is_demo": True,
        "sampling/projected_lap_time_s": 36.035,
    }


def test_non_overlapped_batch_path_never_prepares_or_speculatively_samples() -> None:
    batch = object()
    sampler = _CountingSampler(batch)
    learner = _CountingLearner()
    prefetcher = _BatchPrefetcher(
        cast(Any, SimpleNamespace(sampler=sampler, learner=learner, replay_store=object()))
    )

    first, _, first_wait = prefetcher.next(BatchRequest(batch_size=1))
    second, _, second_wait = prefetcher.next(BatchRequest(batch_size=1))
    prefetcher.close()

    assert first is batch
    assert second is batch
    assert sampler.calls == 2
    assert learner.prepare_calls == 0
    assert first_wait == second_wait == 0.0


def test_distributed_runtimes_reject_short_tokens(tmp_path: Path) -> None:
    config = ActorProcessRequest(tmp_path / "missing.yaml", "127.0.0.1:8787", "actor", "short")
    with pytest.raises(ValueError, match="at least 32 characters"):
        ActorRuntime(config)
    with pytest.raises(ValueError, match="at least 32 characters"):
        Coordinator(
            cast(Any, object()),
            CoordinatorConfig("127.0.0.1:8787", "short", "fingerprint"),
        )


def test_actor_policy_replica_context_cannot_write_run_artifacts() -> None:
    actor = object.__new__(ActorRuntime)
    actor.actor_id = "actor"
    actor.base_dir = Path("project")
    actor.spec = SimpleNamespace(seed=7, artifacts_dir=Path("artifacts"), run_id="run")
    model_factory = object()

    context = actor._replica_context(model_factory)

    assert context == {
        "seed": actor._actor_seed(),
        "model_factory": model_factory,
        "restoring_checkpoint": True,
    }


def test_authentication_requires_the_distributed_token() -> None:
    authenticate(cast(Any, _Context(f"Bearer {_DISTRIBUTED_TOKEN}")), _DISTRIBUTED_TOKEN)
    with pytest.raises(RuntimeError, match="UNAUTHENTICATED"):
        authenticate(cast(Any, _Context("Bearer wrong")), _DISTRIBUTED_TOKEN)


def test_run_fingerprint_covers_semantic_assets(tmp_path: Path) -> None:
    geometry = tmp_path / "geometry.npz"
    pace = tmp_path / "pace.npz"
    geometry.write_bytes(b"geometry-v1")
    pace.write_bytes(b"pace-v1")
    config = _geometry_config(geometry.name, pace.name)
    first = run_fingerprint(RunSpec.model_validate(config), tmp_path)
    config["run_id"] = "run-b"
    assert run_fingerprint(RunSpec.model_validate(config), tmp_path) == first
    pace.write_bytes(b"pace-v2")
    assert run_fingerprint(RunSpec.model_validate(config), tmp_path) != first
    pace.write_bytes(b"pace-v1")
    geometry.write_bytes(b"geometry-v2")
    assert run_fingerprint(RunSpec.model_validate(config), tmp_path) != first


def test_run_fingerprint_covers_first_party_source_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = {
        "api_version": "2.0",
        "run_id": "source-digest",
        "components": _fingerprint_components("trackmaniarl.core.builtins:SmokeLearner"),
    }
    spec = RunSpec.model_validate(config)
    baseline = run_fingerprint(spec, tmp_path)

    monkeypatch.setattr(fingerprint, "_trackmaniarl_source_digest", lambda: "changed")

    assert run_fingerprint(spec, tmp_path) != baseline


def test_run_fingerprint_accepts_nested_component_without_kwargs(tmp_path: Path) -> None:
    components = _fingerprint_components("trackmaniarl.core.builtins:SmokeLearner")
    components["learner"]["kwargs"] = {
        "component": {"class_path": "trackmaniarl.core.builtins:IdentityFeaturePipeline"}
    }
    spec = RunSpec.model_validate(
        {"api_version": "2.0", "run_id": "nested-component", "components": components}
    )

    assert len(run_fingerprint(spec, tmp_path)) == 64


def _geometry_config(geometry_name: str, pace_name: str) -> dict[str, Any]:
    components = _fingerprint_components("trackmaniarl.core.builtins:SmokeLearner")
    components["feature_pipeline"] = _lidar_config(geometry_name, pace_name)
    return {
        "api_version": "2.0",
        "run_id": "run-a",
        "components": components,
        "evaluation": {
            "name": "map",
            "version": "1",
            "maps": [_map_config(geometry_name)],
        },
    }


def _lidar_config(geometry_name: str, pace_name: str) -> dict[str, Any]:
    return {
        "class_path": "trackmaniarl.trackmania.features:LidarFeaturePipeline",
        "kwargs": {
            "config": {
                "geometry_path": geometry_name,
                "pace_reference_path": pace_name,
            }
        },
    }


def _map_config(geometry_name: str) -> dict[str, str]:
    return {
        "id": "map",
        "map_path": "map.Map.Gbx",
        "geometry_path": geometry_name,
        "expected_map_uid": "uid",
    }


def _fingerprint_components(learner: str) -> dict[str, dict[str, Any]]:
    return {
        "learner": {"class_path": learner},
        "replay_store": {"class_path": "trackmaniarl.core.replay:InMemoryReplayStore"},
        "sampler": {"class_path": "trackmaniarl.core.replay:UniformSampler"},
        "feature_pipeline": {"class_path": "trackmaniarl.core.builtins:IdentityFeaturePipeline"},
    }


def test_run_fingerprint_hashes_reexported_implementation_and_effective_parameters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    implementation, helper = _write_fingerprint_package(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    config = _implementation_config()
    implicit_default = run_fingerprint(RunSpec.model_validate(config), tmp_path)
    _assert_parameter_fingerprints(config, tmp_path, implicit_default)
    helper.write_text("WIDTH = 5\n", encoding="utf-8")
    assert run_fingerprint(RunSpec.model_validate(config), tmp_path) != implicit_default
    _change_implementation(implementation)
    assert run_fingerprint(RunSpec.model_validate(config), tmp_path) != implicit_default


def _write_fingerprint_package(tmp_path: Path) -> tuple[Path, Path]:
    package = tmp_path / "fingerprint_package"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    helper = package / "helper.py"
    helper.write_text("WIDTH = 4\n", encoding="utf-8")
    implementation = package / "implementation.py"
    implementation.write_text(
        "from fingerprint_package.helper import WIDTH\n\n"
        "class Component:\n"
        "    def __init__(self, width=WIDTH):\n"
        "        self.width = width\n",
        encoding="utf-8",
    )
    (package / "reexport.py").write_text(
        "from fingerprint_package.implementation import Component\n", encoding="utf-8"
    )
    return implementation, helper


def _implementation_config() -> dict[str, Any]:
    return {
        "api_version": "2.0",
        "run_id": "fingerprint",
        "components": _fingerprint_components("fingerprint_package.reexport:Component"),
    }


def _assert_parameter_fingerprints(config: dict[str, Any], tmp_path: Path, baseline: str) -> None:
    explicit = deepcopy(config)
    explicit["components"]["learner"]["kwargs"] = {"width": 4}
    assert run_fingerprint(RunSpec.model_validate(explicit), tmp_path) == baseline
    changed = deepcopy(config)
    changed["components"]["learner"]["kwargs"] = {"width": 5}
    assert run_fingerprint(RunSpec.model_validate(changed), tmp_path) != baseline


def _change_implementation(path: Path) -> None:
    path.write_text(
        "class Component:\n"
        "    implementation_version = 2\n"
        "    def __init__(self, width=4):\n"
        "        self.width = width\n",
        encoding="utf-8",
    )


def test_run_fingerprint_hashes_declared_wrapper_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    helper = _write_wrapper_package(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    components = _fingerprint_components("fingerprint_wrapper.reexport:Component")
    spec = RunSpec.model_validate(
        {"api_version": "2.0", "run_id": "wrapper", "components": components}
    )
    baseline = run_fingerprint(spec, tmp_path)

    helper.write_text("SENTINEL = 2\n", encoding="utf-8")

    assert run_fingerprint(spec, tmp_path) != baseline


def _write_wrapper_package(tmp_path: Path) -> Path:
    package = tmp_path / "fingerprint_wrapper"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    helper = package / "helper.py"
    helper.write_text("SENTINEL = 1\n", encoding="utf-8")
    (package / "reexport.py").write_text(
        "from functools import partial\n"
        "from fingerprint_wrapper.helper import SENTINEL\n"
        "from trackmaniarl.core.builtins import SmokeLearner\n\n"
        "assert SENTINEL\n"
        "Component = partial(SmokeLearner)\n",
        encoding="utf-8",
    )
    return helper
