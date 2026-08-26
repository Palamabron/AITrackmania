from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from trackmaniarl.core.spec import ComponentSpec, RunSpec
from trackmaniarl.distributed.actor import ActorRuntime


class _ReplicaLearner:
    def __init__(self) -> None:
        self.context: dict[str, Any] | None = None

    def setup(self, context: dict[str, Any]) -> None:
        self.context = context


def _spec(tmp_path: Path) -> RunSpec:
    return RunSpec.model_validate(
        {
            "api_version": "2.0",
            "run_id": "actor-execution",
            "artifacts_dir": str(tmp_path),
            "components": {
                "learner": {
                    "class_path": "tests.fake:Learner",
                    "kwargs": {
                        "execution": {
                            "device": "cuda",
                            "precision": "bfloat16",
                            "compile": True,
                            "compile_mode": "max-autotune",
                            "deterministic": False,
                        }
                    },
                },
                "environment": {"class_path": "tests.fake:Environment"},
                "replay_store": {"class_path": "trackmaniarl.core.replay:InMemoryReplayStore"},
                "sampler": {"class_path": "trackmaniarl.core.replay:UniformSampler"},
                "feature_pipeline": {"class_path": "tests.fake:Pipeline"},
            },
            "distributed": {
                "actor_execution": {
                    "device": "cpu",
                    "precision": "float32",
                    "torch_threads": 2,
                }
            },
        }
    )


def test_actor_components_override_only_replica_execution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec = _spec(tmp_path)
    replica = _ReplicaLearner()
    captured: dict[str, Any] = {}
    thread_counts: list[int] = []

    def instantiate(component: ComponentSpec, **injected: Any) -> Any:
        del injected
        if component.class_path == "tests.fake:Learner":
            captured.update(component.kwargs)
            return replica
        return SimpleNamespace()

    monkeypatch.setattr("trackmaniarl.distributed.actor._instantiate", instantiate)
    monkeypatch.setattr(
        "trackmaniarl.distributed.actor.torch.set_num_threads", thread_counts.append
    )
    actor = object.__new__(ActorRuntime)
    actor.spec = spec
    actor.base_dir = tmp_path
    actor.actor_id = "actor"
    actor._actor_seed = lambda: 7

    actor._components()

    execution = captured["execution"]
    assert execution == {
        "device": "cpu",
        "precision": "float32",
        "compile": False,
        "compile_mode": "max-autotune",
        "deterministic": False,
    }
    assert spec.components.learner.kwargs["execution"]["device"] == "cuda"
    assert spec.components.learner.kwargs["execution"]["compile"] is True
    assert replica.context is not None
    assert replica.context["seed"] == 7
    assert thread_counts == [2]
