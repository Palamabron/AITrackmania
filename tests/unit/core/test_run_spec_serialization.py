"""Run specifications must round-trip between Python, JSON-compatible data and YAML."""

from __future__ import annotations

import warnings

import pytest
from pydantic import BaseModel, ValidationError

from trackmaniarl.core.spec import DistributedSpec, EvaluationSuiteSpec, RunSpec, TrainingSpec


def _run_payload() -> dict[str, object]:
    return {
        "run_id": "round-trip",
        "components": {
            "learner": {"class_path": "trackmaniarl.core.builtins:SmokeLearner"},
            "replay_store": {"class_path": "trackmaniarl.core.replay:InMemoryReplayStore"},
            "sampler": {"class_path": "trackmaniarl.core.replay:UniformSampler"},
            "feature_pipeline": {
                "class_path": "trackmaniarl.core.builtins:IdentityFeaturePipeline"
            },
        },
    }


def test_run_spec_is_frozen_and_round_trips_through_yaml(tmp_path) -> None:
    spec = RunSpec.model_validate({"api_version": "2.0", **_run_payload()})
    path = tmp_path / "run.yaml"
    path.write_text(spec.to_yaml(), encoding="utf-8")
    assert RunSpec.from_yaml(path) == spec


def test_run_spec_requires_explicit_supported_api_version() -> None:
    with pytest.raises(ValidationError, match="api_version"):
        RunSpec.model_validate(_run_payload())
    with pytest.raises(ValidationError, match="api_version"):
        RunSpec.model_validate({"api_version": "1.0", **_run_payload()})


def test_run_spec_json_schema_has_only_serializable_defaults() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        schema = RunSpec.model_json_schema()

    assert schema["properties"]["artifacts_dir"]["format"] == "path"


def test_actor_execution_override_is_explicit_and_structured() -> None:
    assert DistributedSpec().actor_execution is None

    distributed = DistributedSpec.model_validate(
        {
            "actor_execution": {
                "device": "cpu",
                "precision": "bfloat16",
                "torch_threads": 2,
            }
        }
    )

    assert distributed.actor_execution is not None
    assert distributed.actor_execution.device == "cpu"
    assert distributed.actor_execution.precision == "bfloat16"
    assert distributed.actor_execution.torch_threads == 2


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (TrainingSpec, {"updates_per_transition": float("inf")}),
        (DistributedSpec, {"heartbeat_s": float("inf")}),
        (
            EvaluationSuiteSpec,
            {"name": "suite", "version": "1", "time_buckets_s": [1.0, float("inf")]},
        ),
    ],
)
def test_public_numeric_specs_reject_non_finite_values(
    model: type[BaseModel], payload: dict[str, object]
) -> None:
    with pytest.raises(ValidationError, match="finite"):
        model.model_validate(payload)


@pytest.mark.parametrize(
    ("training", "components", "evaluation", "message"),
    [
        (
            {
                "evaluation_stop_min_finish_rate": 0.9,
                "evaluation_stop_median_s": 40.0,
                "evaluation_stop_consecutive_batches": 2,
            },
            {},
            None,
            "evaluate_every_episodes",
        ),
        (
            {
                "evaluate_every_episodes": 1,
                "evaluation_stop_min_finish_rate": 0.9,
                "evaluation_stop_median_s": 40.0,
                "evaluation_stop_consecutive_batches": 2,
            },
            {},
            None,
            "components.evaluator",
        ),
        (
            {
                "evaluate_every_episodes": 1,
                "evaluation_stop_min_finish_rate": 0.9,
                "evaluation_stop_median_s": 40.0,
                "evaluation_stop_consecutive_batches": 2,
            },
            {"evaluator": {"class_path": "trackmaniarl.core.builtins:NullEvaluator"}},
            None,
            "evaluation suite",
        ),
    ],
)
def test_evaluation_stop_requires_an_active_evaluation_path(
    training: dict[str, object],
    components: dict[str, object],
    evaluation: dict[str, object] | None,
    message: str,
) -> None:
    payload = {"api_version": "2.0", **_run_payload(), "training": training}
    component_payload = payload["components"]
    assert isinstance(component_payload, dict)
    payload["components"] = {**component_payload, **components}
    if evaluation is not None:
        payload["evaluation"] = evaluation

    with pytest.raises(ValidationError, match=message):
        RunSpec.model_validate(payload)
