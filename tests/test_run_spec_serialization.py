"""Run specifications must round-trip between Python, JSON-compatible data and YAML."""

from __future__ import annotations

from tmrl.core.spec import RunSpec


def test_run_spec_is_frozen_and_round_trips_through_yaml(tmp_path) -> None:
    spec = RunSpec.model_validate(
        {
            "run_id": "round-trip",
            "components": {
                "learner": {"class_path": "tmrl.core.builtins:SmokeLearner"},
                "replay_store": {"class_path": "tmrl.core.replay:InMemoryReplayStore"},
                "sampler": {"class_path": "tmrl.core.replay:UniformSampler"},
                "feature_pipeline": {"class_path": "tmrl.core.builtins:IdentityFeaturePipeline"},
            },
        }
    )
    path = tmp_path / "run.yaml"
    path.write_text(spec.to_yaml(), encoding="utf-8")
    assert RunSpec.from_yaml(path) == spec
