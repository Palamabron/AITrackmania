from trackmaniarl.observability.wandb_metrics import _event_metrics


def test_wandb_projects_elite_replay_metrics() -> None:
    metrics = _event_metrics(
        "train/update",
        {
            "replay/elite_active_fraction": 0.08,
            "replay/elite_sample_fraction": 0.21,
        },
    )

    assert metrics == {
        "replay/elite_active_fraction": 0.08,
        "replay/elite_sample_fraction": 0.21,
    }


def test_wandb_projects_episode_relabel_count() -> None:
    metrics = _event_metrics("train/episode", {"replay/labeled_transitions": 731})

    assert metrics == {"episode/replay_labeled_transitions": 731}
