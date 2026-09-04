from trackmaniarl.observability.trackers import WandbTracker, _EventContext
from trackmaniarl.observability.wandb_metrics import _event_metrics


def test_offline_pretraining_progress_projection() -> None:
    metrics = _event_metrics(
        "train/offline_pretrain_progress",
        {
            "debug/offline_progress_fraction": 0.25,
            "timing/offline_updates_per_s": 8.5,
            "timing/offline_eta_s": 10.0,
        },
    )

    assert metrics["learner/offline_progress_fraction"] == 0.25
    assert metrics["performance/offline_updates_per_s"] == 8.5
    assert metrics["performance/offline_eta_s"] == 10.0


def test_expert_progress_projection_and_axis() -> None:
    payload = {
        "count": 720.0,
        "demonstrations/completed": 1,
        "demonstrations/count": 23,
        "exact_action_accuracy": 0.6,
        "elapsed_s": 5.0,
        "eta_s": 110.0,
    }
    metrics = _event_metrics("diagnose/expert_progress", payload)
    tracker = WandbTracker.__new__(WandbTracker)
    axis = tracker._event_axis(_EventContext("diagnose/expert_progress", payload, 720))

    assert metrics["expert/demonstrations_completed"] == 1
    assert metrics["expert/exact_action_accuracy"] == 0.6
    assert metrics["expert/eta_s"] == 110.0
    assert axis == {"expert/transitions": 720}
