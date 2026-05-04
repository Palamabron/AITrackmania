"""Wandb logging constants and stats aggregation for training_offline."""

from numbers import Real
from typing import Any

import torch

import tmrl.config.config_objects as cfg_obj

_IS_IQN = getattr(cfg_obj, "ALG_NAME", "") == "IQN"

# Keys that must be present for wandb round-level logging (same as networking.run_with_wandb).
if _IS_IQN:
    _WANDB_ROUND_KEYS: tuple[str, ...] = (
        "loss/iqn_loss",
        "metrics/return_test",
        "metrics/return_train",
        "metrics/episode_length_test",
        "metrics/episode_length_train",
        "eval/return_deterministic",
        "eval/episode_length_deterministic",
        "eval/finish_time_test_s",
        "eval/finished_track_count_test",
        "eval/competition_eliminated",
        "eval/competition_crashes",
    )
else:
    _WANDB_ROUND_KEYS = (
        "losses/actor",
        "losses/critic",
        "metrics/return_test",
        "metrics/return_train",
        "metrics/episode_length_test",
        "metrics/episode_length_train",
        "eval/return_deterministic",
        "eval/episode_length_deterministic",
        "eval/finish_time_test_s",
        "eval/finished_track_count_test",
        "eval/competition_eliminated",
        "eval/competition_crashes",
    )


def _round_stat_to_wandb_log_dict(round_series) -> dict[str, Any]:
    """Build a sanitized dict from a round stat Series for wandb.log (mirrors networking)."""
    log_dict = round_series.to_dict() if hasattr(round_series, "to_dict") else dict(round_series)
    if _IS_IQN:
        # IQN does not optimize actor/critic losses; avoid polluting wandb with NaNs.
        log_dict.pop("losses/actor", None)
        log_dict.pop("losses/critic", None)
    for k, v in list(log_dict.items()):
        is_invalid = v is None or (
            isinstance(v, float) and (v != v or v == float("inf") or v == float("-inf"))
        )
        if is_invalid:
            log_dict[k] = (
                float("nan")
                if k.startswith("losses/")
                else (
                    0.0
                    if k
                    in (
                        "metrics/return_test",
                        "metrics/return_train",
                        "metrics/episode_length_test",
                        "metrics/episode_length_train",
                        "eval/return_deterministic",
                        "eval/episode_length_deterministic",
                        "eval/finish_time_test_s",
                        "eval/finished_track_count_test",
                        "eval/competition_eliminated",
                        "eval/competition_crashes",
                    )
                    else None
                )
            )
    for key in _WANDB_ROUND_KEYS:
        if key not in log_dict or log_dict[key] is None:
            log_dict[key] = float("nan") if key.startswith("losses/") else 0.0
    return log_dict


def _stats_dict_to_numeric(d: dict) -> dict:
    """Convert tensor values in a stats dict to Python scalars so pandas can aggregate."""
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.item() if v.numel() == 1 else float(v.mean().item())
        else:
            out[k] = v
    return out


def _mean_stats_dicts(items: list[dict[str, Any]]) -> dict[str, float]:
    """Fast mean aggregation without pandas DataFrame construction."""
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in items:
        for k, v in row.items():
            if isinstance(v, Real):
                vf = float(v)
                if vf == vf and vf not in (float("inf"), float("-inf")):
                    sums[k] = sums.get(k, 0.0) + vf
                    counts[k] = counts.get(k, 0) + 1
    return {k: (sums[k] / counts[k]) for k in sums if counts.get(k, 0) > 0}
