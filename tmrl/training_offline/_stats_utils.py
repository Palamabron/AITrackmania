"""Wandb logging constants and stats aggregation for training_offline."""

from numbers import Real
from typing import Any

import torch

import tmrl.config.config_objects as cfg_obj


def _is_iqn_algorithm() -> bool:
    """Return True when ``MainConfig`` selects IQN.

    Must not be evaluated at `_stats_utils` import time: `config_objects` imports
    ``TorchTrainingOffline`` before assigning ``ALG_NAME``, so a module-level constant
    would always see an empty algorithm name during that cycle.
    """
    return getattr(cfg_obj, "ALG_NAME", "") == "IQN"


def _wandb_round_keys() -> tuple[str, ...]:
    """Keys required for wandb round-level logging (mirrors networking.run_with_wandb)."""
    if _is_iqn_algorithm():
        return (
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
    return (
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
    """Build a sanitized dict from a round stat Series for ``wandb.log`` (mirrors networking).

    Replaces invalid values (None, NaN, ±inf) with algorithm-appropriate defaults:
    loss keys get ``float('nan')``; metric/eval keys get ``0.0``; anything else
    becomes ``None``. Ensures all keys expected by the wandb dashboard are present.

    Args:
        round_series: A pandas Series or dict of round-level statistics as produced
            by :func:`pandas_dict`.

    Returns:
        dict[str, Any]: A sanitized mapping from metric name to numeric value (or
            ``float('nan')`` / ``0.0``) ready to pass to ``wandb.log``.
    """
    log_dict = round_series.to_dict() if hasattr(round_series, "to_dict") else dict(round_series)
    if _is_iqn_algorithm():
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
    for key in _wandb_round_keys():
        if key not in log_dict or log_dict[key] is None:
            log_dict[key] = float("nan") if key.startswith("losses/") else 0.0
    return log_dict


def _stats_dict_to_numeric(d: dict) -> dict:
    """Convert tensor values in a stats dict to Python scalars so pandas can aggregate.

    Args:
        d: A stats dictionary whose values may be ``torch.Tensor`` or plain scalars.

    Returns:
        dict: A copy of ``d`` where every ``torch.Tensor`` is replaced by a Python
            ``float`` (scalar tensors via ``.item()``; multi-element tensors via ``.mean()``).
    """
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.item() if v.numel() == 1 else float(v.mean().item())
        else:
            out[k] = v
    return out


def _mean_stats_dicts(items: list[dict[str, Any]]) -> dict[str, float]:
    """Fast mean aggregation without pandas DataFrame construction.

    Computes the per-key mean across a list of stat dicts, skipping NaN and ±inf
    values so a single bad batch does not corrupt round-level averages.

    Args:
        items: List of per-batch stat dicts produced by :func:`_stats_dict_to_numeric`.

    Returns:
        dict[str, float]: Mapping from metric name to its mean over all items that
            contributed a finite value. Keys absent in every item are omitted.
    """
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
