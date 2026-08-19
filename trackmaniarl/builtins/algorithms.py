"""Lazy catalogue of first-class TrackmaniaRL 1.0 learners."""

from __future__ import annotations

import importlib
from typing import Any, cast

_ALGORITHMS: dict[str, str] = {
    "proximal_policy_optimization": (
        "trackmaniarl.algorithms.proximal_policy_optimization:ProximalPolicyOptimization"
    ),
    "soft_actor_critic": "trackmaniarl.algorithms.soft_actor_critic:SoftActorCritic",
    "randomized_ensemble_sac": (
        "trackmaniarl.algorithms.randomized_ensemble_sac:RandomizedEnsembleSAC"
    ),
    "truncated_quantile_critic": (
        "trackmaniarl.algorithms.truncated_quantile_critic:TruncatedQuantileCritic"
    ),
    "implicit_quantile_q_learning": (
        "trackmaniarl.algorithms.implicit_quantile_q_learning:ImplicitQuantileQLearning"
    ),
    "stable_discrete_soft_actor_critic": (
        "trackmaniarl.algorithms.stable_discrete_soft_actor_critic:StableDiscreteSoftActorCritic"
    ),
}


def algorithm_class(name: str) -> type[Any]:
    """Resolve one supported built-in algorithm without import-time side effects."""

    try:
        path = _ALGORITHMS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown built-in algorithm {name!r}: {sorted(_ALGORITHMS)}") from exc
    module_name, _, class_name = path.partition(":")
    return cast(type[Any], getattr(importlib.import_module(module_name), class_name))
