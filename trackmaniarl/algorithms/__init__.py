"""First-class TrackmaniaRL 1.0 off-policy learners."""

from trackmaniarl.algorithms.execution import ResolvedTorchExecution, TorchExecutionConfig
from trackmaniarl.algorithms.implicit_quantile_q_learning import ImplicitQuantileQLearning
from trackmaniarl.algorithms.optimization import AdaptiveGradientClipper, GradientClipStats
from trackmaniarl.algorithms.randomized_ensemble_sac import RandomizedEnsembleSAC
from trackmaniarl.algorithms.soft_actor_critic import SoftActorCritic
from trackmaniarl.algorithms.stable_discrete_soft_actor_critic import StableDiscreteSoftActorCritic
from trackmaniarl.algorithms.truncated_quantile_critic import TruncatedQuantileCritic

__all__ = [
    "AdaptiveGradientClipper",
    "GradientClipStats",
    "ImplicitQuantileQLearning",
    "RandomizedEnsembleSAC",
    "ResolvedTorchExecution",
    "SoftActorCritic",
    "StableDiscreteSoftActorCritic",
    "TorchExecutionConfig",
    "TruncatedQuantileCritic",
]
