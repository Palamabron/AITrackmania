"""First-class TrackmaniaRL 2.0 off-policy learners."""

from trackmaniarl.algorithms.execution import ResolvedTorchExecution, TorchExecutionConfig
from trackmaniarl.algorithms.implicit_quantile_q_learning import ImplicitQuantileQLearning
from trackmaniarl.algorithms.optimization import AdaptiveGradientClipper, GradientClipStats
from trackmaniarl.algorithms.proximal_policy_optimization import ProximalPolicyOptimization
from trackmaniarl.algorithms.randomized_ensemble_sac import RandomizedEnsembleSAC
from trackmaniarl.algorithms.soft_actor_critic import SoftActorCritic
from trackmaniarl.algorithms.stable_discrete_soft_actor_critic import StableDiscreteSoftActorCritic
from trackmaniarl.algorithms.truncated_quantile_critic import TruncatedQuantileCritic
from trackmaniarl.algorithms.value_based import DiscreteValueLearner

__all__ = [
    "AdaptiveGradientClipper",
    "DiscreteValueLearner",
    "GradientClipStats",
    "ImplicitQuantileQLearning",
    "ProximalPolicyOptimization",
    "RandomizedEnsembleSAC",
    "ResolvedTorchExecution",
    "SoftActorCritic",
    "StableDiscreteSoftActorCritic",
    "TorchExecutionConfig",
    "TruncatedQuantileCritic",
]
