"""First-class TMRL 1.0 off-policy learners."""

from tmrl.algorithms.execution import ResolvedTorchExecution, TorchExecutionConfig
from tmrl.algorithms.implicit_quantile_q_learning import ImplicitQuantileQLearning
from tmrl.algorithms.randomized_ensemble_sac import RandomizedEnsembleSAC
from tmrl.algorithms.soft_actor_critic import SoftActorCritic
from tmrl.algorithms.stable_discrete_soft_actor_critic import StableDiscreteSoftActorCritic
from tmrl.algorithms.truncated_quantile_critic import TruncatedQuantileCritic

__all__ = [
    "ImplicitQuantileQLearning",
    "RandomizedEnsembleSAC",
    "ResolvedTorchExecution",
    "SoftActorCritic",
    "StableDiscreteSoftActorCritic",
    "TorchExecutionConfig",
    "TruncatedQuantileCritic",
]
