"""Policy and offline-pretraining lifecycle helpers for value learning."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from trackmaniarl.algorithms._torch import polyak_update
from trackmaniarl.algorithms.value_based.policy import (
    DiscreteValuePolicy,
    DiscreteValuePolicyConfig,
)
from trackmaniarl.models.composite import CompositeValueModel

if TYPE_CHECKING:
    from trackmaniarl.algorithms.value_based.learner import DiscreteValueLearner


def masked(learner: DiscreteValueLearner, values: torch.Tensor) -> torch.Tensor:
    if learner.policy_action_ids is None:
        return values
    mask = torch.zeros(values.shape[-1], dtype=torch.bool, device=values.device)
    mask[list(learner.policy_action_ids)] = True
    return values.masked_fill(~mask, -torch.inf)


def sync_target(learner: DiscreteValueLearner) -> bool:
    assert isinstance(learner.model, CompositeValueModel)
    if learner.target_tau:
        polyak_update(learner.model, learner.target_model, learner.target_tau)
        return True
    if learner.update_count % learner.target_update_interval == 0:
        learner.target_model.load_state_dict(learner.model.state_dict(), strict=True)
        return True
    return False


def policy(learner: DiscreteValueLearner) -> DiscreteValuePolicy:
    assert isinstance(learner.model, CompositeValueModel)
    return DiscreteValuePolicy(
        learner.model,
        learner.device,
        DiscreteValuePolicyConfig(
            learner.exploration_epsilon,
            learner.policy_action_ids,
            learner.online_risk,
            learner.evaluation_risk,
            learner.action_selector,
        ),
    )


def begin_offline_pretraining(learner: DiscreteValueLearner) -> None:
    if (
        not learner.freeze_warm_start_during_offline_pretraining
        or learner.model_initialization_checkpoint is None
        or learner._offline_warm_start_requires_grad is not None
    ):
        return
    assert isinstance(learner.model, CompositeValueModel)
    parameters = learner._warm_start_parameters()
    learner._offline_warm_start_requires_grad = tuple(
        (parameter, parameter.requires_grad) for parameter in parameters
    )
    for parameter in parameters:
        parameter.requires_grad_(False)


def end_offline_pretraining(learner: DiscreteValueLearner) -> None:
    state = learner._offline_warm_start_requires_grad
    if state is None:
        return
    for parameter, requires_grad in state:
        parameter.requires_grad_(requires_grad)
    learner._offline_warm_start_requires_grad = None


def warm_start_parameters(learner: DiscreteValueLearner) -> tuple[torch.nn.Parameter, ...]:
    assert isinstance(learner.model, CompositeValueModel)
    parameters: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for name in learner.warm_start_submodules:
        module = cast(torch.nn.Module, getattr(learner.model, name))
        for parameter in module.parameters():
            if id(parameter) not in seen:
                parameters.append(parameter)
                seen.add(id(parameter))
    return tuple(parameters)
