"""Inference policy for composed discrete value models."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, cast

import torch

from trackmaniarl.core.contracts import ActionSelectionRequest, PolicyMode
from trackmaniarl.core.pytree import PyTree, sanitize_finite, tree_map, tree_to_device
from trackmaniarl.models.composite import CompositeValueModel
from trackmaniarl.models.contracts import RiskSpec, ValuePhase


@dataclass(frozen=True, slots=True)
class DiscreteValuePolicyConfig:
    exploration_epsilon: float
    policy_action_ids: tuple[int, ...] | None
    online_risk: RiskSpec
    evaluation_risk: RiskSpec
    action_selector: Any | None = None


class DiscreteValuePolicy:
    def __init__(
        self, model: CompositeValueModel, device: torch.device, config: DiscreteValuePolicyConfig
    ) -> None:
        self.model = deepcopy(model).to(device).eval()
        self.device = device
        self.exploration_epsilon = config.exploration_epsilon
        self.policy_action_ids = config.policy_action_ids
        self.online_risk = config.online_risk
        self.evaluation_risk = config.evaluation_risk
        self.action_selector = config.action_selector
        self._state: PyTree = self.model.initial_policy_state(1, device)
        self.last_q_margin: float | None = None
        self.last_q_max: float | None = None

    def act(self, observation: Any, mode: PolicyMode = PolicyMode.ONLINE) -> int:
        batched = self._prepare_observation(observation)
        with torch.no_grad():
            features, self._state = self.model.policy_step(batched, self._state)
            q_values = self._q_values(features, mode)
            action = self._select_action(q_values, mode)
        return int(action.item())

    def _prepare_observation(self, observation: Any) -> PyTree:
        prepared = tree_to_device(sanitize_finite(observation), self.device)
        return cast(
            PyTree,
            tree_map(
                lambda leaf: leaf.unsqueeze(0) if isinstance(leaf, torch.Tensor) else leaf,
                prepared,
            ),
        )

    def _q_values(self, features: torch.Tensor, mode: PolicyMode) -> torch.Tensor:
        support = self.model.support(features, ValuePhase.EVALUATE)
        risk = self.evaluation_risk if mode is PolicyMode.EVALUATION else self.online_risk
        return self._mask(self.model.expected_all_actions(features, support, risk))

    def _select_action(self, q_values: torch.Tensor, mode: PolicyMode) -> torch.Tensor:
        action = q_values.argmax(dim=-1)
        self._record_gap(q_values)
        if self.action_selector is not None:
            return cast(
                torch.Tensor,
                self.action_selector.select(
                    q_values,
                    action,
                    ActionSelectionRequest(mode, self.exploration_epsilon),
                ),
            )
        if mode is PolicyMode.ONLINE and self._explores():
            return self._random_action()
        return action

    def _explores(self) -> bool:
        return bool(
            self.exploration_epsilon
            and torch.rand((), device=self.device) < self.exploration_epsilon
        )

    def export_state(self) -> Mapping[str, Any]:
        return dict(self.model.state_dict())

    def load_state(self, state: Mapping[str, Any]) -> None:
        self.model.load_state_dict(state, strict=True)

    def set_exploration_epsilon(self, epsilon: float) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("exploration epsilon must be between zero and one")
        self.exploration_epsilon = epsilon

    def reset_episode(self) -> None:
        self._state = self.model.initial_policy_state(1, self.device)
        reset_selector = getattr(self.action_selector, "reset_episode", None)
        if callable(reset_selector):
            reset_selector()

    def _mask(self, values: torch.Tensor) -> torch.Tensor:
        if self.policy_action_ids is None:
            return values
        mask = torch.zeros(values.shape[-1], dtype=torch.bool, device=self.device)
        mask[list(self.policy_action_ids)] = True
        return values.masked_fill(~mask, -torch.inf)

    def _random_action(self) -> torch.Tensor:
        if self.policy_action_ids is None:
            return torch.randint(self.model.action_count, (1,), device=self.device)
        choices = torch.tensor(self.policy_action_ids, device=self.device)
        index = torch.randint(len(choices), (1,), device=self.device)
        return choices[index]

    def _record_gap(self, values: torch.Tensor) -> None:
        if values.shape[-1] < 2:
            self.last_q_margin = None
            self.last_q_max = None
            return
        best, second = values[0].topk(2).values.tolist()
        self.last_q_max = float(best)
        self.last_q_margin = float(best - second)
