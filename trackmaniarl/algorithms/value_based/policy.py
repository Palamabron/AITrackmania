"""Inference policy for composed discrete value models."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any, cast

import torch

from trackmaniarl.core.pytree import PyTree, sanitize_finite, tree_map, tree_to_device
from trackmaniarl.models.composite import CompositeValueModel
from trackmaniarl.models.contracts import RiskSpec, ValuePhase


class DiscreteValuePolicy:
    def __init__(
        self,
        model: CompositeValueModel,
        device: torch.device,
        *,
        exploration_epsilon: float,
        policy_action_ids: tuple[int, ...] | None,
        online_risk: RiskSpec,
        evaluation_risk: RiskSpec,
        action_selector: Any | None = None,
    ) -> None:
        self.model = deepcopy(model).to(device).eval()
        self.device = device
        self.exploration_epsilon = exploration_epsilon
        self.policy_action_ids = policy_action_ids
        self.online_risk = online_risk
        self.evaluation_risk = evaluation_risk
        self.action_selector = action_selector
        self._state: PyTree = self.model.initial_policy_state(1, device)
        self.last_q_margin: float | None = None
        self.last_q_max: float | None = None

    def act(self, observation: Any, *, deterministic: bool = False) -> int:
        prepared = tree_to_device(sanitize_finite(observation), self.device)
        batched = cast(
            PyTree,
            tree_map(
                lambda leaf: leaf.unsqueeze(0) if isinstance(leaf, torch.Tensor) else leaf,
                prepared,
            ),
        )
        with torch.no_grad():
            features, self._state = self.model.policy_step(batched, self._state)
            support = self.model.support(features, ValuePhase.EVALUATE)
            risk = self.evaluation_risk if deterministic else self.online_risk
            q_values = self.model.expected_all_actions(features, support, risk)
            q_values = self._mask(q_values)
            action = q_values.argmax(dim=-1)
            self._record_gap(q_values)
            if self.action_selector is not None:
                action = self.action_selector.select(
                    q_values,
                    action,
                    deterministic=deterministic,
                    epsilon=self.exploration_epsilon,
                )
            elif (
                not deterministic
                and self.exploration_epsilon
                and bool(torch.rand((), device=self.device) < self.exploration_epsilon)
            ):
                action = self._random_action()
        return int(action.item())

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
