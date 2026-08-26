"""Optional objectives for discrete value learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True, slots=True)
class ValueObjectiveContext:
    expected_values: torch.Tensor
    actions: torch.Tensor
    valid: torch.Tensor
    metadata: dict[str, object]
    action_mask: torch.Tensor | None = None


class ValueObjective(Protocol):
    requires_all_actions: bool

    def loss(self, context: ValueObjectiveContext) -> torch.Tensor | None: ...


class DemonstrationMarginObjective:
    requires_all_actions = True

    def __init__(self, margin: float = 0.8, weight: float = 1.0) -> None:
        if margin < 0.0 or weight < 0.0:
            raise ValueError("demonstration margin and weight must be non-negative")
        self.margin = margin
        self.weight = weight

    def loss(self, context: ValueObjectiveContext) -> torch.Tensor | None:
        flags = context.metadata.get("expert_demo_flags", context.metadata.get("demo_flags"))
        if flags is None or not self.weight:
            return None
        demo = torch.as_tensor(
            flags, dtype=torch.bool, device=context.expected_values.device
        ).unsqueeze(1)
        expected = _masked_demo_values(context, demo)
        margins = torch.full_like(expected, self.margin).scatter(
            -1, context.actions.unsqueeze(-1), 0.0
        )
        expert = expected.gather(-1, context.actions.unsqueeze(-1)).squeeze(-1)
        losses = (expected + margins).amax(dim=-1) - expert
        valid = context.valid & demo
        return self.weight * (losses * valid).sum() / valid.sum().clamp_min(1)


class DemonstrationCrossEntropyObjective:
    requires_all_actions = True

    def __init__(self, weight: float = 1.0) -> None:
        if weight < 0.0:
            raise ValueError("demonstration cross-entropy weight must be non-negative")
        self.weight = weight

    def loss(self, context: ValueObjectiveContext) -> torch.Tensor | None:
        flags = context.metadata.get("demo_flags")
        if flags is None or not self.weight:
            return None
        demo = torch.as_tensor(
            flags, dtype=torch.bool, device=context.expected_values.device
        ).unsqueeze(1)
        expected = _masked_demo_values(context, demo)
        leading = context.actions.shape
        losses = torch.nn.functional.cross_entropy(
            expected.reshape(-1, expected.shape[-1]),
            context.actions.reshape(-1),
            reduction="none",
        ).reshape(leading)
        valid = context.valid & demo
        return self.weight * (losses * valid).sum() / valid.sum().clamp_min(1)


def _masked_demo_values(context: ValueObjectiveContext, demo: torch.Tensor) -> torch.Tensor:
    if context.action_mask is None:
        return context.expected_values
    allowed = context.action_mask[context.actions]
    if torch.any(context.valid & demo & ~allowed):
        raise ValueError("demonstration action is excluded by policy_action_ids")
    return context.expected_values.masked_fill(~context.action_mask, -torch.inf)


class PolicyAnchorObjective:
    requires_all_actions = True

    def __init__(self, weight: float = 1.0) -> None:
        if weight < 0.0:
            raise ValueError("policy anchor weight must be non-negative")
        self.weight = weight

    def loss(self, context: ValueObjectiveContext) -> torch.Tensor | None:
        values = context.metadata.get("policy_anchor_q_values")
        if values is None or not self.weight:
            return None
        anchor = torch.as_tensor(
            values,
            dtype=context.expected_values.dtype,
            device=context.expected_values.device,
        )
        if anchor.shape != context.expected_values.shape:
            raise ValueError("policy anchor values must match all-action value shape")
        advantages = context.expected_values - context.expected_values.mean(dim=-1, keepdim=True)
        anchor_advantages = anchor - anchor.mean(dim=-1, keepdim=True)
        losses = torch.nn.functional.smooth_l1_loss(
            advantages, anchor_advantages, reduction="none"
        ).mean(dim=-1)
        valid = context.valid.to(losses.dtype)
        return self.weight * (losses * valid).sum() / valid.sum().clamp_min(1.0)
