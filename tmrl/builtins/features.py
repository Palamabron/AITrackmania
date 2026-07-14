"""Feature pipelines supplied with TMRL for built-in off-policy algorithms."""

from __future__ import annotations

from typing import Any

from tmrl.core.data import Transition
from tmrl.core.pytree import sanitize_finite, tree_collate


class TransitionFeaturePipeline:
    """Collate finite standard transitions for TMRL 1.0 learners."""

    def transform_observation(self, observation: Any) -> Any:
        return sanitize_finite(observation)

    def collate(self, transitions: list[Transition]) -> dict[str, Any]:
        return {
            "observations": tree_collate([item.observation for item in transitions]),
            "actions": tree_collate([item.action for item in transitions]),
            "rewards": tree_collate([item.reward for item in transitions]),
            "next_observations": tree_collate([item.next_observation for item in transitions]),
            "terminated": tree_collate([item.terminated for item in transitions]),
            "truncated": tree_collate([item.truncated for item in transitions]),
        }
