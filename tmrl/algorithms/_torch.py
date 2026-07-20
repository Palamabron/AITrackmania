"""Shared torch implementation details for TMRL 1.0 learners."""

from __future__ import annotations

import json
import random
from collections.abc import Mapping
from contextlib import AbstractContextManager
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from tmrl.algorithms.execution import (
    RequestedDevice,
    ResolvedTorchExecution,
    TorchExecutionConfig,
    resolve_torch_execution,
)
from tmrl.core.data import TrainingBatch
from tmrl.core.pytree import sanitize_finite, tree_to_device


class TorchPolicy:
    """Inference adapter that makes deterministic policy behavior explicit."""

    def __init__(self, actor: nn.Module, device: torch.device, *, discrete: bool = False) -> None:
        self.actor = deepcopy(actor).to(device).eval()
        self.device = device
        self.discrete = discrete

    def act(self, observation: Any, *, deterministic: bool = False) -> Any:
        prepared = tree_to_device(sanitize_finite(observation), self.device)
        if not isinstance(prepared, torch.Tensor):
            raise TypeError(
                "Bundled torch policies require a tensor observation from the feature pipeline"
            )
        with torch.no_grad():
            output = self.actor(prepared, deterministic=deterministic)
        action = output[0] if isinstance(output, tuple) else output
        return action.detach().cpu().numpy()

    def export_state(self) -> Mapping[str, Any]:
        return dict(self.actor.state_dict())

    def load_state(self, state: Mapping[str, Any]) -> None:
        self.actor.load_state_dict(state)


class TorchLearnerBase:
    """Base class for learners backed by a supplied torch model or model factory."""

    def __init__(
        self,
        model: nn.Module | None = None,
        *,
        model_factory: Any | None = None,
        device: str | None = None,
        execution: TorchExecutionConfig | Mapping[str, Any] | None = None,
        seed: int = 0,
    ) -> None:
        # User supplied model bundles intentionally expose algorithm-specific members
        # (actor/q1/q2, critics, q_values). The factory boundary is therefore dynamic.
        self.model: Any = model
        self.model_factory = model_factory
        if execution is not None and device is not None:
            raise ValueError("Use execution.device instead of combining execution with device")
        if isinstance(execution, Mapping):
            execution = TorchExecutionConfig(**execution)
        requested = cast(RequestedDevice, device or "auto")
        self.execution = execution or TorchExecutionConfig(device=requested)
        self.device = torch.device("cpu")
        self.resolved_execution: ResolvedTorchExecution | None = None
        self.scaler: torch.amp.GradScaler | None = None
        self.run_dir: Path | None = None
        self.seed = seed

    def setup(self, context: Mapping[str, Any]) -> None:
        run_dir = context.get("run_dir")
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.resolved_execution = resolve_torch_execution(self.execution)
        self.device = self.resolved_execution.torch_device
        seed = int(context.get("seed", self.seed))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if self.model is None:
            factory = self.model_factory or context.get("model_factory")
            if factory is None:
                raise RuntimeError("Learner needs a torch model or a model_factory component")
            build = getattr(factory, "build", None)
            if not callable(build):
                raise TypeError("model_factory must expose build()")
            self.model = build()
        self.model.to(self.device)
        self.scaler = torch.amp.GradScaler(
            self.device.type,
            enabled=self.resolved_execution.scaler_enabled,
        )
        self._setup_model()

    def autocast(self) -> AbstractContextManager[Any]:
        if self.resolved_execution is None:
            raise RuntimeError("Learner setup() must be called before autocast()")
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.resolved_execution.precision]
        enabled = self.resolved_execution.precision != "float32"
        return torch.autocast(device_type=self.device.type, dtype=dtype, enabled=enabled)

    def execution_manifest(self) -> Mapping[str, object]:
        if self.resolved_execution is None:
            return {
                "requested_device": self.execution.device,
                "requested_precision": self.execution.precision,
                "compile_requested": self.execution.compile,
                "compile_mode": self.execution.compile_mode,
                "resolved": False,
            }
        return {"resolved": True, **self.resolved_execution.manifest()}

    def _record_execution_result(self) -> None:
        if self.run_dir is None or self.resolved_execution is None:
            return
        path = self.run_dir / "manifest.json"
        if not path.is_file():
            return
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["torch_execution"] = dict(self.execution_manifest())
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        temporary.replace(path)

    def _setup_model(self) -> None:
        raise NotImplementedError

    def _batch(self, batch: TrainingBatch) -> TrainingBatch:
        return TrainingBatch(
            data=tree_to_device(batch.data, self.device),
            observations=tree_to_device(batch.observations, self.device),
            actions=tree_to_device(batch.actions, self.device),
            rewards=tree_to_device(batch.rewards, self.device),
            next_observations=tree_to_device(batch.next_observations, self.device),
            terminated=tree_to_device(batch.terminated, self.device),
            truncated=tree_to_device(batch.truncated, self.device),
            bootstrap_discounts=tree_to_device(batch.bootstrap_discounts, self.device),
            transition_ids=batch.transition_ids,
            importance_weights=(
                tree_to_device(batch.importance_weights, self.device)
                if batch.importance_weights is not None
                else None
            ),
            masks=tree_to_device(batch.masks, self.device) if batch.masks is not None else None,
            metadata=batch.metadata,
        )

    @staticmethod
    def _tensor(value: Any, name: str) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a tensor after feature collation")
        return value

    def _rng_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _restore_rng(state: Mapping[str, Any]) -> None:
        if "python" in state:
            random.setstate(state["python"])
        if "numpy" in state:
            np.random.set_state(state["numpy"])
        if "torch" in state:
            torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and "cuda" in state:
            torch.cuda.set_rng_state_all(state["cuda"])


def polyak_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    """Update a target module, including non-parameter state such as BatchNorm buffers."""

    with torch.no_grad():
        for target_parameter, parameter in zip(
            target.parameters(), source.parameters(), strict=True
        ):
            target_parameter.lerp_(parameter, tau)
        # Running statistics are state, not learnable parameters.  Leaving them
        # stale makes targets invalid for otherwise supported BatchNorm models.
        # Copying is the conservative convention used by common SAC implementations.
        for target_buffer, buffer in zip(target.buffers(), source.buffers(), strict=True):
            target_buffer.copy_(buffer)


def weighted_mean(losses: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    """Mean loss with optional normalized importance-sampling weights."""

    if weights is None:
        return losses.mean()
    weights = weights.reshape(-1).to(losses.dtype)
    return (losses.reshape(-1) * weights).sum() / weights.sum().clamp_min(1e-8)
