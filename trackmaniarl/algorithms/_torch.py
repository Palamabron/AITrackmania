"""Shared torch implementation details for RunSpec 2.0 learners."""

from __future__ import annotations

import random
from collections.abc import Mapping
from contextlib import AbstractContextManager
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, Protocol, TypedDict, Unpack, cast

import numpy as np
import torch
from torch import nn

from trackmaniarl.algorithms.execution import (
    ResolvedTorchExecution,
    TorchExecutionConfig,
    resolve_torch_execution,
)
from trackmaniarl.algorithms.torch_batches import transform_batch
from trackmaniarl.core.contracts import PolicyMode
from trackmaniarl.core.data import TrainingBatch
from trackmaniarl.core.pytree import sanitize_finite, tree_collate, tree_map, tree_to_device

type _TransferMode = Literal["blocking", "non_blocking"]


class TorchLearnerOptions(TypedDict, total=False):
    model_factory: Any | None
    execution: TorchExecutionConfig | Mapping[str, Any] | None
    seed: int


def _execution_config(options: TorchLearnerOptions) -> TorchExecutionConfig:
    execution = options.get("execution")
    if isinstance(execution, Mapping):
        execution = TorchExecutionConfig(**execution)
    return execution or TorchExecutionConfig()


class _GradScaler(Protocol):
    def scale(self, outputs: Any) -> Any: ...

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None: ...

    def step(self, optimizer: torch.optim.Optimizer) -> Any: ...

    def update(self) -> None: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...


class TorchPolicy:
    """Inference adapter that makes deterministic policy behavior explicit."""

    def __init__(self, actor: nn.Module, device: torch.device) -> None:
        self.actor = deepcopy(actor).to(device).eval()
        self.device = device

    def act(self, observation: Any, mode: PolicyMode = PolicyMode.ONLINE) -> Any:
        batched = tree_to_device(tree_collate([sanitize_finite(observation)]), self.device)
        with torch.no_grad():
            output = self.actor(batched, mode=mode)
        action = output[0] if isinstance(output, tuple) else output
        action = action[0]
        return action.detach().cpu().numpy()

    def export_state(self) -> Mapping[str, Any]:
        return dict(self.actor.state_dict())

    def load_state(self, state: Mapping[str, Any]) -> None:
        self.actor.load_state_dict(state)


def evaluated_actor_state(
    state: Mapping[str, Any], model: nn.Module, policy_state: Mapping[str, Any]
) -> dict[str, Any]:
    actor = cast(Any, model).actor
    if set(policy_state) != set(actor.state_dict()):
        raise ValueError("evaluated policy state does not match the learner actor")
    result = deepcopy(dict(state))
    for name in ("model", "target_model"):
        module_state = dict(cast(Mapping[str, Any], result[name]))
        for key, value in policy_state.items():
            module_state[f"actor.{key}"] = deepcopy(value)
        result[name] = module_state
    return result


class TorchLearnerBase:
    """Base class for learners backed by a supplied torch model or model factory."""

    def __init__(
        self,
        model: nn.Module | None = None,
        **options: Unpack[TorchLearnerOptions],
    ) -> None:
        # User supplied model bundles intentionally expose algorithm-specific members
        # (actor/q1/q2, critics, q_values). The factory boundary is therefore dynamic.
        self.model: Any = model
        self.model_factory = options.get("model_factory")
        self.execution = _execution_config(options)
        self.device = torch.device("cpu")
        self.resolved_execution: ResolvedTorchExecution | None = None
        self.scaler: _GradScaler | None = None
        self._transfer_stream: torch.cuda.Stream | None = None
        self.run_dir: Path | None = None
        self._restoring_checkpoint = False
        self.seed = options.get("seed", 0)

    def setup(self, context: Mapping[str, Any]) -> None:
        self._setup_runtime(context)
        self._seed_runtime(context)
        self._build_model(context)
        self._setup_scaler()
        self._setup_model()

    def _setup_runtime(self, context: Mapping[str, Any]) -> None:
        run_dir = context.get("run_dir")
        self.run_dir = Path(run_dir) if run_dir is not None else None
        restoring = context.get("restoring_checkpoint", False)
        if not isinstance(restoring, bool):
            raise TypeError("restoring_checkpoint must be a bool")
        self._restoring_checkpoint = restoring
        self.resolved_execution = resolve_torch_execution(self.execution)
        self.device = self.resolved_execution.torch_device
        torch.use_deterministic_algorithms(self.execution.deterministic)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = self.execution.deterministic
            torch.backends.cudnn.benchmark = not self.execution.deterministic

    def _seed_runtime(self, context: Mapping[str, Any]) -> None:
        seed = int(context.get("seed", self.seed))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_model(self, context: Mapping[str, Any]) -> None:
        if self.model is None:
            factory = self.model_factory or context.get("model_factory")
            if factory is None:
                raise RuntimeError("Learner needs a torch model or a model_factory component")
            build = getattr(factory, "build", None)
            if not callable(build):
                raise TypeError("model_factory must expose build()")
            self.model = build()
        self.model.to(self.device)

    def _setup_scaler(self) -> None:
        if self.resolved_execution is None:
            raise RuntimeError("Learner execution must resolve before scaler setup")
        self.scaler = cast(Any, torch.amp).GradScaler(
            self.device.type,
            enabled=self.resolved_execution.scaler_enabled,
        )
        if self.resolved_execution.backend in {"cuda", "rocm"}:
            self._transfer_stream = cast(Any, torch.cuda).Stream(device=self.device)

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

    def _optimize(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer) -> None:
        if self.scaler is None:
            raise RuntimeError("Learner setup() must be called before optimize()")
        optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()

    def execution_manifest(self) -> Mapping[str, object]:
        if self.resolved_execution is None:
            return {
                "requested_device": self.execution.device,
                "requested_precision": self.execution.precision,
                "deterministic": self.execution.deterministic,
                "resolved": False,
            }
        return {"resolved": True, **self.resolved_execution.manifest()}

    def _setup_model(self) -> None:
        raise NotImplementedError

    def _batch(self, batch: TrainingBatch) -> TrainingBatch:
        self._validate_batch_layout(batch)
        event = batch.metadata.get("_trackmaniarl_transfer_event")
        if event is not None:
            torch.cuda.current_stream(self.device).wait_event(event)
            return replace(
                batch,
                metadata={
                    key: value
                    for key, value in batch.metadata.items()
                    if key
                    not in {
                        "_trackmaniarl_transfer_event",
                        "_trackmaniarl_transfer_started",
                    }
                },
            )
        return self._move_batch(batch, "blocking")

    def _validate_batch_layout(self, batch: TrainingBatch) -> None:
        if getattr(self, "supports_sequence_training", None) is not False:
            return
        configured = batch.metadata.get("sequence_length")
        rewards_are_sequential = isinstance(batch.rewards, torch.Tensor) and batch.rewards.ndim > 1
        if (isinstance(configured, int) and configured > 1) or rewards_are_sequential:
            raise ValueError(f"{type(self).__name__} requires sequence_length=1")

    def prepare_batch(self, batch: TrainingBatch) -> TrainingBatch:
        if self._transfer_stream is None:
            return batch
        pinned = self._pin_batch(batch)
        started = cast(Any, torch.cuda).Event(enable_timing=True)
        event = cast(Any, torch.cuda).Event(enable_timing=True)
        with torch.cuda.stream(self._transfer_stream):
            started.record()
            staged = self._move_batch(pinned, "non_blocking")
            event.record()
        return replace(
            staged,
            metadata={
                **staged.metadata,
                "_trackmaniarl_transfer_event": event,
                "_trackmaniarl_transfer_started": started,
            },
        )

    def _move_batch(self, batch: TrainingBatch, mode: _TransferMode) -> TrainingBatch:
        def move(value: Any) -> Any:
            return tree_to_device(value, self.device, mode=mode)

        return transform_batch(batch, move)

    @staticmethod
    def _pin_batch(batch: TrainingBatch) -> TrainingBatch:
        def pin(value: Any) -> Any:
            return tree_map(
                lambda leaf: (
                    leaf.pin_memory()
                    if isinstance(leaf, torch.Tensor)
                    and leaf.device.type == "cpu"
                    and not leaf.is_pinned()
                    else leaf
                ),
                value,
            )

        return transform_batch(batch, pin)

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

    def _scaler_state(self) -> dict[str, Any]:
        if self.scaler is None:
            raise RuntimeError("Learner setup() must be called before checkpointing")
        return self.scaler.state_dict()

    def _restore_scaler(self, state: object) -> None:
        if self.scaler is None:
            raise RuntimeError("Learner setup() must be called before restoring")
        if not isinstance(state, Mapping):
            raise ValueError("checkpoint is missing gradient scaler state")
        self.scaler.load_state_dict(dict(state))

    @staticmethod
    def _restore_rng(state: object) -> None:
        if not isinstance(state, Mapping):
            raise ValueError("checkpoint is missing RNG state")
        required = {"python", "numpy", "torch"}
        missing = required - state.keys()
        if missing:
            raise ValueError(f"checkpoint RNG state is missing: {', '.join(sorted(missing))}")
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
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
