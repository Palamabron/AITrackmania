"""Offline behavior cloning for compact TrackMania action contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

from tmrl.core.contracts import FeaturePipeline, ModelFactory, Policy
from tmrl.trackmania.actions import select_brake_tap_actions
from tmrl.trackmania.demonstrations import Demonstration, load_demonstration
from tmrl.trackmania.iqn import _LidarObservationEncoder


class LidarBehaviorCloningModel(nn.Module):
    """Categorical policy over an explicit compact action set and frame history."""

    def __init__(
        self,
        *,
        action_ids: tuple[int, ...],
        telemetry_dim: int = 26,
        history_length: int = 1,
        spatial_bins: int = 12,
        burn_in: int = 0,
        lidar_channels: int = 4,
        telemetry_group_dims: tuple[int, ...] | None = None,
        encoder_hidden_dim: int = 192,
        encoder_output_dim: int = 256,
    ) -> None:
        super().__init__()
        self.action_ids = tuple(action_ids)
        select_brake_tap_actions(self.action_ids)
        self.action_count = len(self.action_ids)
        self.encoder = _LidarObservationEncoder(
            telemetry_dim=telemetry_dim,
            history_length=history_length,
            spatial_bins=spatial_bins,
            burn_in=burn_in,
            lidar_channels=lidar_channels,
            telemetry_group_dims=telemetry_group_dims,
            hidden_dim=encoder_hidden_dim,
            output_dim=encoder_output_dim,
        )
        self.head = nn.Linear(self.encoder.output_dim, self.action_count)

    def forward(self, observation: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return cast(torch.Tensor, self.head(self.encoder(observation)))


class LidarBehaviorCloningModelFactory:
    def __init__(
        self,
        *,
        action_ids: tuple[int, ...],
        telemetry_dim: int = 26,
        history_length: int = 1,
        spatial_bins: int = 12,
        burn_in: int = 0,
        lidar_channels: int = 4,
        telemetry_group_dims: tuple[int, ...] | None = None,
        encoder_hidden_dim: int = 192,
        encoder_output_dim: int = 256,
    ) -> None:
        self.action_ids = tuple(action_ids)
        self.telemetry_dim = telemetry_dim
        self.history_length = history_length
        self.spatial_bins = spatial_bins
        self.burn_in = burn_in
        self.lidar_channels = lidar_channels
        self.telemetry_group_dims = telemetry_group_dims
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_output_dim = encoder_output_dim

    def build(self) -> LidarBehaviorCloningModel:
        return LidarBehaviorCloningModel(
            action_ids=self.action_ids,
            telemetry_dim=self.telemetry_dim,
            history_length=self.history_length,
            spatial_bins=self.spatial_bins,
            burn_in=self.burn_in,
            lidar_channels=self.lidar_channels,
            telemetry_group_dims=self.telemetry_group_dims,
            encoder_hidden_dim=self.encoder_hidden_dim,
            encoder_output_dim=self.encoder_output_dim,
        )


class BehaviorCloningPolicy:
    def __init__(self, model: LidarBehaviorCloningModel, device: torch.device) -> None:
        self.model = model
        self.device = device

    def reset_episode(self) -> None:
        pass

    def act(self, observation: Mapping[str, torch.Tensor], *, deterministic: bool = False) -> int:
        del deterministic
        batched = {key: value.to(self.device).unsqueeze(0) for key, value in observation.items()}
        with torch.inference_mode():
            return int(self.model(batched).argmax(dim=-1).item())


class BehaviorCloningLearner:
    """Categorical learner used only by the offline ``tmrl bc-train`` command."""

    def __init__(
        self,
        model: LidarBehaviorCloningModel | None = None,
        *,
        model_factory: ModelFactory | None = None,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.01,
        max_steps: int = 20_000,
        validation_interval: int = 100,
        early_stopping_patience: int = 30,
        lr_scheduler_factor: float = 0.3,
        lr_scheduler_patience: int = 5,
        min_learning_rate: float = 1e-6,
        gradient_clip_norm: float = 5.0,
        horizontal_flip_augmentation: bool = False,
        execution: Mapping[str, Any] | None = None,
        seed: int = 0,
    ) -> None:
        if (
            learning_rate <= 0.0
            or weight_decay < 0.0
            or min_learning_rate < 0.0
            or gradient_clip_norm <= 0.0
        ):
            raise ValueError("behavior cloning optimizer parameters are invalid")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        if (
            min(max_steps, validation_interval, early_stopping_patience, lr_scheduler_patience) < 1
            or not 0.0 < lr_scheduler_factor < 1.0
        ):
            raise ValueError("behavior cloning schedule parameters must be positive")
        self.model = model
        self.model_factory = model_factory
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.max_steps = max_steps
        self.validation_interval = validation_interval
        self.early_stopping_patience = early_stopping_patience
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_patience = lr_scheduler_patience
        self.min_learning_rate = min_learning_rate
        self.gradient_clip_norm = gradient_clip_norm
        self.horizontal_flip_augmentation = horizontal_flip_augmentation
        self.execution = dict(execution or {})
        self.seed = seed
        self.device = torch.device("cpu")
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None

    def setup(self, context: Mapping[str, Any]) -> None:
        factory = self.model_factory or context.get("model_factory")
        if self.model is None:
            if factory is None:
                raise ValueError("BehaviorCloningLearner requires model_factory")
            self.model = factory.build()
        requested = str(self.execution.get("device", "cpu"))
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("behavior cloning requested CUDA but torch.cuda is unavailable")
        self.device = torch.device(requested)
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.lr_scheduler_factor,
            patience=self.lr_scheduler_patience,
            min_lr=self.min_learning_rate,
        )

    def train_batch(
        self,
        observations: Mapping[str, torch.Tensor],
        labels: torch.Tensor,
        class_weights: torch.Tensor,
    ) -> dict[str, float]:
        if self.model is None or self.optimizer is None:
            raise RuntimeError("BehaviorCloningLearner.setup must run before training")
        self.model.train()
        logits = self.model(_to_device(observations, self.device))
        targets = labels.to(self.device)
        weights = class_weights.to(self.device)
        loss = functional.cross_entropy(
            logits, targets, weight=weights, label_smoothing=self.label_smoothing
        )
        self.optimizer.zero_grad(set_to_none=True)
        cast(Any, loss).backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.gradient_clip_norm
        )
        self.optimizer.step()
        accuracy = (logits.argmax(dim=-1) == targets).float().mean()
        return {
            "loss": float(loss.detach()),
            "accuracy": float(accuracy.detach()),
            "gradient_norm": float(gradient_norm.detach()),
            "learning_rate": self.current_learning_rate(),
        }

    def evaluate_batch(
        self,
        observations: Mapping[str, torch.Tensor],
        labels: torch.Tensor,
        class_weights: torch.Tensor,
    ) -> tuple[float, int, int, torch.Tensor, torch.Tensor]:
        if self.model is None:
            raise RuntimeError("BehaviorCloningLearner.setup must run before evaluation")
        self.model.eval()
        with torch.inference_mode():
            logits = self.model(_to_device(observations, self.device))
            targets = labels.to(self.device)
            loss = functional.cross_entropy(logits, targets, weight=class_weights.to(self.device))
            predicted = logits.argmax(dim=-1)
            correct_mask = predicted == targets
            correct = int(correct_mask.sum().item())
            action_count = logits.shape[-1]
            per_action_count = torch.bincount(targets, minlength=action_count)
            per_action_correct = torch.bincount(targets[correct_mask], minlength=action_count)
        return (
            float(loss),
            correct,
            int(targets.numel()),
            per_action_correct.cpu(),
            per_action_count.cpu(),
        )

    def policy(self) -> Policy:
        if self.model is None:
            raise RuntimeError("BehaviorCloningLearner.setup must run before policy")
        self.model.eval()
        return BehaviorCloningPolicy(self.model, self.device)

    def step_scheduler(self, validation_loss: float) -> float:
        if self.scheduler is None:
            raise RuntimeError("BehaviorCloningLearner.setup must run before scheduling")
        self.scheduler.step(validation_loss)
        return self.current_learning_rate()

    def current_learning_rate(self) -> float:
        if self.optimizer is None:
            raise RuntimeError("BehaviorCloningLearner.setup must run before reading learning rate")
        return float(self.optimizer.param_groups[0]["lr"])

    def update(self, batch: Any) -> Mapping[str, float]:
        del batch
        raise RuntimeError("BehaviorCloningLearner only supports tmrl bc-train")

    def state_dict(self) -> Mapping[str, Any]:
        if self.model is None or self.optimizer is None or self.scheduler is None:
            raise RuntimeError("BehaviorCloningLearner.setup must run before checkpointing")
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if self.model is None or self.optimizer is None or self.scheduler is None:
            raise RuntimeError("BehaviorCloningLearner.setup must run before restoring")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        scheduler = state.get("scheduler")
        if scheduler is not None:
            self.scheduler.load_state_dict(scheduler)


@dataclass(frozen=True, slots=True)
class BehaviorCloningLap:
    observations: tuple[Mapping[str, torch.Tensor], ...]
    labels: torch.Tensor


def load_behavior_cloning_laps(
    paths: Sequence[Path],
    pipeline: FeaturePipeline,
    action_ids: tuple[int, ...],
    *,
    expected_action_repeat_frames: int | None = None,
) -> list[BehaviorCloningLap]:
    """Convert full demonstration laps into compact supervised examples."""

    mapping = {action: index for index, action in enumerate(action_ids)}
    laps: list[BehaviorCloningLap] = []
    for path in paths:
        demo = load_demonstration(path)
        _validate_demonstration_contract(demo, pipeline, path, expected_action_repeat_frames)
        reset = getattr(pipeline, "reset_episode", None)
        if callable(reset):
            reset()
        labels = []
        observations = []
        for frame, action in zip(demo.frames[:-1], demo.actions, strict=True):
            source_action = int(action)
            if source_action not in mapping:
                raise ValueError(
                    f"demo {path} contains action {source_action} outside compact action IDs"
                )
            observation = pipeline.transform_observation(frame)
            if not isinstance(observation, Mapping):
                raise TypeError("behavior cloning requires mapping lidar observations")
            observations.append({key: value.detach().clone() for key, value in observation.items()})
            labels.append(mapping[source_action])
        laps.append(BehaviorCloningLap(tuple(observations), torch.tensor(labels, dtype=torch.long)))
    if len(laps) < 3:
        raise ValueError("behavior cloning requires at least three complete demonstration laps")
    return laps


def _validate_demonstration_contract(
    demonstration: Demonstration,
    pipeline: FeaturePipeline,
    path: Path,
    expected_action_repeat_frames: int | None,
) -> None:
    geometry = getattr(pipeline, "geometry", None)
    if geometry is not None:
        if demonstration.map_uid != geometry.map_uid:
            raise ValueError(
                f"demo {path} map UID {demonstration.map_uid!r} does not match "
                f"feature geometry {geometry.map_uid!r}"
            )
        if demonstration.geometry_sha256 != geometry.sha256:
            raise ValueError(f"demo {path} was recorded against a different geometry asset")
    if (
        expected_action_repeat_frames is not None
        and demonstration.action_repeat_frames != expected_action_repeat_frames
    ):
        raise ValueError(
            f"demo {path} action repeat {demonstration.action_repeat_frames} does not match "
            f"environment action repeat {expected_action_repeat_frames}"
        )


def split_behavior_cloning_laps(
    laps: Sequence[BehaviorCloningLap], seed: int
) -> tuple[list[BehaviorCloningLap], list[BehaviorCloningLap]]:
    """Split complete laps into an 80/20 deterministic train/validation partition."""

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(laps), generator=generator).tolist()
    validation_count = max(1, round(len(laps) * 0.2))
    validation = [laps[index] for index in order[:validation_count]]
    training = [laps[index] for index in order[validation_count:]]
    return training, validation


def augment_behavior_cloning_laps(
    laps: Sequence[BehaviorCloningLap], action_ids: tuple[int, ...]
) -> list[BehaviorCloningLap]:
    """Add a reflected copy of each local-frame demonstration lap."""

    mapping = _horizontal_flip_action_indices(action_ids)
    reflected = [
        BehaviorCloningLap(
            tuple(horizontal_flip_observation(observation) for observation in lap.observations),
            mapping[lap.labels],
        )
        for lap in laps
    ]
    return [*laps, *reflected]


def horizontal_flip_observation(
    observation: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Reflect the v59 local TrackMania observation across the car forward axis."""

    lidar = observation["lidar"]
    telemetry = observation["telemetry"]
    if lidar.shape[-2] != 8 or telemetry.shape[-1] != 46:
        raise ValueError("horizontal flip requires the 8-channel, 46-feature BC observation")
    reflected_lidar = lidar.clone()
    reflected_lidar[..., 0, :] = -lidar[..., 2, :]
    reflected_lidar[..., 1, :] = lidar[..., 3, :]
    reflected_lidar[..., 2, :] = -lidar[..., 0, :]
    reflected_lidar[..., 3, :] = lidar[..., 1, :]
    reflected_lidar[..., 4, :] = -lidar[..., 4, :]
    reflected_lidar[..., 5, :] = lidar[..., 5, :]
    reflected_telemetry = telemetry.clone()
    reflected_telemetry[..., 6] = -telemetry[..., 6]
    reflected_telemetry[..., 10] = telemetry[..., 11]
    reflected_telemetry[..., 11] = telemetry[..., 10]
    reflected_telemetry[..., 12] = telemetry[..., 13]
    reflected_telemetry[..., 13] = telemetry[..., 12]
    reflected_telemetry[..., 18] = -telemetry[..., 18]
    reflected_telemetry[..., 19] = -telemetry[..., 19]
    reflected_telemetry[..., 22] = -telemetry[..., 22]
    reflected_telemetry[..., 29] = -telemetry[..., 29]
    reflected_telemetry[..., 31] = -telemetry[..., 31]
    reflected_telemetry[..., 32] = -telemetry[..., 32]
    reflected_telemetry[..., 34] = -telemetry[..., 36]
    reflected_telemetry[..., 35] = telemetry[..., 37]
    reflected_telemetry[..., 36] = -telemetry[..., 34]
    reflected_telemetry[..., 37] = telemetry[..., 35]
    reflected_telemetry[..., 39] = -telemetry[..., 39]
    reflected_telemetry[..., 41] = -telemetry[..., 41]
    return {
        "lidar": reflected_lidar,
        "lidar_mask": observation["lidar_mask"].clone(),
        "telemetry": reflected_telemetry,
    }


def _horizontal_flip_action_indices(action_ids: tuple[int, ...]) -> torch.Tensor:
    _, table = select_brake_tap_actions(action_ids)
    mirrored: list[int] = []
    for control in table:
        match = next(
            (
                index
                for index, candidate in enumerate(table)
                if np.array_equal(candidate[:2], control[:2])
                and np.isclose(candidate[2], -control[2])
            ),
            None,
        )
        if match is None:
            raise ValueError("horizontal flip requires left-right paired compact actions")
        mirrored.append(match)
    return torch.tensor(mirrored, dtype=torch.long)


def flatten_behavior_cloning_laps(
    laps: Sequence[BehaviorCloningLap], indices: torch.Tensor | None = None
) -> tuple[list[Mapping[str, torch.Tensor]], torch.Tensor]:
    observations = [observation for lap in laps for observation in lap.observations]
    labels = torch.cat([lap.labels for lap in laps])
    if indices is None:
        return observations, labels
    return [observations[int(index)] for index in indices], labels[indices]


def class_weights(labels: torch.Tensor, action_count: int) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=action_count).float()
    if bool((counts == 0).any()):
        raise ValueError("every compact action must appear in behavior cloning training laps")
    weights = counts.rsqrt()
    return (weights / weights.mean()).clamp(0.5, 3.0)


def collate_behavior_cloning(
    observations: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    if not observations:
        raise ValueError("behavior cloning batch must not be empty")
    return {
        key: torch.stack([observation[key] for observation in observations])
        for key in observations[0]
    }


def clone_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-copy tensor state before the next optimizer update mutates it."""

    return deepcopy(dict(state))


def _to_device(
    observations: Mapping[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in observations.items()}
