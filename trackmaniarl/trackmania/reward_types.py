"""Structured reward output for TrackMania transitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

type TerminalReason = Literal[
    "off_track",
    "finished",
    "time_limit",
    "no_progress",
    "slow_progress",
]


@dataclass(frozen=True, slots=True)
class RewardResult:
    reward: float
    terminated: bool
    reason: str | None
    time_reward: float = 0.0
    pbrs_reward: float = 0.0
    progress_reward: float = 0.0
    projected_velocity_reward: float = 0.0
    projected_speed_reward: float = 0.0
    steering_delta_reward: float = 0.0
    time_attack_terminal_reward: float = 0.0
    pace_reward: float = 0.0
    terminal_reward: float = 0.0
    collision_reward: float = 0.0
    collided: bool = False
    collision_detected: bool = False
    potential_progress: float = 0.0
    projected_velocity_mps: float = 0.0
    projected_velocity_ratio: float = 0.0
    reference_time_s: float = 0.0
    time_debt_s: float = 0.0
    nearest_distance_m: float = 0.0
    accepted_progress_delta_m: float = 0.0
    window_progress_m: float = 0.0
    steps_since_progress: int = 0


@dataclass(frozen=True, slots=True)
class TransitionInput:
    position: np.ndarray
    finish_ui_active: bool
    velocity: np.ndarray | None
    race_time_ms: float | None
    collision: bool
    steering: float | None


@dataclass(frozen=True, slots=True)
class AdvanceRequest:
    nearest: int
    point: np.ndarray
    time_budget_s: float


@dataclass(frozen=True, slots=True)
class RewardTerms:
    time_reward: float
    progress_potential: float
    progress_reward: float
    projected_velocity_reward: float
    projected_speed_reward: float
    steering_delta_reward: float
    projected_velocity_mps: float
    projected_velocity_ratio: float


@dataclass(frozen=True, slots=True)
class TerminalRequest:
    reason: TerminalReason
    terms: RewardTerms
    terminal_reward: float
    time_attack_terminal_reward: float = 0.0


@dataclass(frozen=True, slots=True)
class PaceValues:
    reward: float
    reference_time_s: float
    time_debt_s: float


@dataclass(frozen=True, slots=True)
class CollisionRequest:
    result: RewardResult
    collision: bool
    race_time_s: float | None
