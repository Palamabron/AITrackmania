"""Reward and termination logic for TrackMania 2020 RL.

RewardFunction computes step reward and termination from position, speed,
and inputs. It uses a reference trajectory and optional track boundaries
for progress, speed-along-track, and off-track/stall detection.
"""

import atexit
import dataclasses
import math
import os
import pickle
import shutil
import socket
import tempfile
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from loguru import logger

from tmrl.config._internal.spacing_lookahead import points_number_from_spacing_config
from tmrl.custom.tm.utils.track_features import TrackFeatureProvider, discrete_curvature_xz

OFF_TRACK_PROGRESS_ZERO_MULTIPLIER = 2.0
SPEED_REWARD_MIN_KMH = 5.0
TIME_STEP_SECONDS = 0.05
DUMMY_TRAJECTORY_THRESHOLD = 2
MIN_ROAD_HALF_WIDTH_M = 0.5
DEFAULT_ROAD_HALF_WIDTH_M = 12.0
DT_MILLISECOND_THRESHOLD = 1.0
NEAR_FINISH_PROGRESS_THRESHOLD = 0.97
NEAR_FINISH_MIN_SPEED_KMH = 10.0


@dataclass
class EpisodeState:
    """Episode-scoped state grouped for safe reset."""

    cur_idx: int = 0
    prev_idx: int = 0
    step_counter: int = 0
    failure_counter: int = 0
    episode_reward: float = 0.0
    furthest_race_progress: float = 0.0
    furthest_reached_idx: int = 0
    new_lap: bool = False
    lap_cur_cooldown: int = 0
    last_progress_step: int = 0
    term_reason: str | None = None
    logged_run_this_episode: bool = False
    prev_pos: np.ndarray | None = None


def _resample_polyline_by_arc_length(points: np.ndarray, num_points: int) -> np.ndarray:
    """Resample a polyline to a fixed number of points uniformly by arc length.

    Args:
        points: Array of shape (N, 3) or (N, 2) forming the polyline.
        num_points: Desired number of points in the resampled polyline.

    Returns:
        Resampled polyline of shape (num_points, ...) and dtype float64.
    """
    points = np.asarray(points, dtype=np.float64)
    num_input = len(points)
    if num_input <= 1 or num_points <= 1:
        return points.copy()
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative_length = np.zeros(num_input)
    np.cumsum(segment_lengths, out=cumulative_length[1:])
    total_length = float(cumulative_length[-1])
    if total_length <= 0:
        return points.copy()
    # np.interp requires strictly increasing xp; drop duplicate arc positions.
    _, unique_idx = np.unique(cumulative_length, return_index=True)
    unique_idx = np.sort(unique_idx)
    cumulative_length = cumulative_length[unique_idx]
    points = points[unique_idx]
    if len(points) <= 1:
        return points.copy()
    total_length = float(cumulative_length[-1])
    arc_positions = np.linspace(0.0, total_length, num_points, endpoint=True)
    resampled = np.zeros((num_points, points.shape[1]), dtype=np.float64)
    for dim in range(points.shape[1]):
        resampled[:, dim] = np.interp(arc_positions, cumulative_length, points[:, dim])
    return resampled


def _polyline_length(points: np.ndarray) -> float:
    """Total arc length of a polyline."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def _extend_polyline_straight(points: np.ndarray, extra_m: float, spacing_m: float) -> np.ndarray:
    """Append straight samples from the last segment direction."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2 or extra_m <= 0.0:
        return pts.copy()
    tangent = pts[-1] - pts[-2]
    norm = float(np.linalg.norm(tangent))
    if norm <= 1e-9:
        return pts.copy()
    unit = tangent / norm
    spacing = max(0.1, float(spacing_m))
    n_new = max(1, math.ceil(extra_m / spacing))
    step = extra_m / n_new
    offsets = step * np.arange(1, n_new + 1, dtype=np.float64)[:, None]
    extra = pts[-1] + unit * offsets
    return np.vstack([pts, extra])


def _ensure_wandb_api_key(api_key: str) -> None:
    """Set WANDB_API_KEY env var if not already present and a key was provided."""
    if "WANDB_API_KEY" not in os.environ and api_key:
        os.environ["WANDB_API_KEY"] = api_key


class RewardFunction:
    """Reward and termination logic for TrackMania 2020 RL.

    Uses OpenPlanet API data. Rewards progress along a reference trajectory,
    speed along the track, and applies penalties for crash and constant penalty.
    Handles termination (stall, off-track, end-of-track).
    """

    def __init__(
        self,
        reward_data_path: str,
        nb_obs_forward: int = 8,
        nb_obs_backward: int = 8,
        max_dist_from_traj: float = 23.5,
        crash_penalty: float = 0.5,
        constant_penalty: float = 0.0,
        *,
        # --- Track geometry ---
        require_track_boundary_pickles: bool = False,
        track_path_left: str = "",
        track_path_right: str = "",
        # --- Reward tuning (from RewardConfig.model_dump()) ---
        reward_config: dict[str, Any] | None = None,
        # --- Timing ---
        time_step_duration: float = 0.05,
        # --- Operational ---
        points_distance: float = 1.0,
        lap_cooldown: int = 0,
        config_file_path: str = "",
        # --- W&B logging ---
        use_wandb: bool = False,
        wandb_project: str = "tmrl",
        wandb_entity: str = "tmrl",
        wandb_run_id: str = "",
        wandb_api_key: str = "",
        wandb_config: dict[str, Any] | None = None,
    ) -> None:
        rc = reward_config or {}

        self.reward_data_path = reward_data_path
        if not os.path.isfile(reward_data_path):
            raise FileNotFoundError(
                f"Reward trajectory missing: {reward_data_path}. "
                "Set environment.map_name and record reward_<map>.pkl under TmrlData/reward/."
            )
        with open(reward_data_path, "rb") as f:
            self.data = pickle.load(f)
        self._dummy_trajectory = len(self.data) <= DUMMY_TRAJECTORY_THRESHOLD

        self.datalen = len(self.data)
        self._reward_extended_for_boundaries = False

        if require_track_boundary_pickles:
            if not os.path.isfile(track_path_left):
                raise FileNotFoundError(
                    f"Strict track boundaries require left track pickle: missing {track_path_left}"
                )
            if not os.path.isfile(track_path_right):
                raise FileNotFoundError(
                    "Strict track boundaries require right track pickle: missing "
                    f"{track_path_right}"
                )
            with open(track_path_left, "rb") as f:
                self.left_track = np.asarray(pickle.load(f), dtype=np.float64)
            with open(track_path_right, "rb") as f:
                self.right_track = np.asarray(pickle.load(f), dtype=np.float64)
        else:
            if not os.path.isfile(track_path_left):
                self.left_track = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
            else:
                with open(track_path_left, "rb") as f:
                    self.left_track = np.asarray(pickle.load(f), dtype=np.float64)

            if not os.path.isfile(track_path_right):
                self.right_track = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
            else:
                with open(track_path_right, "rb") as f:
                    self.right_track = np.asarray(pickle.load(f), dtype=np.float64)

        if len(self.data) >= 2 and len(self.left_track) >= 2 and len(self.right_track) >= 2:
            reward_len_m = _polyline_length(self.data)
            left_len_m = _polyline_length(self.left_track)
            right_len_m = _polyline_length(self.right_track)
            boundary_len_m = max(left_len_m, right_len_m)
            if boundary_len_m > reward_len_m + 0.25:
                mean_seg = reward_len_m / max(1, len(self.data) - 1)
                spacing_m = max(0.2, mean_seg)
                extra_m = boundary_len_m - reward_len_m
                self.data = _extend_polyline_straight(
                    self.data, extra_m=extra_m, spacing_m=spacing_m
                )
                self.datalen = len(self.data)
                self._reward_extended_for_boundaries = True
                logger.info(
                    "Extended reward trajectory to match boundaries (+{:.2f}m, {} points).",
                    extra_m,
                    self.datalen,
                )

        self._has_boundaries = (
            len(self.left_track) >= 2 and len(self.right_track) >= 2 and self.datalen >= 2
        )
        if self._has_boundaries:
            # Keep raw XZ coords before resampling for trajectory-aligned
            # boundary lookup (the resampled versions are only used by
            # track_features.py for observation computation).
            _raw_left_xz = self.left_track[:, [0, 2]].astype(np.float64).copy()
            _raw_right_xz = self.right_track[:, [0, 2]].astype(np.float64).copy()

            if len(self.left_track) != self.datalen:
                self.left_track = _resample_polyline_by_arc_length(self.left_track, self.datalen)
            if len(self.right_track) != self.datalen:
                self.right_track = _resample_polyline_by_arc_length(self.right_track, self.datalen)
            self._left_xz = self.left_track[:, [0, 2]].astype(np.float64).copy()
            self._right_xz = self.right_track[:, [0, 2]].astype(np.float64).copy()

            # Align boundary geometry to the reward trajectory so that
            # _road_center_xz[i] represents the road cross-section at the
            # same physical location as self.data[i].  Without this, arc-
            # length resampled boundaries can be misaligned with the
            # trajectory (e.g. different starting direction) and indexing
            # by best_index gives the wrong road segment.
            from scipy.spatial import cKDTree

            traj_xz = self.data[:, [0, 2]].astype(np.float64)
            _, left_nn = cKDTree(_raw_left_xz).query(traj_xz)
            _, right_nn = cKDTree(_raw_right_xz).query(traj_xz)
            aligned_left = _raw_left_xz[left_nn]
            aligned_right = _raw_right_xz[right_nn]
            self._road_center_xz = (aligned_left + aligned_right) / 2.0
            self._road_half_widths = np.linalg.norm(aligned_left - aligned_right, axis=1) / 2.0
            self._road_half_widths = np.maximum(self._road_half_widths, MIN_ROAD_HALF_WIDTH_M)
            _hw = self._road_half_widths
            logger.info(
                "Track boundaries loaded: {} points, road half-width "
                "min={:.1f}m mean={:.1f}m max={:.1f}m (aligned to trajectory via KDTree)",
                self.datalen,
                float(_hw.min()),
                float(_hw.mean()),
                float(_hw.max()),
            )
        else:
            self._left_xz = np.zeros((2, 2), dtype=np.float64)
            self._right_xz = np.zeros((2, 2), dtype=np.float64)
            self._road_center_xz = np.zeros((2, 2), dtype=np.float64)
            self._road_half_widths = np.ones(2, dtype=np.float64) * DEFAULT_ROAD_HALF_WIDTH_M

        self.cur_idx = 0
        self.prev_idx = 0
        self.nb_obs_forward = nb_obs_forward
        self.nb_obs_backward = nb_obs_backward
        self._tracking_lookahead_min = int(
            rc.get("tracking_lookahead_min", max(nb_obs_forward, 24))
        )
        self._tracking_lookahead_max = int(
            rc.get("tracking_lookahead_max", max(self._tracking_lookahead_min, 64))
        )
        self._tracking_margin_points = int(rc.get("tracking_lookahead_margin_points", 4))

        self.max_dist_from_traj = float(rc.get("max_stray", max_dist_from_traj))

        raw_dt = float(time_step_duration)
        if raw_dt >= DT_MILLISECOND_THRESHOLD:
            self._time_step_duration = raw_dt / 1000.0
        else:
            self._time_step_duration = raw_dt

        no_prog_sec = float(rc.get("min_seconds_before_failure", 0.0))
        if no_prog_sec > 0.0:
            self._max_no_progress_steps = max(
                1,
                int(round(no_prog_sec / self._time_step_duration)),  # noqa: RUF046
            )
            steps_if_005 = max(1, int(round(no_prog_sec / 0.05)))  # noqa: RUF046
            if no_prog_sec >= 1.0 and self._max_no_progress_steps < steps_if_005 // 2:
                logger.warning(
                    "No-progress steps {} low for {:.1f}s (dt={:.3f}s); dt=0.05s -> {} steps.",
                    self._max_no_progress_steps,
                    no_prog_sec,
                    self._time_step_duration,
                    steps_if_005,
                )
                self._time_step_duration = 0.05
                self._max_no_progress_steps = steps_if_005
            self._use_time_no_progress = True
            self._last_progress_step = 0
            logger.info(
                "Reward: no-progress episode cutoff after {:.3f}s sim time "
                "(~{} env steps at dt={:.4f}s). {}",
                no_prog_sec,
                self._max_no_progress_steps,
                self._time_step_duration,
                config_file_path,
            )
        else:
            self._max_no_progress_steps = 0
            self._use_time_no_progress = False
            self._last_progress_step = 0
            logger.info(
                "Reward: min_seconds_before_failure=0 → no stagnant-progress timeout. {}",
                config_file_path,
            )

        # Slow-progress cutoff: terminate when the car gains less track distance
        # than min_progress_rate (m/s) averaged over a sliding window. The binary
        # no-progress timeout above is evaded by arbitrarily slow creep (any
        # reward_progress > 0 resets its timer); a rate-based check is not.
        self._min_progress_rate_mps = max(0.0, float(rc.get("min_progress_rate", 0.0)))
        _spw_sec = float(rc.get("slow_progress_window_seconds", 5.0))
        if self._min_progress_rate_mps > 0.0 and _spw_sec > 0.0:
            self._slow_progress_window_steps = max(1, round(_spw_sec / self._time_step_duration))
            logger.info(
                "Reward: slow-progress cutoff below {:.1f} m/s averaged over {:.1f}s "
                "(~{} env steps).",
                self._min_progress_rate_mps,
                _spw_sec,
                self._slow_progress_window_steps,
            )
        else:
            self._slow_progress_window_steps = 0
        self._slow_progress_dist_history: deque[float] = deque(
            maxlen=max(2, self._slow_progress_window_steps + 1)
        )

        _ot_grace_sec = float(rc.get("off_track_seconds_before_failure", 0.5))
        if _ot_grace_sec <= 0.0:
            self._off_track_grace_steps = 0
        else:
            self._off_track_grace_steps = max(1, round(_ot_grace_sec / self._time_step_duration))
        logger.info(
            "Reward: off-track grace {:.3f}s sim time (~{} env steps at dt={:.4f}s). {}",
            _ot_grace_sec,
            self._off_track_grace_steps,
            self._time_step_duration,
            config_file_path,
        )

        self.step_counter = 0
        self.failure_counter = 0
        self._prev_pos: np.ndarray | None = None

        self.average_distance = self.calculate_average_distance()
        self._cumulative_dist = np.zeros(max(1, self.datalen))
        if self.datalen > 1:
            diffs = np.linalg.norm(np.diff(self.data, axis=0), axis=1)
            np.cumsum(diffs, out=self._cumulative_dist[1:])
        self._total_traj_length = (
            max(1.0, float(self._cumulative_dist[-1])) if self.datalen >= 1 else 1.0
        )

        self._progress_reward_full_lap = float(rc.get("progress_reward_full_lap", 200.0))
        self._speed_reward_weight = float(rc.get("speed_reward_weight", 0.25))
        self._speed_reward_exponent = float(rc.get("speed_reward_exponent", 1.0))
        self._speed_reward_alignment_floor = float(rc.get("speed_reward_alignment_floor", 0.0))
        self._max_speed_kmh = float(rc.get("max_speed_kmh", 100.0))
        self._constant_penalty = float(rc.get("constant_penalty", constant_penalty))
        self._drift_reward_weight = float(rc.get("drift_reward_weight", 0.0))
        self._drift_optimal_angle_deg = float(rc.get("drift_optimal_angle_deg", 12.0))
        self._drift_sigma_deg = float(rc.get("drift_sigma_deg", 8.0))
        self._drift_threshold_kmh = float(rc.get("drift_threshold_kmh", 80.0))
        self._drift_curvature_threshold = float(rc.get("drift_curvature_threshold", 0.01))
        self._max_track_width = float(rc.get("max_track_width", 35.0))
        self.crash_penalty = float(rc.get("crash_penalty", crash_penalty))
        self._reward_clip_floor = float(rc.get("reward_clip_floor", 5.0))
        self._reward_scale = float(rc.get("reward_scale", 1.0))
        self._end_of_track_reward = float(rc.get("end_of_track_reward", 10.0))
        self._projected_velocity_scale = float(rc.get("projected_velocity_scale", 0.0))
        self._track_curvature_obs = bool(rc.get("track_curvature_obs", False))
        self._progress_min_alignment = float(rc.get("progress_min_alignment", 0.0))
        self._velocity_alignment_reward_weight = float(
            rc.get("velocity_alignment_reward_weight", 0.0)
        )
        self._drift_weight_start = float(
            rc.get("drift_reward_weight_start", self._drift_reward_weight)
        )
        self._drift_weight_end = float(rc.get("drift_reward_weight_end", 0.0))
        self._drift_anneal_steps = int(rc.get("drift_anneal_steps", 0))
        self._rear_slip_activation = float(rc.get("REAR_SLIP_ACTIVATION", 0.5))

        self._cornering_speed_bonus = float(rc.get("cornering_speed_bonus", 0.0))
        self._cornering_curvature_threshold = float(rc.get("cornering_curvature_threshold", 0.01))

        self._progress_accel_bonus = float(rc.get("progress_acceleration_bonus", 0.0))
        self._progress_ema: float = 0.0

        self._cte_penalty_weight = float(rc.get("cte_penalty_weight", 0.0))
        self._cte_penalty_exponent = float(rc.get("cte_penalty_exponent", 2.0))

        self._boundary_penalty_weight = float(rc.get("boundary_penalty_weight", 0.0))
        self._boundary_penalty_start = float(rc.get("boundary_penalty_start", 0.85))
        self._boundary_crash_penalty = float(rc.get("boundary_crash_penalty", 0.0))
        self._wall_hug_penalty_factor = float(rc.get("wall_hug_penalty_factor", 0.0))
        self._wall_hug_speed_threshold = float(rc.get("wall_hug_speed_threshold", 10.0))
        self._wall_hug_lateral_threshold = float(rc.get("wall_hug_lateral_threshold", 0.85))
        self._terminal_failure_penalty = float(rc.get("terminal_failure_penalty", 0.0))

        self._global_env_steps: int = 0

        self._term_reason: str | None = None
        self._lap_cooldown_init = lap_cooldown
        self.lap_cur_cooldown = lap_cooldown
        self.new_lap = False
        self.episode_reward = 0.0
        self.furthest_race_progress = 0.0
        self.furthest_reached_idx = 0
        self._logged_run_this_episode = False
        self._use_wandb = use_wandb
        self._checkpoint_stride = max(
            1, min(len(self.data), int(points_distance / max(self.average_distance, 0.01)))
        )
        _track_pct = float(rc.get("track_look_ahead_pct", 0.0))
        _track_spacing = float(rc.get("track_point_spacing_m", 0.0))
        self._points_number: int | None
        if _track_pct > 0 and _track_spacing > 0 and self.datalen > 1:
            self._points_number = points_number_from_spacing_config(
                self._total_traj_length, _track_pct, _track_spacing
            )
            self._point_spacing_m = _track_spacing
        else:
            self._point_spacing_m = 0.0
            self._points_number = None

        self._debug_reward = bool(
            rc.get("debug_reward_components", rc.get("DEBUG_REWARD_COMPONENTS", False))
        )
        self._debug_log_interval = int(
            rc.get("debug_log_interval", rc.get("DEBUG_LOG_INTERVAL", 100))
        )
        self._reset_debug_accumulators()
        self._set_episode_state(EpisodeState(lap_cur_cooldown=self._lap_cooldown_init))
        self.track_feature_provider = TrackFeatureProvider(self)

        if self._use_wandb:
            import wandb

            wandb_dir = tempfile.mkdtemp()
            atexit.register(shutil.rmtree, wandb_dir, ignore_errors=True)
            _ensure_wandb_api_key(wandb_api_key)

            def _init_worker_wandb(attempt: int) -> bool:
                # Primary id matches the historical convention: "<run.name> WORKER"
                if attempt == 0:
                    run_id = f"{wandb_run_id} WORKER"
                else:
                    # Collision-safe fallback for distributed workers across hosts/processes.
                    host = socket.gethostname() or "unknown-host"
                    short_uuid = uuid.uuid4().hex[:8]
                    run_id = f"{wandb_run_id} WORKER-{host}-{os.getpid()}-{short_uuid}"
                try:
                    wandb.init(
                        project=wandb_project,
                        entity=wandb_entity,
                        id=run_id,
                        name=f"{wandb_run_id} worker",
                        config=wandb_config or {},
                        job_type="worker",
                        dir=wandb_dir,
                        resume="allow",
                    )
                except Exception as exc:
                    logger.warning("wandb worker init failed (attempt {}): {}", attempt, exc)
                    return False
                if wandb.run is None:
                    logger.warning("wandb worker init returned no active run (attempt {})", attempt)
                    return False
                logger.info(
                    "W&B worker run active: id={} project={!r} entity={!r} url={}",
                    wandb.run.id,
                    wandb_project,
                    wandb_entity,
                    wandb.run.get_url() if hasattr(wandb.run, "get_url") else "(no url)",
                )
                return True

            # One retry with a unique id; keeps training usable even if id collides in W&B.
            if not _init_worker_wandb(0) and not _init_worker_wandb(1):
                self._use_wandb = False

    def get_n_next_checkpoints_xy(
        self, position: list[float] | np.ndarray, number_of_next_points: int
    ) -> list[float]:
        """Compatibility wrapper. Prefer `track_feature_provider.get_n_next_checkpoints_xy`."""
        return self.track_feature_provider.get_n_next_checkpoints_xy(
            position, number_of_next_points
        )

    def _nearest_index_in_window(
        self, pos: np.ndarray, start_idx: int, end_idx: int
    ) -> tuple[int, float]:
        """Find nearest trajectory index to `pos` in the half-open window [start_idx, end_idx)."""
        if end_idx <= start_idx:
            idx = min(max(start_idx, 0), self.datalen - 1)
            return idx, float(np.linalg.norm(pos - self.data[idx]))

        segment = self.data[start_idx:end_idx]
        dists = np.linalg.norm(segment - pos, axis=1)
        best_local = int(np.argmin(dists))
        best_index = start_idx + best_local
        min_dist = float(dists[best_local])
        return best_index, min_dist

    def _tracking_lookahead_points(self, speed_kmh: float) -> int:
        """Adaptive trajectory-tracking lookahead independent from observation horizon."""
        spacing = max(self.average_distance, 0.2)
        meters_per_step = max(0.0, speed_kmh / 3.6) * self._time_step_duration
        dynamic_points = math.ceil(meters_per_step / spacing) + self._tracking_margin_points
        lookahead = max(self._tracking_lookahead_min, dynamic_points)
        return min(self._tracking_lookahead_max, lookahead)

    def get_track_info(
        self, position: list[float] | np.ndarray, points_number: int
    ) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
        """Compatibility wrapper. Prefer `track_feature_provider.get_track_info`."""
        return self.track_feature_provider.get_track_info(position, points_number)

    def calculate_average_distance(self) -> float:
        """Mean segment length between consecutive trajectory points."""
        if len(self.data) < 2:
            return 0.0
        distances = np.linalg.norm(np.diff(self.data, axis=0), axis=1)
        if distances.size == 0:
            return 0.0
        mean_dist = float(np.mean(distances))
        if not np.isfinite(mean_dist):
            return 0.0
        return mean_dist

    def _reset_debug_accumulators(self):
        """Resets debug metric accumulators."""
        self._dbg_speeds_kmh = []
        self._dbg_speed_rewards = []
        self._dbg_progress_rewards = []
        self._dbg_projected_rewards = []
        self._dbg_cte_penalties = []
        self._dbg_boundary_penalties = []
        self._dbg_lateral_ratios = []
        self._dbg_crash_steps = 0
        self._dbg_end_of_track_awarded = False

    _EPISODE_FIELD_MAP: ClassVar[dict[str, str]] = {
        "last_progress_step": "_last_progress_step",
        "term_reason": "_term_reason",
        "logged_run_this_episode": "_logged_run_this_episode",
        "prev_pos": "_prev_pos",
    }

    def _set_episode_state(self, state: EpisodeState) -> None:
        """Apply grouped episode state fields to instance attributes."""
        for f in dataclasses.fields(state):
            attr_name = self._EPISODE_FIELD_MAP.get(f.name, f.name)
            setattr(self, attr_name, getattr(state, f.name))

    def _log_episode_debug_summary(self):
        """Logs a per-component reward breakdown for the finished episode."""
        if not self._dbg_speeds_kmh:
            return
        steps = len(self._dbg_speeds_kmh)
        speeds = np.array(self._dbg_speeds_kmh)
        logger.info(f" Episode summary ({steps} steps):")
        logger.info(
            f"  Speed km/h: mean={float(np.mean(speeds)):.1f} max={float(np.max(speeds)):.1f}"
        )

        progress_sum = (
            float(np.sum(self._dbg_progress_rewards)) if self._dbg_progress_rewards else 0.0
        )
        speed_sum = float(np.sum(self._dbg_speed_rewards)) if self._dbg_speed_rewards else 0.0
        projected_sum = (
            float(np.sum(self._dbg_projected_rewards)) if self._dbg_projected_rewards else 0.0
        )

        const_sum = -self._constant_penalty * steps
        boundary_sum = (
            float(np.sum(self._dbg_boundary_penalties)) if self._dbg_boundary_penalties else 0.0
        )
        eot = (
            self._end_of_track_reward if getattr(self, "_dbg_end_of_track_awarded", False) else 0.0
        )

        logger.info(
            f"  progress: {progress_sum:+.1f}  |  speed: {speed_sum:+.1f}  |"
            f"  projected_v: {projected_sum:+.1f}  |  constant: {const_sum:+.1f}"
            f"  |  boundary: {boundary_sum:+.1f}"
        )
        approx = progress_sum + speed_sum + projected_sum + const_sum + boundary_sum + eot
        logger.info(f"  end_of_track: {eot:+.1f}  |  approx_total: {approx:+.1f}")

        if self._dbg_lateral_ratios:
            lr = np.array(self._dbg_lateral_ratios)
            n_over_05 = int(np.sum(lr > 0.5))
            n_over_07 = int(np.sum(lr > 0.7))
            n_over_085 = int(np.sum(lr > 0.85))
            n_over_10 = int(np.sum(lr > 1.0))
            logger.info(
                f"  boundary detail: {len(lr)} steps checked, "
                f"lateral_ratio mean={float(np.mean(lr)):.3f} max={float(np.max(lr)):.3f}  |  "
                f">0.5: {n_over_05}  |  >0.7: {n_over_07}  |  >0.85: {n_over_085}  |  >1.0: {n_over_10} steps"  # noqa: E501
            )

    def _current_drift_weight(self) -> float:
        if self._drift_anneal_steps > 0:
            frac = min(1.0, self._global_env_steps / self._drift_anneal_steps)
            return self._drift_weight_start + frac * (
                self._drift_weight_end - self._drift_weight_start
            )
        return self._drift_reward_weight

    def compute_reward(
        self,
        pos,
        velocity_xyz: np.ndarray | list[float] | tuple[float, float, float] | None = None,
        dir_xyz: np.ndarray | list[float] | tuple[float, float, float] | None = None,
        surface_materials: list[int] | tuple[int, int, int, int] | np.ndarray | None = None,
        wheel_slips: list[float] | tuple[float, float, float, float] | np.ndarray | None = None,
        crashed: bool = False,
        speed: float | None = None,
        next_cp: bool = False,
        next_lap: bool = False,
        end_of_track: bool = False,
        input_brake: float | None = None,
        aim_yaw: float | None = None,
        input_steer: float | None = None,
        gear: float | None = None,
        slip_angle_deg: float | None = None,
    ) -> tuple[float, bool, int, float]:
        """
        Computes the reward and termination status for the current step.

        Args:
            pos: Current position.
            velocity_xyz: World velocity vector (x, y, z) from telemetry if available.
            dir_xyz: Forward direction vector (x, y, z) from telemetry if available.
            surface_materials: Surface IDs under FL/FR/RL/RR wheels.
            wheel_slips: Slip coefficients for FL/FR/RL/RR wheels.
            crashed (bool): Whether the agent crashed. Defaults to False.
            speed (float, optional): Current speed in km/h. Defaults to None.
            next_cp (bool): Whether a checkpoint was passed. Defaults to False.
            next_lap (bool): Whether a lap was completed. Defaults to False.
            end_of_track (bool): Whether the end of track was reached. Defaults to False.
            input_brake (float, optional): Current brake input. Defaults to None.
            aim_yaw (float, optional): Current vehicle yaw. Defaults to None.
            input_steer (float, optional): Current steering input. Defaults to None.
            gear (float, optional): Current gear. Defaults to None.
            slip_angle_deg (float, optional): Slip angle (deg) for drift reward. Defaults to None.

        Returns:
            Tuple of (reward, terminated, failure_counter, episode_reward).
        """
        terminated = False
        self.step_counter += 1
        self.prev_idx = self.cur_idx
        _speed_kmh = float(speed) if speed is not None else 0.0

        pos = np.asarray(pos, dtype=np.float64).reshape(3)
        start_fwd = self.cur_idx
        tracking_lookahead = self._tracking_lookahead_points(_speed_kmh)
        end_fwd = min(self.cur_idx + tracking_lookahead, self.datalen)
        best_index, min_dist = self._nearest_index_in_window(pos, start_fwd, end_fwd)
        reward_progress = 0.0
        if self.datalen > 1 and self._total_traj_length > 0:
            idx_furthest = min(self.furthest_reached_idx, self.datalen - 1)
            dist_furthest = self._cumulative_dist[idx_furthest]
            dist_best = self._cumulative_dist[min(best_index, self.datalen - 1)]
            distance_gained = max(0.0, float(dist_best - dist_furthest))
            reward_progress = float(
                distance_gained * (self._progress_reward_full_lap / self._total_traj_length)
            )
            if min_dist > OFF_TRACK_PROGRESS_ZERO_MULTIPLIER * self.max_dist_from_traj:
                reward_progress = 0.0
            else:
                self.furthest_reached_idx = max(self.furthest_reached_idx, best_index)
        if reward_progress > 0:
            self._last_progress_step = self.step_counter

        if best_index == self.cur_idx:
            start_bwd = max(0, self.cur_idx - self.nb_obs_backward + 1)
            end_bwd = self.cur_idx + 1
            best_index, min_dist = self._nearest_index_in_window(pos, start_bwd, end_bwd)
        self.cur_idx = best_index
        self._global_env_steps += 1

        _speed_reward_added = 0.0
        _computed_slip_deg: float | None = None

        heading_xz: np.ndarray | None = None
        if dir_xyz is not None:
            d3 = np.asarray(dir_xyz, dtype=np.float64).reshape(3)
            d_xz = d3[[0, 2]]
            d_norm = np.linalg.norm(d_xz)
            if d_norm > 1e-6:
                heading_xz = d_xz / d_norm

        motion_xz: np.ndarray | None = None
        if velocity_xyz is not None:
            v3 = np.asarray(velocity_xyz, dtype=np.float64).reshape(3)
            v_xz = v3[[0, 2]]
            v_norm = np.linalg.norm(v_xz)
            if v_norm > 1e-6:
                motion_xz = v_xz / v_norm

        if heading_xz is None and aim_yaw is not None:
            car_yaw = float(aim_yaw)
            heading_xz = np.array([np.sin(car_yaw), np.cos(car_yaw)], dtype=np.float64)

        if motion_xz is not None:
            car_dir = motion_xz
        elif heading_xz is not None:
            car_dir = heading_xz
        else:
            if self._prev_pos is not None:
                delta = (pos - self._prev_pos)[[0, 2]]
                delta_norm = np.linalg.norm(delta)
                if delta_norm > 1e-6:
                    car_dir = delta / delta_norm
                else:
                    car_dir = np.array([0.0, 1.0], dtype=np.float64)
            else:
                car_dir = np.array([0.0, 1.0], dtype=np.float64)
        self._prev_pos = pos.copy()  # always track so next step can use position delta

        if heading_xz is not None and motion_xz is not None:
            cross = heading_xz[0] * motion_xz[1] - heading_xz[1] * motion_xz[0]
            dot = float(np.dot(heading_xz, motion_xz))
            _computed_slip_deg = abs(math.degrees(math.atan2(cross, dot)))

        is_airborne = False
        if surface_materials is not None:
            mats = np.asarray(surface_materials).reshape(-1)
            if mats.size >= 4:
                is_airborne = bool(np.all(mats[:4] == 0))

        next_idx = min(best_index + 1, self.datalen - 1)
        alignment_effective = 0.0
        track_alignment = 0.0
        if next_idx > best_index:
            track_vec = self.data[next_idx] - self.data[best_index]
        elif best_index > 0:
            track_vec = self.data[best_index] - self.data[best_index - 1]
        else:
            track_vec = None
        if track_vec is not None:
            track_vec_xz = track_vec[[0, 2]]
            norm = np.linalg.norm(track_vec_xz)
            if norm > 0:
                track_dir = track_vec_xz / norm
                alignment = np.dot(track_dir, car_dir)
                track_alignment = float(alignment)
                base_alignment = max(
                    getattr(self, "_speed_reward_alignment_floor", 0.0),
                    max(0.0, alignment),
                )
                alignment_effective = max(0.5, base_alignment) if is_airborne else base_alignment

        reward = reward_progress

        _projected_velocity_reward = 0.0
        # Speed-shaped components are gated by progress: only credit when the
        # agent covers new ground (reward_progress > 0). Otherwise fast motion
        # over already-visited track is farmable.
        if self._projected_velocity_scale > 0.0 and _speed_kmh != 0.0 and reward_progress > 0.0:
            speed_ms = _speed_kmh / 3.6
            _projected_velocity_reward = (
                self._projected_velocity_scale
                * speed_ms
                * track_alignment
                * self._time_step_duration
            )
            reward += _projected_velocity_reward

        if (
            _speed_kmh > SPEED_REWARD_MIN_KMH
            and min_dist <= self.max_dist_from_traj
            and self._speed_reward_weight > 0
            and reward_progress > 0.0
        ):
            useful_speed_factor = (_speed_kmh / self._max_speed_kmh) * alignment_effective
            if useful_speed_factor > 0:
                exp = getattr(self, "_speed_reward_exponent", 1.0)
                _speed_reward_added = self._speed_reward_weight * (useful_speed_factor**exp)
                reward += _speed_reward_added

        track_curvature_abs = 0.0
        if self.datalen > 2 and (self._drift_reward_weight > 0 or self._cornering_speed_bonus > 0):
            i0 = max(0, best_index - 1)
            i2 = min(self.datalen - 1, best_index + 1)
            if i0 < i2:
                track_curvature_abs = abs(
                    discrete_curvature_xz(
                        self.data[i0],
                        self.data[best_index],
                        self.data[i2],
                        signed=False,
                    )
                )

        _effective_slip = slip_angle_deg if slip_angle_deg is not None else _computed_slip_deg
        allow_drift_bonus = track_curvature_abs >= self._drift_curvature_threshold
        if (
            reward_progress > 0
            and wheel_slips is not None
            and _speed_kmh >= self._drift_threshold_kmh
            and allow_drift_bonus
        ):
            ws = np.asarray(wheel_slips, dtype=np.float64).reshape(-1)
            if ws.size >= 4:
                rear_slip_avg = 0.5 * (abs(float(ws[2])) + abs(float(ws[3])))
                drift_w = self._current_drift_weight()
                if drift_w > 0 and rear_slip_avg > self._rear_slip_activation:
                    reward += drift_w * min(1.0, rear_slip_avg)
        elif (
            reward_progress > 0
            and _effective_slip is not None
            and _speed_kmh >= self._drift_threshold_kmh
            and allow_drift_bonus
        ):
            _effective_slip = abs(float(_effective_slip))
            drift_w = self._current_drift_weight()
            if drift_w > 0:
                opt = self._drift_optimal_angle_deg
                sigma = max(1e-3, self._drift_sigma_deg)
                drift_bonus = drift_w * math.exp(
                    -((_effective_slip - opt) ** 2) / (2.0 * sigma * sigma)
                )
                reward += drift_bonus

        # --- Cornering speed bonus ---
        if (
            reward_progress > 0
            and self._cornering_speed_bonus > 0
            and track_curvature_abs > self._cornering_curvature_threshold
        ):
            speed_frac = min(1.0, _speed_kmh / max(1.0, self._max_speed_kmh))
            reward += (
                self._cornering_speed_bonus * speed_frac * track_curvature_abs * alignment_effective
            )

        # --- Progress acceleration bonus ---
        if self._progress_accel_bonus > 0 and reward_progress > 0:
            self._progress_ema = 0.9 * self._progress_ema + 0.1 * reward_progress
            if reward_progress > self._progress_ema:
                reward += self._progress_accel_bonus * (reward_progress - self._progress_ema)

        race_progress = self.compute_race_progress()
        if race_progress > self.furthest_race_progress:
            self.furthest_race_progress = race_progress

        if (
            getattr(self, "_use_time_no_progress", False)
            and race_progress >= NEAR_FINISH_PROGRESS_THRESHOLD
            and _speed_kmh >= NEAR_FINISH_MIN_SPEED_KMH
            and not end_of_track
        ):
            self._last_progress_step = self.step_counter

        if (
            getattr(self, "_use_time_no_progress", False)
            and (self.step_counter - self._last_progress_step) >= self._max_no_progress_steps
        ):
            terminated = True
            self._term_reason = "no_progress_timeout"

        # Rate-based stall detection: catches slow creep that resets the binary
        # timer above on every tiny forward gain. Uses the cumulative distance at
        # the furthest reached index (monotone, immune to backward jitter).
        if self._slow_progress_window_steps > 0 and self.datalen > 1:
            idx_f = min(self.furthest_reached_idx, self.datalen - 1)
            hist = self._slow_progress_dist_history
            hist.append(float(self._cumulative_dist[idx_f]))
            near_finish_grace = (
                race_progress >= NEAR_FINISH_PROGRESS_THRESHOLD
                and _speed_kmh >= NEAR_FINISH_MIN_SPEED_KMH
            )
            window_required_m = (
                self._min_progress_rate_mps
                * self._slow_progress_window_steps
                * self._time_step_duration
            )
            if (
                len(hist) == hist.maxlen
                and not end_of_track
                and not near_finish_grace
                and (hist[-1] - hist[0]) < window_required_m
            ):
                terminated = True
                self._term_reason = "slow_progress"

        if min_dist > self._max_track_width and self.step_counter > self._off_track_grace_steps:
            terminated = True
            self._term_reason = "off_track"

        cte_penalty = 0.0
        if self._cte_penalty_weight > 0.0:
            norm_dist = min_dist / max(1.0, self._max_track_width / 2.0)
            cte_penalty = self._cte_penalty_weight * (norm_dist**self._cte_penalty_exponent)
            reward -= cte_penalty

        # --- Boundary proximity penalties (requires loaded track boundary geometry) ---
        # Grace period: skip boundary penalties for the first ~1s so the spawn
        # position (which may sit outside the boundary polyline) doesn't
        # immediately terminate or dominate the reward signal.
        _boundary_soft_pen = 0.0
        _boundary_crash_pen = 0.0
        _boundary_grace = self.step_counter <= self._off_track_grace_steps
        if (
            self._has_boundaries
            and not _boundary_grace
            and (
                self._boundary_penalty_weight > 0
                or self._boundary_crash_penalty > 0
                or self._wall_hug_penalty_factor > 0
            )
        ):
            pos_xz = pos[[0, 2]]
            bi = min(best_index, len(self._road_center_xz) - 1)
            center = self._road_center_xz[bi]
            half_w = float(self._road_half_widths[bi])

            dist_from_center = float(np.linalg.norm(pos_xz - center))
            lateral_ratio = dist_from_center / max(half_w, MIN_ROAD_HALF_WIDTH_M)

            _bp_start = self._boundary_penalty_start
            if self._boundary_penalty_weight > 0 and _bp_start < lateral_ratio <= 1.0:
                frac = (lateral_ratio - _bp_start) / (1.0 - _bp_start)
                _boundary_soft_pen += self._boundary_penalty_weight * frac * frac

            if lateral_ratio > 1.0 and self._boundary_crash_penalty > 0:
                terminated = True
                self._term_reason = "boundary_crash"
                _boundary_crash_pen = self._boundary_crash_penalty

            if (
                self._wall_hug_penalty_factor > 0
                and _speed_kmh >= self._wall_hug_speed_threshold
                and lateral_ratio > self._wall_hug_lateral_threshold
            ):
                _boundary_soft_pen += (
                    self._wall_hug_penalty_factor * _speed_kmh / self._max_speed_kmh
                )

            reward -= _boundary_soft_pen
            if self._debug_reward:
                self._dbg_lateral_ratios.append(lateral_ratio)

        if crashed:
            reward -= abs(self.crash_penalty)

        if self._constant_penalty > 0:
            reward -= self._constant_penalty

        if end_of_track:
            if self.datalen > 1 and self._total_traj_length > 0:
                dist_best = self._cumulative_dist[min(best_index, self.datalen - 1)]
                remaining_dist = max(0.0, float(self._total_traj_length - dist_best))
                reward += remaining_dist * (
                    self._progress_reward_full_lap / self._total_traj_length
                )
            reward += self._end_of_track_reward
            self._dbg_end_of_track_awarded = True

        if self._debug_reward:
            self._dbg_speeds_kmh.append(_speed_kmh)
            self._dbg_progress_rewards.append(reward_progress)
            self._dbg_speed_rewards.append(_speed_reward_added)
            self._dbg_projected_rewards.append(_projected_velocity_reward)
            self._dbg_cte_penalties.append(-cte_penalty)
            self._dbg_boundary_penalties.append(-(_boundary_soft_pen + _boundary_crash_pen))
        reward = max(-self._reward_clip_floor, reward)
        # Failure terminations must not be free: without a terminal cost,
        # "creep until the no-progress timeout" bootstraps to V~0 and remains a
        # safe local optimum (v7 collapsed into exactly this). Applied after the
        # clip floor: the floor bounds accumulated per-step shaping, not
        # one-time terminal events. end_of_track is never penalized.
        if (
            terminated
            and not end_of_track
            and self._term_reason == "boundary_crash"
            and _boundary_crash_pen > 0.0
        ):
            reward -= _boundary_crash_pen
        elif (
            terminated
            and not end_of_track
            and self._terminal_failure_penalty > 0.0
            and (self._term_reason != "boundary_crash" or self._boundary_crash_penalty <= 0.0)
        ):
            reward -= self._terminal_failure_penalty
        reward = reward * self._reward_scale

        if terminated and self._term_reason is None:
            self._term_reason = "unknown_term"
        if end_of_track and self._term_reason is None:
            self._term_reason = "end_of_track"

        if getattr(self, "_use_time_no_progress", False):
            self.failure_counter = self.step_counter - self._last_progress_step
        else:
            self.failure_counter = 0

        self.episode_reward += reward

        return reward, terminated, self.failure_counter, self.episode_reward

    def log_model_run(self, terminated: bool, end_of_track: bool, truncated: bool = False) -> None:
        """Log episode outcome to console and optionally to Weights & Biases.

        Args:
            terminated: Whether the episode was terminated (stall, off-track, etc.).
            end_of_track: Whether the agent reached the end of the track.
            truncated: Whether the episode ended due to a time/sample cap (Gymnasium truncation).
        """
        episode_done = bool(terminated or end_of_track or truncated)
        if episode_done and not self._logged_run_this_episode:
            self._logged_run_this_episode = True
            if end_of_track:
                self.furthest_race_progress = 1.0

            run_time_seconds = self.step_counter * getattr(
                self, "_time_step_duration", TIME_STEP_SECONDS
            )
            term_reason = getattr(self, "_term_reason", None)
            if truncated and term_reason is None:
                term_reason = "truncated"
            logger.info(
                "Total reward of the run: {:.4f} (Steps: {}, Time: {:.2f}s, reason: {}, "
                "progress: {:.3f}, idx: {}/{})",
                self.episode_reward,
                self.step_counter,
                run_time_seconds,
                term_reason,
                self.furthest_race_progress,
                self.cur_idx,
                self.datalen,
            )
            if self._debug_reward:
                self._log_episode_debug_summary()

            if self._use_wandb:
                import wandb

                if wandb.run is None:
                    logger.warning("wandb.log skipped: no active W&B run on worker")
                    return

                self._episode_count = getattr(self, "_episode_count", 0) + 1
                log_dict: dict[str, float | int | str] = {
                    "run/reward": self.episode_reward,
                    "run/time_seconds": run_time_seconds,
                    "run/steps": self.step_counter,
                    "run/best_race_progress": self.furthest_race_progress,
                    "run/cur_idx_frac": self.cur_idx / max(1, self.datalen - 1),
                    "run/episode_count": self._episode_count,
                    "run/finish_time": run_time_seconds if end_of_track else 0.0,
                    "run/finished_track": int(end_of_track),
                    "run/truncated": int(truncated),
                }
                if term_reason is not None:
                    log_dict["run/term_reason"] = term_reason
                wandb.log(log_dict)

    def compute_race_progress(self) -> float:
        """Current race progress as fraction of trajectory length (0.0 to 1.0)."""
        if len(self.data) <= 1:
            return 0.0
        return min(1.0, max(0.0, self.cur_idx / (len(self.data) - 1)))

    def reset(self) -> None:
        """Reset reward state for a new episode."""
        self._set_episode_state(EpisodeState(lap_cur_cooldown=self._lap_cooldown_init))
        self._progress_ema = 0.0
        self._slow_progress_dist_history.clear()
        if self._debug_reward:
            self._reset_debug_accumulators()
