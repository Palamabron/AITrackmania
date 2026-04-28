"""Reward and termination logic for TrackMania 2020 RL.

RewardFunction computes step reward and termination from position, speed,
and inputs. It uses a reference trajectory and optional track boundaries
for progress, speed-along-track, and off-track/stall detection.
"""

import atexit
import math
import os
import pickle
import shutil
import socket
import tempfile
import uuid
from typing import Any

import numpy as np
from loguru import logger

from tmrl.config.spacing_lookahead import points_number_from_spacing_config

OFF_TRACK_PROGRESS_ZERO_MULTIPLIER = 2.0
SPEED_REWARD_MIN_KMH = 5.0
TIME_STEP_SECONDS = 0.05
DUMMY_TRAJECTORY_THRESHOLD = 2
MIN_ROAD_HALF_WIDTH_M = 0.5
DEFAULT_ROAD_HALF_WIDTH_M = 12.0
DT_MILLISECOND_THRESHOLD = 1.0


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
    arc_positions = np.linspace(0.0, total_length, num_points, endpoint=True)
    result = []
    segment_index = 0
    for arc_s in arc_positions:
        if arc_s >= total_length:
            result.append(points[-1].copy())
            continue
        while segment_index + 1 < num_input and cumulative_length[segment_index + 1] < arc_s:
            segment_index += 1
        if segment_index + 1 >= num_input:
            result.append(points[-1].copy())
            continue
        seg_start = cumulative_length[segment_index]
        seg_end = cumulative_length[segment_index + 1]
        interp = (arc_s - seg_start) / (seg_end - seg_start) if seg_end > seg_start else 0.0
        interp = np.clip(interp, 0.0, 1.0)
        result.append((1.0 - interp) * points[segment_index] + interp * points[segment_index + 1])
    return np.array(result, dtype=np.float64)


def _ensure_wandb_api_key(api_key: str) -> None:
    """Set WANDB_API_KEY env var if not already present and a key was provided."""
    if "WANDB_API_KEY" not in os.environ and api_key:
        os.environ["WANDB_API_KEY"] = api_key


class RewardFunction:
    """Reward and termination logic for TrackMania 2020 RL.

    Uses OpenPlanet API data. Rewards progress along a reference trajectory,
    speed along the track, and applies penalties for crash, steering jerk,
    and constant penalty. Handles termination (stall, off-track, end-of-track).
    """

    def __init__(
        self,
        reward_data_path: str,
        nb_obs_forward: int = 8,
        nb_obs_backward: int = 8,
        max_dist_from_traj: float = 23.5,
        crash_penalty: float = 10.0,
        constant_penalty: float = 0.0,
        *,
        # --- Track geometry ---
        is_lidar: bool = False,
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

        if is_lidar:
            if not os.path.isfile(track_path_left):
                raise FileNotFoundError(
                    f"LIDAR requires left track boundary: missing {track_path_left}"
                )
            if not os.path.isfile(track_path_right):
                raise FileNotFoundError(
                    f"LIDAR requires right track boundary: missing {track_path_right}"
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

        self._has_boundaries = (
            len(self.left_track) >= 2 and len(self.right_track) >= 2 and self.datalen >= 2
        )
        if self._has_boundaries:
            if len(self.left_track) != self.datalen:
                self.left_track = _resample_polyline_by_arc_length(self.left_track, self.datalen)
            if len(self.right_track) != self.datalen:
                self.right_track = _resample_polyline_by_arc_length(self.right_track, self.datalen)
            self._left_xz = self.left_track[:, [0, 2]].astype(np.float64).copy()
            self._right_xz = self.right_track[:, [0, 2]].astype(np.float64).copy()
            self._road_center_xz = (self._left_xz + self._right_xz) / 2.0
            self._road_half_widths = np.linalg.norm(self._left_xz - self._right_xz, axis=1) / 2.0
            self._road_half_widths = np.maximum(self._road_half_widths, MIN_ROAD_HALF_WIDTH_M)
            _hw = self._road_half_widths
            logger.info(
                "Track boundaries loaded: {} points, road half-width "
                "min={:.1f}m mean={:.1f}m max={:.1f}m",
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
        self._max_track_width = float(rc.get("max_track_width", 35.0))
        self.crash_penalty = float(rc.get("crash_penalty", 2.0))
        self._reward_clip_floor = float(rc.get("reward_clip_floor", 5.0))
        self._reward_scale = float(rc.get("reward_scale", 1.0))
        self._end_of_track_reward = float(rc.get("end_of_track_reward", 10.0))
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

            if not _init_worker_wandb(0):
                # One retry with a unique id; keeps training usable even if id collides in W&B.
                if not _init_worker_wandb(1):
                    self._use_wandb = False

    def get_n_next_checkpoints_xy(
        self, position: list[float] | np.ndarray, number_of_next_points: int
    ) -> list[float]:
        """Next N checkpoint (x, z) coordinates relative to position, scaled by 10.

        Args:
            position: Current (x, y, z) position of the agent.
            number_of_next_points: Number of future trajectory points to return.

        Returns:
            Flattened list of (delta_x, delta_z) per point, each scaled by 10.
        """
        max_idx = len(self.data) - 1
        next_indices = [
            min(self.cur_idx + step * self._checkpoint_stride, max_idx)
            for step in range(1, number_of_next_points + 1)
        ]
        route_to_next_poses = []
        for pos_index in next_indices:
            for axis in (0, -1):
                route_to_next_poses.append((self.data[pos_index][axis] - position[axis]) * 10.0)
        return route_to_next_poses

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

    def get_track_info(
        self, position: list[float] | np.ndarray, points_number: int
    ) -> (
        tuple[list[float], list[float], list[float]]
        | tuple[list[float], list[float], list[float], list[float]]
    ):
        """Track boundary (left, center, right) positions relative to current position.

        Args:
            position: Current (x, y, z) position of the agent.
            points_number: Number of look-ahead points (ignored if spacing-based points used).

        Returns:
            Tuple of (left_positions, center_positions, right_positions), each
            a flat list of relative coordinates.
        """
        max_idx = min(len(self.data), len(self.left_track), len(self.right_track)) - 1
        if (getattr(self, "_point_spacing_m", 0) or 0) > 0 and (
            getattr(self, "_points_number", 0) or 0
        ) > 0:
            next_indices = []
            cur_idx = self.cur_idx
            cur_dist = self._cumulative_dist[min(cur_idx, self.datalen - 1)]
            n_pts = self._points_number or 0
            for _ in range(n_pts):
                target_dist = cur_dist + self._point_spacing_m
                while (
                    cur_idx < self.datalen - 1 and self._cumulative_dist[cur_idx + 1] <= target_dist
                ):
                    cur_idx += 1
                if cur_idx < self.datalen - 1:
                    cur_idx += 1
                    cur_dist = self._cumulative_dist[cur_idx]
                else:
                    cur_dist = self._cumulative_dist[-1] + self._point_spacing_m
                next_indices.append(min(cur_idx, max_idx))
            points_number = len(next_indices)
        else:
            next_indices = [
                self.cur_idx + step * self._checkpoint_stride + 1 for step in range(points_number)
            ]
            for idx in range(len(next_indices)):
                if next_indices[idx] > max_idx:
                    next_indices[idx] = max_idx

        left_positions, center_positions, right_positions = [], [], []
        for pos_index in next_indices:
            for axis in (0, -1):
                left_val = self.left_track[pos_index][axis]
                right_val = self.right_track[pos_index][axis]
                center_val = (left_val + right_val) / 2.0
                left_positions.append(left_val - position[axis])
                center_positions.append(center_val - position[axis])
                right_positions.append(right_val - position[axis])

        if getattr(self, "_track_curvature_obs", False) and len(next_indices) > 0:
            curvatures = []
            for _k, pos_index in enumerate(next_indices):
                i0 = max(0, pos_index - 1)
                i1 = pos_index
                i2 = min(self.datalen - 1, pos_index + 1)
                if i0 >= i2 or self.datalen < 2:
                    curvatures.append(0.0)
                    continue
                p0 = self.data[i0]
                p1 = self.data[i1]
                p2 = self.data[i2]
                v1 = np.array([p1[0] - p0[0], p1[2] - p0[2]], dtype=np.float64)
                v2 = np.array([p2[0] - p1[0], p2[2] - p1[2]], dtype=np.float64)
                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)
                if n1 < 1e-9 or n2 < 1e-9:
                    curvatures.append(0.0)
                    continue
                v1, v2 = v1 / n1, v2 / n2
                dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
                angle = math.acos(dot)
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                sign = 1.0 if cross >= 0 else -1.0
                arc = 0.5 * (n1 + n2)
                kappa = sign * (angle / arc) if arc > 1e-9 else 0.0
                curvatures.append(float(kappa))
            return left_positions, center_positions, right_positions, curvatures
        return left_positions, center_positions, right_positions

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
        self._dbg_crash_steps = 0
        self._dbg_end_of_track_awarded = False

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

        const_sum = -self._constant_penalty * steps
        eot = (
            self._end_of_track_reward if getattr(self, "_dbg_end_of_track_awarded", False) else 0.0
        )

        logger.info(
            f"  progress: {progress_sum:+.1f}  |  speed: {speed_sum:+.1f}  |"
            f"  constant: {const_sum:+.1f}"
        )
        approx = progress_sum + speed_sum + const_sum + eot
        logger.info(f"  end_of_track: {eot:+.1f}  |  approx_total: {approx:+.1f}")

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
        end_fwd = min(self.cur_idx + self.nb_obs_forward, self.datalen)
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
            d_xz = np.array([d3[0], d3[2]], dtype=np.float64)
            d_norm = np.linalg.norm(d_xz)
            if d_norm > 1e-6:
                heading_xz = d_xz / d_norm

        motion_xz: np.ndarray | None = None
        if velocity_xyz is not None:
            v3 = np.asarray(velocity_xyz, dtype=np.float64).reshape(3)
            v_xz = np.array([v3[0], v3[2]], dtype=np.float64)
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
            # Fall back to position-delta direction from previous step.
            if self._prev_pos is not None:
                delta = pos[[0, 2]] - self._prev_pos[[0, 2]]
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
        if next_idx > best_index:
            track_vec = self.data[next_idx] - self.data[best_index]
            track_vec_xz = np.array([track_vec[0], track_vec[2]], dtype=np.float64)
            norm = np.linalg.norm(track_vec_xz)
            if norm > 0:
                track_dir = track_vec_xz / norm
                alignment = np.dot(track_dir, car_dir)
                if is_airborne:
                    alignment_effective = 1.0
                else:
                    alignment_effective = max(
                        getattr(self, "_speed_reward_alignment_floor", 0.0),
                        max(0.0, alignment),
                    )

        reward = reward_progress

        if (
            _speed_kmh > SPEED_REWARD_MIN_KMH
            and min_dist <= self.max_dist_from_traj
            and self._speed_reward_weight > 0
        ):
            useful_speed_factor = (_speed_kmh / self._max_speed_kmh) * alignment_effective
            if useful_speed_factor > 0:
                exp = getattr(self, "_speed_reward_exponent", 1.0)
                _speed_reward_added = self._speed_reward_weight * (useful_speed_factor**exp)
                reward += _speed_reward_added

        _effective_slip = slip_angle_deg if slip_angle_deg is not None else _computed_slip_deg
        if wheel_slips is not None and _speed_kmh >= self._drift_threshold_kmh:
            ws = np.asarray(wheel_slips, dtype=np.float64).reshape(-1)
            if ws.size >= 4:
                rear_slip_avg = 0.5 * (abs(float(ws[2])) + abs(float(ws[3])))
                drift_w = self._current_drift_weight()
                if drift_w > 0 and rear_slip_avg > self._rear_slip_activation:
                    reward += drift_w * min(1.0, rear_slip_avg)
        elif _effective_slip is not None and _speed_kmh >= self._drift_threshold_kmh:
            _effective_slip = abs(float(_effective_slip))
            drift_w = self._current_drift_weight()
            if drift_w > 0:
                opt = self._drift_optimal_angle_deg
                sigma = max(1e-3, self._drift_sigma_deg)
                drift_bonus = drift_w * math.exp(
                    -((_effective_slip - opt) ** 2) / (2.0 * sigma * sigma)
                )
                reward += drift_bonus

        if (
            getattr(self, "_use_time_no_progress", False)
            and (self.step_counter - self._last_progress_step) >= self._max_no_progress_steps
        ):
            terminated = True
            self._term_reason = "no_progress_timeout"

        if min_dist > self._max_track_width and self.step_counter > self._off_track_grace_steps:
            terminated = True
            self._term_reason = "off_track"

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

        reward = max(-self._reward_clip_floor, reward)
        reward = reward * self._reward_scale

        if terminated and self._term_reason is None:
            self._term_reason = "unknown_term"
        if end_of_track and self._term_reason is None:
            self._term_reason = "end_of_track"

        if getattr(self, "_use_time_no_progress", False):
            self.failure_counter = self.step_counter - self._last_progress_step
        else:
            self.failure_counter = 0
        race_progress = self.compute_race_progress()
        if race_progress > self.furthest_race_progress:
            self.furthest_race_progress = race_progress

        self.episode_reward += reward

        return reward, terminated, self.failure_counter, self.episode_reward

    def log_model_run(
        self, terminated: bool, end_of_track: bool, truncated: bool = False
    ) -> None:
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
                "Total reward of the run: {:.4f} (Steps: {}, Time: {:.2f}s, reason: {})",
                self.episode_reward,
                self.step_counter,
                run_time_seconds,
                term_reason,
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
        self.cur_idx = 0
        self.prev_idx = 0
        self.furthest_reached_idx = 0
        self.step_counter = 0
        self.failure_counter = 0
        self._prev_pos = None
        self._last_progress_step = 0
        self._term_reason = None
        self.episode_reward = 0.0
        self.new_lap = False
        self.lap_cur_cooldown = self._lap_cooldown_init
        self.furthest_race_progress = 0.0
        self._logged_run_this_episode = False
        if self._debug_reward:
            self._reset_debug_accumulators()
