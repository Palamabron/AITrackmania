# standard library imports
import atexit
import math
import os
import pickle
import shutil
import tempfile
import time

# third-party imports
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LinearRegression

import tmrl.config.config_constants as cfg
import wandb


class RewardFunction:
    """
    Computes a reward from the Openplanet API for Trackmania 2020.
    """

    def __init__(
        self,
        reward_data_path,
        nb_obs_forward=8,
        nb_obs_backward=8,
        nb_zero_rew_before_failure=10,
        min_nb_steps_before_failure=int(2.5 * 20),
        max_dist_from_traj=23.5,
        crash_penalty=10.0,
        constant_penalty=0.0,
        low_threshold=10,
        high_threshold=250,
    ):
        """
        Instantiates a reward function for TM2020.

        Args:
            reward_data_path: path where the trajectory file is stored
            nb_obs_forward: max distance of allowed cuts (positions in the trajectory)
            nb_obs_backward: same for rewinding the reward to a previously visited position
            nb_zero_rew_before_failure: after this many steps with no reward, episode ends
            min_nb_steps_before_failure: episode must have at least this many steps before failure
            max_dist_from_traj: reward is 0 if car is further than this from the demo trajectory
        """
        self.reward_data_path = reward_data_path
        if not os.path.exists(reward_data_path):
            logger.warning(
                f"Reward trajectory not found at {reward_data_path}. "
                "Using dummy trajectory; episode will NOT end from distance/progress (no reset). "
                "Record a trajectory to that path to enable trajectory-based rewards and failure."
            )
            self.data = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])  # dummy reward
            self._dummy_trajectory = True
        else:
            with open(reward_data_path, "rb") as f:
                self.data = pickle.load(f)
            self._dummy_trajectory = len(self.data) <= 2
            if self._dummy_trajectory:
                logger.warning(
                    f"Reward file {reward_data_path} has only {len(self.data)} points; "
                    "treating as dummy (no distance/progress termination)."
                )

        if not os.path.exists(cfg.TRACK_PATH_LEFT):
            logger.debug(f" track not found at path:{cfg.TRACK_PATH_LEFT}")
            self.left_track = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])  # dummy
        else:
            with open(cfg.TRACK_PATH_LEFT, "rb") as f:
                self.left_track = pickle.load(f)

        if not os.path.exists(cfg.TRACK_PATH_RIGHT):
            logger.debug(f" track not found at path:{cfg.TRACK_PATH_RIGHT}")
            self.right_track = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])  # dummy
        else:
            with open(cfg.TRACK_PATH_RIGHT, "rb") as f:
                self.right_track = pickle.load(f)

        self.cur_idx = 0
        self.prev_idx = 0
        self.nb_obs_forward = nb_obs_forward
        self.nb_obs_backward = nb_obs_backward
        # Guard against zero/invalid failure thresholds from config migration mismatches.
        # This prevents immediate terminations (e.g. failure_counter at step 1/2).
        cfg_min_steps = int(cfg.REWARD_CONFIG.get("MIN_STEPS", 0))
        self.min_nb_steps_before_failure = max(1, int(min_nb_steps_before_failure), cfg_min_steps)
        self.nb_zero_rew_before_failure = max(1, int(nb_zero_rew_before_failure))
        self.max_dist_from_traj = max_dist_from_traj
        self.step_counter = 0
        self.failure_counter = 0
        self.low_speed_steps = 0  # consecutive steps with speed below stall threshold
        self.datalen = len(self.data)
        self.average_distance = self.calculate_average_distance()
        # Cumulative arc length along trajectory (density-robust progress)
        self._cumulative_dist = np.zeros(max(1, self.datalen))
        if self.datalen > 1:
            diffs = np.linalg.norm(np.diff(self.data, axis=0), axis=1)
            np.cumsum(diffs, out=self._cumulative_dist[1:])
        self._total_traj_length = (
            max(1.0, float(self._cumulative_dist[-1])) if self.datalen >= 1 else 1.0
        )
        # Deviation threshold scales with point spacing; sparse trajectories less penalized
        self._deviation_threshold = max(17.5, self.average_distance * 1.5)
        # Speed bonus only when on track; reckless penalty when off track + high speed
        speed_safe_ratio = float(cfg.REWARD_CONFIG.get("SPEED_SAFE_DEVIATION_RATIO", 0.6))
        self._speed_safe_deviation = speed_safe_ratio * self.max_dist_from_traj
        self._reckless_speed_threshold = float(cfg.REWARD_CONFIG.get("RECKLESS_SPEED_THRESHOLD", 120))
        self._reckless_penalty_factor = float(cfg.REWARD_CONFIG.get("RECKLESS_PENALTY_FACTOR", 0.002))
        self.speed_bonus = cfg.SPEED_BONUS
        # Wall-hugging / slow-progress penalty params
        self._wall_hug_speed_threshold = float(
            cfg.REWARD_CONFIG.get("WALL_HUG_SPEED_THRESHOLD", 10.0)
        )
        self._wall_hug_penalty_factor = float(
            cfg.REWARD_CONFIG.get("WALL_HUG_PENALTY_FACTOR", 0.005)
        )
        self._proximity_reward_shaping = float(
            cfg.REWARD_CONFIG.get("PROXIMITY_REWARD_SHAPING", 0.5)
        )
        self._reward_scale = float(cfg.REWARD_CONFIG.get("REWARD_SCALE", 3.0))
        self._no_progress_steps = 0
        self.crash_penalty = crash_penalty
        # self.crash_counter = 1
        self.constant_penalty = constant_penalty
        self.lap_cur_cooldown = cfg.LAP_COOLDOWN
        # self.checkpoint_cur_cooldown = cfg.CHECKPOINT_COOLDOWN
        self.crash_cur_cooldown = cfg.CRASH_COOLDOWN
        self.new_lap = False
        self.near_finish = False
        self.new_checkpoint = False
        self.episode_reward = 0.0
        self.reward_sum_list = []
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.cur_distance = 0
        self.prev_distance = 0
        self.window_size = 40
        self.cooldown = self.window_size // 4
        self.change_cooldown = self.cooldown
        self.n = max(
            1, min(len(self.data), int(cfg.POINTS_DISTANCE / max(self.average_distance, 0.01)))
        )  # intervals of ~10.25m
        self.i = 0
        self.min_value = int(cfg.MIN_NB_ZERO_REW_BEFORE_FAILURE)
        self.max_value = int(cfg.MAX_NB_ZERO_REW_BEFORE_FAILURE)
        if self.min_value <= 0 and self.max_value <= 0:
            # Align oscillation with configured countdown when legacy ENV keys are absent.
            self.min_value = self.nb_zero_rew_before_failure
            self.max_value = self.nb_zero_rew_before_failure
        self.min_value = max(1, self.min_value)
        self.max_value = max(self.min_value, self.max_value)
        self.mid_value = (self.max_value + self.min_value) / 2
        self.amplitude = (self.max_value - self.min_value) / 2
        self.oscillation_period = cfg.OSCILLATION_PERIOD
        self.index_divider = 100.0 / self.datalen
        print(f"n: {self.n}")
        self.furthest_race_progress = 0
        self.medium_speed_bonus = cfg.SPEED_BONUS / 2
        self._logged_run_this_episode = False  # avoid repeating "Total reward" until next reset
        self._use_wandb = getattr(cfg, "WANDB_WORKER", True)

        if cfg.WANDB_DEBUG_REWARD:
            self.send_reward = []

        if self._use_wandb:
            wandb_dir = tempfile.mkdtemp()  # prevent wandb from polluting the home directory
            atexit.register(
                shutil.rmtree, wandb_dir, ignore_errors=True
            )  # clean up after wandb atexit handler finishes
            wandb_initialized = False
            err_cpt = 0
            while not wandb_initialized:
                try:
                    wandb.init(
                        project=cfg.WANDB_PROJECT,
                        entity=cfg.WANDB_ENTITY,
                        id=cfg.WANDB_RUN_ID + " WORKER",
                        config=cfg.create_config(),
                        job_type="worker",
                        dir=wandb_dir,
                    )
                    wandb_initialized = True
                except Exception as e:
                    err_cpt += 1
                    logger.warning(f"wandb error {err_cpt}: {e}")
                    if err_cpt > 10:
                        logger.warning("Could not connect to wandb, aborting.")
                        exit()
                    else:
                        time.sleep(10.0)

        self.i = 0

    def get_n_next_checkpoints_xy(self, pos, number_of_next_points: int):
        """
        Retrieves the positions of next checkpoints based on the current position.
        """
        next_indices = [self.cur_idx + i * self.n for i in range(1, number_of_next_points + 1)]
        for i in range(len(next_indices)):
            if next_indices[i] >= len(self.data):
                next_indices[i] = len(self.data) - 1
        route_to_next_poses = []
        for pos_index in next_indices:
            for i in (0, -1):
                route_to_next_poses.append((self.data[pos_index][i] - pos[i]) * 10.0)

        return route_to_next_poses

    def get_track_info(self, pos, points_number):
        """
        Fetches track info (left, center, right positions) for the next checkpoints.
        """
        next_indices = [self.cur_idx + i * self.n + 1 for i in range(points_number)]
        left_track_positions, center_track_positions, right_track_positions = [], [], []
        # Clamp to valid range for all arrays we index (data, left_track, right_track)
        max_idx = min(len(self.data), len(self.left_track), len(self.right_track)) - 1
        for i in range(len(next_indices)):
            if next_indices[i] > max_idx:
                next_indices[i] = max_idx

        for pos_index in next_indices:
            for i in (0, -1):
                left = self.left_track[pos_index][i]
                right = self.right_track[pos_index][i]

                center = (left + right) / 2.0

                left_track_positions.append(left - pos[i])
                center_track_positions.append(center - pos[i])
                right_track_positions.append(right - pos[i])

                # left_track_positions.append((self.data[pos_index][i] - pos[i]))
                # left_track_positions.append((self.data[pos_index][i] - pos[i]))
                # left_track_positions.append((self.data[pos_index][i] - pos[i]))

        return left_track_positions, center_track_positions, right_track_positions

    def calculate_average_distance(self):
        """
        Computes the average distance between consecutive points in the track data.
        """
        # Calculate the Euclidean distance between consecutive points in the trajectory
        distances = np.linalg.norm(np.diff(self.data, axis=0), axis=1)

        # Compute the average distance
        average_distance = np.mean(distances)

        return average_distance

    def compute_reward(
        self,
        pos,
        crashed: bool = False,
        speed: float | None = None,
        next_cp: bool = False,
        next_lap: bool = False,
        end_of_track: bool = False,
    ):
        """
        Calculates the reward based on the car's position, speed, and track progress.
        Handles penalties for crashes, speed bonuses, lap completions, etc.

        Reward scale: progress is scaled so a full lap is ~100 raw; then
        reward = tanh(REWARD_SCALE * raw). So raw values > ~0.4 already saturate
        (tanh(3*0.4) ≈ 0.83). END_OF_TRACK_REWARD and LAP_REWARD only need to be
        on the order of 1.0–2.0 to give a strong finish signal; 100 is excessive.
        """
        terminated = False
        term_reason = None
        self.step_counter += 1
        self.prev_idx = self.cur_idx
        min_dist = np.inf
        index = self.cur_idx
        temp = self.nb_obs_forward
        best_index = 0
        while True:
            dist = np.linalg.norm(pos - self.data[index])
            if dist <= min_dist:
                min_dist = float(dist)
                best_index = index
                temp = self.nb_obs_forward
            index += 1
            temp -= 1
            # stop condition
            if (
                index >= self.datalen or temp <= 0
            ):  # if trajectory complete or cuts counter depleted
                # When far from trajectory we used to set best_index = cur_idx so we "didn't move",
                # which prevented any progress when spawn and trajectory start don't align. Now we
                # keep best_index as the closest point so cur_idx can advance (e.g. car drives
                # toward the trajectory). Progress reward for this step is zeroed below.
                break  # we found the best index and can break the while loop

        # Progress reward: use distance along trajectory (density-robust), scale so full lap ~100
        if self.datalen <= 1 or self._total_traj_length <= 0:
            reward = 0.0
            distance_gained = 0.0
        else:
            dist_cur = self._cumulative_dist[min(self.cur_idx, self.datalen - 1)]
            dist_best = self._cumulative_dist[min(best_index, self.datalen - 1)]
            distance_gained = max(0.0, float(dist_best - dist_cur))
            reward = float(distance_gained * (100.0 / self._total_traj_length))
            if min_dist > 2.0 * self.max_dist_from_traj:
                reward = 0.0
            elif min_dist > self._speed_safe_deviation and self._proximity_reward_shaping > 0:
                proximity_ratio = 1.0 - min(
                    1.0,
                    (min_dist - self._speed_safe_deviation)
                    / max(self.max_dist_from_traj - self._speed_safe_deviation, 1.0),
                )
                reward *= max(1.0 - self._proximity_reward_shaping, proximity_ratio)

        if (
            best_index == self.cur_idx
        ):  # if the best index didn't change, we rewind (more Markovian reward)
            min_dist = np.inf
            index = self.cur_idx
            while True:
                dist = np.linalg.norm(pos - self.data[index])
                if dist <= min_dist:
                    min_dist = float(dist)
                    best_index = index
                    temp = self.nb_obs_backward
                index -= 1
                temp -= 1
                # stop condition
                if index <= 0 or temp <= 0:
                    break
            if not getattr(self, "_dummy_trajectory", False):
                _speed_kmh = float(speed) if speed is not None else 0.0
                _stall_threshold_kmh = 5.0
                if (
                    _speed_kmh < _stall_threshold_kmh
                    and self.step_counter > self.min_nb_steps_before_failure
                ):
                    self.failure_counter += 1
                    if self.failure_counter > self.nb_zero_rew_before_failure:
                        terminated = True
                        term_reason = "failure_counter"
                else:
                    self.failure_counter = 0
                # Wall-hug detection: moving but no trajectory progress while off-track
                if (
                    _speed_kmh > self._wall_hug_speed_threshold
                    and min_dist > self._speed_safe_deviation
                ):
                    self._no_progress_steps += 1
                    wall_hug_penalty = (
                        self._wall_hug_penalty_factor
                        * self._no_progress_steps
                        * (min_dist / max(self._speed_safe_deviation, 1.0))
                    )
                    reward -= wall_hug_penalty
                    if (
                        self._no_progress_steps > self.nb_zero_rew_before_failure * 3
                        and self.step_counter > self.min_nb_steps_before_failure
                    ):
                        terminated = True
                        term_reason = "wall_hug_no_progress"
                else:
                    self._no_progress_steps = max(0, self._no_progress_steps - 1)
            else:
                self.failure_counter = 0
        else:
            self.failure_counter = 0
            self._no_progress_steps = max(0, self._no_progress_steps - 2)
        self.cur_idx = best_index

        if self.episode_reward != 0.0:
            reward -= abs(self.constant_penalty)
            _speed_kmh = float(speed) if speed is not None else 0.0
            # Speed bonus only when on track (min_dist <= safe deviation)
            if min_dist <= self._speed_safe_deviation and self.speed_bonus > 0:
                if _speed_kmh > cfg.SPEED_MIN_THRESHOLD:
                    speed_reward = (_speed_kmh - cfg.SPEED_MIN_THRESHOLD) * self.speed_bonus
                    reward += speed_reward
                elif _speed_kmh > cfg.SPEED_MEDIUM_THRESHOLD:
                    speed_reward = (
                        _speed_kmh - cfg.SPEED_MEDIUM_THRESHOLD * 0.75
                    ) * self.speed_bonus * 2
                    reward += speed_reward
            elif _speed_kmh < -0.5:
                penalty = 1 / (1 + np.exp(-0.1 * _speed_kmh - 3)) - 1
                reward += penalty
            # Reckless penalty: high speed + far from trajectory (off track)
            if (
                min_dist > self._deviation_threshold
                and _speed_kmh > self._reckless_speed_threshold
                and self._reckless_penalty_factor > 0
            ):
                reckless_penalty = (
                    self._reckless_penalty_factor
                    * (_speed_kmh - self._reckless_speed_threshold)
                    * (min_dist / max(self._deviation_threshold, 1.0))
                )
                reward -= reckless_penalty

        if not getattr(self, "_dummy_trajectory", False):
            _speed_kmh = float(speed) if speed is not None else 0.0
            _stall_threshold_kmh = 5.0
            # Only terminate for stray (off trajectory) when moving slowly; avoid resetting at speed
            _stray_speed_kmh = 15.0
            if (
                self.step_counter > self.min_nb_steps_before_failure
                and min_dist > self.max_dist_from_traj
                and _speed_kmh < _stray_speed_kmh
            ):
                terminated = True
                term_reason = "stray"

            # End episode when car is stuck (speed near zero for too long)
            if _speed_kmh < _stall_threshold_kmh:
                self.low_speed_steps += 1
                if (
                    self.step_counter > self.min_nb_steps_before_failure
                    and self.low_speed_steps >= self.nb_zero_rew_before_failure
                ):
                    terminated = True
                    if term_reason is None:
                        term_reason = "low_speed_steps"
            else:
                self.low_speed_steps = 0
        else:
            self.low_speed_steps = 0

        # deviation_penalty: threshold/strength scale with trajectory density
        deviation_penalty_applied = 0.0
        if min_dist > self._deviation_threshold:
            excess = min_dist - self._deviation_threshold
            scale = max(self.average_distance, 1.0)
            deviation_penalty_applied = abs((2 / (1 + np.exp(-0.15 * (excess / scale)))) - 1)
            reward -= deviation_penalty_applied
        # Off-track speed penalty: penalize any driving far from trajectory
        _speed_kmh_ot = float(speed) if speed is not None else 0.0
        if (
            min_dist > self._speed_safe_deviation
            and _speed_kmh_ot > 5.0
            and self._wall_hug_penalty_factor > 0
        ):
            off_track_speed_penalty = (
                self._wall_hug_penalty_factor
                * 0.5
                * (_speed_kmh_ot / 50.0)
                * (min_dist / max(self._speed_safe_deviation, 1.0))
            )
            reward -= off_track_speed_penalty

        if next_lap and self.cur_idx > self.prev_idx:
            self.new_lap = True

        if self.cur_idx > int(len(self.data) * 0.9925) and self.cur_idx > self.prev_idx:
            self.near_finish = True

        if self.new_lap and self.lap_cur_cooldown > 0:
            reward += cfg.LAP_REWARD * self.lap_cur_cooldown / cfg.LAP_COOLDOWN
            self.lap_cur_cooldown -= 1
            print(f"lap reward added: {reward}")

        if next_cp:
            reward += cfg.CHECKPOINT_REWARD
            print(f"checkpoint reward added: {reward}")
            # self.new_checkpoint = False

        if self.near_finish and self.lap_cur_cooldown > 0:
            near_finish_bonus = (
                (cfg.LAP_COOLDOWN - self.lap_cur_cooldown)
                / cfg.LAP_COOLDOWN
                * cfg.END_OF_TRACK_REWARD
            )
            reward += near_finish_bonus
            self.lap_cur_cooldown -= 1
            print(f"finish reward added: {near_finish_bonus}")

        if self.near_finish or self.new_lap and 5 < self.cur_idx < len(self.data) * 0.1:
            self.new_lap = False
            self.near_finish = False

        if crashed:
            reward -= abs(self.crash_penalty)
            # self.crash_counter += 1

        # When the agent made progress along the trajectory, do not allow penalties
        # to push the step reward below zero (so progressing runs get positive totals).
        if distance_gained > 0:
            reward = max(0.0, reward)

        # Scale reward then soft-clip: preserves gradient contrast better than raw tanh
        reward = reward * self._reward_scale
        reward = math.tanh(reward)
        race_progress = self.compute_race_progress()

        if race_progress > self.furthest_race_progress:
            self.furthest_race_progress = race_progress

        if cfg.WANDB_DEBUG_REWARD:
            self.send_reward.append(reward)

        self.episode_reward += reward

        return reward, terminated, self.failure_counter, self.episode_reward

    def log_model_run(self, terminated, end_of_track):
        """
        Logs the run summary and reward-related parameters to Weights & Biases (wandb).
        Only logs once per episode; rtgym may call this repeatedly while done=True before reset.
        """
        if (terminated or end_of_track) and not self._logged_run_this_episode:
            self._logged_run_this_episode = True
            if end_of_track:
                self.furthest_race_progress = 1.0
            # Always print and log to wandb once per episode (including zero reward)
            print(f"Total reward of the run: {self.episode_reward}")
            if self._use_wandb:
                self._episode_count = getattr(self, "_episode_count", 0) + 1
                if (
                    cfg.WANDB_DEBUG_REWARD
                    and getattr(self, "send_reward", None)
                    and len(self.send_reward) > 0
                ):
                    send_reward_df = pd.DataFrame({"Reward": self.send_reward})
                    summary_stats = send_reward_df.describe()
                    summary_stats = summary_stats.reset_index()
                    q1_value = float(
                        summary_stats.loc[summary_stats["index"] == "25%", "Reward"].iloc[0]
                    )
                    q2_value = float(
                        summary_stats.loc[summary_stats["index"] == "50%", "Reward"].iloc[0]
                    )
                    q3_value = float(
                        summary_stats.loc[summary_stats["index"] == "75%", "Reward"].iloc[0]
                    )
                    mean_value = float(
                        summary_stats.loc[summary_stats["index"] == "mean", "Reward"].iloc[0]
                    )
                    max_value = float(
                        summary_stats.loc[summary_stats["index"] == "max", "Reward"].iloc[0]
                    )
                    min_value = float(
                        summary_stats.loc[summary_stats["index"] == "min", "Reward"].iloc[0]
                    )
                    count_value = float(
                        summary_stats.loc[summary_stats["index"] == "count", "Reward"].iloc[0]
                    )
                    std_value = float(
                        summary_stats.loc[summary_stats["index"] == "std", "Reward"].iloc[0]
                    )
                    wandb.log(
                        {
                            "run/Run reward": self.episode_reward,
                            "run/Q1": q1_value,
                            "run/Q2": q2_value,
                            "run/Q3": q3_value,
                            "run/mean": mean_value,
                            "run/max": max_value,
                            "run/min": min_value,
                            "run/count": count_value,
                            "run/std": std_value,
                            "run/best race progress": self.furthest_race_progress,
                        },
                        step=self._episode_count,
                    )
                    self.send_reward.clear()
                else:
                    wandb.log({"Run reward": self.episode_reward}, step=self._episode_count)
            if self.episode_reward != 0.0:
                self.reward_sum_list.append(self.episode_reward)
                # wandb.log({"Run reward": self.reward_sum})
                # self.change_min_nb_steps_before_failure()
                self.i = self.i + 1
                if self.oscillation_period <= 0:
                    self.nb_zero_rew_before_failure = int(self.mid_value)
                else:
                    self.nb_zero_rew_before_failure = int(
                        self.mid_value
                        + self.amplitude * np.sin(2 * np.pi * self.i / self.oscillation_period)
                    )
                self.nb_zero_rew_before_failure = max(1, int(self.nb_zero_rew_before_failure))
                # Steps with zero reward allowed before ending episode (tuned from run)
                logger.debug(
                    f"failure_countdown (next): {self.nb_zero_rew_before_failure} "
                    "(steps with 0 reward before failure)"
                )

    def compute_race_progress(self):
        """
        Computes the current race progress based on the car's position in the track.
        """
        return self.cur_idx / len(self.data)

    def calculate_ema(self, alpha: float = 0.25):
        """
        Calculates the Exponential Moving Average (EMA) of the reward sum list.
        """
        ema_values = [self.reward_sum_list[0]]
        for i in range(1, len(self.reward_sum_list)):
            ema = alpha * self.reward_sum_list[i] + (1 - alpha) * ema_values[-1]
            ema_values.append(ema)
        return ema_values

    @staticmethod
    def check_linear_coefficent(data):
        x = np.arange(len(data)).reshape(-1, 1)
        y = np.array(data).reshape(-1, 1)
        model = LinearRegression().fit(x, y)
        return model.coef_[0][0]

    def change_min_nb_steps_before_failure(self):
        """
        Checks the linear coefficient of a given data sequence.
        """
        if len(self.reward_sum_list) <= self.window_size * 2 or abs(self.episode_reward) < 1:
            return
        if self.change_cooldown <= 0:
            ema_values = self.calculate_ema()
            print(f"ema_values: {ema_values}")
            corr = self.check_linear_coefficent(ema_values[self.window_size :])
            print(f"corr: {corr}")
            if corr <= 0.05:
                if self.min_nb_steps_before_failure <= 270:
                    self.min_nb_steps_before_failure += 2
            elif corr >= 0.095:
                if self.min_nb_steps_before_failure >= 108:
                    self.min_nb_steps_before_failure -= 8
            self.change_cooldown = self.cooldown
        else:
            self.change_cooldown -= 1
        print(f"current min_nb_steps_before_failure: {self.min_nb_steps_before_failure}")

    def reset(self):
        """
        Resets the reward function for a new episode.
        Adjusts the minimum number of steps before failure based on the trend of rewards.
        """

        self.cur_idx = 0
        self.prev_idx = 0
        self.step_counter = 0
        self.failure_counter = 0
        self.low_speed_steps = 0
        self._no_progress_steps = 0
        self.episode_reward = 0.0
        # self.crash_counter = 1
        self.lap_cur_cooldown = cfg.LAP_COOLDOWN
        # self.checkpoint_cur_cooldown = cfg.CHECKPOINT_COOLDOWN

        self.furthest_race_progress = 0
        self._logged_run_this_episode = False
