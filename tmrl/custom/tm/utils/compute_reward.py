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
import logging

import pandas as pd

import tmrl.config.config_constants as cfg
from sklearn.linear_model import LinearRegression
import wandb

logging.basicConfig(level=logging.INFO)


class RewardFunction:
    """
    Computes a reward from the Openplanet API for Trackmania 2020.
    """

    def __init__(self,
                 reward_data_path,
                 nb_obs_forward=8,
                 nb_obs_backward=8,
                 nb_zero_rew_before_failure=10,
                 min_nb_steps_before_failure=int(2.5 * 20),
                 max_dist_from_traj=23.5,
                 crash_penalty=10.0,
                 constant_penalty=0.0,
                 low_threshold=10,
                 high_threshold=250):
        """
        Instantiates a reward function for TM2020.

        Args:
            reward_data_path: path where the trajectory file is stored
            nb_obs_forward: max distance of allowed cuts (as a number of positions in the trajectory)
            nb_obs_backward: same thing but for when rewinding the reward to a previously visited position
            nb_zero_rew_before_failure: after this number of steps with no reward, episode is terminated
            min_nb_steps_before_failure: the episode must have at least this number of steps before failure
            max_dist_from_traj: the reward is 0 if the car is further than this distance from the demo trajectory
        """
        if not os.path.exists(reward_data_path):
            logging.debug(f" reward not found at path:{reward_data_path}")
            self.data = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])  # dummy reward
        else:
            with open(reward_data_path, 'rb') as f:
                self.data = pickle.load(f)

        if not os.path.exists(cfg.TRACK_PATH_LEFT):
            logging.debug(f" reward not found at path:{cfg.TRACK_PATH_LEFT}")
            self.left_track = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])  # dummy reward
        else:
            with open(cfg.TRACK_PATH_LEFT, 'rb') as f:
                self.left_track = pickle.load(f)

        if not os.path.exists(cfg.TRACK_PATH_RIGHT):
            logging.debug(f" reward not found at path:{cfg.TRACK_PATH_RIGHT}")
            self.right_track = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])  # dummy reward
        else:
            with open(cfg.TRACK_PATH_RIGHT, 'rb') as f:
                self.right_track = pickle.load(f)

        self.cur_idx = 0
        self.prev_idx = 0
        self.nb_obs_forward = nb_obs_forward
        self.nb_obs_backward = nb_obs_backward
        self.nb_zero_rew_before_failure = nb_zero_rew_before_failure
        self.min_nb_steps_before_failure = min_nb_steps_before_failure
        self.max_dist_from_traj = max_dist_from_traj
        self.step_counter = 0
        self.failure_counter = 0
        self.datalen = len(self.data)
        self.speed_bonus = cfg.SPEED_BONUS
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
        self.average_distance = self.calculate_average_distance()
        self.n = max(1, min(len(self.data),
                            int(cfg.POINTS_DISTANCE / max(self.average_distance, 0.01))))  # intervals of ~10.25m
        self.i = 0
        self.min_value = cfg.MIN_NB_ZERO_REW_BEFORE_FAILURE
        self.max_value = cfg.MAX_NB_ZERO_REW_BEFORE_FAILURE
        self.mid_value = (self.max_value + self.min_value) / 2
        self.amplitude = (self.max_value - self.min_value) / 2
        self.oscillation_period = cfg.OSCILLATION_PERIOD
        self.index_divider = 100. / self.datalen
        print(f"n: {self.n}")
        self.furthest_race_progress = 0
        self.medium_speed_bonus = cfg.SPEED_BONUS / 2

        if cfg.WANDB_DEBUG_REWARD:
            self.send_reward = []

        wandb_dir = tempfile.mkdtemp()  # prevent wandb from polluting the home directory
        atexit.register(shutil.rmtree, wandb_dir, ignore_errors=True)  # clean up after wandb atexit handler finishes
        wandb_initialized = False
        err_cpt = 0
        while not wandb_initialized:
            try:
                wandb.init(
                    project=cfg.WANDB_PROJECT,
                    entity=cfg.WANDB_ENTITY,
                    id=cfg.WANDB_RUN_ID + " WORKER",
                    config=cfg.create_config(),
                    job_type="worker"
                )
                wandb_initialized = True
            except Exception as e:
                err_cpt += 1
                logging.warning(f"wandb error {err_cpt}: {e}")
                if err_cpt > 10:
                    logging.warning(f"Could not connect to wandb, aborting.")
                    exit()
                else:
                    time.sleep(10.0)

        self.i = 0

    def get_n_next_checkpoints_xy(self, pos, number_of_next_points: int):
        '''
        Retrieves the positions of next checkpoints based on the current position.
        '''
        next_indices = [self.cur_idx + i * self.n for i in range(1, number_of_next_points + 1)]
        for i in range(len(next_indices)):
            if next_indices[i] >= len(self.data):
                next_indices[i] = len(self.data) - 1
        route_to_next_poses = []
        for pos_index in next_indices:
            for i in (0, -1):
                route_to_next_poses.append((self.data[pos_index][i] - pos[i]) * 10.)

        return route_to_next_poses

    def get_track_info(self, pos, points_number):
        '''
        Fetches information about the track (left, center, right positions) for the next checkpoints.
        '''
        next_indices = [self.cur_idx + i * self.n + 1 for i in range(points_number)]
        left_track_positions, center_track_positions, right_track_positions = [], [], []
        for i in range(len(next_indices)):
            if next_indices[i] >= len(self.data):
                next_indices[i] = len(self.data) - 1

        for pos_index in next_indices:
            for i in (0, -1):
                left = self.left_track[pos_index][i]
                right = self.right_track[pos_index][i]

                center = (left + right) / 2.

                left_track_positions.append(left - pos[i])
                center_track_positions.append(center - pos[i])
                right_track_positions.append(right - pos[i])

                # left_track_positions.append((self.data[pos_index][i] - pos[i]))
                # left_track_positions.append((self.data[pos_index][i] - pos[i]))
                # left_track_positions.append((self.data[pos_index][i] - pos[i]))

        return left_track_positions, center_track_positions, right_track_positions

    def calculate_average_distance(self):
        '''
        Computes the average distance between consecutive points in the track data.
        '''
        # Calculate the Euclidean distance between consecutive points in the trajectory
        distances = np.linalg.norm(np.diff(self.data, axis=0), axis=1)

        # Compute the average distance
        average_distance = np.mean(distances)

        return average_distance

    def compute_reward(self, pos, crashed: bool = False, speed: float = None,
                       next_cp: bool = False, next_lap: bool = False, end_of_tack: bool = False):
        '''
        Calculates the reward based on the car's position, speed, and track progress.
        Handles penalties for crashes, speed bonuses, lap completions, etc.
        '''
        terminated = False
        self.step_counter += 1
        self.prev_idx = self.cur_idx
        min_dist = np.inf
        index = self.cur_idx
        temp = self.nb_obs_forward
        best_index = 0
        while True:
            dist = np.linalg.norm(pos - self.data[index])
            if dist <= min_dist:
                min_dist = dist
                best_index = index
                temp = self.nb_obs_forward
            index += 1
            temp -= 1
            # stop condition
            if index >= self.datalen or temp <= 0:  # if trajectory complete or cuts counter depleted
                # We check that we are not too far from the demo trajectory:
                if min_dist > self.max_dist_from_traj:
                    best_index = self.cur_idx  # if so, consider we didn't move

                break  # we found the best index and can break the while loop

        # The reward is then proportional to the number of passed indexes (i.e., track distance):
        reward = (best_index - self.cur_idx) / 100.0

        if best_index == self.cur_idx:  # if the best index didn't change, we rewind (more Markovian reward)
            min_dist = np.inf
            index = self.cur_idx
            while True:
                dist = np.linalg.norm(pos - self.data[index])
                if dist <= min_dist:
                    min_dist = dist
                    best_index = index
                    temp = self.nb_obs_backward
                index -= 1
                temp -= 1
                # stop condition
                if index <= 0 or temp <= 0:
                    break
            if self.step_counter > self.min_nb_steps_before_failure:
                self.failure_counter += 1
                if self.failure_counter > self.nb_zero_rew_before_failure:
                    terminated = True

        else:
            self.failure_counter = 0
        self.cur_idx = best_index

        if self.episode_reward != 0.0:
            reward -= abs(self.constant_penalty)
            if speed > cfg.SPEED_MIN_THRESHOLD:
                speed_reward = (speed - cfg.SPEED_MIN_THRESHOLD) * self.speed_bonus  # x / 250 * 0.04 = 0.00016 * x
                reward += speed_reward
                # print(f"speed_reward: {speed_reward}")

            elif speed > cfg.SPEED_MEDIUM_THRESHOLD:
                speed_reward = (speed - cfg.SPEED_MEDIUM_THRESHOLD * 0.75) * self.speed_bonus * 2
                reward += speed_reward
            elif speed < -0.5:
                penalty = 1 / (1 + np.exp(-0.1 * speed - 3)) - 1
                reward += penalty

        if min_dist > self.max_dist_from_traj:
            terminated = True

        # deviation_penalty
        if min_dist > 17.5:
            # print(f"before: {reward}")
            reward -= abs((2 / (1 + np.exp(-0.025 * (min_dist - 12.)))) - 1)

            # print(f"after: {reward}")

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
            near_finish_bonus = (cfg.LAP_COOLDOWN - self.lap_cur_cooldown) / cfg.LAP_COOLDOWN * cfg.END_OF_TRACK_REWARD
            reward += near_finish_bonus
            self.lap_cur_cooldown -= 1
            print(f"finish reward added: {near_finish_bonus}")

        if self.near_finish or self.new_lap and 5 < self.cur_idx < len(self.data) * 0.1:
            self.new_lap = False
            self.near_finish = False

        if speed > cfg.SPEED_MIN_THRESHOLD:
            speed_reward = (speed - cfg.SPEED_MIN_THRESHOLD) * self.speed_bonus  # x / 250 * 0.04 = 0.00016 * x
            reward += speed_reward
        elif speed < -0.5:

            penalty = 1 / (1 + np.exp(-0.1 * speed - 3)) - 1
            reward += penalty
        elif speed > 1.0:
            speed_reward = (speed / 250) * 0.04
            reward += speed_reward

        if crashed:
            reward -= abs(self.crash_penalty)
            # self.crash_counter += 1

        # clipping reward (maps values above 6 and below -6 to 1 and -1)
        # reward = math.tanh(6 / (1 + np.exp(-0.7 * reward)) - 3)
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
         Logs the summary of the run, updating reward-related parameters and logging information to Weights & Biases (wandb).
        """
        if terminated or end_of_track:
            if end_of_track:
                self.furthest_race_progress = 1.0
            print(f"Total reward of the run: {self.episode_reward}")
            if self.episode_reward != 0.0:
                self.reward_sum_list.append(self.episode_reward)
                # wandb.log({"Run reward": self.reward_sum})
                # self.change_min_nb_steps_before_failure()
                self.i = self.i + 1
                self.nb_zero_rew_before_failure = int(
                    self.mid_value + self.amplitude * np.sin(2 * np.pi * self.i / self.oscillation_period))
                print(f"nb_zero_rew_before_failure: {self.nb_zero_rew_before_failure}")
                if cfg.WANDB_DEBUG_REWARD:
                    send_reward_df = pd.DataFrame({"Reward": self.send_reward})
                    summary_stats = send_reward_df.describe()
                    summary_stats = summary_stats.reset_index()

                    q1_value = float(summary_stats.loc[summary_stats['index'] == '25%', 'Reward'].iloc[0])
                    q2_value = float(summary_stats.loc[summary_stats['index'] == '50%', 'Reward'].iloc[0])
                    q3_value = float(summary_stats.loc[summary_stats['index'] == '75%', 'Reward'].iloc[0])
                    mean_value = float(summary_stats.loc[summary_stats['index'] == 'mean', 'Reward'].iloc[0])
                    max_value = float(summary_stats.loc[summary_stats['index'] == 'max', 'Reward'].iloc[0])
                    min_value = float(summary_stats.loc[summary_stats['index'] == 'min', 'Reward'].iloc[0])
                    count_value = float(summary_stats.loc[summary_stats['index'] == 'count', 'Reward'].iloc[0])
                    std_value = float(summary_stats.loc[summary_stats['index'] == 'std', 'Reward'].iloc[0])

                    wandb.log(
                        {
                            "run/Run reward": self.episode_reward,
                            "run/Q1": q1_value, "run/Q2": q2_value, "run/Q3": q3_value, "run/mean": mean_value,
                            "run/max": max_value, "run/min": min_value,
                            "run/count": count_value, "run/std": std_value,
                            "run/best race progress": self.furthest_race_progress
                        }
                    )
                    self.send_reward.clear()
                else:
                    wandb.log({"Run reward": self.episode_reward})

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
        '''
        Checks the linear coefficient of a given data sequence.
        '''
        if len(self.reward_sum_list) <= self.window_size * 2 or abs(self.episode_reward) < 1:
            return
        if self.change_cooldown <= 0:
            ema_values = self.calculate_ema()
            print(f"ema_values: {ema_values}")
            corr = self.check_linear_coefficent(ema_values[self.window_size:])
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
        self.episode_reward = 0.0
        # self.crash_counter = 1
        self.lap_cur_cooldown = cfg.LAP_COOLDOWN
        # self.checkpoint_cur_cooldown = cfg.CHECKPOINT_COOLDOWN

        self.furthest_race_progress = 0