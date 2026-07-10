"""RolloutWorker: deploys the current policy in the environment."""

import datetime
import itertools
import math
import os
import time
from collections.abc import Callable
from typing import Any

import gymnasium
import numpy as np
from loguru import logger
from tlspyo import Endpoint

import tmrl.config as cfg
import tmrl.config.config_objects as cfg_obj
import tmrl.config.constants as cfg_c
from tmrl.networking.buffer import Buffer
from tmrl.networking.utils import (
    _maybe_log_reward_on_rollout_truncation,
    _parse_worker_send_chunk_size,
    print_with_timestamp,
)


class RolloutWorker:
    """Actor.

    A `RolloutWorker` deploys the current policy in the environment.
    A `RolloutWorker` may connect to a `Server` to which it sends buffered experience.
    Alternatively, it may exist in standalone mode for deployment.
    """

    def __init__(
        self,
        env_cls,
        actor_module_cls,
        sample_compressor: Callable[..., Any] | None = None,
        device="cpu",
        max_samples_per_episode=np.inf,
        model_path=cfg.MODEL_PATH_WORKER,
        obs_preprocessor: Callable[..., Any] | None = None,
        crc_debug=False,
        model_path_history=cfg.MODEL_PATH_SAVE_HISTORY,
        model_history=cfg.MODEL_HISTORY,
        standalone=False,
        server_ip=None,
        server_port=cfg.PORT,
        password=cfg.PASSWORD,
        local_port=cfg.LOCAL_PORT_WORKER,
        header_size=cfg.HEADER_SIZE,
        max_buf_len=cfg.BUFFER_SIZE,
        security=cfg.SECURITY,
        keys_dir=cfg.CREDENTIALS_DIRECTORY,
        hostname=cfg.HOSTNAME,
    ):
        """
        Args:
            env_cls (type): class of the Gymnasium environment (subclass of tmrl.envs.GenericGymEnv)
            actor_module_cls (type): module class for the policy (tmrl.actor.ActorModule subclass)
            sample_compressor (callable): compressor for samples over the Internet; when not `None`,
                must take (prev_act, obs, rew, terminated, truncated, info) and return same order;
                works with a decompression scheme in the Memory class.
            device (str): device on which the policy is running
            max_samples_per_episode (int): if an episode gets longer than this, it is reset
            model_path (str): path where a local copy of the policy will be stored
            obs_preprocessor (callable): if not None, (obs) -> modified observation
            crc_debug (bool): useful for debugging custom pipelines; leave to False otherwise
            model_path_history (str): path to policy history (omit .tmod); leave default recommended
            model_history (int): save policy every this many new policies (0: not saved)
            standalone (bool): if True, the worker will not try to connect to a server
            server_ip (str): ip of the central server
            server_port (int): public port of the central server
            password (str): tlspyo password
            local_port (int): tlspyo local communication port; usually, leave this to the default
            header_size (int): tlspyo header size (bytes)
            max_buf_len (int): tlspyo max number of messages in buffer
            security (str): tlspyo security type (None or "TLS")
            keys_dir (str): tlspyo credentials directory; usually, leave this to the default
            hostname (str): tlspyo hostname; usually, leave this to the default
        """
        self.obs_preprocessor = obs_preprocessor
        self.get_local_buffer_sample = sample_compressor
        self.env = env_cls()
        obs_space = self.env.observation_space
        _obs_dim = (
            sum(math.prod(s.shape or ()) for s in obs_space.spaces)
            if isinstance(obs_space, gymnasium.spaces.Tuple)
            else math.prod(obs_space.shape or ())
        )
        logger.info(
            " Worker env: interface={}, observation_space total_dim={}, "
            "POINTS_NUMBER={}, USE_RNN_MODEL={}",
            cfg_obj.INTERFACE_DISPLAY_NAME,
            _obs_dim,
            cfg_c.POINTS_NUMBER,
            cfg_c.USE_RNN_MODEL,
        )
        act_space = self.env.action_space
        self.model_path = model_path
        self.model_path_history = model_path_history
        self.device = device
        self.actor = actor_module_cls(observation_space=obs_space, action_space=act_space).to(
            self.device
        )
        self.standalone = standalone
        if os.path.isfile(self.model_path):
            logger.debug(f"Loading model from {self.model_path}")
            self.actor = self.actor.load(self.model_path, device=self.device)
        else:
            logger.debug(f"No model found at {self.model_path}")
        self.buffer = Buffer()
        self.max_samples_per_episode = max_samples_per_episode
        self.crc_debug = crc_debug
        self.model_history = model_history
        self._cur_hist_cpt = 0
        self.model_cpt = 0

        self.debug_ts_cpt = 0
        self.debug_ts_res_cpt = 0

        self.sde_sample_freq = int(cfg.SDE_SAMPLE_FREQ)
        self.sde_step_counter = 0

        self.start_time = time.time()
        self.server_ip = server_ip if server_ip is not None else "127.0.0.1"
        self._worker_send_chunk_size = _parse_worker_send_chunk_size(
            os.environ.get("TMRL_WORKER_SEND_CHUNK_SIZE")
        )

        print_with_timestamp(f"server IP: {self.server_ip}")

        if not self.standalone:
            self.__endpoint = Endpoint(
                ip_server=self.server_ip,
                port=server_port,
                password=password,
                groups="workers",
                local_com_port=local_port,
                header_size=header_size,
                max_buf_len=max_buf_len,
                security=security,
                keys_dir=keys_dir,
                hostname=hostname,
                deserializer_mode="synchronous",
            )
        else:
            self.__endpoint = None

    def act(self, obs, test=False):
        """
        Select an action based on observation `obs`

        Args:
            obs (nested structure): observation
            test (bool): directly passed to the `act()` method of the `ActorModule`

        Returns:
            numpy.array: action computed by the `ActorModule`
        """
        action = self.actor.act_(obs, test=test)
        return action

    def reset(self, collect_samples):
        """
        Starts a new episode.

        Args:
            collect_samples (bool): if True, samples are buffered and sent to the `Server`

        Returns:
            Tuple:
            (nested structure: observation retrieved from the environment,
            dict: information retrieved from the environment)
        """
        obs = None
        try:
            act = self.env.unwrapped.default_action
        except AttributeError:
            act = None

        if hasattr(self.actor, "reset_noise"):
            self.actor.reset_noise(1)
        if hasattr(self.actor, "reset_explore_state"):
            self.actor.reset_explore_state()
        self.sde_step_counter = 0

        new_obs, info = self.env.reset()
        if self.obs_preprocessor is not None:
            new_obs = self.obs_preprocessor(new_obs)
        rew = 0.0
        terminated, truncated = False, False
        if collect_samples:
            if self.crc_debug:
                self.debug_ts_cpt += 1
                self.debug_ts_res_cpt = 0
                info["crc_sample"] = (obs, act, new_obs, rew, terminated, truncated)
                info["crc_sample_ts"] = (self.debug_ts_cpt, self.debug_ts_res_cpt)
            if self.get_local_buffer_sample:
                sample = self.get_local_buffer_sample(
                    act, new_obs, rew, terminated, truncated, info
                )
            else:
                sample = act, new_obs, rew, terminated, truncated, info
            self.buffer.append_sample(sample)
        return new_obs, info

    def step(self, obs, test, collect_samples, last_step=False):
        """
        Performs a full RL transition.

        A full RL transition is obs -> act -> (new_obs, rew, terminated, truncated, info).
        In Real-Time RL, act is appended to a buffer that is part of new_obs (real-time delays).

        Args:
            obs (nested structure): previous observation
            test (bool): passed to the `act()` method of the `ActorModule`
            collect_samples (bool): if True, samples are buffered and sent to the `Server`
            last_step (bool): if True and `terminated` is False, `truncated` will be set to True

        Returns:
            Tuple:
            (nested structure: new observation,
            float: new reward,
            bool: episode termination signal,
            bool: episode truncation signal,
            dict: information dictionary)
        """
        self.sde_step_counter += 1
        if (
            getattr(self.actor, "sde", None) is not None
            and self.sde_step_counter % self.sde_sample_freq == 0
        ):
            self.actor.reset_noise(1)

        act = self.act(obs, test=test)
        new_obs, rew, terminated, truncated, info = self.env.step(act)
        if isinstance(info, dict) and info.get("crashed", False):
            penalty = float(info.get("crash_penalty", cfg.REWARD_CONFIG.get("crash_penalty", 0.5)))
            logger.info("Car crashed (penalty -{} already applied by the reward function)", penalty)

        if self.obs_preprocessor is not None:
            new_obs = self.obs_preprocessor(new_obs)
        if collect_samples:
            if last_step and not terminated:
                truncated = True
                if isinstance(info, dict):
                    # Interfaces that implement RewardFunction logging (boundary lidar / vision / …)
                    # so worker-side wandb metrics flush on forced episode truncation.
                    info["env_truncated"] = True
            if self.crc_debug:
                self.debug_ts_cpt += 1
                self.debug_ts_res_cpt += 1
                info["crc_sample"] = (obs, act, new_obs, rew, terminated, truncated)
                info["crc_sample_ts"] = (self.debug_ts_cpt, self.debug_ts_res_cpt)
            if self.get_local_buffer_sample:
                sample = self.get_local_buffer_sample(
                    act, new_obs, rew, terminated, truncated, info
                )
            else:
                sample = act, new_obs, rew, terminated, truncated, info
            self.buffer.append_sample(sample)
        if collect_samples and truncated:
            _maybe_log_reward_on_rollout_truncation(self.env, info)
        return new_obs, rew, terminated, truncated, info

    def collect_train_episode(self, max_samples=None):
        """
        Collects up to `max_samples` training transitions (reset to terminated or truncated).

        Stores the episode and training return in the worker's local Buffer.
        for sending to the `Server`.

        Args:
            max_samples (int): if not terminated after this many steps, reset and set truncated.
        """
        if max_samples is None:
            max_samples = self.max_samples_per_episode

        iterator = range(max_samples) if max_samples != np.inf else itertools.count()

        ret = 0.0
        steps = 0
        obs, _ = self.reset(collect_samples=True)
        for i in iterator:
            obs, rew, terminated, truncated, _ = self.step(
                obs=obs, test=False, collect_samples=True, last_step=i == max_samples - 1
            )
            ret += rew
            steps += 1
            if terminated or truncated:
                break
        self.buffer.stat_train_return = ret
        self.buffer.stat_train_steps = steps
        self._maybe_apply_finish_time_bonus()

    def _maybe_apply_finish_time_bonus(self):
        """Spread the finish-time speed bonus over the buffered episode when it
        ended on the finish line (no-op unless reward.time_bonus_scale > 0)."""
        if self.buffer.memory and self.buffer.stat_train_steps > 0:
            last_info = self.buffer.memory[-1][5]
            if isinstance(last_info, dict) and last_info.get("end_of_track", False):
                time_bonus_scale = float(cfg.REWARD_CONFIG.get("time_bonus_scale", 0.0))
                reward_scale = float(cfg.REWARD_CONFIG.get("reward_scale", 1.0))
                if time_bonus_scale > 0 and reward_scale > 0:
                    self.buffer.apply_speed_bonus(time_bonus_scale * reward_scale)

    def run_episodes(self, max_samples_per_episode=None, nb_episodes=np.inf, train=False):
        """
        Runs `nb_episodes` episodes.

        Args:
            max_samples_per_episode (int): same as run_episode
            nb_episodes (int): total number of episodes to collect
            train (bool): same as run_episode
        """
        if max_samples_per_episode is None:
            max_samples_per_episode = self.max_samples_per_episode

        iterator = range(nb_episodes) if nb_episodes != np.inf else itertools.count()

        for _ in iterator:
            self.run_episode(max_samples_per_episode, train=train)

    def _run_deterministic_test_episodes(self, max_samples, n_episodes):
        """Run deterministic eval: competition-style (TMRL test rules) or legacy mean over runs."""
        if cfg_c.BEST_CHECKPOINT_LAP_TIME:
            self._run_competition_style_test_episodes(max_samples, n_episodes)
        else:
            self._run_legacy_deterministic_test_episodes(max_samples, n_episodes)

    def _run_legacy_deterministic_test_episodes(self, max_samples, n_episodes):
        """Mean return/steps; mean finish time over episodes that reached end of track."""
        returns = []
        steps_list = []
        finish_times = []
        for _ in range(n_episodes):
            self.run_episode(max_samples, train=False)
            returns.append(self.buffer.stat_test_return)
            steps_list.append(self.buffer.stat_test_steps)
            if getattr(self.buffer, "stat_test_finished_track", False):
                finish_times.append(self.buffer.stat_test_finish_time)
        self.buffer.stat_test_return = float(np.mean(returns))
        self.buffer.stat_test_steps = float(np.mean(steps_list))
        self.buffer.stat_test_finish_time = float(np.mean(finish_times)) if finish_times else 0.0
        self.buffer.stat_test_finished_count = len(finish_times)
        self.buffer.stat_test_competition_eliminated = False
        self.buffer.stat_test_competition_crashes = n_episodes - len(finish_times)
        if finish_times:
            logger.info(
                "Test runs (epsilon=0) finish times (s): {}  (finished {}/{} episodes)",
                [round(t, 2) for t in finish_times],
                len(finish_times),
                n_episodes,
            )

    def _run_competition_style_test_episodes(self, max_samples, n_episodes):
        """Eval aligned with TMRL competition.

        N attempts, crash discards run, +penalty to next finish.

        After ``COMPETITION_EVAL_MAX_CRASHES`` crashes the eval is eliminated (no valid mean time).
        Mean time is over successful runs only (crashed attempts are excluded from the mean).
        """
        penalty_s = float(cfg_c.COMPETITION_EVAL_CRASH_PENALTY_S)
        max_crashes = int(cfg_c.COMPETITION_EVAL_MAX_CRASHES)
        penalty_carry = 0.0
        crashes = 0
        eliminated = False
        returns: list[float] = []
        steps_list: list[float] = []
        finish_times_adjusted: list[float] = []
        raw_finish_times: list[float] = []

        for _attempt in range(n_episodes):
            if eliminated:
                break
            self.run_episode(max_samples, train=False)
            returns.append(self.buffer.stat_test_return)
            steps_list.append(self.buffer.stat_test_steps)
            finished = bool(getattr(self.buffer, "stat_test_finished_track", False))
            raw_t = float(getattr(self.buffer, "stat_test_finish_time", 0.0))
            if finished and raw_t > 0.0:
                adj = raw_t + penalty_carry
                finish_times_adjusted.append(adj)
                raw_finish_times.append(raw_t)
                penalty_carry = 0.0
            else:
                crashes += 1
                penalty_carry += penalty_s
                if crashes >= max_crashes:
                    eliminated = True
                    break

        self.buffer.stat_test_competition_eliminated = eliminated
        self.buffer.stat_test_competition_crashes = crashes
        self.buffer.stat_test_return = float(np.mean(returns)) if returns else 0.0
        self.buffer.stat_test_steps = float(np.mean(steps_list)) if steps_list else 0.0
        if eliminated or not finish_times_adjusted:
            self.buffer.stat_test_finish_time = 0.0
            self.buffer.stat_test_finished_count = 0
            self.buffer.stat_test_finished_track = False
            logger.info(
                "Competition eval: eliminated={} crashes={}/{}  (no mean lap time)",
                eliminated,
                crashes,
                max_crashes,
            )
        else:
            mean_adj = float(np.mean(finish_times_adjusted))
            self.buffer.stat_test_finish_time = mean_adj
            self.buffer.stat_test_finished_count = len(finish_times_adjusted)
            self.buffer.stat_test_finished_track = len(finish_times_adjusted) > 0
            logger.info(
                "Competition eval: mean lap (penalty-adjusted)={:.2f}s from {}/{} finishes, "
                "crashes={}, raw times (s): {}",
                mean_adj,
                len(finish_times_adjusted),
                n_episodes,
                crashes,
                [round(t, 2) for t in raw_finish_times],
            )

    def run_episode(self, max_samples=None, train=False):
        """
        Collects up to `max_samples` test transitions (reset to terminated or truncated).

        Args:
            max_samples (int): at most this many samples per episode.
                If the episode is longer, it is forcefully reset and `truncated` is set to True.
            train (bool): whether the episode is a training or a test episode.
                `step` is called with `test=not train`.
        """
        if max_samples is None:
            max_samples = self.max_samples_per_episode

        iterator = range(max_samples) if max_samples != np.inf else itertools.count()

        ret = 0.0
        steps = 0
        saw_end_of_track = False
        obs, info = self.reset(collect_samples=False)
        for _ in iterator:
            obs, rew, terminated, truncated, info = self.step(
                obs=obs, test=not train, collect_samples=False
            )
            ret += rew
            steps += 1
            if isinstance(info, dict) and info.get("end_of_track"):
                saw_end_of_track = True
            if terminated or truncated:
                break
        self.buffer.stat_test_return = ret
        self.buffer.stat_test_steps = float(steps)
        if not train:
            dt = float(cfg.RTGYM_TIME_STEP_DURATION)
            end_of_track = saw_end_of_track or (
                bool(info.get("end_of_track", False)) if isinstance(info, dict) else False
            )
            self.buffer.stat_test_finish_time = (steps * dt) if end_of_track else 0.0
            self.buffer.stat_test_finished_track = end_of_track

    def run(self, test_episode_interval=0, nb_episodes=np.inf, verbose=True, expert=False):
        """
        Runs the worker for `nb_episodes` episodes.

        Sends episodes to the Server and checks for new weights between episodes.
        For sync/fine-grained sampling use other APIs; for deployment use run_episodes.

        Args:
            test_episode_interval (int): test episode every N train episodes; 0 to disable.
            nb_episodes (int): max train episodes to collect.
            verbose (bool): whether to log INFO messages.
            expert (bool): if True, send samples only, no model updates nor test episodes.
        """

        iterator = range(nb_episodes) if nb_episodes != np.inf else itertools.count()

        if expert:
            if not verbose:
                for _ in iterator:
                    self.collect_train_episode(self.max_samples_per_episode)
                    self.send_and_clear_buffer()
                    self.ignore_actor_weights()
            else:
                for _ in iterator:
                    print_with_timestamp("collecting expert episode")
                    self.collect_train_episode(self.max_samples_per_episode)
                    print_with_timestamp("copying buffer for sending")
                    self.send_and_clear_buffer()
                    self.ignore_actor_weights()
        elif not verbose:
            if not test_episode_interval:
                for _ in iterator:
                    self.collect_train_episode(self.max_samples_per_episode)
                    self.send_and_clear_buffer()
                    self.update_actor_weights(verbose=False)
            else:
                n_test_per_eval = getattr(cfg, "RW_TEST_EPISODES_PER_EVAL", 5)
                for episode in iterator:
                    if episode % test_episode_interval == 0 and not self.crc_debug:
                        self._run_deterministic_test_episodes(
                            self.max_samples_per_episode, n_test_per_eval
                        )
                    self.collect_train_episode(self.max_samples_per_episode)
                    self.send_and_clear_buffer()
                    self.update_actor_weights(verbose=False)
        else:
            n_test_per_eval = getattr(cfg, "RW_TEST_EPISODES_PER_EVAL", 5)
            for episode in iterator:
                if (
                    test_episode_interval
                    and episode % test_episode_interval == 0
                    and not self.crc_debug
                ):
                    print_with_timestamp(f"running {n_test_per_eval} deterministic test episode(s)")
                    self._run_deterministic_test_episodes(
                        self.max_samples_per_episode, n_test_per_eval
                    )
                print_with_timestamp("collecting train episode")
                self.collect_train_episode(self.max_samples_per_episode)
                print_with_timestamp("copying buffer for sending")
                self.send_and_clear_buffer()
                print_with_timestamp("checking for new weights")
                self.update_actor_weights(verbose=True)

    def run_synchronous(
        self,
        test_episode_interval=0,
        nb_steps=np.inf,
        initial_steps=1,
        max_steps_per_update=np.inf,
        end_episodes=True,
        verbose=False,
    ):
        """
        Collects `nb_steps` steps while synchronizing with the Trainer.

        For traditional (non-real-time) envs that can be stepped fast.
        For rtgym with wait_on_done, set end_episodes to True.

        Note: Test episode collection requires ``end_episodes=True``; set
            ``test_episode_interval=0`` to disable it.

        Args:
            test_episode_interval (int): run deterministic test episodes every N training
                episodes; 0 to disable. Requires ``end_episodes=True``.
            nb_steps (int): total steps to collect (after initial_steps).
            initial_steps (int): steps before waiting for first model update.
            max_steps_per_update (float): max steps per model from Server (can be non-integer).
            end_episodes (bool): if True, wait for episode end before send/wait; else pause.
            verbose (bool): whether to log INFO messages.
        """

        if verbose:
            logger.info(f"Collecting {initial_steps} initial steps")

        iteration = 0
        done = False
        while iteration < initial_steps:
            steps = 0
            ret = 0.0
            obs, _ = self.reset(collect_samples=True)
            done = False
            iteration += 1
            while not done and (end_episodes or iteration < initial_steps):
                obs, rew, terminated, truncated, _ = self.step(
                    obs=obs,
                    test=False,
                    collect_samples=True,
                    last_step=steps == self.max_samples_per_episode - 1,
                )
                iteration += 1
                steps += 1
                ret += rew
                done = terminated or truncated
            self.buffer.stat_train_return = ret
            self.buffer.stat_train_steps = steps
            if verbose:
                logger.info("Sending buffer (initial steps)")
            self.send_and_clear_buffer()

        i_model = 1

        ratio = (iteration + 1) / i_model
        while ratio > max_steps_per_update:
            if verbose:
                logger.info(
                    f"Ratio {ratio} > {max_steps_per_update}, sending buffer checking updates"
                )
            self.send_and_clear_buffer()
            i_model += self.update_actor_weights(verbose=verbose, blocking=True)
            ratio = (iteration + 1) / i_model

        iteration = 0
        episode = 0
        steps = 0
        ret = 0.0

        while iteration < nb_steps:
            if done:
                if (
                    test_episode_interval > 0
                    and episode % test_episode_interval == 0
                    and end_episodes
                ):
                    n_test_per_eval = getattr(cfg, "RW_TEST_EPISODES_PER_EVAL", 5)
                    if verbose:
                        print_with_timestamp(
                            f"running {n_test_per_eval} deterministic test episode(s)"
                        )
                    self._run_deterministic_test_episodes(
                        self.max_samples_per_episode, n_test_per_eval
                    )
                obs, _ = self.reset(collect_samples=True)
                done = False
                iteration += 1
                steps = 0
                ret = 0.0
                episode += 1

            while not done and (end_episodes or ratio <= max_steps_per_update):
                obs, rew, terminated, truncated, _ = self.step(
                    obs=obs,
                    test=False,
                    collect_samples=True,
                    last_step=steps == self.max_samples_per_episode - 1,
                )
                iteration += 1
                steps += 1
                ret += rew

                done = terminated or truncated

                if not end_episodes:
                    ratio = (iteration + 1) / i_model
                    while ratio > max_steps_per_update:
                        if verbose:
                            logger.info(
                                f"Ratio {ratio} > {max_steps_per_update}, sending buffer (no eoe)"
                            )
                        if not done:
                            if verbose:
                                logger.info("Sending buffer (no eoe)")
                            self.send_and_clear_buffer()
                        i_model += self.update_actor_weights(verbose=verbose, blocking=True)
                        ratio = (iteration + 1) / i_model

            if end_episodes:
                ratio = (iteration + 1) / i_model
                while ratio > max_steps_per_update:
                    if verbose:
                        logger.info(f"Ratio {ratio} > {max_steps_per_update}, sending buffer (eoe)")
                    if not done:
                        if verbose:
                            logger.info("Sending buffer (eoe)")
                        self.send_and_clear_buffer()
                    i_model += self.update_actor_weights(verbose=verbose, blocking=True)
                    ratio = (iteration + 1) / i_model

            self.buffer.stat_train_return = ret
            self.buffer.stat_train_steps = steps
            if done and end_episodes:
                self._maybe_apply_finish_time_bonus()
            if verbose:
                logger.info(
                    f"Sending buffer - DEBUG ratio {ratio} iteration {iteration} i_model {i_model}"
                )
            self.send_and_clear_buffer()

    def run_env_benchmark(self, nb_steps, test=False, verbose=True):
        """
        Benchmarks the environment.

        This method is only compatible with rtgym_ environments.
        The rtgym config must have the "benchmark" option set to True.

        .. _rtgym: https://github.com/yannbouteiller/rtgym

        Args:
            nb_steps (int): number of steps to perform to compute the benchmark
            test (bool): whether the actor is called in test or train mode
            verbose (bool): whether to log INFO messages
        """
        if nb_steps == np.inf or nb_steps < 0:
            raise RuntimeError(f"Invalid number of steps: {nb_steps}")

        obs, _ = self.reset(collect_samples=False)
        for _ in range(nb_steps):
            obs, _rew, terminated, truncated, _ = self.step(
                obs=obs, test=test, collect_samples=False
            )
            if terminated or truncated:
                break
        res = self.env.unwrapped.benchmarks()
        if verbose:
            print_with_timestamp(f"Benchmark results:\n{res}")
        return res

    def send_and_clear_buffer(self):
        """Snapshot the local buffer, clear it, then send the payload to the Server.

        The buffer is snapshotted and cleared atomically under the buffer lock before
        any network I/O, to avoid a race where the serializer sees a cleared Buffer and
        the trainer receives 0 samples.

        If the payload exceeds ``_worker_send_chunk_size`` samples, it is split into
        multiple ``produce`` calls. Episode-level statistics (return, steps, finish
        time, etc.) are attached only to the last chunk so they are not double-counted
        by the trainer.

        Empty-memory payloads (e.g. containing only deterministic-test stats) are still
        sent so that metric state propagates to the trainer.

        No-op if the worker is in standalone mode (no endpoint).
        """
        if self.__endpoint is None:
            return
        # Snapshot first, then clear local buffer to avoid races where async serializer
        # sees a cleared Buffer object and trainer receives 0 samples.
        payload = Buffer(maxlen=self.buffer.maxlen)
        with self.buffer._guarded():
            payload.memory = self.buffer.memory
            payload.stat_train_return = self.buffer.stat_train_return
            payload.stat_test_return = self.buffer.stat_test_return
            payload.stat_train_steps = self.buffer.stat_train_steps
            payload.stat_test_steps = self.buffer.stat_test_steps
            payload.stat_test_finish_time = getattr(self.buffer, "stat_test_finish_time", 0.0)
            payload.stat_test_finished_track = getattr(
                self.buffer, "stat_test_finished_track", False
            )
            payload.stat_test_finished_count = getattr(self.buffer, "stat_test_finished_count", 0)
            payload.stat_test_competition_eliminated = getattr(
                self.buffer, "stat_test_competition_eliminated", False
            )
            payload.stat_test_competition_crashes = getattr(
                self.buffer, "stat_test_competition_crashes", 0
            )
            self.buffer.clear()

        t_send_start = time.perf_counter()
        n_samples = len(payload.memory)
        if n_samples == 0:
            # Keep metric-only messages (e.g. deterministic test stats) behavior.
            self.__endpoint.produce(payload, "trainers")
            return

        chunk_size = self._worker_send_chunk_size
        if n_samples <= chunk_size:
            self.__endpoint.produce(payload, "trainers")
        else:
            for start in range(0, n_samples, chunk_size):
                end = min(start + chunk_size, n_samples)
                chunk = Buffer(maxlen=self.buffer.maxlen)
                chunk.memory = payload.memory[start:end]
                # Keep episode-level stats only on the last chunk.
                if end == n_samples:
                    chunk.stat_train_return = payload.stat_train_return
                    chunk.stat_test_return = payload.stat_test_return
                    chunk.stat_train_steps = payload.stat_train_steps
                    chunk.stat_test_steps = payload.stat_test_steps
                    chunk.stat_test_finish_time = payload.stat_test_finish_time
                    chunk.stat_test_finished_track = payload.stat_test_finished_track
                    chunk.stat_test_finished_count = payload.stat_test_finished_count
                    chunk.stat_test_competition_eliminated = (
                        payload.stat_test_competition_eliminated
                    )
                    chunk.stat_test_competition_crashes = payload.stat_test_competition_crashes
                self.__endpoint.produce(chunk, "trainers")

        elapsed = time.perf_counter() - t_send_start
        logger.info(
            " Sent {} sample(s) to server in {:.3f}s (chunk_size={})",
            n_samples,
            elapsed,
            chunk_size,
        )

    def update_actor_weights(self, verbose=True, blocking=False):
        """
        Updates the actor with new weights received from the `Server` when available.

        Args:
            verbose (bool): whether to log INFO messages.
            blocking (bool): if True, blocks until a model is received; otherwise, can be a no-op.

        Returns:
            int: number of new actor models received from the Server (the latest is used).
        """
        if self.__endpoint is None:
            return 0
        weights_list = self.__endpoint.receive_all(blocking=blocking)
        nb_received = len(weights_list)
        if nb_received > 0:
            weights = weights_list[-1]
            if self.model_history:
                self._cur_hist_cpt += 1
                if self._cur_hist_cpt == self.model_history:
                    x = datetime.datetime.now()
                    with open(
                        self.model_path_history + str(x.strftime("%d_%m_%Y_%H_%M_%S")) + ".tmod",
                        "wb",
                    ) as f:
                        f.write(weights)
                    self._cur_hist_cpt = 0
                    if verbose:
                        print_with_timestamp("model weights saved in history")
            if hasattr(self.actor, "load_from_bytes"):
                loaded = self.actor.load_from_bytes(weights, device=self.device)
                if isinstance(loaded, bool):
                    loaded_ok = loaded
                elif loaded is None:
                    loaded_ok = True
                else:
                    # Generic ActorModule implementations may return a new actor instance.
                    self.actor = loaded
                    loaded_ok = True
            else:
                with open(self.model_path, "wb") as f:
                    f.write(weights)
                self.actor = self.actor.load(self.model_path, device=self.device)
                loaded_ok = True
            if verbose:
                if loaded_ok:
                    print_with_timestamp("model weights have been updated")
                else:
                    print_with_timestamp(
                        "model weights NOT applied (shape mismatch); keeping previous weights"
                    )
        return nb_received

    def ignore_actor_weights(self):
        """
        Clears the buffer of weights received from the `Server`.

        This is useful for expert RolloutWorkers, because all RolloutWorkers receive weights.

        Returns:
            int: number of new (ignored) actor models received from the Server.
        """
        if self.__endpoint is None:
            return 0
        weights_list = self.__endpoint.receive_all(blocking=False)
        nb_received = len(weights_list)
        return nb_received
