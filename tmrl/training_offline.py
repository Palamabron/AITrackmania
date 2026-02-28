# standard library imports
import datetime
import time
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Number
from typing import Any

# third-party imports
import gymnasium
import numpy as np
import torch
from loguru import logger

# local imports
import tmrl.config.config_constants as cfg
from tmrl.tools.player_runs import poll_player_runs_for_injection
from tmrl.util import pandas_dict

__docformat__ = "google"


def _observation_space_from_sample(observation) -> gymnasium.spaces.Space:
    """Build a gymnasium observation space from a single observation (e.g. tuple of arrays).

    Use this when the replay buffer already has data so the model is built with the same
    observation shape as the data (avoids LayerNorm / backbone shape mismatch).
    """
    if isinstance(observation, (list, tuple)):
        spaces_list = []
        for s in observation:
            arr = np.asarray(s)
            spaces_list.append(
                gymnasium.spaces.Box(
                    low=np.float32(-np.inf),
                    high=np.float32(np.inf),
                    shape=arr.shape,
                    dtype=arr.dtype,
                )
            )
        return gymnasium.spaces.Tuple(tuple(spaces_list))
    else:
        arr = np.asarray(observation)
        return gymnasium.spaces.Box(
            low=np.float32(-np.inf),
            high=np.float32(np.inf),
            shape=arr.shape,
            dtype=arr.dtype,
        )


def _observation_dim(space: gymnasium.spaces.Space) -> int:
    """Total dimension of an observation space (Tuple of Box or single Box)."""
    if isinstance(space, gymnasium.spaces.Tuple):
        return sum(int(np.prod(s.shape)) for s in space.spaces)
    return int(np.prod(space.shape))


def _one_obs_from_batch(batch_obs) -> np.ndarray | tuple:
    """Extract a single observation (numpy) from batch observation (tensor or tuple of tensors)."""
    if isinstance(batch_obs, (list, tuple)):
        return tuple(
            t[0].cpu().numpy() if hasattr(t, "cpu") else np.asarray(t[0]) for t in batch_obs
        )
    if hasattr(batch_obs, "cpu"):
        return batch_obs[0].cpu().numpy()
    return np.asarray(batch_obs[0])


def _stats_dict_to_numeric(d: dict) -> dict:
    """Convert tensor values in a stats dict to Python scalars so pandas can aggregate."""
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.item() if v.numel() == 1 else float(v.mean().item())
        else:
            out[k] = v
    return out


def _mean_stats_dicts(items: list[dict[str, Any]]) -> dict[str, float]:
    """Fast mean aggregation without pandas DataFrame construction."""
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in items:
        for k, v in row.items():
            if isinstance(v, Number):
                vf = float(v)
                # skip NaN/Inf to mimic skipna behavior
                if vf == vf and vf not in (float("inf"), float("-inf")):
                    sums[k] = sums.get(k, 0.0) + vf
                    counts[k] = counts.get(k, 0) + 1
    return {k: (sums[k] / counts[k]) for k in sums if counts.get(k, 0) > 0}


@dataclass(eq=False)
class TrainingOffline:
    """
    Training wrapper for off-policy algorithms.

    Args:
        env_cls (type): dummy env class for obs/action spaces, or (obs_space, act_space) tuple.
        memory_cls (type): class of the replay memory
        training_agent_cls (type): class of the training agent
        epochs (int): total epochs; agent saved every epoch
        rounds (int): rounds per epoch; statistics every round
        steps (int): training steps per round
        update_model_interval (int): steps between model broadcasts
        update_buffer_interval (int): steps between retrieving buffered samples
        max_training_steps_per_env_step (float): training pauses when above this ratio
        sleep_between_buffer_retrieval_attempts (float): sleep when waiting for samples
        python_profiling (bool): if True, run_epoch is profiled and printed each epoch
        agent_scheduler (callable): if not None, f(Agent, epoch) at start of each epoch
        start_training (int): min samples in replay buffer before starting training
        device (str): device for memory to collate samples
    """

    env_cls: type[Any] | None = None  # GenericGymEnv or (observation_space, action_space)
    memory_cls: type[Any] | None = None  # = TorchMemory  # replay memory
    training_agent_cls: type[Any] | None = None  # = TrainingAgent  # training agent
    epochs: int = 10  # total number of epochs, we save the agent every epoch
    rounds: int = 50  # number of rounds per epoch, we generate statistics every round
    steps: int = 2000  # number of training steps per round
    update_model_interval: int = 100  # number of training steps between model broadcasts
    update_buffer_interval: int = (
        100  # number of training steps between retrieving buffered samples
    )
    max_training_steps_per_env_step: float = 1.0  # training will pause when above this ratio
    sleep_between_buffer_retrieval_attempts: float = (
        1.0  # algorithm will sleep for this amount of time when waiting
    )
    # for needed incoming samples
    agent_scheduler: Callable[..., Any] | None = (
        None  # if not None, must be of the form f(Agent, epoch), called at the beginning of
    )
    # each epoch
    start_training: int = (
        0  # minimum number of samples in the replay buffer before starting training
    )
    device: str | None = None  # device on which the model of the TrainingAgent will live
    python_profiling: bool = (
        False  # if True, run_epoch will be profiled and the profiling will be printed at the end
    )
    # of each epoch
    pytorch_profiling: bool = False
    total_updates = 0

    def __post_init__(self):
        """
        Initializes memory and spaces. The agent is built from actual replay data
        (buffer or first batch), not from env/config, so observation dimension
        always matches the data used for training.
        """
        device = self.device or "cpu"
        self.epoch = 0
        assert (
            self.memory_cls is not None
            and self.training_agent_cls is not None
            and self.env_cls is not None
        )
        self.memory = self.memory_cls(nb_steps=self.steps, device=device)
        if isinstance(self.env_cls, tuple):
            _, action_space = self.env_cls
        else:
            with self.env_cls() as env:
                action_space = env.action_space
        self._action_space = action_space
        # Build agent only when we have data: observation_space is inferred from
        # buffer/batch so it never depends on config (e.g. POINTS_NUMBER / track length).
        if len(self.memory) > 0:
            prev_obs, *_ = self.memory.get_transition(0)
            observation_space = _observation_space_from_sample(prev_obs)
            dim = _observation_dim(observation_space)
            logger.info(
                " Inferred observation_space from replay buffer at init (dim=%s), building agent.",
                dim,
            )
            self.agent = self.training_agent_cls(
                observation_space=observation_space,
                action_space=action_space,
                device=device,
            )
        else:
            self.agent = None
            logger.info(
                " Replay buffer empty at init; agent will be built from first available data "
                "(buffer or batch) so observation_space matches training data."
            )
        self.total_samples = len(self.memory)
        self._injected_player_run_ids: set[str] = set()
        self._best_return_train: float = float("-inf")
        self._best_epoch: int = -1
        self._perf_acc = dict(
            sample_s=0.0,
            update_buffer_s=0.0,
            train_s=0.0,
            wait_ratio_s=0.0,
            broadcast_s=0.0,
            batches=0,
        )
        logger.info(f" Initial total_samples:{self.total_samples}")
        if cfg.PLAYER_RUNS_ONLINE_INJECTION:
            from pathlib import Path

            _pr_path = Path(cfg.PLAYER_RUNS_SOURCE_PATH)
            logger.info(
                " Player runs online injection: SOURCE_PATH={} (exists={})",
                cfg.PLAYER_RUNS_SOURCE_PATH,
                _pr_path.exists(),
            )

    def _ensure_agent_from_data(self, batch=None):
        """
        Build or rebuild the agent so observation_space matches the data we train on.
        When a batch is provided, if its observation dim differs from the current agent,
        the agent is rebuilt from the batch (handles mixed buffer: e.g. 369 from workers,
        363 from player runs).
        """
        if batch is not None:
            one_obs = _one_obs_from_batch(batch[0])
            batch_obs_space = _observation_space_from_sample(one_obs)
            batch_dim = _observation_dim(batch_obs_space)
            if self.agent is not None:
                current_dim = _observation_dim(self.agent.observation_space)
                if batch_dim == current_dim:
                    return
                logger.warning(
                    " Observation dim from batch (%s) != agent (%s); rebuilding agent from batch.",
                    batch_dim,
                    current_dim,
                )
            observation_space = batch_obs_space
            dim = batch_dim
        elif self.agent is not None:
            return
        elif len(self.memory) > 0:
            prev_obs, *_ = self.memory.get_transition(0)
            if isinstance(prev_obs, (list, tuple)):
                one_obs = tuple(
                    (t.cpu().numpy() if hasattr(t, "cpu") else np.asarray(t)).squeeze()
                    for t in prev_obs
                )
            else:
                arr = prev_obs.cpu().numpy() if hasattr(prev_obs, "cpu") else np.asarray(prev_obs)
                one_obs = arr.squeeze()
            observation_space = _observation_space_from_sample(one_obs)
            dim = _observation_dim(observation_space)
        else:
            return
        logger.info(
            " Building agent from data (observation dim=%s); observation_space independent of config.",
            dim,
        )
        device = self.device or "cpu"
        self.agent = self.training_agent_cls(
            observation_space=observation_space,
            action_space=self._action_space,
            device=device,
        )

    def update_buffer(self, interface):
        """
        Updates the memory buffer by appending new data.
        Args: interface (an object with a method retrieve_buffer to get new data)
        Actions:
        Retrieves buffer data from the interface and appends it to the memory.
        Updates the count of total samples.
        """
        buffer = interface.retrieve_buffer()
        self.memory.append(buffer)
        self.total_samples += len(buffer)

        self._ensure_agent_from_data()

        if not cfg.PLAYER_RUNS_ONLINE_INJECTION:
            return

        demo_buffer, imported_ids, imported_files = poll_player_runs_for_injection(
            source_dir=cfg.PLAYER_RUNS_SOURCE_PATH,
            seen_run_ids=self._injected_player_run_ids,
            max_files=cfg.PLAYER_RUNS_MAX_FILES_PER_UPDATE,
            consume_on_read=cfg.PLAYER_RUNS_CONSUME_ON_READ,
        )
        if len(demo_buffer) > 0:
            repeat = cfg.PLAYER_RUNS_DEMO_INJECTION_REPEAT
            for _ in range(repeat):
                self.memory.append(demo_buffer)
                self.total_samples += len(demo_buffer)
            logger.info(
                " Injected {} player-run sample(s) from {} file(s), repeat x{} "
                "(effective: {}). run_ids={}",
                len(demo_buffer),
                len(imported_files),
                repeat,
                len(demo_buffer) * repeat,
                sorted(imported_ids),
            )

    def check_ratio(self, interface) -> float:
        """
        Checks the ratio of updates to total samples and waits for new samples if needed.
         Args: interface (an object to retrieve buffer data)
         Actions:
         Ratio of updates to total samples; if over limit or -1, waits for new samples.
        """
        ratio = (
            self.total_updates / self.total_samples
            if self.total_samples > 0.0 and self.total_samples >= self.start_training
            else -1.0
        )
        waited_s = 0.0
        if ratio > self.max_training_steps_per_env_step or ratio == -1.0:
            logger.info(
                " Waiting for new samples (total_samples={}, need >= {} to start)",
                self.total_samples,
                self.start_training,
            )
            wait_attempts = 0
            t_wait_start = time.perf_counter()
            while ratio > self.max_training_steps_per_env_step or ratio == -1.0:
                samples_before = self.total_samples
                self.update_buffer(interface)
                if self.total_samples > samples_before:
                    logger.info(
                        " Received {} samples from server (total: {})",
                        self.total_samples - samples_before,
                        self.total_samples,
                    )
                ratio = (
                    self.total_updates / self.total_samples
                    if self.total_samples > 0.0 and self.total_samples >= self.start_training
                    else -1.0
                )
                if ratio > self.max_training_steps_per_env_step or ratio == -1.0:
                    wait_attempts += 1
                    if wait_attempts % 10 == 1 and wait_attempts > 1:
                        logger.info(
                            " Still waiting for samples (total_samples={}, attempt ~{})",
                            self.total_samples,
                            wait_attempts,
                        )
                    time.sleep(self.sleep_between_buffer_retrieval_attempts)
            logger.info(" Resuming training")
            waited_s = time.perf_counter() - t_wait_start
        return waited_s

    def run_round(self, interface, stats_training, t_sample_prev):
        """
        Run one round of training (multiple batches), update buffer and optionally broadcast model.

        Steps:
            1. Every update_buffer_interval steps, pull buffer from interface into replay memory
               and refresh end-of-episode indices and reward sums.
            2. For each batch in memory: call agent.train(), aggregate stats,
            increment total_updates.
            3. Every update_model_interval steps, broadcast current actor weights via interface.
            4. After each batch, call check_ratio to optionally wait for more samples.

        Args:
            interface: Object to retrieve buffer data and broadcast model (e.g. Trainer link).
            stats_training: List to append per-batch training stats (returns, durations, etc.).
            t_sample_prev: Timestamp of previous sample (used for sampling duration in stats).
        """
        num_elements = 5
        step_size = max(1, int(self.steps / (num_elements - 1)))
        batch_index_checkpoints = {i * step_size for i in range(num_elements)}
        for batch_index in range(self.steps):
            t_sample_start = time.perf_counter()
            batch = self.memory.sample()
            t_sample = time.time()
            self._perf_acc["sample_s"] += time.perf_counter() - t_sample_start

            if self.total_updates % self.update_buffer_interval == 0:
                # retrieve local buffer in replay memory
                t_update_buffer_start = time.perf_counter()
                self.update_buffer(interface)
                self._perf_acc["update_buffer_s"] += time.perf_counter() - t_update_buffer_start

            t_update_buffer = time.time()

            if self.total_updates == 0:
                logger.info("starting training")

            if batch_index in batch_index_checkpoints:
                logger.info(
                    f"batch {batch_index}/{self.steps} finished at: {datetime.datetime.now()}"
                )

            self._ensure_agent_from_data(batch=batch)

            t_train_start = time.perf_counter()
            stats_training_dict = self.agent.train(batch, self.epoch, batch_index, len(self.memory))
            self._perf_acc["train_s"] += time.perf_counter() - t_train_start

            t_train = time.time()

            stats_training_dict["return_test"] = self.memory.stat_test_return
            stats_training_dict["return_train"] = self.memory.stat_train_return
            stats_training_dict["episode_length_test"] = self.memory.stat_test_steps
            stats_training_dict["episode_length_train"] = self.memory.stat_train_steps
            stats_training_dict["sampling_duration"] = t_sample - t_sample_prev
            stats_training_dict["training_step_duration"] = t_train - t_update_buffer
            if hasattr(self.memory, "last_sample_demo_fraction"):
                stats_training_dict["debug/demo_fraction_in_batch"] = float(
                    self.memory.last_sample_demo_fraction
                )
            stats_training += (_stats_dict_to_numeric(stats_training_dict),)
            self.total_updates += 1
            self._perf_acc["batches"] += 1
            if self.total_updates % self.update_model_interval == 0:
                # broadcast model weights
                t_broadcast_start = time.perf_counter()
                interface.broadcast_model(self.agent.get_actor())
                self._perf_acc["broadcast_s"] += time.perf_counter() - t_broadcast_start
            self._perf_acc["wait_ratio_s"] += self.check_ratio(interface)

            t_sample_prev = time.time()

    def run_epoch(self, interface):
        """Run one epoch: multiple rounds of training, then increment epoch counter.

        Steps:
            1. Optionally run agent_scheduler(agent, epoch) if set.
            2. For each round: check_ratio (wait for samples if needed), then run_round.
            3. Collect round stats (memory size, round time, idle/update/train times).
            4. If python_profiling is True, run pyinstrument and log profile.
            5. Increment epoch and return list of round stats.

        Args:
            interface: Object to retrieve buffer data and broadcast model.

        Returns:
            List of per-round stat dicts (e.g. round_time, memory_len, return_test).
        """
        stats = []
        self._ensure_agent_from_data()

        if (
            self.agent_scheduler is not None
            and callable(self.agent_scheduler)
            and self.agent is not None
        ):
            self.agent_scheduler(self.agent, self.epoch)

        for rnd in range(self.rounds):
            logger.info(
                f"=== epoch {self.epoch}/{self.epochs} ".ljust(20, "=")
                + f" round {rnd}/{self.rounds} ".ljust(50, "=")
            )
            logger.debug(f"(Training): current memory size:{len(self.memory)}")

            stats_training = []

            t0 = time.time()
            self.check_ratio(interface)
            t1 = time.time()

            if self.python_profiling:
                from pyinstrument import Profiler

                pro = Profiler()
                pro.start()

            t2 = time.time()

            t_sample_prev = t2

            self.run_round(interface, stats_training, t_sample_prev)

            t3 = time.time()

            round_time = t3 - t0
            idle_time = t1 - t0
            update_buf_time = t2 - t1
            train_time = t3 - t2
            logger.debug(
                f"round_time:{round_time}, idle:{idle_time}, update_buf:{update_buf_time}, "
                f"train_time:{train_time}"
            )
            stats += (
                pandas_dict(
                    memory_len=len(self.memory),
                    round_time=round_time,
                    idle_time=idle_time,
                    **_mean_stats_dicts(stats_training),
                ),
            )

            logger.info(stats[-1].add_prefix("  ").to_string() + "\n")
            if self._perf_acc["batches"] > 0:
                batches = float(self._perf_acc["batches"])
                logger.info(
                    " Perf avg [ms/batch] sample={:.2f} update_buf={:.2f} train={:.2f} "
                    "broadcast={:.2f} wait_ratio={:.2f} (batches={})",
                    1000.0 * self._perf_acc["sample_s"] / batches,
                    1000.0 * self._perf_acc["update_buffer_s"] / batches,
                    1000.0 * self._perf_acc["train_s"] / batches,
                    1000.0 * self._perf_acc["broadcast_s"] / batches,
                    1000.0 * self._perf_acc["wait_ratio_s"] / batches,
                    self._perf_acc["batches"],
                )

            if self.python_profiling:
                pro.stop()
                logger.info(pro.output_text(unicode=True, color=False, show_all=True))

        self._maybe_save_best_checkpoint(stats)
        self.epoch += 1
        return stats

    def _maybe_save_best_checkpoint(self, stats):
        """Save actor weights when epoch-average return_train improves."""
        try:
            returns = [s.get("return_train", float("nan")) for s in stats if hasattr(s, "get")]
            valid = [r for r in returns if r == r]  # filter nan
            if not valid:
                return
            mean_ret = sum(valid) / len(valid)
            if mean_ret > self._best_return_train and mean_ret > 0:
                self._best_return_train = mean_ret
                self._best_epoch = self.epoch
                best_path = cfg.WEIGHTS_FOLDER / "best_actor.pth"
                import torch

                torch.save(self.agent.get_actor().state_dict(), str(best_path))
                logger.info(
                    " New best return_train={:.2f} at epoch {} -> saved {}",
                    mean_ret,
                    self.epoch,
                    best_path,
                )
        except Exception as e:
            logger.warning(" Failed to save best checkpoint: {}", e)


class TorchTrainingOffline(TrainingOffline):
    """
    TrainingOffline for trainers based on PyTorch.

    This class implements automatic device selection with PyTorch.
    """

    def __init__(
        self,
        env_cls: type[Any] | None = None,
        memory_cls: type[Any] | None = None,
        training_agent_cls: type[Any] | None = None,
        epochs: int = 10,
        rounds: int = 50,
        steps: int = 2000,
        update_model_interval: int = 100,
        update_buffer_interval: int = 100,
        max_training_steps_per_env_step: float = 1.0,
        sleep_between_buffer_retrieval_attempts: float = 1.0,
        python_profiling: bool = False,
        pytorch_profiling: bool = False,
        agent_scheduler: Callable[..., Any] | None = None,
        start_training: int = 0,
        device: str | None = None,
    ):
        """
        Same as TrainingOffline; device=None selects automatically for torch.

        Args:
            env_cls (type): dummy env class or (observation_space, action_space) tuple
            memory_cls (type): replay memory class
            training_agent_cls (type): training agent class
            epochs (int): total epochs
            rounds (int): rounds per epoch
            steps (int): training steps per round
            update_model_interval (int): steps between model broadcasts
            update_buffer_interval (int): steps between retrieving buffered samples
            max_training_steps_per_env_step (float): pause training when above this ratio
            sleep_between_buffer_retrieval_attempts (float): sleep when waiting for samples
            python_profiling (bool): profile run_epoch and print at end of each epoch
            agent_scheduler (callable): if not None, f(Agent, epoch) at start of each epoch
            start_training (int): min samples in replay buffer before training
            device (str): device for memory (None = auto)
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(
            env_cls,
            memory_cls,
            training_agent_cls,
            epochs,
            rounds,
            steps,
            update_model_interval,
            update_buffer_interval,
            max_training_steps_per_env_step,
            sleep_between_buffer_retrieval_attempts,
            agent_scheduler,
            start_training,
            device,
            python_profiling,
            pytorch_profiling,
        )
