# standard library imports
import datetime
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# third-party imports
import torch
from loguru import logger
from pandas import DataFrame

# local imports
from tmrl.util import pandas_dict

__docformat__ = "google"


def _stats_dict_to_numeric(d: dict) -> dict:
    """Convert tensor values in a stats dict to Python scalars so pandas can aggregate."""
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.item() if v.numel() == 1 else float(v.mean().item())
        else:
            out[k] = v
    return out


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
        Initializes various attributes and objects after the instance is created.
        Args: self (instance of the class)
        Actions:
        Sets epoch to 0. Inits memory (memory_cls) with nb_steps and device.
        Gets observation_space and action_space from env_cls.
        Inits agent (training_agent_cls) with those spaces and device.
        Logs the initial total samples in the memory.
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
            observation_space, action_space = self.env_cls
        else:
            with self.env_cls() as env:
                observation_space, action_space = env.observation_space, env.action_space
        self.agent = self.training_agent_cls(
            observation_space=observation_space, action_space=action_space, device=device
        )
        self.total_samples = len(self.memory)
        logger.info(f" Initial total_samples:{self.total_samples}")

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

    def check_ratio(self, interface):
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
        if ratio > self.max_training_steps_per_env_step or ratio == -1.0:
            logger.info(
                " Waiting for new samples (total_samples={}, need >= {} to start)",
                self.total_samples,
                self.start_training,
            )
            wait_attempts = 0
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
        for batch_index, batch in enumerate(self.memory):  # this samples a fixed number of batches
            t_sample = time.time()

            if self.total_updates % self.update_buffer_interval == 0:
                # retrieve local buffer in replay memory
                self.update_buffer(interface)
                self.memory.end_episodes_indices = self.memory.find_zero_rewards_indices(
                    self.memory.data[self.memory.rewards_index]
                )
                self.memory.reward_sums = [
                    self.memory.data[self.memory.rewards_index][index]["reward_sum"]
                    for index in self.memory.end_episodes_indices
                ]

            t_update_buffer = time.time()

            if self.total_updates == 0:
                logger.info("starting training")

            num_elements = 5

            # Calculate the step size between elements
            step_size = int(self.steps / (num_elements - 1))

            # Create a list of five equally spaced elements
            batch_index_checkpoints = [i * step_size for i in range(num_elements)]

            if batch_index in batch_index_checkpoints:
                logger.info(
                    f"batch {batch_index}/{self.steps} finished at: {datetime.datetime.now()}"
                )

            stats_training_dict = self.agent.train(batch, self.epoch, batch_index, len(self.memory))

            t_train = time.time()

            stats_training_dict["return_test"] = self.memory.stat_test_return
            stats_training_dict["return_train"] = self.memory.stat_train_return
            stats_training_dict["episode_length_test"] = self.memory.stat_test_steps
            stats_training_dict["episode_length_train"] = self.memory.stat_train_steps
            stats_training_dict["sampling_duration"] = t_sample - t_sample_prev
            stats_training_dict["training_step_duration"] = t_train - t_update_buffer
            stats_training += (_stats_dict_to_numeric(stats_training_dict),)
            self.total_updates += 1
            if self.total_updates % self.update_model_interval == 0:
                # broadcast model weights
                interface.broadcast_model(self.agent.get_actor())
            self.check_ratio(interface)

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

        if self.agent_scheduler is not None and callable(self.agent_scheduler):
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
                    **DataFrame(stats_training).mean(skipna=True),
                ),
            )

            logger.info(stats[-1].add_prefix("  ").to_string() + "\n")

            if self.python_profiling:
                pro.stop()
                logger.info(pro.output_text(unicode=True, color=False, show_all=True))

        # if len(self.memory.end_episodes_indices) > 1:
        # print(f"end_episodes_indices: {self.memory.end_episodes_indices}")
        # print(f"reward_sums: {self.memory.reward_sums}")

        self.epoch += 1
        return stats


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
