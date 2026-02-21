# standard library imports
import datetime
import logging
import time
from dataclasses import dataclass

# third-party imports
import torch
from pandas import DataFrame

# local imports
from util import pandas_dict

logging.basicConfig(level=logging.INFO)
__docformat__ = "google"


@dataclass(eq=False)
class TrainingOffline:
    """
    Training wrapper for off-policy algorithms.

    Args:
        env_cls (type): class of a dummy environment, used only to retrieve observation and action spaces if needed.
        Alternatively, this can be a tuple of the form (observation_space, action_space).
        memory_cls (type): class of the replay memory
        training_agent_cls (type): class of the training agent
        epochs (int): total number of epochs, we save the agent every epoch
        rounds (int): number of rounds per epoch, we generate statistics every round
        steps (int): number of training steps per round
        update_model_interval (int): number of training steps between model broadcasts
        update_buffer_interval (int): number of training steps between retrieving buffered samples
        max_training_steps_per_env_step (float): training will pause when above this ratio
        sleep_between_buffer_retrieval_attempts (float): algorithm will sleep for this amount of time when waiting for
        needed incoming samples
        python_profiling (bool): if True, run_epoch will be profiled and the profiling will be printed at the end of each epoch
        agent_scheduler (callable): if not None, must be of the form f(Agent, epoch), called at the beginning of each epoch
        start_training (int): minimum number of samples in the replay buffer before starting training
        device (str): device on which the memory will collate training samples
    """

    env_cls: type = None  # =GenericGymEnv - dummy environment used only to retrieve observation and action spaces if
    # needed
    memory_cls: type = None  # = TorchMemory  # replay memory
    training_agent_cls: type = None  # = TrainingAgent  # training agent
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
    agent_scheduler: callable = (
        None  # if not None, must be of the form f(Agent, epoch), called at the beginning of
    )
    # each epoch
    start_training: int = (
        0  # minimum number of samples in the replay buffer before starting training
    )
    device: str = None  # device on which the model of the TrainingAgent will live
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
        Sets the initial epoch to 0.
        Initializes memory using a defined memory class (memory_cls), passing in parameters like nb_steps and device.
        Retrieves observation_space and action_space from the environment class (env_cls).
        Initializes agent using a training agent class (training_agent_cls) with the obtained observation_space, action_space, and device.
        Logs the initial total samples in the memory.
        """
        device = self.device
        self.epoch = 0
        self.memory = self.memory_cls(nb_steps=self.steps, device=device)
        if type(self.env_cls) == tuple:
            observation_space, action_space = self.env_cls
        else:
            with self.env_cls() as env:
                observation_space, action_space = env.observation_space, env.action_space
        self.agent = self.training_agent_cls(
            observation_space=observation_space, action_space=action_space, device=device
        )
        self.total_samples = len(self.memory)
        logging.info(f" Initial total_samples:{self.total_samples}")

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
         Calculates the ratio of updates to total samples and checks if it exceeds a defined limit.
         If the ratio exceeds the limit or is initially -1, it waits for new samples before resuming training.
        """
        ratio = (
            self.total_updates / self.total_samples
            if self.total_samples > 0.0 and self.total_samples >= self.start_training
            else -1.0
        )
        if ratio > self.max_training_steps_per_env_step or ratio == -1.0:
            logging.info(" Waiting for new samples")
            while ratio > self.max_training_steps_per_env_step or ratio == -1.0:
                # wait for new samples
                self.update_buffer(interface)
                ratio = (
                    self.total_updates / self.total_samples
                    if self.total_samples > 0.0 and self.total_samples >= self.start_training
                    else -1.0
                )
                if ratio > self.max_training_steps_per_env_step or ratio == -1.0:
                    time.sleep(self.sleep_between_buffer_retrieval_attempts)
            logging.info(" Resuming training")

    def run_round(self, interface, stats_training, t_sample_prev):
        """
        Runs a round of training using the memory data in batches.
        Args:
        interface (an object to retrieve buffer data)
        stats_training (a list to store training statistics)
        t_sample_prev (time of the previous sample)
        Actions:
        Loops through batches in memory and performs training using an agent.
        Logs information related to batch checkpoints, training duration, and various statistics.
        Updates model weights and checks the update-to-sample ratio.
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
                logging.info("starting training")

            num_elements = 5

            # Calculate the step size between elements
            step_size = int(self.steps / (num_elements - 1))

            # Create a list of five equally spaced elements
            batch_index_checkpoints = [i * step_size for i in range(num_elements)]

            if batch_index in batch_index_checkpoints:
                logging.info(
                    f"batch {batch_index} out of {self.steps} has finished at: {datetime.datetime.now()}"
                )

            stats_training_dict = self.agent.train(batch, self.epoch, batch_index, len(self.memory))

            t_train = time.time()

            stats_training_dict["return_test"] = self.memory.stat_test_return
            stats_training_dict["return_train"] = self.memory.stat_train_return
            stats_training_dict["episode_length_test"] = self.memory.stat_test_steps
            stats_training_dict["episode_length_train"] = self.memory.stat_train_steps
            stats_training_dict["sampling_duration"] = t_sample - t_sample_prev
            stats_training_dict["training_step_duration"] = t_train - t_update_buffer
            stats_training += (stats_training_dict,)
            self.total_updates += 1
            if self.total_updates % self.update_model_interval == 0:
                # broadcast model weights
                interface.broadcast_model(self.agent.get_actor())
            self.check_ratio(interface)

            t_sample_prev = time.time()

    def run_epoch(self, interface):
        """
        Runs multiple rounds within an epoch.
        Args: interface (an object to retrieve buffer data)
        Actions:
        Manages the execution of multiple rounds within an epoch, calling run_round() for each round.
        Collects and logs statistics related to memory size, round time, training time, and more.
        Increments the epoch count at the end.
        """
        stats = []
        state = None

        if self.agent_scheduler is not None and callable(self.agent_scheduler):
            self.agent_scheduler(self.agent, self.epoch)

        for rnd in range(self.rounds):
            logging.info(
                f"=== epoch {self.epoch}/{self.epochs} ".ljust(20, "=")
                + f" round {rnd}/{self.rounds} ".ljust(50, "=")
            )
            logging.debug(f"(Training): current memory size:{len(self.memory)}")

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
            logging.debug(
                f"round_time:{round_time}, idle_time:{idle_time}, update_buf_time:{update_buf_time}, train_time:{train_time}"
            )
            stats += (
                pandas_dict(
                    memory_len=len(self.memory),
                    round_time=round_time,
                    idle_time=idle_time,
                    **DataFrame(stats_training).mean(skipna=True),
                ),
            )

            logging.info(stats[-1].add_prefix("  ").to_string() + "\n")

            if self.python_profiling:
                pro.stop()
                logging.info(pro.output_text(unicode=True, color=False, show_all=True))

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
        env_cls: type = None,
        memory_cls: type = None,
        training_agent_cls: type = None,
        epochs: int = 10,
        rounds: int = 50,
        steps: int = 2000,
        update_model_interval: int = 100,
        update_buffer_interval: int = 100,
        max_training_steps_per_env_step: float = 1.0,
        sleep_between_buffer_retrieval_attempts: float = 1.0,
        python_profiling: bool = False,
        pytorch_profiling: bool = False,
        agent_scheduler: callable = None,
        start_training: int = 0,
        device: str = None,
    ):
        """
        Same arguments as `TrainingOffline`, but when `device` is `None` it is selected automatically for torch.

        Args:
            env_cls (type): class of a dummy environment, used only to retrieve observation and action spaces if needed. Alternatively, this can be a tuple of the form (observation_space, action_space).
            memory_cls (type): class of the replay memory
            training_agent_cls (type): class of the training agent
            epochs (int): total number of epochs, we save the agent every epoch
            rounds (int): number of rounds per epoch, we generate statistics every round
            steps (int): number of training steps per round
            update_model_interval (int): number of training steps between model broadcasts
            update_buffer_interval (int): number of training steps between retrieving buffered samples
            max_training_steps_per_env_step (float): training will pause when above this ratio
            sleep_between_buffer_retrieval_attempts (float): algorithm will sleep for this amount of time when waiting for needed incoming samples
            profiling (bool): if True, run_epoch will be profiled and the profiling will be printed at the end of each epoch
            agent_scheduler (callable): if not None, must be of the form f(Agent, epoch), called at the beginning of each epoch
            start_training (int): minimum number of samples in the replay buffer before starting training
            device (str): device on which the memory will collate training samples (None for automatic)
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
