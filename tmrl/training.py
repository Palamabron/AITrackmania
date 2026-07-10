from abc import ABC, abstractmethod


class TrainingAgent(ABC):
    """
    Training algorithm.

    CAUTION: When overriding `__init__`, don't forget to call `super().__init__` in the subclass.
    """

    def __init__(self, observation_space, action_space, device):
        """
        Args:
            observation_space (gymnasium.spaces.Space): observation space
            action_space (gymnasium.spaces.Space): action space
            device (str): device for training
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device

    @abstractmethod
    def train(
        self,
        batch,
        epoch: int | None = None,
        batch_index: int | None = None,
        iters: int | None = None,
    ):
        """Execute one gradient update step.

        Args:
            batch: A tuple ``(prev_obs, action, reward, new_obs, terminated, truncated[, ...])``.
            epoch (int | None): Current training epoch index.
            batch_index (int | None): Step index within the current round.
            iters (int | None): Current replay buffer size (``len(memory)``).

        Returns:
            dict: A mapping from metric name to value for logging (e.g. to wandb).
                Return an empty dict if no metrics are available.
        """
        raise NotImplementedError

    @abstractmethod
    def get_actor(self):
        """
        Returns the current ActorModule to be broadcast to the RolloutWorkers.

        Returns:
             ActorModule: current actor to be broadcast
        """
        raise NotImplementedError
