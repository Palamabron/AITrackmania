import pickle
from abc import ABC, abstractmethod
from io import BytesIO

import numpy as np
import torch

from tmrl.util import collate_torch

__docformat__ = "google"


class ActorModule(ABC):
    """
    Implement this interface for the RolloutWorker(s) to interact with your policy.

    .. note::
       If overidden, the __init()__ definition must at least take the two following arguments
       (args or kwargs): `observation_space` and `action_space`.
       When overriding `__init__`, don't forget to call `super().__init__` in the subclass.
    """

    def __init__(self, observation_space, action_space):
        """
        Args:
            observation_space (gymnasium.spaces.Space): observation space
                (here for your convenience)
            action_space (gymnasium.spaces.Space): action space
                (here for your convenience)
        """
        self.observation_space = observation_space
        self.action_space = action_space
        super().__init__()

    def save(self, path):
        """
        Save your `ActorModule` on the hard drive.

        If not implemented, `save` defaults to `pickle.dump(obj=self, ...)`.

        You need to override this method if your `ActorModule` is not picklable.

        .. note::
           Everything needs to be saved into a single binary file.
           `tmrl` reads this file and transfers its content over network.

        Args:
            path (pathlib.Path): a filepath to save your `ActorModule` to
        """
        with open(path, "wb") as f:
            pickle.dump(obj=self, file=f)

    def load(self, path, device):
        """
        Load and return an instance of your `ActorModule` from the hard drive.

        This method loads your `ActorModule` from the binary file saved by your
        implementation of `save`.

        If not implemented, `load` defaults to returning this output of pickle.load(...).
        By default, the `device` argument is ignored (but you may use it in your impl).

        You need to override this method if your ActorModule is not picklable.

        .. note::
           You can use this to load attributes and return self, or return a new instance.

        Args:
            path (pathlib.Path): a filepath to load your ActorModule from
            device: device to load relevant attributes to (e.g., "cpu" or "cuda:0")

        Returns:
            ActorModule: An instance of your ActorModule
        """
        with open(path, "rb") as f:
            res = pickle.load(file=f)
        return res

    def to_device(self, device):
        """
        Set the `ActorModule`'s relevant attributes to the designated device.

        By default, this method is a no-op and returns `self`.

        Args:
            device: the device where to move relevant attributes (e.g., `"cpu"` or `"cuda:0"`)

        Returns:
            an `ActorModule` whose relevant attributes are moved to `device` (can be `self`)
        """
        return self

    def save_to_bytes(self) -> bytes:
        """Serialize actor to bytes for network transport."""
        return pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)

    def load_from_bytes(self, payload: bytes, device):
        """Deserialize actor from bytes (default pickle-based implementation)."""
        return pickle.loads(payload)

    @abstractmethod
    def act(self, obs, test=False):
        """
        Must compute an action from an observation.

        Args:
            obs (object): the observation
            test (bool): True at test time, False otherwise

        Returns:
            numpy.array: the computed action
        """
        raise NotImplementedError

    def act_(self, obs, test=False):
        return self.act(obs, test=test)


class TorchActorModule(ActorModule, torch.nn.Module, ABC):
    """
    Partial implementation of `ActorModule` as a `torch.nn.Module`.

    You can implement this instead of `ActorModule` when using PyTorch.
    `TorchActorModule` is a subclass of `torch.nn.Module` and may implement `forward()`.
    Typically, your implementation of `act()` can call `forward()` with gradients turned off.

    When using `TorchActorModule`, the `act` method receives observations collated on `device`,
    with an additional dimension corresponding to the batch size.

    .. note::
       If overidden, the __init()__ definition must at least take the two following arguments
       (args or kwargs): `observation_space` and `action_space`.
       When overriding `__init__`, don't forget to call `super().__init__` in the subclass.
    """

    def __init__(self, observation_space, action_space, device="cpu"):
        """
        Args:
            observation_space (gymnasium.spaces.Space): observation space
                (here for your convenience)
            action_space (gymnasium.spaces.Space): action space
                (here for your convenience)
            device: device where your model lives and where observations for
                `act` are collated
        """
        super().__init__(observation_space, action_space)  # ActorModule
        # super().__init__()  # torch.nn.Module
        self.device = device

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, device):
        self.device = device
        self.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        return self

    def save_to_bytes(self) -> bytes:
        buffer = BytesIO()
        torch.save(self.state_dict(), buffer)
        return buffer.getvalue()

    def load_from_bytes(self, payload: bytes, device):
        self.device = device
        buffer = BytesIO(payload)
        try:
            state = torch.load(buffer, map_location=self.device, weights_only=True)
            try:
                self.load_state_dict(state)
            except RuntimeError:
                # Trainer may wrap the actor (e.g. _AsymmetricActorAdapter) which
                # adds an 'actor.' prefix to every key.  Strip it and retry.
                prefix = "actor."
                stripped = {
                    k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in state.items()
                }
                if stripped != state:
                    self.load_state_dict(stripped)
                else:
                    raise
        except RuntimeError as e:
            if "size mismatch" in str(e) or "Missing key" in str(e) or "shape" in str(e).lower():
                from loguru import logger

                logger.warning(
                    "Ignoring incompatible weights from server (shape mismatch). "
                    "This is normal after a config change or trainer reset -- the worker "
                    "will keep its current weights until the trainer broadcasts compatible ones. "
                    "If this persists, restart server -> trainer -> worker."
                )
                return self
            raise
        return self

    def act_(self, obs, test=False):
        obs = collate_torch([obs], device=self.device)
        with torch.no_grad():
            action = self.act(obs, test=test)
        if action is not None:
            action = np.nan_to_num(action, nan=0.0)
            np.clip(action, -1.0, 1.0, out=action)
        return action

    # noinspection PyMethodOverriding
    def to(self, device):
        self.device = device
        return super().to(device=device)

    def to_device(self, device):
        return self.to(device)
