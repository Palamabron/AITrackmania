"""TorchMemory: partial Memory implementation that collates samples into PyTorch tensors."""

from abc import ABC
from collections.abc import Callable
from typing import Any

from tmrl.memory.base import Memory
from tmrl.util import collate_torch


class TorchMemory(Memory, ABC):
    """
    Partial implementation of the `Memory` class collating samples into batched torch tensors.

    .. note::
       When overriding `__init__`, don't forget to call `super().__init__` in the subclass.
       Your `__init__` method needs to take at least all the arguments of the superclass.
    """

    def __init__(
        self,
        device,
        nb_steps,
        sample_preprocessor: Callable[..., Any] | None = None,
        memory_size=1000000,
        batch_size=256,
        dataset_path="",
        crc_debug=False,
        n_step_return=1,
    ):
        """
        Args:
            device (str): output tensors will be collated to this device
            nb_steps (int): number of steps per round
            sample_preprocessor (callable): can be used for data augmentation
            memory_size (int): size of the circular buffer
            batch_size (int): batch size of the output tensors
            dataset_path (str): an offline dataset may be provided here to initialize the memory
            crc_debug (bool): False usually, True when using CRC debugging of the pipeline
            n_step_return (int): number of steps for n-step TD returns (1 = single-step)
        """
        super().__init__(
            memory_size=memory_size,
            batch_size=batch_size,
            dataset_path=dataset_path,
            nb_steps=nb_steps,
            sample_preprocessor=sample_preprocessor,
            crc_debug=crc_debug,
            device=device,
            n_step_return=n_step_return,
        )

    def collate(self, batch, device):
        return collate_torch(batch, device)

    def clear(self) -> None:
        """Remove all transitions from the memory."""
        self.data = []
