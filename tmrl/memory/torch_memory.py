"""TorchMemory: partial Memory implementation that collates samples into PyTorch tensors."""

from abc import ABC
from collections.abc import Callable
from typing import Any

from tmrl.memory.base import Memory
from tmrl.util import collate_torch


class TorchMemory(Memory, ABC):
    """Partial ``Memory`` implementation that collates samples into PyTorch tensors.

    Implements :meth:`collate` using :func:`~tmrl.util.collate_torch`, which
    stacks each field of a sample list into a batched ``torch.Tensor`` and moves
    the result to ``self.device``.  Subclasses must still implement
    :meth:`append_buffer`, :meth:`__len__`, and :meth:`get_transition`.

    Note:
        When subclassing, call ``super().__init__`` and accept at least all
        arguments of the base :class:`~tmrl.memory.base.Memory` class.
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
        """Initialize TorchMemory.

        Args:
            device: Device to which output tensors are collated (e.g. ``"cpu"``
                or ``"cuda"``).
            nb_steps: Number of sampling steps per training round.
            sample_preprocessor: Optional data-augmentation callable applied to
                each sampled batch element.
            memory_size: Maximum number of transitions in the circular buffer.
            batch_size: Number of transitions per sampled batch.
            dataset_path: Path to an offline dataset pickle to preload on init.
            crc_debug: When ``True``, run CRC integrity checks on each sample.
            n_step_return: Number of consecutive steps for n-step TD returns.
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
        """Collate a list of transition tuples into batched PyTorch tensors.

        Args:
            batch: List of ``(prev_obs, new_act, rew, new_obs, terminated,
                truncated, info)`` tuples.
            device: Target device for the resulting tensors.

        Returns:
            tuple: Batched tensors ``(prev_obs, new_act, rew, new_obs, terminated,
                truncated)`` collated on ``device``.
        """
        return collate_torch(batch, device)

    def clear(self) -> None:
        """Remove all transitions from the memory."""
        self.data = []
