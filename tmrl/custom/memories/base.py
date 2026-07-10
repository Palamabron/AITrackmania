"""Base memory classes for TrackMania reinforcement learning."""

from collections.abc import Callable
from typing import Any

import numpy as np

from tmrl.custom.memories._internal.enums import BufferField, GenericField
from tmrl.custom.memories._internal.sampling_utils import (
    canonical_replay_action_vector,
    configure_discrete_steer_bins,
)
from tmrl.memory import TorchMemory
from tmrl.registry import MEMORIES


def last_true_in_list(li: list[bool]) -> int | None:
    """Find the index of the last ``True`` value in a boolean list.

    Used to locate the most recent end-of-episode (EOE) marker within a
    history window so that :func:`replace_hist_before_eoe` can pad the
    observation/action history correctly.

    Args:
        li: List of boolean values to scan from right to left.

    Returns:
        int | None: Index of the last ``True`` entry, or ``None`` if no
            ``True`` was found.
    """
    for i in reversed(range(len(li))):
        if li[i]:
            return i
    return None


def replace_hist_before_eoe(hist: list, eoe_idx_in_hist: int) -> None:
    """Pad history entries before an episode boundary with the post-boundary value.

    When a history window (image stack, action buffer) spans an episode boundary
    at ``eoe_idx_in_hist``, entries before that boundary would otherwise carry
    stale values from the previous episode.  This function overwrites them with
    ``hist[eoe_idx_in_hist + 1]``, propagating the first post-boundary frame
    backward so the network sees a clean start-of-episode context.

    The in-place replacement walks backwards from ``eoe_idx_in_hist`` to index 0,
    so all replaced entries receive the same post-boundary value (not a chain
    from unrelated earlier history).

    Args:
        hist: List whose leading entries should be replaced; modified in place.
        eoe_idx_in_hist: Index of the episode boundary within ``hist``.
            Must satisfy ``0 <= eoe_idx_in_hist < len(hist)``.

    Raises:
        ValueError: If ``eoe_idx_in_hist`` is out of range (greater than
            ``len(hist) - 1``).
    """
    last_idx = len(hist) - 1
    if eoe_idx_in_hist > last_idx:
        raise ValueError(
            f"replace_hist_before_eoe: eoe_idx_in_hist ({eoe_idx_in_hist}) > last_idx ({last_idx})"
        )
    if 0 <= eoe_idx_in_hist < last_idx:
        for i in reversed(range(len(hist))):
            if i <= eoe_idx_in_hist:
                hist[i] = hist[i + 1]


def enforce_demo_batch_fraction(
    result: np.ndarray,
    item_is_demo_flags: np.ndarray,
    demo_min: float,
    demo_max: float,
) -> np.ndarray:
    """Swap sampled indices so the demo share of the batch lands in [demo_min, demo_max].

    Enforcement is skipped when the buffer contains only demo or only non-demo
    items (nothing to swap in).

    Args:
        result: Sampled item indices (modified in place and returned).
        item_is_demo_flags: Boolean array over all valid item indices; True = demo.
        demo_min: Minimum demo fraction of the batch (0-1).
        demo_max: Maximum demo fraction of the batch (0-1).

    Returns:
        The adjusted index array.
    """
    batch_size = len(result)
    if batch_size == 0 or (demo_min <= 0.0 and demo_max >= 1.0):
        return result
    demo_items = np.flatnonzero(item_is_demo_flags)
    non_demo_items = np.flatnonzero(~item_is_demo_flags)
    if demo_items.size == 0 or non_demo_items.size == 0:
        return result

    min_demo = int(np.ceil(demo_min * batch_size))
    max_demo = int(np.floor(demo_max * batch_size))
    max_demo = max(min_demo, min(max_demo, batch_size))

    demo_positions = np.flatnonzero(item_is_demo_flags[result])
    non_demo_positions = np.flatnonzero(~item_is_demo_flags[result])

    if demo_positions.size < min_demo and non_demo_positions.size > 0:
        need = int(min(min_demo - demo_positions.size, non_demo_positions.size))
        replace_positions = np.random.choice(non_demo_positions, size=need, replace=False)
        replacements = np.random.choice(demo_items, size=need, replace=demo_items.size < need)
        result[replace_positions] = replacements
        demo_positions = np.flatnonzero(item_is_demo_flags[result])

    if demo_positions.size > max_demo:
        excess = int(demo_positions.size - max_demo)
        replace_positions = np.random.choice(demo_positions, size=excess, replace=False)
        replacements = np.random.choice(
            non_demo_items, size=excess, replace=non_demo_items.size < excess
        )
        result[replace_positions] = replacements

    return result


@MEMORIES.register("generic")
class GenericTorchMemory(TorchMemory):
    """Generic torch-based memory for simple replay buffer scenarios.

    supports_nstep = True signals to config_objects that this memory implements
    memory-side n-step return accumulation and may be used with algorithm.n_steps > 1.

    Supports memory-side n-step returns: when ``n_step_return > 1``, each sampled
    transition carries the discounted reward sum over up to n consecutive steps
    (never crossing episode boundaries), the observation n steps ahead, and the
    effective window length in ``info["n_step_effective"]``.

    When ``discrete_n_steer_bins > 0`` (discrete IQN/SDSAC pipeline), continuous
    ``(3,)`` ``[gas, brake, steer]`` actions (e.g. injected human demos) are
    quantized to discrete indices at append time so the action column stays
    homogeneous (mixed scalar/(3,) shapes would crash ``collate_torch``).
    """

    supports_nstep: bool = True

    def __init__(
        self,
        memory_size: int = 1_000_000,
        batch_size: int = 1,
        dataset_path: str = "",
        nb_steps: int = 1,
        sample_preprocessor: Callable[..., Any] | None = None,
        crc_debug: bool = False,
        device: str = "cpu",
        discrete_n_steer_bins: int = 0,
        n_step_return: int = 1,
        gamma: float | None = None,
        demo_min_batch_fraction: float = 0.0,
        demo_max_batch_fraction: float = 1.0,
    ):
        """Initialize GenericTorchMemory.

        Args:
            memory_size: Maximum number of transitions in the circular buffer.
            batch_size: Number of transitions per sampled batch.
            dataset_path: Path to an offline dataset pickle to preload on init.
            nb_steps: Number of sampling steps per training round.
            sample_preprocessor: Optional data-augmentation callable applied to
                each sampled batch element.
            crc_debug: When ``True``, run CRC integrity checks on each sample.
                Incompatible with ``n_step_return > 1``.
            device: Target device for the collated output tensors.
            discrete_n_steer_bins: Number of steer bins for the discrete action
                pipeline; ``0`` = continuous actions.
            n_step_return: Number of consecutive steps for n-step TD returns.
                When ``> 1``, ``gamma`` must be provided.
            gamma: Discount factor for n-step reward accumulation.  Required
                when ``n_step_return > 1``.
            demo_min_batch_fraction: Minimum demo fraction of each sampled batch.
            demo_max_batch_fraction: Maximum demo fraction of each sampled batch.

        Raises:
            ValueError: If ``n_step_return < 1``, or if ``n_step_return > 1``
                and ``gamma`` is ``None``, or if ``crc_debug=True`` and
                ``n_step_return > 1``.
        """
        configure_discrete_steer_bins(discrete_n_steer_bins)
        self.discrete_n_steer_bins = int(discrete_n_steer_bins)
        self._discrete_action_table: list[Any] | None = None
        n_step_return = int(n_step_return)
        if n_step_return < 1:
            raise ValueError(f"n_step_return must be >= 1, got {n_step_return}")
        if n_step_return > 1:
            if gamma is None:
                raise ValueError(
                    "GenericTorchMemory requires gamma (the algorithm's discount factor) "
                    "when n_step_return > 1, so memory-side n-step rewards are discounted "
                    "consistently with the Bellman backup."
                )
            if crc_debug:
                raise ValueError(
                    "crc_debug is incompatible with n_step_return > 1 "
                    "(CRC checks assume 1-step transitions)."
                )
        self.gamma = float(gamma) if gamma is not None else 1.0
        self.demo_min_batch_fraction = max(0.0, min(1.0, float(demo_min_batch_fraction)))
        self.demo_max_batch_fraction = max(0.0, min(1.0, float(demo_max_batch_fraction)))
        if self.demo_max_batch_fraction < self.demo_min_batch_fraction:
            self.demo_max_batch_fraction = self.demo_min_batch_fraction
        self.last_sample_demo_fraction = 0.0
        # Cached per-data-index demo flags; invalidated whenever self.data changes.
        self._demo_flags_cache: np.ndarray | None = None
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

    def _canonical_discrete_action(self, action: Any) -> Any:
        """Quantize continuous ``(3,)`` actions to discrete indices (discrete pipeline only).

        Worker rollouts already store scalar integer indices and pass through
        unchanged; injected demos carry continuous ``[gas, brake, steer]`` rows.
        """
        a = np.asarray(action)
        if a.ndim == 0 and np.issubdtype(a.dtype, np.integer):
            return action
        flat = a.reshape(-1)
        if flat.size == 1 and np.issubdtype(a.dtype, np.integer):
            return np.int64(flat[0])
        if flat.size == 3:
            from tmrl.custom.tm.utils.control.discrete import (
                build_brake_tap_action_table,
                continuous_control_to_discrete_index,
            )

            if self._discrete_action_table is None:
                _, self._discrete_action_table = build_brake_tap_action_table(
                    n_steer=self.discrete_n_steer_bins
                )
            idx = continuous_control_to_discrete_index(
                flat.astype(np.float32), self._discrete_action_table
            )
            return np.int64(idx)
        raise ValueError(
            "GenericTorchMemory (discrete pipeline) received an action that is neither a "
            f"scalar integer index nor a (3,) control vector: shape={a.shape}, dtype={a.dtype}."
        )

    def _stored_action(self, idx: int) -> Any:
        """Read an action from storage, healing legacy continuous rows in place.

        Buffers pickled into checkpoints before append-time quantization existed
        can still hold ``(3,)`` demo actions; fix them lazily on first read.
        """
        action = self.data[GenericField.ACTIONS][idx]
        if self.discrete_n_steer_bins > 0:
            a = np.asarray(action)
            if not (a.ndim == 0 and np.issubdtype(a.dtype, np.integer)):
                action = self._canonical_discrete_action(action)
                self.data[GenericField.ACTIONS][idx] = action
        return action

    def append_buffer(self, buffer: Any) -> None:
        """Append a buffer of samples to the memory."""
        bf = BufferField
        if self.discrete_n_steer_bins > 0:
            actions = [self._canonical_discrete_action(b[bf.ACTION]) for b in buffer.memory]
        else:
            actions = [b[bf.ACTION] for b in buffer.memory]
        data_fields = [
            actions,
            [b[bf.OBSERVATION] for b in buffer.memory],
            [b[bf.REWARD] for b in buffer.memory],
            [b[bf.TERMINATED] for b in buffer.memory],
            [b[bf.TRUNCATED] for b in buffer.memory],
            [b[bf.INFO] for b in buffer.memory],
            [b[bf.TERMINATED] or b[bf.TRUNCATED] for b in buffer.memory],
        ]

        if self.__len__() > 0:
            for i, d in enumerate(data_fields):
                self.data[i] += d
        else:
            self.data = list(data_fields)

        to_trim = int(self.__len__() - self.memory_size)
        if to_trim > 0:
            for i in range(len(data_fields)):
                self.data[i] = self.data[i][to_trim:]
        self._demo_flags_cache = None

    def __len__(self) -> int:
        """Return the number of valid transitions in memory."""
        if len(self.data) == 0:
            return 0
        res = len(self.data[0]) - 1
        return max(0, res)

    def clear(self) -> None:
        """Remove all transitions from the memory."""
        self.data = []
        self._demo_flags_cache = None
        self.last_sample_demo_fraction = 0.0

    def mark_episode_boundary(self) -> None:
        """Mark the last stored entry as an episode end (truncation).

        Called by the trainer around demo injection so 1-step transitions and
        n-step windows never span the seam between unrelated streams (worker
        rollouts vs injected demo laps).
        """
        field = GenericField
        if len(self.data) == 0 or len(self.data[field.TRUNCATED]) == 0:
            return
        self.data[field.TRUNCATED][-1] = True
        self.data[field.DONE][-1] = True

    def _demo_flags(self) -> np.ndarray:
        """Boolean demo flag per data index (cached until the data changes)."""
        if self._demo_flags_cache is None:
            infos = self.data[GenericField.INFO] if len(self.data) > 0 else []
            self._demo_flags_cache = np.fromiter(
                (isinstance(e, dict) and bool(e.get("is_demo", False)) for e in infos),
                dtype=bool,
                count=len(infos),
            )
        return self._demo_flags_cache

    def _item_demo_flags(self, max_item: int) -> np.ndarray:
        """Boolean demo flag per item index in [0, max_item)."""
        # Item ``i`` is the transition into data index ``i + 1``.
        return self._demo_flags()[1 : max_item + 1]

    def _max_start_item(self) -> int:
        """Exclusive upper bound for valid transition start indices."""
        length = self.__len__()
        if self.n_step_return > 1:
            return max(0, length - self.n_step_return + 1)
        return length

    def sample_indices(self):
        """Sample transition start indices, enforcing the demo batch fraction.

        Draws ``batch_size`` uniform random indices from ``[0, max_start)`` and
        then calls :func:`enforce_demo_batch_fraction` to swap indices until the
        demo fraction of the batch satisfies
        ``[demo_min_batch_fraction, demo_max_batch_fraction]``.

        Returns:
            numpy.ndarray | tuple: Array of ``int64`` start indices, or ``()``
                when the buffer is empty.
        """
        max_start = self._max_start_item()
        if max_start <= 0:
            self.last_sample_demo_fraction = 0.0
            return ()
        result = np.random.randint(0, max_start, size=self.batch_size, dtype=np.int64)
        item_flags = self._item_demo_flags(max_start)
        result = enforce_demo_batch_fraction(
            result,
            item_flags,
            self.demo_min_batch_fraction,
            self.demo_max_batch_fraction,
        )
        self.last_sample_demo_fraction = (
            float(item_flags[result].mean()) if len(result) > 0 else 0.0
        )
        return result

    def get_transition(self, item: int) -> tuple:
        """Get a transition, applying n-step accumulation when configured.

        For ``n_step_return == 1`` returns the raw 1-step transition at ``item``.
        For ``n_step_return > 1`` accumulates a discounted reward sum over up to
        ``n`` consecutive steps forward from ``item``, stopping early at an
        episode boundary (``done=True``).  The terminal observation and
        ``terminated``/``truncated`` flags are taken from the last accumulated
        step.

        When the starting transition at ``item`` is terminal, the method
        resamples from valid transitions (maintaining demo/non-demo category when
        possible) up to ``max_retries`` times before raising.

        Args:
            item: Transition start index in ``[0, len(self))``.

        Returns:
            tuple: ``(prev_obs, new_act, rew, new_obs, terminated, truncated, info)``
                where ``rew`` is a ``numpy.float32`` scalar.

        Raises:
            RuntimeError: If no non-terminal starting transition is found after
                ``max_retries`` attempts.
        """
        field = GenericField
        n = int(self.n_step_return)
        resample_high = max(1, self._max_start_item())

        # Bounded retries to avoid excessive loops on large buffers
        max_retries = min(100, max(10, self.__len__()))
        item_flags = self._item_demo_flags(resample_high)
        want_demo = bool(item_flags[item]) if item < len(item_flags) else None
        if want_demo is True:
            resample_candidates = np.flatnonzero(item_flags)
        elif want_demo is False:
            resample_candidates = np.flatnonzero(~item_flags)
        else:
            resample_candidates = None
        for _attempt in range(max_retries):
            if not self.data[field.DONE][item]:
                break
            if resample_candidates is None:
                item = np.random.randint(0, resample_high)
            elif resample_candidates.size > 0:
                item = int(np.random.choice(resample_candidates))
            else:
                item = np.random.randint(0, resample_high)
        else:
            done_count = sum(self.data[field.DONE])
            raise RuntimeError(
                f"Failed to sample non-terminal transition after {max_retries} attempts. "
                f"Buffer has {done_count}/{self.__len__()} done=True transitions. "
                "This suggests a data quality issue or environment that always "
                "terminates immediately."
            )

        idx_last = item

        if n <= 1:
            idx_now = item + 1
            info = self.data[field.INFO][idx_now]
            info = dict(info) if isinstance(info, dict) else {}
            info["n_step_effective"] = 1
            return (
                self.data[field.OBSERVATIONS][idx_last],
                self._stored_action(idx_now),
                self.data[field.REWARDS][idx_now],
                self.data[field.OBSERVATIONS][idx_now],
                self.data[field.TERMINATED][idx_now],
                self.data[field.TRUNCATED][idx_now],
                info,
            )

        # Accumulate discounted rewards forward from idx_last, stopping at the
        # episode boundary so returns never leak across episodes.
        rewards = self.data[field.REWARDS]
        dones = self.data[field.DONE]
        n_step_reward = 0.0
        n_eff = 0
        for k in range(n):
            idx_k = idx_last + 1 + k
            n_step_reward += (self.gamma**k) * float(rewards[idx_k])
            n_eff = k + 1
            if dones[idx_k]:
                break
        idx_now = idx_last + n_eff

        # info of the first step in the window labels the transition (is_demo etc.).
        info = self.data[field.INFO][idx_last + 1]
        info = dict(info) if isinstance(info, dict) else {}
        info["n_step_effective"] = n_eff
        return (
            self.data[field.OBSERVATIONS][idx_last],
            self._stored_action(idx_last + 1),
            np.float32(n_step_reward),
            self.data[field.OBSERVATIONS][idx_now],
            self.data[field.TERMINATED][idx_now],
            self.data[field.TRUNCATED][idx_now],
            info,
        )


@MEMORIES.register("tm_base")
class MemoryTM(TorchMemory):
    """Base class for TrackMania replay memories with temporal structure."""

    #: Index into ``self.data`` for per-step ``info`` dicts
    #: (subclasses must set if demo mixing applies).
    info_field_index: int | None = None

    def __init__(
        self,
        memory_size: int | None = None,
        batch_size: int | None = None,
        dataset_path: str = "",
        imgs_obs: int = 4,
        act_buf_len: int = 1,
        nb_steps: int = 1,
        sample_preprocessor: Callable[..., Any] | None = None,
        crc_debug: bool = False,
        device: str = "cpu",
        discrete_n_steer_bins: int = 0,
        demo_min_batch_fraction: float = 0.0,
        demo_max_batch_fraction: float = 1.0,
        n_step_return: int = 1,
    ):
        """Initialize MemoryTM.

        Computes ``min_samples``, ``start_imgs_offset``, and ``start_acts_offset``
        from ``imgs_obs`` and ``act_buf_len`` before calling ``super().__init__``.
        These offsets govern the history-window alignment used in
        ``get_transition`` implementations.

        Args:
            memory_size: Maximum number of transitions in the circular buffer.
            batch_size: Number of transitions per sampled batch.
            dataset_path: Path to an offline dataset pickle to preload on init.
            imgs_obs: Number of consecutive image frames per observation.
            act_buf_len: Number of past actions included per observation.
            nb_steps: Number of sampling steps per training round.
            sample_preprocessor: Optional data-augmentation callable.
            crc_debug: When ``True``, run CRC integrity checks on each sample.
            device: Target device for the collated output tensors.
            discrete_n_steer_bins: Number of steer bins for the discrete action
                pipeline; ``0`` = continuous actions.
            demo_min_batch_fraction: Minimum demo fraction of each sampled batch.
            demo_max_batch_fraction: Maximum demo fraction of each sampled batch.
            n_step_return: Number of consecutive steps for n-step TD returns.
        """
        configure_discrete_steer_bins(discrete_n_steer_bins)
        self.discrete_n_steer_bins = int(discrete_n_steer_bins)
        self.imgs_obs = imgs_obs
        self.act_buf_len = act_buf_len
        self.min_samples = max(self.imgs_obs, self.act_buf_len)
        self.start_imgs_offset = max(0, self.min_samples - self.imgs_obs)
        self.start_acts_offset = max(0, self.min_samples - self.act_buf_len)
        self.demo_min_batch_fraction = max(0.0, min(1.0, float(demo_min_batch_fraction)))
        self.demo_max_batch_fraction = max(0.0, min(1.0, float(demo_max_batch_fraction)))
        if self.demo_max_batch_fraction < self.demo_min_batch_fraction:
            self.demo_max_batch_fraction = self.demo_min_batch_fraction
        self.last_sample_demo_fraction = 0.0
        self._demo_flags_cache: np.ndarray | None = None
        self._demo_flags_cache_len: int = 0
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

    def append_buffer(self, buffer):
        """Append a buffer of samples - must be implemented by subclasses."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Return the number of valid transitions in memory."""
        if len(self.data) == 0:
            return 0
        res = len(self.data[0]) - self.min_samples - 1
        return max(0, res)

    @staticmethod
    def _is_demo_info_entry(info_entry: Any) -> bool:
        """Return True if an info dict carries ``is_demo=True``.

        Args:
            info_entry: Entry from the info column of ``self.data``.

        Returns:
            bool: ``True`` when ``info_entry`` is a dict with ``is_demo`` truthy.
        """
        if not isinstance(info_entry, dict):
            return False
        return bool(info_entry.get("is_demo", False))

    def _info_field_index(self) -> int | None:
        """Return the validated info-column index, or ``None`` when unavailable.

        Reads ``self.info_field_index``, casts to ``int``, and guards against
        out-of-range values given the current ``self.data`` length.

        Returns:
            int | None: Validated column index, or ``None`` when
                ``info_field_index`` is ``None`` or ``self.data`` is too short.
        """
        idx = self.info_field_index
        if idx is None:
            return None
        idx = int(idx)
        if len(self.data) == 0 or idx < 0 or idx >= len(self.data):
            return None
        return idx

    def _item_is_demo(self, item: int) -> bool:
        """Return True if the transition at ``item`` is labelled as a demo.

        Looks up ``self.data[info_field_index][item + min_samples]`` and
        delegates to :meth:`_is_demo_info_entry`.

        Args:
            item: Transition item index in ``[0, len(self))``.

        Returns:
            bool: ``True`` when the transition is a demo, ``False`` otherwise
                (including when the info column is unavailable or out of range).
        """
        info_field_index = self._info_field_index()
        if info_field_index is None:
            return False
        idx_now = item + self.min_samples
        info_stream = self.data[info_field_index]
        if idx_now < 0 or idx_now >= len(info_stream):
            return False
        return self._is_demo_info_entry(info_stream[idx_now])

    def _set_last_sample_demo_fraction(self, indices, flags: np.ndarray | None = None) -> None:
        """Update ``self.last_sample_demo_fraction`` from a sample of indices.

        Args:
            indices: Sampled item indices.
            flags: Optional pre-computed boolean demo-flag array indexed by item.
                When provided, the fraction is computed as a vectorised mean.
                When ``None``, each index is looked up individually via
                :meth:`_item_is_demo`.
        """
        if len(indices) == 0:
            self.last_sample_demo_fraction = 0.0
            return
        if flags is not None:
            self.last_sample_demo_fraction = float(flags[indices].mean())
        else:
            demo_count = sum(1 for idx in indices if self._item_is_demo(int(idx)))
            self.last_sample_demo_fraction = float(demo_count) / float(len(indices))

    def sample_indices(self):
        """Sample transition indices, enforcing the configured demo batch fraction.

        Draws ``batch_size`` uniform random indices from ``[0, length)`` and,
        when a valid ``info_field_index`` is present and the demo fraction bounds
        are non-trivial, calls :func:`enforce_demo_batch_fraction` to swap
        indices until the demo share of the batch is within
        ``[demo_min_batch_fraction, demo_max_batch_fraction]``.

        Returns:
            numpy.ndarray | tuple: Array of ``int64`` item indices, or ``()``
                when the buffer is empty.
        """
        length = len(self)
        if length <= 0:
            self.last_sample_demo_fraction = 0.0
            return ()

        demo_min = self.demo_min_batch_fraction
        demo_max = self.demo_max_batch_fraction
        result = np.random.randint(0, length, size=self.batch_size, dtype=np.int64)
        if self._info_field_index() is not None and (demo_min > 0.0 or demo_max < 1.0):
            if self._demo_flags_cache is None or self._demo_flags_cache_len != length:
                self._demo_flags_cache = np.fromiter(
                    (self._item_is_demo(i) for i in range(length)), dtype=bool, count=length
                )
                self._demo_flags_cache_len = length
            result = enforce_demo_batch_fraction(result, self._demo_flags_cache, demo_min, demo_max)
            self._set_last_sample_demo_fraction(result, flags=self._demo_flags_cache)
        else:
            self._set_last_sample_demo_fraction(result)
        return result

    def __getitem__(self, item):
        """Retrieve a transition and apply discrete action normalization.

        Delegates to :meth:`~tmrl.memory.base.Memory.__getitem__` and then
        converts ``new_act`` to the canonical ``(3,)`` float32 replay format
        via :func:`canonical_replay_action_vector`.

        Args:
            item: Transition index.

        Returns:
            tuple: ``(prev_obs, new_act, rew, new_obs, terminated, truncated, info)``
                with ``new_act`` as a ``(3,)`` float32 array.
        """
        prev_obs, new_act, rew, new_obs, terminated, truncated, info = super().__getitem__(item)
        new_act = canonical_replay_action_vector(new_act, self.discrete_n_steer_bins)
        return prev_obs, new_act, rew, new_obs, terminated, truncated, info

    def get_transition(self, item: int):
        """Return a transition tuple at position ``item`` — implemented by subclasses.

        Subclasses must read ``self.data`` fields using the appropriate enum and
        handle episode-boundary padding via
        :func:`~tmrl.custom.memories.base.replace_hist_before_eoe`.

        Args:
            item: Transition item index in ``[0, len(self))``.

        Returns:
            tuple: ``(prev_obs, new_act, rew, new_obs, terminated, truncated, info)``.

        Raises:
            NotImplementedError: Always — must be overridden by subclasses.
        """
        raise NotImplementedError
