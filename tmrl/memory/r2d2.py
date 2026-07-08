"""R2D2Memory: episode-aware, PER-enabled memory for recurrent RL."""

import random
from abc import ABC
from collections.abc import Callable
from typing import Any

import numpy as np
from loguru import logger

from tmrl.custom.memories._internal.sampling_utils import (
    canonical_replay_action_vector,
    configure_discrete_steer_bins,
)
from tmrl.memory.base import Memory
from tmrl.util import collate_torch


class R2D2Memory(Memory, ABC):
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
        rewards_index: int = 18,
        r2d2_rewind: float = 0.5,
        per_td_enabled: bool = False,
        per_td_alpha: float = 0.6,
        per_td_beta: float = 0.4,
        per_td_eps: float = 1e-6,
        r2d2_num_sequences: int = 0,
        r2d2_sequence_length: int = 0,
        player_runs_per_alpha: float = 0.0,
        fog_decay_temperature: float = 0.0,
        demo_min_batch_fraction: float = 0.0,
        demo_max_batch_fraction: float = 1.0,
        discrete_n_steer_bins: int = 0,
        n_step_return: int = 1,
    ):
        configure_discrete_steer_bins(discrete_n_steer_bins)
        self.discrete_n_steer_bins = int(discrete_n_steer_bins)
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
        self.rewards_index = rewards_index
        self.previous_episode = 0
        self.end_episodes_indices: list[int] = []
        self.chosen_episode = 0
        self.burn_ins = (20, 40)
        self.isNewEpisode = True
        self.chosen_burn_in = 0
        self.reward_sums: list[float] = []
        self.episode_demo_flags: list[bool] = []
        self.indices: list[int] = []
        self.cur_idx = 0
        self.batch_size = batch_size
        self.rewind = r2d2_rewind
        assert 0.1 <= self.rewind <= 0.9, "R2D2 REWIND CONST SHOULD BE BETWEEN 0.1 AND 0.9"
        self.last_sample_demo_fraction = 0.0
        if not hasattr(self, "min_samples"):
            self.min_samples = 0
        self._episode_metadata_dirty = True
        self.per_td_enabled = per_td_enabled
        self.per_td_alpha = per_td_alpha
        self.per_td_beta = per_td_beta
        self.per_td_eps = per_td_eps
        self.r2d2_num_sequences = r2d2_num_sequences
        self.r2d2_sequence_length = r2d2_sequence_length
        self.player_runs_per_alpha = player_runs_per_alpha
        self.fog_decay_temperature = fog_decay_temperature
        self.demo_min_batch_fraction = demo_min_batch_fraction
        self.demo_max_batch_fraction = demo_max_batch_fraction
        self.priorities: list[float] = []
        self._last_per_is_weights: list[float] = []
        if len(self.data) > 0 and len(self.data[0]) > 0:
            self.priorities = [1.0] * len(self.data[0])

    def __getitem__(self, item):
        prev_obs, new_act, rew, new_obs, terminated, truncated, info = super().__getitem__(item)
        new_act = canonical_replay_action_vector(new_act, self.discrete_n_steer_bins)
        return prev_obs, new_act, rew, new_obs, terminated, truncated, info

    def _extend_priorities(self, n: int) -> None:
        """Extend priorities for n new buffer entries (max of current or 1.0)."""
        if not self.per_td_enabled:
            return
        max_p = max(self.priorities) if self.priorities else 1.0
        self.priorities.extend([max_p] * n)

    def _trim_priorities(self, to_trim: int) -> None:
        """Trim the first to_trim entries from priorities (after buffer trim)."""
        if self.per_td_enabled and self.priorities and to_trim > 0:
            self.priorities = self.priorities[to_trim:]

    def clear(self) -> None:
        """Remove all transitions and reset R2D2 episode/priority state."""
        self.data = []
        self.end_episodes_indices = []
        self.reward_sums = []
        self.episode_demo_flags = []
        self.priorities = []
        self._last_per_is_weights = []
        self._episode_metadata_dirty = True
        self.last_sample_demo_fraction = 0.0

    def update_priorities(self, indices: tuple[int, ...], td_errors: np.ndarray) -> None:
        """Update priorities for the given transition indices (item space) using TD errors."""
        if not self.per_td_enabled or not self.priorities:
            return
        eps = self.per_td_eps
        td_flat = np.asarray(td_errors).flatten()
        min_samp = getattr(self, "min_samples", 0)
        for k, item in enumerate(indices):
            if k >= len(td_flat):
                break
            buf_idx = item + min_samp
            if 0 <= buf_idx < len(self.priorities):
                self.priorities[buf_idx] = float(np.abs(td_flat[k]) + eps)

    def _sample_indices_td_priority(self) -> tuple[int, ...] | None:
        """Sample indices with probability proportional to (sum of priorities)^PER_TD_ALPHA."""
        if not self.per_td_enabled or not self.priorities:
            self._last_per_is_weights = []
            return None
        num_seq = self.r2d2_num_sequences
        seq_len = self.r2d2_sequence_length
        if num_seq <= 0 or seq_len <= 0 or num_seq * seq_len != self.batch_size:
            self._last_per_is_weights = []
            return None
        self._refresh_episode_metadata()
        if len(self.end_episodes_indices) == 0:
            self._last_per_is_weights = []
            return None
        min_samp = getattr(self, "min_samples", 0)
        max_item = len(self) - 1
        if max_item < seq_len - 1:
            self._last_per_is_weights = []
            return None
        alpha = self.per_td_alpha
        eps = self.per_td_eps
        prev_end_buf = -1
        valid_starts: list[tuple[int, int]] = []
        seq_weights: list[float] = []
        for end_buf in self.end_episodes_indices:
            start_item = max(0, prev_end_buf + 1 - min_samp)
            end_item = min(end_buf - min_samp, max_item)
            for start in range(start_item, end_item - seq_len + 2):
                if start + seq_len - 1 <= end_item:
                    buf_start = start + min_samp
                    priority_sum = sum(
                        self.priorities[buf_start + j]
                        for j in range(seq_len)
                        if buf_start + j < len(self.priorities)
                    )
                    valid_starts.append((start, start + seq_len))
                    seq_weights.append((priority_sum + eps) ** alpha)
            prev_end_buf = end_buf
        if len(valid_starts) == 0:
            self._last_per_is_weights = []
            return None
        total_w = sum(seq_weights)
        if total_w <= 0:
            seq_weights = [1.0] * len(valid_starts)
            total_w = len(seq_weights)
        probs = [w / total_w for w in seq_weights]
        chosen = np.random.choice(len(valid_starts), size=num_seq, replace=True, p=probs)
        indices: list[int] = []
        beta = float(self.per_td_beta)
        n_sequences = max(1, len(valid_starts))
        is_weights: list[float] = []
        # Normalize by max weight in batch to bound gradients; do not use mean/sum normalization.
        for i in chosen:
            s, e = valid_starts[i]
            indices.extend(range(s, e))
            p_i = max(float(probs[i]), 1e-12)
            w_i = (1.0 / (n_sequences * p_i)) ** beta
            is_weights.extend([w_i] * seq_len)
        self._last_per_is_weights = is_weights
        result = tuple(indices)
        self._set_last_sample_demo_fraction(result)
        return result

    @staticmethod
    def _is_demo_info_entry(info_entry: Any) -> bool:
        if not isinstance(info_entry, dict):
            return False
        value = info_entry.get("is_demo", False)
        return bool(value)

    def _set_last_sample_demo_fraction(self, indices: tuple[int, ...]) -> None:
        if len(indices) == 0 or len(self.data) <= self.rewards_index:
            self.last_sample_demo_fraction = 0.0
            return
        info_index = self.rewards_index + 1
        if info_index < len(self.data):
            info_stream = self.data[info_index]
        else:
            info_stream = self.data[self.rewards_index]
        demo_count = 0
        total_count = 0
        for idx in indices:
            idx_now = idx + self.min_samples
            if 0 <= idx_now < len(info_stream):
                total_count += 1
                if self._is_demo_info_entry(info_stream[idx_now]):
                    demo_count += 1
        self.last_sample_demo_fraction = (
            float(demo_count) / float(total_count) if total_count > 0 else 0.0
        )

    def collate(self, batch, device):
        """
        Method in Memory and its subclasses.
        Used to collate a batch of data onto a specified device.
        Calls an external function collate_torch and returns its result.
        """
        return collate_torch(batch, device)

    @staticmethod
    def find_zero_rewards_indices(reward_sums):
        """
        Finds indices where reward sum transitions from non-zero to zero.
        reward_sums can be a list of floats or a list of dicts with "reward_sum" key.
        """
        zero_rewards_indices = []
        prev_reward_sum = None

        for i, entry in enumerate(reward_sums):
            if isinstance(entry, dict):
                reward_sum = float(entry.get("reward_sum", 0.0))
            else:
                reward_sum = float(entry)
            if prev_reward_sum is not None and reward_sum == 0.0 and prev_reward_sum != 0.0:
                zero_rewards_indices.append(i - 1)

            prev_reward_sum = reward_sum

        return zero_rewards_indices

    def _refresh_episode_metadata(self) -> None:
        """Refresh cached episode metadata once per buffer append, not per sampled batch."""
        if not self._episode_metadata_dirty:
            return
        if len(self.data) <= self.rewards_index:
            self.end_episodes_indices = []
            self.reward_sums = []
            self.episode_demo_flags = []
            self._episode_metadata_dirty = False
            return
        reward_stream = self.data[self.rewards_index]
        self.end_episodes_indices = self.find_zero_rewards_indices(reward_stream)
        self.reward_sums = []
        for index in self.end_episodes_indices:
            ent = reward_stream[index]
            self.reward_sums.append(
                float(ent["reward_sum"]) if isinstance(ent, dict) else float(ent)
            )
        info_index = self.rewards_index + 1
        if info_index < len(self.data):
            info_stream = self.data[info_index]
            self.episode_demo_flags = [
                self._is_demo_info_entry(info_stream[index]) for index in self.end_episodes_indices
            ]
        else:
            self.episode_demo_flags = [
                self._is_demo_info_entry(reward_stream[index])
                if isinstance(reward_stream[index], dict)
                else False
                for index in self.end_episodes_indices
            ]
        self._episode_metadata_dirty = False

    def append(self, buffer):
        super().append(buffer)
        if len(buffer) > 0:
            self._episode_metadata_dirty = True

    @staticmethod
    def normalize_list(input_list):
        """
        Normalizes a list of values between 0 and 1.
        Handles cases where the range of values is zero to prevent division by zero.
        """
        min_val = min(input_list)
        max_val = max(input_list)

        if min_val == max_val:
            return [0.0] * len(input_list)

        normalized_list = [(x - min_val) / (max_val - min_val) for x in input_list]

        return normalized_list

    def _sample_indices_iid_sequences(self) -> tuple[int, ...] | None:
        """
        Sample B independent sequences of length L from different episodes (i.i.d.).
        Returns None if disabled; caller falls back to contiguous sampling.
        """
        num_seq = self.r2d2_num_sequences
        seq_len = self.r2d2_sequence_length
        if num_seq <= 0 or seq_len <= 0 or num_seq * seq_len != self.batch_size:
            return None
        self._refresh_episode_metadata()
        if len(self.end_episodes_indices) == 0:
            return None
        min_samp = getattr(self, "min_samples", 0)
        max_item = len(self) - 1
        if max_item < seq_len - 1:
            return None
        prev_end_buf = -1
        episode_ranges = []
        episode_rewards = []
        episode_is_demo = []
        for i, end_buf in enumerate(self.end_episodes_indices):
            start_item = max(0, prev_end_buf + 1 - min_samp)
            end_item = min(end_buf - min_samp, max_item)
            if end_item >= start_item and end_item - start_item + 1 >= seq_len:
                episode_ranges.append((start_item, end_item))
                episode_rewards.append(self.reward_sums[i] if i < len(self.reward_sums) else 0.0)
                episode_is_demo.append(
                    self.episode_demo_flags[i] if i < len(self.episode_demo_flags) else False
                )
            prev_end_buf = end_buf
        if len(episode_ranges) == 0:
            return None
        weights = episode_rewards if episode_rewards else [1.0] * len(episode_ranges)
        if sum(weights) <= 0:
            weights = (
                self.normalize_list(weights) if len(weights) > 1 else [1.0] * len(episode_ranges)
            )
        per_alpha = self.player_runs_per_alpha
        if per_alpha > 0:
            _eps = 1e-6
            weights = [(max(0.0, float(w)) + _eps) ** per_alpha for w in weights]
        if sum(weights) <= 0:
            weights = [1.0] * len(episode_ranges)

        fog_temp = float(self.fog_decay_temperature)
        if fog_temp > 0 and len(episode_ranges) > 1:
            n_ep_total = len(episode_ranges)
            recency = np.array([i / max(1, n_ep_total - 1) for i in range(n_ep_total)])
            log_recency = fog_temp * recency
            log_recency -= log_recency.max()
            fog_weights = np.exp(log_recency)
            for i in range(len(weights)):
                weights[i] = float(weights[i]) * float(fog_weights[i])
            w_sum = sum(weights)
            if w_sum <= 0:
                weights = [1.0] * len(episode_ranges)

        # Prefer diverse episodes: sample without replacement when num_seq <= num episodes
        n_ep = len(episode_ranges)
        if num_seq <= n_ep:
            ep_order = random.sample(range(n_ep), num_seq)
        else:
            ep_order = list(random.sample(range(n_ep), n_ep))
            ep_order += random.choices(range(n_ep), weights=weights, k=num_seq - n_ep)
            random.shuffle(ep_order)
        demo_eps = [i for i in range(n_ep) if episode_is_demo[i]]
        non_demo_eps = [i for i in range(n_ep) if not episode_is_demo[i]]

        # Enforce DEMO_MIN_BATCH_FRACTION: guarantee a floor of demo sequences
        demo_min_frac = self.demo_min_batch_fraction
        if demo_min_frac > 0 and demo_eps and non_demo_eps:
            min_demo_seqs = max(1, int(demo_min_frac * num_seq))
            current_demo = [j for j, ep in enumerate(ep_order) if episode_is_demo[ep]]
            if len(current_demo) < min_demo_seqs:
                non_demo_positions = [j for j, ep in enumerate(ep_order) if not episode_is_demo[ep]]
                need = min_demo_seqs - len(current_demo)
                if need > 0 and non_demo_positions:
                    replace_positions = random.sample(
                        non_demo_positions, min(need, len(non_demo_positions))
                    )
                    for pos in replace_positions:
                        ep_order[pos] = random.choice(demo_eps)

        # Enforce DEMO_MAX_BATCH_FRACTION: cap demo sequences
        demo_max_frac = self.demo_max_batch_fraction
        if demo_max_frac < 1.0 and demo_eps:
            max_demo_seqs = max(1, int(demo_max_frac * num_seq))
            if non_demo_eps:
                demo_positions = [j for j, ep in enumerate(ep_order) if episode_is_demo[ep]]
                if len(demo_positions) > max_demo_seqs:
                    for pos in demo_positions[max_demo_seqs:]:
                        ep_order[pos] = random.choice(non_demo_eps)
        indices: list[int] = []
        for ep_idx in ep_order:
            (start_item, end_item) = episode_ranges[ep_idx]
            max_start = end_item - seq_len + 1
            if max_start < start_item:
                continue
            start = random.randint(start_item, max_start) if max_start > start_item else start_item
            indices.extend(range(start, start + seq_len))
        if len(indices) != self.batch_size:
            return None
        result = tuple(indices)
        self._set_last_sample_demo_fraction(result)
        return result

    def sample_indices(self):
        """
        Generates indices for sampling from the memory based on various conditions.
        When PER_TD_ENABLED, samples with probability proportional to
        (sum of TD-error priorities)^alpha.
        Otherwise when R2D2_NUM_SEQUENCES and R2D2_SEQUENCE_LENGTH are set, samples
        B independent sequences of length L (by reward or uniformly).
        """
        if self.per_td_enabled:
            td_result = self._sample_indices_td_priority()
            if td_result is not None:
                return td_result
            self._last_per_is_weights = []
        iid_result = self._sample_indices_iid_sequences()
        if iid_result is not None:
            self._last_per_is_weights = []
            return iid_result
        self._last_per_is_weights = []
        self._refresh_episode_metadata()
        batch_size = self.batch_size

        if len(self.end_episodes_indices) == 0:
            if self.cur_idx == 0:
                self.cur_idx += max(1, int(batch_size * self.rewind))

                result = tuple(range(0, self.cur_idx))
                self._set_last_sample_demo_fraction(result)
                return result
            else:
                if self.cur_idx + batch_size < len(self):
                    result = tuple(range(self.cur_idx, self.cur_idx + batch_size))
                    self.cur_idx += int(batch_size * self.rewind)
                    self._set_last_sample_demo_fraction(result)
                    return result
                else:
                    result = tuple(range(len(self) - batch_size, len(self)))
                    self.cur_idx = 0
                    self._set_last_sample_demo_fraction(result)
                    return result
        else:
            if self.isNewEpisode:
                if len(self.reward_sums) == 1:
                    self.chosen_episode = self.end_episodes_indices[0]
                    self.previous_episode = 0
                else:
                    sampling_weights = (
                        list(self.reward_sums)
                        if sum(self.reward_sums) > 0
                        else self.normalize_list(self.reward_sums)
                    )
                    per_alpha = self.player_runs_per_alpha
                    if per_alpha > 0:
                        _eps = 1e-6
                        sampling_weights = [
                            (max(0.0, float(w)) + _eps) ** per_alpha for w in sampling_weights
                        ]
                    if sum(sampling_weights) <= 0:
                        sampling_weights = [1.0] * len(self.end_episodes_indices)
                    self.chosen_episode = random.choices(
                        self.end_episodes_indices,
                        weights=sampling_weights,
                        k=1,
                    )[0]
                    previous_episode_index = (
                        sorted(self.end_episodes_indices).index(self.chosen_episode) - 1
                    )
                    if previous_episode_index < 0:
                        self.previous_episode = 0
                    else:
                        self.previous_episode = self.end_episodes_indices[previous_episode_index]

                episode_length = self.chosen_episode - self.previous_episode
                self.chosen_burn_in = random.randint(self.burn_ins[0], self.burn_ins[1])

                if episode_length <= batch_size + self.chosen_burn_in:
                    range_length = min(batch_size, episode_length - self.chosen_burn_in)

                    start_idx = max(self.previous_episode, self.chosen_episode - range_length)

                    end_idx = min(self.chosen_episode - 1, start_idx + range_length)

                    if end_idx - start_idx < batch_size:
                        start_idx = max(self.previous_episode, end_idx - batch_size)

                    result = tuple(range(start_idx, end_idx))
                    if len(result) == 0:
                        start_idx = max(self.previous_episode, self.chosen_episode - batch_size)
                        result = tuple(range(start_idx, self.chosen_episode))[:batch_size]
                    self._set_last_sample_demo_fraction(result)
                    return result
                else:
                    if self.previous_episode < 0:
                        self.previous_episode = 0

                    self.cur_idx = self.previous_episode + self.chosen_burn_in
                    result = tuple(range(self.cur_idx, self.cur_idx + batch_size))

                    self.cur_idx += batch_size
                    self.isNewEpisode = False
                    self._set_last_sample_demo_fraction(result)
                    return result
            else:
                self.cur_idx -= int(batch_size * self.rewind)

                if self.cur_idx + batch_size >= self.chosen_episode:
                    self.isNewEpisode = True

                    result = tuple(range(self.chosen_episode - batch_size, self.chosen_episode))
                    self.cur_idx = self.chosen_episode
                    self._set_last_sample_demo_fraction(result)
                    return result
                else:
                    self.isNewEpisode = False
                    result = tuple(range(self.cur_idx, self.cur_idx + batch_size))
                    self.cur_idx += batch_size
                    self._set_last_sample_demo_fraction(result)
                    return result

    def __len__(self):
        if len(self.data) == 0:
            return 0
        res = len(self.data[0]) - self.min_samples - 1
        if res < 0:
            return 0
        else:
            return res

    def sample(self):
        """
        Samples data from the memory using the generated indices from sample_indices.
        Collates the sampled data into a batch using the collate method and returns it.
        When PER_TD_ENABLED, each sample's info dict includes
        'batch_indices' (item index) for priority updates.
        """
        indices = self.sample_indices()
        if len(indices) == 0:
            n = len(self)
            if n == 0:
                raise RuntimeError("Cannot sample from empty replay memory")
            batch_size = min(self.batch_size, n)
            indices = tuple(random.sample(range(n), batch_size))
            if int(getattr(self, "n_step_return", 1)) > 1 and not getattr(
                self, "_warned_iid_nstep", False
            ):
                logger.warning(
                    "R2D2 memory fell back to i.i.d. sampling while n_step_return={} > 1: "
                    "batch rows are not consecutive transitions, so algorithms computing "
                    "n-step returns along the batch axis (TQC/SDSAC) would mix unrelated "
                    "samples. Ensure sequence sampling preconditions hold (num_sequences * "
                    "sequence_length == batch_size and enough complete episodes).",
                    self.n_step_return,
                )
                self._warned_iid_nstep = True
        per_td = self.per_td_enabled
        is_weights = getattr(self, "_last_per_is_weights", [])
        batch = []
        for pos, idx in enumerate(indices):
            prev_obs, new_act, rew, new_obs, terminated, truncated, info = self[idx]
            if per_td and isinstance(info, dict):
                info = {**info, "batch_indices": idx}
                if len(is_weights) == len(indices):
                    info["is_weight"] = np.float32(is_weights[pos])
            batch.append((prev_obs, new_act, rew, new_obs, terminated, truncated, info))
        batch = self.collate(batch, self.device)
        return batch
