"""Unit tests for GenericTorchMemory: memory-side n-step returns and demo batch fractions."""

from __future__ import annotations

import numpy as np
import pytest
from tmrl.custom.memories.base import GenericTorchMemory, enforce_demo_batch_fraction
from tmrl.custom.memories.enums import GenericField
from tmrl.networking import Buffer


def _entry(
    rew: float,
    terminated: bool = False,
    truncated: bool = False,
    is_demo: bool = False,
    obs_val: float = 0.0,
):
    """One replay entry in TMRL convention: (act, obs, rew, terminated, truncated, info)."""
    obs = (np.full(4, obs_val, dtype=np.float32),)
    info = {"is_demo": True} if is_demo else {}
    return (np.int64(1), obs, np.float32(rew), bool(terminated), bool(truncated), info)


def _make_memory(n_step_return: int = 1, gamma: float | None = None, batch_size: int = 4):
    """Build a GenericTorchMemory with default settings and optional n-step configuration.

    Args:
        n_step_return: n-step window size (1 = standard 1-step).
        gamma: Discount factor; required when n_step_return > 1.
        batch_size: Number of transitions per sampled batch.

    Returns:
        An unpopulated ``GenericTorchMemory`` instance.
    """
    return GenericTorchMemory(
        memory_size=10_000,
        batch_size=batch_size,
        dataset_path="",
        nb_steps=1,
        sample_preprocessor=None,
        crc_debug=False,
        device="cpu",
        discrete_n_steer_bins=0,
        n_step_return=n_step_return,
        gamma=gamma,
    )


def _fill(memory: GenericTorchMemory, entries) -> None:
    """Append a list of sample entries to memory via a temporary Buffer.

    Args:
        memory: Target memory to populate.
        entries: Sequence of ``(action, obs, reward, terminated, truncated, info)`` tuples.
    """
    buf = Buffer()
    for e in entries:
        buf.append_sample(e)
    memory.append_buffer(buf)


def _two_episode_entries(terminated_mid: bool = True):
    """Episode 1: rewards 1,2,3,4 ending at data index 4; episode 2: rewards 10,20,30,40.

    Data index:   0     1    2    3    4(done)   5     6     7     8     9
    Reward:       0     1    2    3    4         0     10    20    30    40
    """
    end_kwargs = {"terminated": True} if terminated_mid else {"truncated": True}
    return [
        _entry(0.0, obs_val=0.0),
        _entry(1.0, obs_val=1.0),
        _entry(2.0, obs_val=2.0),
        _entry(3.0, obs_val=3.0),
        _entry(4.0, obs_val=4.0, **end_kwargs),
        _entry(0.0, obs_val=5.0),
        _entry(10.0, obs_val=6.0),
        _entry(20.0, obs_val=7.0),
        _entry(30.0, obs_val=8.0),
        _entry(40.0, obs_val=9.0),
    ]


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_nstep_requires_gamma():
    """Constructing GenericTorchMemory with n_step_return > 1 and gamma=None raises ValueError."""
    with pytest.raises(ValueError, match="gamma"):
        _make_memory(n_step_return=3, gamma=None)


def test_nstep_rejects_crc_debug():
    """n-step mode is incompatible with crc_debug=True and raises ValueError at construction."""
    with pytest.raises(ValueError, match="crc_debug"):
        GenericTorchMemory(n_step_return=3, gamma=0.9, crc_debug=True)


def test_nstep_rejects_non_positive():
    """n_step_return=0 is rejected at construction time with a ValueError."""
    with pytest.raises(ValueError, match="n_step_return"):
        _make_memory(n_step_return=0)


# ---------------------------------------------------------------------------
# 1-step path (unchanged semantics + n_step_effective metadata)
# ---------------------------------------------------------------------------


def test_one_step_transition_layout():
    """1-step get_transition returns the correct 7-tuple layout with n_step_effective=1 in info."""
    memory = _make_memory(n_step_return=1)
    _fill(memory, _two_episode_entries())

    prev_obs, act, rew, new_obs, terminated, truncated, info = memory.get_transition(1)
    assert float(prev_obs[0][0]) == 1.0
    assert float(new_obs[0][0]) == 2.0
    assert float(rew) == 2.0
    assert not terminated and not truncated
    assert info["n_step_effective"] == 1


def test_one_step_getitem_passes_metadata_through():
    """__getitem__ surfaces n_step_effective and is_demo in the returned info dict."""
    memory = _make_memory(n_step_return=1)
    _fill(memory, _two_episode_entries())

    *_, info = memory[1]
    assert info["n_step_effective"] == 1
    assert info["is_demo"] is False


# ---------------------------------------------------------------------------
# n-step accumulation
# ---------------------------------------------------------------------------


def test_nstep_full_window_mid_episode():
    """3-step return accumulates the correct discounted sum when no episode boundary intervenes."""
    gamma = 0.5
    memory = _make_memory(n_step_return=3, gamma=gamma)
    _fill(memory, _two_episode_entries())

    prev_obs, _act, rew, new_obs, terminated, truncated, info = memory.get_transition(0)
    assert float(prev_obs[0][0]) == 0.0
    # R = r1 + 0.5*r2 + 0.25*r3 = 1 + 1 + 0.75
    assert float(rew) == pytest.approx(1.0 + 0.5 * 2.0 + 0.25 * 3.0)
    assert float(new_obs[0][0]) == 3.0
    assert not terminated and not truncated
    assert info["n_step_effective"] == 3


def test_nstep_stops_at_termination():
    """The n-step window truncates at a terminated step and sets terminated=True in the output."""
    gamma = 0.5
    memory = _make_memory(n_step_return=3, gamma=gamma)
    _fill(memory, _two_episode_entries(terminated_mid=True))

    # Start at item 2: steps are data indices 3 (rew 3) and 4 (rew 4, terminated).
    _prev, _act, rew, new_obs, terminated, truncated, info = memory.get_transition(2)
    assert float(rew) == pytest.approx(3.0 + 0.5 * 4.0)
    assert float(new_obs[0][0]) == 4.0
    assert bool(terminated) and not bool(truncated)
    assert info["n_step_effective"] == 2


def test_nstep_stops_at_truncation():
    """The n-step window truncates at a truncated step analogously to termination."""
    memory = _make_memory(n_step_return=3, gamma=0.5)
    _fill(memory, _two_episode_entries(terminated_mid=False))

    _prev, _act, rew, _new_obs, terminated, truncated, info = memory.get_transition(2)
    assert float(rew) == pytest.approx(3.0 + 0.5 * 4.0)
    assert not bool(terminated) and bool(truncated)
    assert info["n_step_effective"] == 2


def test_nstep_never_leaks_across_episodes():
    """Windows starting in episode 1 must stop at the boundary, never touching episode 2."""
    memory = _make_memory(n_step_return=3, gamma=1.0)
    _fill(memory, _two_episode_entries())

    # item -> (expected reward sum, expected n_eff, expected terminated)
    expected = {
        0: (1.0 + 2.0 + 3.0, 3, False),
        1: (2.0 + 3.0 + 4.0, 3, True),  # window ends exactly on the terminal step
        2: (3.0 + 4.0, 2, True),
        3: (4.0, 1, True),
    }
    for item, (exp_rew, exp_n_eff, exp_term) in expected.items():
        _prev, _act, rew, _obs, terminated, _trunc, info = memory.get_transition(item)
        assert float(rew) == pytest.approx(exp_rew)
        assert info["n_step_effective"] == exp_n_eff
        assert bool(terminated) is exp_term


def test_nstep_resamples_terminal_start_within_bounds():
    """get_transition resamples away from terminal-start items, staying within the valid window."""
    memory = _make_memory(n_step_return=3, gamma=0.5)
    _fill(memory, _two_episode_entries())
    np.random.seed(0)

    max_start = memory._max_start_item()
    for _ in range(50):
        # Item 4 is a terminal start; get_transition must resample a valid one.
        prev_obs, *_ = memory.get_transition(4)
        assert float(prev_obs[0][0]) != 4.0
    assert max_start == len(memory) - memory.n_step_return + 1


def test_nstep_sample_indices_respect_window_bound():
    """sample_indices never draws from positions where the n-step window cannot complete."""
    memory = _make_memory(n_step_return=3, gamma=0.5, batch_size=64)
    _fill(memory, _two_episode_entries())
    np.random.seed(1)

    for _ in range(20):
        indices = memory.sample_indices()
        assert len(indices) == 64
        assert all(0 <= i < memory._max_start_item() for i in indices)


def test_nstep_full_sample_collates_metadata():
    """sample() returns a 7-element batch tuple with n_step_effective collated as a tensor."""
    memory = _make_memory(n_step_return=3, gamma=0.5, batch_size=8)
    _fill(memory, _two_episode_entries())
    np.random.seed(2)

    batch = memory.sample()
    assert len(batch) == 7
    info = batch[6]
    n_eff = info["n_step_effective"]
    assert n_eff.shape == (8,)
    assert bool(((n_eff >= 1) & (n_eff <= 3)).all())


# ---------------------------------------------------------------------------
# Episode boundary sealing (demo injection seam)
# ---------------------------------------------------------------------------


def test_mark_episode_boundary_seals_seam():
    """mark_episode_boundary seals the worker/demo seam so n-step windows never leak across it."""
    memory = _make_memory(n_step_return=3, gamma=1.0)
    worker_entries = [_entry(1.0, obs_val=float(i)) for i in range(6)]
    _fill(memory, worker_entries)

    memory.mark_episode_boundary()
    demo_entries = [_entry(100.0, is_demo=True, obs_val=100.0 + i) for i in range(6)]
    _fill(memory, demo_entries)

    seam_idx = 5  # last worker entry
    assert memory.data[GenericField.DONE][seam_idx] is True
    assert memory.data[GenericField.TRUNCATED][seam_idx] is True

    # Windows starting before the seam stop at it: no demo rewards (100) leak in.
    for item in range(seam_idx):
        _prev, _act, rew, *_ = memory.get_transition(item)
        assert float(rew) < 100.0


# ---------------------------------------------------------------------------
# Demo batch fraction enforcement
# ---------------------------------------------------------------------------


def _demo_mixed_memory(batch_size: int, demo_min: float, demo_max: float):
    """Build a GenericTorchMemory with 200 regular and 50 demo entries for demo-fraction tests.

    Args:
        batch_size: Number of transitions per sampled batch.
        demo_min: Minimum demo fraction enforced during sampling.
        demo_max: Maximum demo fraction enforced during sampling.

    Returns:
        A populated ``GenericTorchMemory`` with demo-fraction constraints configured.
    """
    memory = GenericTorchMemory(
        memory_size=10_000,
        batch_size=batch_size,
        device="cpu",
        n_step_return=1,
        demo_min_batch_fraction=demo_min,
        demo_max_batch_fraction=demo_max,
    )
    entries = [_entry(1.0, obs_val=float(i)) for i in range(200)]
    entries += [_entry(2.0, is_demo=True, obs_val=500.0 + i) for i in range(50)]
    _fill(memory, entries)
    return memory


def test_demo_fraction_floor_and_cap():
    """sample_indices keeps the demo fraction within [demo_min, demo_max] on every draw."""
    np.random.seed(3)
    memory = _demo_mixed_memory(batch_size=100, demo_min=0.2, demo_max=0.4)

    for _ in range(20):
        indices = memory.sample_indices()
        flags = memory._item_demo_flags(memory._max_start_item())
        demo_count = int(flags[indices].sum())
        assert 20 <= demo_count <= 40
        assert memory.last_sample_demo_fraction == pytest.approx(demo_count / 100.0)


def test_demo_fraction_disabled_keeps_uniform_sampling():
    """With demo_min=0 and demo_max=1 sampling is unconstrained; the fraction is still logged."""
    np.random.seed(4)
    memory = _demo_mixed_memory(batch_size=100, demo_min=0.0, demo_max=1.0)
    indices = memory.sample_indices()
    assert len(indices) == 100
    # Fraction logging still works without enforcement.
    assert 0.0 <= memory.last_sample_demo_fraction <= 1.0


def test_demo_fraction_floor_with_scarce_demos():
    """Floor enforcement may need replacement with repetition when demos are scarce."""
    np.random.seed(5)
    memory = GenericTorchMemory(
        memory_size=10_000,
        batch_size=50,
        device="cpu",
        n_step_return=1,
        demo_min_batch_fraction=0.5,
        demo_max_batch_fraction=1.0,
    )
    entries = [_entry(1.0) for _ in range(100)]
    entries += [_entry(2.0, is_demo=True) for _ in range(5)]
    _fill(memory, entries)

    indices = memory.sample_indices()
    flags = memory._item_demo_flags(memory._max_start_item())
    assert int(flags[indices].sum()) >= 25


def test_enforce_demo_batch_fraction_noop_without_demos():
    flags = np.zeros(100, dtype=bool)
    result = np.arange(10, dtype=np.int64)
    out = enforce_demo_batch_fraction(result, flags, 0.2, 0.5)
    assert np.array_equal(out, np.arange(10))


def test_demo_flags_cache_invalidated_on_append():
    memory = _demo_mixed_memory(batch_size=10, demo_min=0.0, demo_max=1.0)
    flags_before = memory._demo_flags()
    n_before = int(flags_before.sum())
    _fill(memory, [_entry(1.0, is_demo=True) for _ in range(10)])
    flags_after = memory._demo_flags()
    assert int(flags_after.sum()) == n_before + 10


# ---------------------------------------------------------------------------
# Discrete action canonicalization (demo injection vs worker rollouts)
# ---------------------------------------------------------------------------


def _discrete_memory(batch_size: int = 8) -> GenericTorchMemory:
    """Build a GenericTorchMemory configured for 13 discrete steering bins.

    Args:
        batch_size: Number of transitions per sampled batch.

    Returns:
        An unpopulated ``GenericTorchMemory`` with discrete-action quantization enabled.
    """
    return GenericTorchMemory(
        memory_size=10_000,
        batch_size=batch_size,
        device="cpu",
        discrete_n_steer_bins=13,
        n_step_return=1,
    )


def _continuous_entry(control, rew: float = 1.0, is_demo: bool = True):
    """Build a sample with a (3,) continuous gas/brake/steer control vector.

    Args:
        control: Gas/brake/steer triple as a list or array.
        rew: Reward value.
        is_demo: Whether to flag this entry as a demonstration in info.

    Returns:
        A ``(action, obs, reward, terminated, truncated, info)`` tuple.
    """
    obs = (np.zeros(4, dtype=np.float32),)
    info = {"is_demo": is_demo}
    return (np.asarray(control, dtype=np.float32), obs, np.float32(rew), False, False, info)


def test_mixed_demo_and_worker_actions_are_homogeneous_and_collate():
    """Worker int indices + demo (3,) controls must coexist in one batch (the
    pre-fix behavior crashed collate_torch on the first mixed batch)."""
    memory = _discrete_memory(batch_size=16)
    worker_entries = [_entry(1.0, obs_val=float(i)) for i in range(20)]
    demo_entries = [
        _continuous_entry([1.0, 0.0, 0.0]),  # full gas, straight
        _continuous_entry([0.0, 1.0, -1.0]),  # full brake, full left
        _continuous_entry([1.0, 0.0, 1.0]),  # full gas, full right
    ] * 5
    _fill(memory, worker_entries)
    _fill(memory, demo_entries)

    from tmrl.custom.memories.enums import GenericField as GF

    for a in memory.data[GF.ACTIONS]:
        arr = np.asarray(a)
        assert arr.ndim == 0 and np.issubdtype(arr.dtype, np.integer)

    np.random.seed(6)
    batch = memory.sample()  # would raise ValueError in collate_torch before the fix
    assert batch[1].shape == (16,)


def test_demo_action_quantization_roundtrip():
    """Quantized demo actions map back to controls close to the originals."""
    from tmrl.custom.tm.utils.control.discrete import (
        build_brake_tap_action_table,
        discrete_index_to_control,
    )

    memory = _discrete_memory()
    controls = [[1.0, 0.0, 0.0], [0.0, 1.0, -1.0], [1.0, 0.0, 1.0]]
    _fill(memory, [_continuous_entry(c) for c in controls])

    _, table = build_brake_tap_action_table(n_steer=13)
    from tmrl.custom.memories.enums import GenericField as GF

    for stored, original in zip(memory.data[GF.ACTIONS], controls, strict=True):
        recovered = discrete_index_to_control(int(stored), table)
        assert recovered[0] == pytest.approx(original[0])  # gas
        assert recovered[1] == pytest.approx(original[1])  # brake (never tap sentinel)
        assert abs(float(recovered[2]) - original[2]) <= 1.0 / 12.0 + 1e-6  # steer bin width


def test_invalid_action_shape_raises():
    """Actions with wrong shape (not scalar int or (3,) float) raise ValueError on append."""
    memory = _discrete_memory()
    bad = (
        np.zeros(5, dtype=np.float32),
        (np.zeros(4, dtype=np.float32),),
        np.float32(0.0),
        False,
        False,
        {},
    )
    buf = Buffer()
    buf.append_sample(bad)
    with pytest.raises(ValueError, match="neither a"):
        memory.append_buffer(buf)


def test_continuous_pipeline_actions_untouched():
    """With discrete_n_steer_bins=0 (SAC/TQC), (3,) actions stay continuous."""
    memory = GenericTorchMemory(memory_size=100, batch_size=2, device="cpu")
    _fill(memory, [_continuous_entry([0.5, 0.0, 0.2], is_demo=False) for _ in range(4)])
    from tmrl.custom.memories.enums import GenericField as GF

    for a in memory.data[GF.ACTIONS]:
        assert np.asarray(a).shape == (3,)


def test_legacy_checkpoint_actions_healed_on_read():
    """Buffers pickled before append-time quantization may hold (3,) demo rows;
    get_transition must heal them in place instead of crashing collate."""
    from tmrl.custom.memories.enums import GenericField as GF

    memory = _discrete_memory(batch_size=4)
    _fill(memory, [_entry(1.0, obs_val=float(i)) for i in range(6)])
    # Simulate a legacy checkpoint: overwrite stored actions with raw continuous rows.
    memory.data[GF.ACTIONS][2] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    memory.data[GF.ACTIONS][3] = np.array([0.0, 1.0, -1.0], dtype=np.float32)

    for item in (1, 2):
        _prev, act, *_ = memory.get_transition(item)
        arr = np.asarray(act)
        assert arr.ndim == 0 and np.issubdtype(arr.dtype, np.integer)

    stored = np.asarray(memory.data[GF.ACTIONS][2])
    assert stored.ndim == 0 and np.issubdtype(stored.dtype, np.integer)
