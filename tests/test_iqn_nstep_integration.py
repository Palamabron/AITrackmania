"""Integration test: IQN train() consuming memory-side n-step batches (boundary-lidar layout)."""

from __future__ import annotations

import gymnasium.spaces as spaces
import numpy as np
import pytest
from tmrl.custom.custom_algorithms.iqn import IQNAgent
from tmrl.custom.memories.base import GenericTorchMemory
from tmrl.networking import Buffer


def _boundary_obs_space() -> spaces.Tuple:
    """TM2020InterfaceBoundary.get_observation_space() plus the two rtgym
    act_in_obs slots (act_buf_len=2) that production observations carry."""
    return spaces.Tuple(
        (
            spaces.Box(low=-300, high=300, shape=(60,)),
            spaces.Box(low=0.0, high=1000.0, shape=(1,)),
            spaces.Box(low=0.0, high=6, shape=(1,)),
            spaces.Box(low=0.0, high=np.inf, shape=(1,)),
            spaces.Box(low=-100, high=100.0, shape=(1,)),
            spaces.Box(low=-1, high=1.0, shape=(1,)),
            spaces.Box(low=0.0, high=1, shape=(4,)),
            spaces.Box(low=0.0, high=1, shape=(1,)),
            spaces.Box(low=0.0, high=1.0, shape=(1,)),
            spaces.Discrete(78),
            spaces.Discrete(78),
        )
    )


def _random_obs(rng: np.random.Generator) -> tuple:
    return (
        rng.uniform(-1, 1, size=60).astype(np.float32),
        rng.uniform(0, 1, size=1).astype(np.float32),
        rng.uniform(0, 1, size=1).astype(np.float32),
        rng.uniform(0, 1, size=1).astype(np.float32),
        rng.uniform(-1, 1, size=1).astype(np.float32),
        rng.uniform(-1, 1, size=1).astype(np.float32),
        rng.uniform(0, 1, size=4).astype(np.float32),
        np.zeros(1, dtype=np.float32),
        np.zeros(1, dtype=np.float32),
        # rtgym action-buffer slots, scaled to [0, 1] by the obs preprocessor
        np.asarray(rng.uniform(0, 1), dtype=np.float32),
        np.asarray(rng.uniform(0, 1), dtype=np.float32),
    )


def _make_agent(n_steps: int, lr: float = 3.0e-5, **overrides) -> IQNAgent:
    return IQNAgent(
        observation_space=_boundary_obs_space(),
        action_space=spaces.Discrete(78),
        hidden_dim=64,
        num_blocks=1,
        n_cos=16,
        dueling=True,
        n_actions=78,
        n_quantiles_train=8,
        n_quantiles_target=8,
        n_quantiles_eval=8,
        gamma=0.995,
        lr=lr,
        n_steps=n_steps,
        double_dqn=True,
        target_update_freq=100,
        epsilon_schedule_mode="linear",
        epsilon_start=0.3,
        epsilon_end=0.01,
        epsilon_decay_steps=1000.0,
        epsilon_cosine_t0=100.0,
        epsilon_cosine_tmult=1.5,
        epsilon_cosine_decay=0.8,
        epsilon_cosine_initial_amplitude=0.1,
        epsilon_cosine_floor_fraction=0.5,
        epsilon_cosine_floor_steps=0,
        explore_repeat_steps=1,
        weight_decay=0.0,
        adam_eps=1e-8,
        grad_clip=2.0,
        huber_kappa=0.7,
        soft_target_tau=0.005,
        log_target_stats=True,
        sort_quantiles=True,
        monotonicity_regularization=False,
        monotonicity_lambda=0.01,
        munchausen_enabled=False,
        munchausen_alpha=0.02,
        munchausen_tau=0.2,
        munchausen_clip_min=-0.3,
        munchausen_clip_max=0.0,
        iqn_n_steer_bins=13,
        backup_clip_range=0.0,
        reward_normalize_scale=1.0,
        mixed_precision=False,
        mixed_precision_dtype="bf16",
        seed=42,
        device="cpu",
        split_track_observation=True,
        track_encoder="conv1d",
        use_simbav2=True,
        **overrides,
    )


def _filled_memory(
    n_step_return: int,
    batch_size: int,
    demo: bool = False,
    demo_action: int | None = None,
) -> GenericTorchMemory:
    memory = GenericTorchMemory(
        memory_size=10_000,
        batch_size=batch_size,
        device="cpu",
        discrete_n_steer_bins=13,
        n_step_return=n_step_return,
        gamma=0.995,
    )
    rng = np.random.default_rng(7)
    buf = Buffer()
    for i in range(60):
        terminated = i in (29, 59)  # two episode ends inside the stream
        action = np.int64(demo_action if demo_action is not None else rng.integers(0, 78))
        buf.append_sample(
            (
                action,
                _random_obs(rng),
                np.float32(rng.uniform(0.0, 1.0)),
                bool(terminated),
                False,
                {"is_demo": True} if demo else {},
            )
        )
    memory.append_buffer(buf)
    return memory


@pytest.mark.parametrize("n_steps", [1, 3])
def test_iqn_train_step_on_generic_memory_batch(n_steps):
    agent = _make_agent(n_steps=n_steps)
    memory = _filled_memory(n_step_return=n_steps, batch_size=16)
    np.random.seed(11)

    batch = memory.sample()
    assert batch[6]["n_step_effective"].shape == (16,)

    stats = agent.train(batch, 0, 0, len(memory))

    assert np.isfinite(stats["loss/iqn_loss"])
    assert stats.get("debug/nan_detected", 0.0) == 0.0
    assert np.isfinite(stats["q/mean_q"])
    # gamma**n_eff bootstrap: targets must stay in a sane range for ~[0,1] rewards.
    assert abs(stats["debug/target_mean"]) < 100.0


def test_iqn_rejects_invalid_n_steps():
    with pytest.raises(ValueError, match="n_steps"):
        _make_agent(n_steps=0)


def test_batch_info_carries_is_demo_flags():
    memory = _filled_memory(n_step_return=1, batch_size=16, demo=True)
    np.random.seed(3)
    batch = memory.sample()
    flags = batch[6]["is_demo"]
    assert flags.shape == (16,)
    assert bool(flags.all())


def test_bc_margin_loss_aligns_argmax_to_demo_action():
    """DQfD margin loss must (a) be reported, (b) push the argmax of demo states
    toward the demo action over training steps."""
    demo_action = 7
    agent = _make_agent(
        n_steps=1,
        lr=1.0e-3,
        bc_lambda=1.0,
        bc_anneal_steps_end=0,  # static lambda
        bc_margin=0.5,
    )
    memory = _filled_memory(n_step_return=1, batch_size=16, demo=True, demo_action=demo_action)
    np.random.seed(5)

    first_match = None
    last_stats = None
    for _ in range(60):
        batch = memory.sample()
        last_stats = agent.train(batch, 0, 0, len(memory))
        if first_match is None:
            first_match = last_stats["debug/demo_argmax_match"]

    assert last_stats is not None
    assert np.isfinite(last_stats["loss/bc_margin"])
    assert last_stats["loss/bc_margin"] >= 0.0
    assert last_stats["bc/bc_lambda"] == 1.0
    assert last_stats["debug/demo_argmax_match"] > 0.9, (
        f"argmax should align to the demo action (start={first_match:.2f}, "
        f"end={last_stats['debug/demo_argmax_match']:.2f})"
    )


def test_bc_margin_disabled_by_default():
    agent = _make_agent(n_steps=1)
    memory = _filled_memory(n_step_return=1, batch_size=16, demo=True)
    np.random.seed(9)
    stats = agent.train(memory.sample(), 0, 0, len(memory))
    assert stats["bc/bc_lambda"] == 0.0
    assert stats["loss/bc_margin"] == 0.0


def test_bc_lambda_zero_kills_anneal_schedule():
    """bc_lambda=0 must disable margin loss even when anneal steps are configured."""
    agent = _make_agent(
        n_steps=1,
        bc_lambda=0.0,
        bc_lambda_start=1.0,
        bc_anneal_steps_end=1_000_000,
    )
    memory = _filled_memory(n_step_return=1, batch_size=16, demo=True)
    stats = agent.train(memory.sample(), 0, 0, len(memory))
    assert stats["bc/bc_lambda"] == 0.0
    assert stats["loss/bc_margin"] == 0.0


def test_iqn_warns_when_memory_only_provides_one_step_metadata():
    """n_steps>1 with memory-side n_step_return=1 must log a one-time warning."""
    agent = _make_agent(n_steps=3)
    agent._warned_missing_n_step_metadata = False
    memory = _filled_memory(n_step_return=1, batch_size=8)
    np.random.seed(21)
    agent.train(memory.sample(), 0, 0, len(memory))
    assert agent._warned_missing_n_step_metadata
