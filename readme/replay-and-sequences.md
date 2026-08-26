# Replay, n-step returns and recurrent sequences

Replay is an explicit RunSpec component. `InMemoryReplayStore` owns transition
storage and episode links; a sampler decides which eligible IDs become a
`TrainingBatch`. The learner returns optional priority feedback. This separation
keeps uniform, prioritized, recurrent and demonstration-aware experiments
comparable without changing the environment.

See [algorithms](algorithms.md) for learner compatibility and
[configuration](configuration.md) for the complete field reference.

<p align="center">
  <img src="../docs/diagrams/replay-sequence-preview.svg" alt="Replay sequence window with burn-in, learning positions, n-step targets and sequence priority" width="900">
</p>

[Editable diagram](../docs/diagrams/replay-sequence.excalidraw) ·
[local preview](../docs/diagrams/replay-sequence-preview.html)

## Replay choices

| Sampler | Selection | Sequences | Priority feedback | Typical use |
| --- | --- | --- | --- | --- |
| `UniformSampler` | Uniform without replacement from complete n-step starts | `sequence_length=1` only | Ignored | Feed-forward baseline and smoke tests |
| `PrioritizedSampler` | Proportional PER with replacement, optional uniform mixture and expert strata | Yes, with `InMemoryReplayStore` | Applied | Recommended value-learning replay |
| `SequenceSampler` | Uniform complete contiguous windows without replacement | Required, length at least 2 | Ignored | Recurrent ablation without PER |
| `DemoMixSampler` | Explicit bounded demo/online counts without replacement | `sequence_length=1` only | Ignored | Simple demonstration mixing |
| `OnPolicySequenceSampler` | Latest complete contiguous rollout | Yes; batch size 1, n-step 1 | Ignored | PPO only |

`InMemoryReplayStore` is a fixed-capacity ring with stable transition IDs,
episode and step links, an incremental eligibility index and an append revision.
Demonstration transitions are protected from ordinary online eviction; a store
whose capacity cannot keep them safely raises instead of silently deleting
them.

## N-step targets and boundaries

For a start transition `t`, the materialized return is

```text
G_t^(n) = r_t + gamma * r_(t+1) + ... + gamma^(k-1) * r_(t+k-1)
target  = G_t^(n) + bootstrap_discount * V(s_(t+k))
```

where `k <= n`. A true terminal sets the bootstrap discount to zero. A
truncation ends the stored episode but retains the bootstrap convention carried
by the transition batch. A horizon may end early at either boundary and never
continues into the next episode. A non-terminal start is eligible only when its
required continuation exists; this prevents a still-arriving rollout tail from
being trained as if it were terminal.

`training.n_step` is a positive integer and `training.gamma` is in `[0, 1]`.
For sequence learning, runtime validation requires
`n_step < sequence_length`. N-step return construction and the learner use the
same gamma from the `BatchRequest`.

## Proportional prioritized replay

TrackmaniaRL implements proportional [Prioritized Experience
Replay](https://arxiv.org/abs/1511.05952). After priority feedback `delta_i`,

```text
p_i = abs(delta_i) + priority_epsilon
q_i = p_i ** alpha
P_PER(i) = q_i / sum_j(q_j)
P(i) = (1 - uniform_mix) * P_PER(i) + uniform_mix / N
w_i = (N * P(i)) ** (-beta)
w_i_normalized = w_i / max_batch(w)
```

| Parameter | Type/default | Meaning and failure mode |
| --- | --- | --- |
| `components.sampler.kwargs.alpha` | float, `0.6` | Priority exponent. `0` is uniform; a large value concentrates training on few samples. Must be non-negative. |
| `components.sampler.kwargs.beta` | float, `0.4` | Default importance-sampling correction. `0` disables correction; `1` fully corrects the configured sampling distribution. |
| `components.sampler.kwargs.priority_epsilon` | float, `1e-6` | Positive floor added after absolute value, keeping every active item sampleable. |
| `components.sampler.kwargs.uniform_mix` | float, `0.0` | Probability-mixture weight in `[0, 1]`; prevents complete priority concentration but weakens prioritization. |
| `training.beta` | float or null, `null` | Per-run override. When set, it takes precedence over sampler `beta` in every batch. |
| `training.per_beta_final` | float or null, `null` | Final beta in `[0, 1]`; requires `training.beta`. |
| `training.per_beta_anneal_transitions` | positive int or null | Linear annealing duration. If omitted, `total_transitions` is used. |

New eligible transitions receive the largest priority seen so far. Stale
updates for an evicted transition ID are ignored. Sampler RNG, priority arrays,
eligibility frontier and store contents are checkpointed at a consistent
learner boundary; exact resume restores them before further sampling.

For a sequence, `DiscreteValueLearner` aggregates valid absolute TD errors as

```text
sequence_priority = 0.9 * max(abs(TD)) + 0.1 * mean(abs(TD))
```

and assigns that value to the transition ID at the final position of the
sampled history window. The max term keeps rare large errors visible while the
mean term represents the rest of the window. Masks exclude non-learning
positions from both statistics.

### Elite and expert controls

These are sampling controls, not reward terms:

| Parameter | Default | Exact effect |
| --- | --- | --- |
| `elite_time_s` | `null` | A transition whose `sampling/projected_lap_time_s` is at most this threshold is elite. Positive seconds. |
| `elite_priority_boost` | `1.0` | Multiplies elite proportional weight after `p ** alpha`; must be at least 1. |
| `expert_demo_time_s` | `null` | A demo transition at or below this projected-lap threshold enters the expert stratum. Positive seconds. |
| `expert_fraction` | `0.0` | Rounded fraction of each batch drawn from the expert stratum. Requires `expert_demo_time_s`; both expert and non-expert pools must satisfy the requested counts. |

Elite boosting composes with `uniform_mix`. When expert stratification is
active, reported sampling probabilities include the selected stratum fraction,
so importance weights correct the distribution that was actually used.

`DemoMixSampler` is simpler: `min_demo_fraction` and `max_demo_fraction` bound
the number of transitions marked by `info.is_demo` or `info.source == "demo"`.
It raises when either the demo or online pool cannot fill the requested batch.
Use PER's expert stratum when priority learning and a fixed expert proportion
are both required.

## Recurrent sequence contract

`training.sequence_length` is the replay time dimension. A sequence sampler
accepts only full, unique, contiguous histories from one identified episode:

- transition IDs are consecutive;
- `episode_id` is present and unchanged;
- `step` increments when present;
- the preceding transition is neither terminal nor truncated;
- every learning position has its complete n-step horizon.

Windows at the beginning of an episode that lack real history are ineligible.
The recurrent samplers do not simulate context by repeating the first
observation. Returned masks have shape `(batch, time)` and the current complete
windows contain all `true` values; the learner still applies them to losses and
priority aggregation.

`components.learner.kwargs.burn_in` reconstructs temporal context on the prefix
without training through it. It must satisfy

```text
0 <= burn_in < training.sequence_length
training.n_step < training.sequence_length
```

The final window position remains a learning anchor with its own materialized
n-step target. Intermediate learning positions stop before the unavailable
right edge. GRU and Mamba temporal cores reconstruct state from the sampled
observations; recurrent state is not stored in replay.

`components.feature_pipeline.kwargs.history_length` and replay sequences are
two different ways to supply history. Runtime validation rejects both values
above one because that would stack a history of histories. For GRU/Mamba replay
set feature `history_length: 1`; for a feed-forward model using a fixed feature
stack keep `training.sequence_length: 1`.

## R2D2-style scope

The project provides **R2D2-style recurrent replay**, not a complete
implementation of [Recurrent Experience Replay in Distributed Reinforcement
Learning](https://willdabney.com/publication/r2d2/).

Implemented similarities include distributed actors, complete recurrent replay
windows, burn-in, n-step targets, sequence-level max/mean priorities, policy
version metadata and soft/hard policy-lag controls. Important differences are:

- recurrent states are reconstructed from observations rather than stored;
- overlap arises from sampled windows rather than a fixed actor unroll schedule;
- the priority mixture is `0.9 max + 0.1 mean`, not claimed paper equivalence;
- R2D2 value rescaling is not part of the learner;
- actor and learner scheduling, network architecture and optimizer defaults are
  TrackmaniaRL contracts rather than the paper's exact configuration.

Use that narrower name in experiment reports and record the full RunSpec.

## Configuration recipes

The following are RunSpec fragments. Start from the validated
`readme/examples/builtin-smoke.yaml` or generated Trackmania RunSpec and replace
the shown sections.

### Feed-forward IQN or FQF with PER

```yaml
components:
  replay_store:
    class_path: trackmaniarl.core.replay:InMemoryReplayStore
    kwargs: {capacity: 1000000}
  sampler:
    class_path: trackmaniarl.core.replay:PrioritizedSampler
    kwargs:
      alpha: 0.6
      beta: 0.4
      priority_epsilon: 1.0e-6
      uniform_mix: 0.01

training:
  batch_size: 512
  sequence_length: 1
  n_step: 3
  gamma: 0.995
  beta: 0.4
  per_beta_final: 1.0
  per_beta_anneal_transitions: 2000000
```

Select `ImplicitQuantileHead` plus `RandomQuantileStrategy` for IQN, or
`FractionProposalHead` plus `FractionProposalStrategy` for FQF as shown in the
[algorithm recipes](algorithms.md). The replay contract is the same.

### GRU or Mamba with sequence PER

```yaml
components:
  learner:
    class_path: trackmaniarl.algorithms.value_based:DiscreteValueLearner
    kwargs:
      burn_in: 16
  sampler:
    class_path: trackmaniarl.core.replay:PrioritizedSampler
    kwargs: {alpha: 0.6, beta: 0.4, uniform_mix: 0.01}
  feature_pipeline:
    class_path: trackmaniarl.trackmania.features:LidarFeaturePipeline
    kwargs:
      geometry_path: assets/my-map.geometry.npz
      expected_map_uid: my-map-uid
      history_length: 1

training:
  batch_size: 32
  sequence_length: 80
  n_step: 5
  gamma: 0.995
  beta: 0.4
```

Use `trackmaniarl.models.temporal:GruTemporalCore`, or the experimental
`trackmaniarl.models.temporal:MambaTemporalCore`. Mamba's technique originates
in [Mamba: Linear-Time Sequence Modeling with Selective State
Spaces](https://arxiv.org/abs/2312.00752); the TrackmaniaRL block is a reusable
temporal component, not a reproduction of that paper's language-model setup.

### Demo-mixed single-step replay

```yaml
components:
  sampler:
    class_path: trackmaniarl.core.replay:DemoMixSampler
    kwargs:
      min_demo_fraction: 0.25
      max_demo_fraction: 0.25
      seed: 0

training:
  sequence_length: 1
  n_step: 3
```

Validate every complete RunSpec and run a bounded gate before training:

```bash
uv run trackmaniarl validate run.yaml
uv run trackmaniarl smoke run.yaml --transitions 100
uv run trackmaniarl train run.yaml
```

The smoke command is a live Trackmania gate for a Trackmania environment. A
game-free RunSpec can use `validate` and its deterministic unit/integration
tests without presenting those as a live game result.
