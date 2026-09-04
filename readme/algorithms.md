# Algorithms

This page describes the algorithm implementations in the current TrackmaniaRL
2.0 tree. It is a support contract, not a claim that a class reproduces every
detail, hyperparameter, benchmark, or result from the paper whose name it uses.
The paper links identify the closest primary reference.

For new discrete-value work, use
`trackmaniarl.algorithms.value_based:DiscreteValueLearner`. Scalar Q, QR-DQN,
IQN, and FQF are model compositions trained by that one learner.

The YAML blocks below are deliberately labelled **fragments**. Merge the shown
keys into a complete RunSpec 2.0 file; keep the environment, evaluator, map,
geometry, logging, and distributed sections from the generated project unless
the fragment replaces them. See the [configuration guide](configuration.md),
[replay guide](replay-and-sequences.md), [reward guide](rewards.md), and
[imitation-learning guide](imitation-learning.md) for those shared contracts.

## Support matrix

| Algorithm | Current status and public entry point | Action and training contract | Replay and sequences | Execution | Checkpoint and warm start |
| --- | --- | --- | --- | --- | --- |
| Standard Q / DQN | Supported scalar composition of `DiscreteValueLearner`; built-in key `discrete_value` | Discrete indices; off-policy online RL; the learner still uses a Double-DQN target | Uniform or prioritized replay; single-step or contiguous sequences | Local and distributed | Exact v2 learner/replay/sampler resume; named composite submodule warm start |
| Double + Dueling | Supported behavior of the unified learner plus `ScalarQHead(mode: dueling)`; not a separate class | Discrete indices; online network selects and target network evaluates | Same as scalar Q | Local and distributed | Same as unified scalar Q |
| QR-DQN | Supported composition of `DiscreteValueLearner` | Discrete distributional value learning with fixed quantile locations | Uniform or prioritized replay; single-step or sequences | Local and distributed | Exact v2 resume and composite warm start |
| IQN | Supported composition of `DiscreteValueLearner`; generated Trackmania default | Discrete implicit quantile value learning; optional upper-CVaR action evaluation | Uniform or prioritized replay; single-step or sequences | Local and distributed | Exact v2 resume and composite warm start |
| FQF | Supported composition of `DiscreteValueLearner` | Discrete learned-fraction distributional value learning | Uniform or prioritized replay; single-step or sequences | Local and distributed | Exact v2 resume, fraction optimizer state, and composite warm start |
| SAC | Public learner `SoftActorCritic`; built-in key `soft_actor_critic`; custom model factory required | Continuous off-policy actor-critic | Uniform or PER; `sequence_length: 1` only | Local and distributed when the custom model is installed on every participant | Exact resume; no built-in partial warm start |
| REDQ / randomized-ensemble SAC | Public learner `RandomizedEnsembleSAC`; built-in key `randomized_ensemble_sac`; custom model factory required | Continuous off-policy ensemble actor-critic | Uniform or PER; `sequence_length: 1` only | Local and distributed | Exact resume including target RNG/update count; no built-in partial warm start |
| TQC | Public learner `TruncatedQuantileCritic`; built-in key `truncated_quantile_critic`; first-party telemetry factory | Continuous off-policy distributional actor-critic | Uniform or PER; `sequence_length: 1` only | Local and distributed | Exact resume including alpha; no built-in partial warm start |
| PPO | Public learner `ProximalPolicyOptimization`; built-in key `proximal_policy_optimization`; first-party telemetry factory | Continuous on-policy GAE/PPO | Latest contiguous rollout through `OnPolicySequenceSampler` | Local `trackmaniarl train` only; distributed learner/actor rejects it | Exact local resume at restart-safe episode boundaries; no partial warm start |
| Stable discrete SAC | **SD-SAC-inspired experimental** `StableDiscreteSoftActorCritic`; custom model factory required | Discrete off-policy categorical actor with twin all-action critics | Uniform or PER; `sequence_length: 1` only | Local and distributed | Exact resume including alpha; no built-in partial warm start |
| Behavior cloning (BC) | Public Trackmania offline-supervised lifecycle, `BehaviorCloningLearner` | Compact discrete categorical policy; offline demonstrations/recovery data | No RL replay or WAL; deterministic lap/episode split and contiguous feature histories | Local `bc-train`; closed-loop `bc-benchmark` | `bc-latest.pt` exact resume; `bc-best-validation.pt` promotion candidate; compatible encoder/temporal tensors can warm-start unified RL |
| DAgger | Public `dagger-collect` data-collection workflow, not a learner | Student compact discrete BC policy plus trajectory-tracking teacher | Writes weighted recovery `.npz`; feed it back to BC with `--recovery` | Local live Trackmania only | Starts from a BC checkpoint; output is a dataset, not a resumable optimizer checkpoint |
| DQfD-style objectives | Opt-in `DemonstrationMarginObjective` and `DemonstrationCrossEntropyObjective` inside `DiscreteValueLearner`; **not full DQfD** | Discrete off-policy TD plus demo-only auxiliary losses; offline pretraining is available | Demo-aware replay flags; PER supports sequences, `DemoMixSampler` is single-step | Offline pretrain, then local or distributed off-policy RL | Unified v2 exact resume; objectives are part of the checkpoint contract |

“Distributed” in the table means the off-policy learner exposes a replicable
policy and can use `trackmaniarl learner`/`trackmaniarl actor`. Every machine
must also be able to import the same custom model factory and must share the
same RunSpec fingerprint and action/feature contracts.

## Shared target, replay, and action semantics

- `TrainingBatch.bootstrap_discounts` already contains the n-step discount and
  terminal mask. Learners must not infer termination from rewards. Truncation
  behavior is therefore owned by replay construction.
- `training.gamma`, `training.n_step`, `training.sequence_length`, and
  `training.beta` are replay request settings. They are not learner constructor
  arguments.
- `UniformSampler` is single-step. `SequenceSampler` and `PrioritizedSampler`
  can return contiguous sequences. `DemoMixSampler` is single-step.
- The unified value learner masks invalid sequence positions, applies burn-in
  through the temporal core, and gives PER one priority per sampled sequence.
  `training.n_step` must be smaller than `training.sequence_length`.
- The unified discrete policy returns a model action index. With the standard
  78-action Trackmania head, `policy_action_ids` masks a subset of those global
  indices; it does not remap a compact head.
- Continuous action bounds belong to the actor. The first-party TQC and PPO
  telemetry factories use `[accelerator, brake, steer]` bounds
  `[0, 0, -1]` to `[1, 1, 1]`. The SAC and REDQ learners do not clamp an
  incorrectly configured custom actor.
- A complete local or distributed resume restores replay and sampler state for
  off-policy runs. `--model-initialization-checkpoint` is a warm start, not a
  resume, and is implemented by the unified value learner.

## Unified discrete value family

All five configurations in this section use quantile/scalar regression on the
chosen action, n-step bootstrap discounts, online-network action selection,
target-network evaluation, importance weights, and PER priority feedback.
`target_tau > 0` performs a Polyak update every learner update; `target_tau: 0`
uses a hard copy every `target_update_interval` updates.

Common metrics are `loss/value`, `loss/total`, `loss/objectives`,
`gradients/norm`, `debug/trained_positions`,
`debug/target_synced_fraction`, and `timing/update_s`. Periodic diagnostics add
selected/target Q statistics, absolute TD error, n-step return, bootstrap-zero
fraction, and action entropy.

### Standard Q / DQN

**Status and intuition.** This is the supported scalar-Q baseline: one value
per action and Smooth L1 TD regression. It is called “DQN” as a model family,
but its bootstrap is always Double-DQN-style, so this is not the original DQN
target from the paper.

**Contract.** `ModelContract.DISCRETE_VALUE`; discrete global action indices;
online off-policy training. Uniform replay and PER are valid. Identity, GRU,
or Mamba temporal cores can be used with the corresponding sampler and sequence
settings. Local and distributed training are supported.

**State, parameters, and failures.** Important knobs are `learning_rate`,
`target_update_interval`, `target_tau`, `gradient_clip_norm`, `burn_in`,
`exploration_epsilon`, `policy_action_ids`, and `value_rescaling`. Resume checks
the composite architecture fingerprint and restores online/target models,
optimizer, scaler, RNG, objective state, and adaptive-clipper schedule. A head
dimension mismatch, an excluded/out-of-range action, an invalid sequence/burn-in
combination, or a different architecture fingerprint fails explicitly.

**YAML fragment — scalar Q baseline:**

```yaml
components:
  learner:
    class_path: trackmaniarl.algorithms.value_based:DiscreteValueLearner
    kwargs:
      learning_rate: 1.0e-4
      target_tau: 0.0
      target_update_interval: 1000
      exploration_epsilon: 0.1
  model_factory:
    class_path: trackmaniarl.models.factory:CompositeValueModelFactory
    kwargs:
      encoder:
        class_path: trackmaniarl.trackmania.encoders:LidarSensorEncoder
        kwargs:
          config: {output_dim: 256}
      temporal:
        class_path: trackmaniarl.models.temporal:IdentityTemporalCore
        kwargs: {input_dim: 256}
      head:
        class_path: trackmaniarl.models.heads:ScalarQHead
        kwargs: {feature_dim: 256, action_count: 78, mode: standard}
      strategy:
        class_path: trackmaniarl.models.strategies:ScalarValueStrategy
  replay_store:
    class_path: trackmaniarl.core.replay:InMemoryReplayStore
  sampler:
    class_path: trackmaniarl.core.replay:PrioritizedSampler
training:
  batch_size: 512
  sequence_length: 1
  n_step: 3
  gamma: 0.995
  beta: 0.4
```

**Command:** `uv run trackmaniarl validate run.yaml`, then
`uv run trackmaniarl train run.yaml`.

**Primary paper:** [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602).

### Double DQN with a dueling head

**Status and intuition.** Double action selection is mandatory in the unified
learner: the online model chooses `argmax_a Q(s', a)` and the target model
evaluates that action. Dueling is an independent head option that decomposes
value and advantage and subtracts mean advantage. It is supported for scalar,
fixed-quantile, and implicit-quantile heads; it is not a separate learner and
does not imply Rainbow.

**Contract, replay, execution, and state.** These are identical to the selected
unified scalar/quantile composition: discrete off-policy, uniform/PER,
single-step or sequences, local/distributed, exact v2 resume, and named
encoder/temporal warm start. There is no dueling-specific metric; monitor the
common Q, TD-error, action-entropy, and gradient metrics. Invalid head/action
dimensions and incompatible checkpoints fail at setup/load.

**YAML fragment — replace the scalar head above:**

```yaml
components:
  model_factory:
    kwargs:
      head:
        class_path: trackmaniarl.models.heads:ScalarQHead
        kwargs: {feature_dim: 256, action_count: 78, mode: dueling}
      strategy:
        class_path: trackmaniarl.models.strategies:ScalarValueStrategy
```

**Command:** `uv run trackmaniarl validate run.yaml`, then
`uv run trackmaniarl train run.yaml`.

**Primary papers:** [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
and [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581).

### QR-DQN

**Status and intuition.** Supported unified composition. A fixed set of
uniform quantile midpoints represents the return distribution and is trained
with pairwise quantile Huber loss. Action selection uses the weighted expected
return, with the same Double target as the other unified variants.

**Contract, replay, execution, and state.** Discrete off-policy
`DISCRETE_VALUE`; uniform/PER; single-step or sequences; local/distributed;
exact v2 resume and named composite warm start. `quantile_count` must be at
least two and must match between `FixedQuantileHead` and
`FixedQuantileStrategy`. More quantiles increase the pairwise loss tensor and
memory cost. Use common value/TD/gradient metrics; there are no separate QR
metrics.

**YAML fragment — QR-DQN head and strategy:**

```yaml
components:
  model_factory:
    kwargs:
      head:
        class_path: trackmaniarl.models.heads:FixedQuantileHead
        kwargs:
          config:
            feature_dim: 256
            action_count: 78
            quantile_count: 32
            dueling: true
      strategy:
        class_path: trackmaniarl.models.strategies:FixedQuantileStrategy
        kwargs: {quantile_count: 32}
```

**Command:** `uv run trackmaniarl validate run.yaml`, then
`uv run trackmaniarl train run.yaml`.

**Primary paper:** [Distributional Reinforcement Learning with Quantile Regression](https://arxiv.org/abs/1710.10044).

### IQN

**Status and intuition.** Supported unified composition and the generated
Trackmania default. IQN samples quantile fractions during training/target
construction and evaluates deterministic uniform midpoints for policy action
selection. The implicit head embeds fractions with cosine features.

**Contract, replay, execution, and state.** Discrete off-policy
`DISCRETE_VALUE`; uniform/PER; single-step or contiguous recurrent sequences;
local/distributed; exact v2 resume and named composite warm start. Key knobs are
the train/target/evaluation quantile counts, `cosine_count`, optional `dueling`,
and `online_quantile_distortion`/`evaluation_quantile_distortion` with
`upper_cvar_alpha`. Counts below two, incompatible dimensions, bad masks, or
non-matching fingerprints fail. Quantile count directly controls compute and
memory; upper-CVaR is a deliberate risk-seeking upper-tail distortion, not a
neutral paper baseline.

**YAML fragment — IQN head and strategy:**

```yaml
components:
  learner:
    class_path: trackmaniarl.algorithms.value_based:DiscreteValueLearner
    kwargs:
      exploration_epsilon: 0.1
      online_quantile_distortion: neutral
      evaluation_quantile_distortion: neutral
  model_factory:
    kwargs:
      head:
        class_path: trackmaniarl.models.heads:ImplicitQuantileHead
        kwargs:
          config: {feature_dim: 256, action_count: 78, cosine_count: 64, dueling: true}
      strategy:
        class_path: trackmaniarl.models.strategies:RandomQuantileStrategy
        kwargs:
          train_quantile_count: 32
          target_quantile_count: 32
          evaluation_quantile_count: 32
```

**Command:** `uv run trackmaniarl validate run.yaml`, then
`uv run trackmaniarl train run.yaml`.

**Primary paper:** [Implicit Quantile Networks for Distributional Reinforcement Learning](https://arxiv.org/abs/1806.06923).

### FQF

**Status and intuition.** Supported unified composition. FQF uses the same
implicit quantile head as IQN, but a proposal network learns fraction masses.
Quantile regression receives detached fraction points; a separate analytical
fraction loss receives detached quantile values. This prevents either optimizer
from updating the other path through an unintended gradient.

**Contract, replay, execution, and state.** Discrete off-policy
`DISCRETE_VALUE`; uniform/PER; single-step or sequences; local/distributed.
Exact v2 checkpoints include both main and fraction optimizer states. Key knobs
are `fraction_count`, `fraction_learning_rate`, `entropy_coefficient`, and
`fraction_gradient_clip_norm`. Monitor `loss/fraction`, `fraction/entropy`,
`fraction/effective_count`, min/max mass, and Wasserstein-gradient diagnostics
in addition to common metrics. Fraction collapse, an overly large fraction
learning rate, mismatched feature dimensions, or a checkpoint missing the
strategy optimizer fails or destabilizes training.

**YAML fragment — FQF learner, head, and strategy:**

```yaml
components:
  learner:
    class_path: trackmaniarl.algorithms.value_based:DiscreteValueLearner
    kwargs:
      learning_rate: 1.0e-4
      fraction_learning_rate: 1.0e-7
      fraction_gradient_clip_norm: 10.0
  model_factory:
    kwargs:
      head:
        class_path: trackmaniarl.models.heads:ImplicitQuantileHead
        kwargs:
          config: {feature_dim: 256, action_count: 78, cosine_count: 64, dueling: true}
      strategy:
        class_path: trackmaniarl.models.strategies:LearnedFractionStrategy
        kwargs: {feature_dim: 256, fraction_count: 32, entropy_coefficient: 1.0e-3}
```

**Command:** `uv run trackmaniarl validate run.yaml`, then
`uv run trackmaniarl train run.yaml`.

**Primary paper:** [Fully Parameterized Quantile Function for Distributional Reinforcement Learning](https://arxiv.org/abs/1911.02140).

## Continuous actor-critic learners

### Soft Actor-Critic (SAC)

**Status and intuition.** `SoftActorCritic` is a public SAC-v2-style twin-critic
learner with a squashed stochastic actor, minimum target Q, Polyak targets,
optional learned temperature, and PER feedback. TrackmaniaRL does not bundle a
`CONTINUOUS_ACTOR_CRITIC` model factory, so this is a public extension API, not
a turnkey Trackmania baseline.

**Contract.** The custom model must expose `actor`, `q1`, and `q2`; the actor
returns `(action, log_probability)`, and each critic returns one scalar per
observation/action pair. Actions are continuous and their bounds live in the
actor. Training is online off-policy with `sequence_length: 1`; uniform replay
or PER works. The replicable actor policy supports local and distributed runs
when the factory is installed everywhere.

**State, parameters, metrics, and failures.** Exact resume stores online/target
models, actor/critic/alpha optimizers, log-alpha, scaler, and RNG. There is no
partial warm-start loader. Key parameters are `learning_rate`, `target_tau`,
`entropy_coefficient`, `target_entropy`, and
`learn_entropy_coefficient`. Metrics are `loss/actor`, `loss/critic`,
`loss/entropy`, and `state/alpha`. Missing model members, a sequence length above
one, incorrect action bounds/shapes, or incompatible model state fails; bad
reward scale/target entropy commonly drives alpha or Q values out of range.

**YAML fragment — SAC with a required user model:**

```yaml
components:
  learner:
    class_path: trackmaniarl.algorithms.soft_actor_critic:SoftActorCritic
    kwargs:
      learning_rate: 3.0e-4
      target_tau: 0.005
      entropy_coefficient: 0.2
      learn_entropy_coefficient: true
  model_factory:
    # Project component; declare ModelContract.CONTINUOUS_ACTOR_CRITIC.
    class_path: my_trackmania_agent.models:SacModelFactory
  replay_store:
    class_path: trackmaniarl.core.replay:InMemoryReplayStore
  sampler:
    class_path: trackmaniarl.core.replay:PrioritizedSampler
  feature_pipeline:
    class_path: trackmaniarl.trackmania.features:TelemetryFeaturePipeline
training:
  sequence_length: 1
  n_step: 3
  gamma: 0.995
  beta: 0.4
```

**Command:** after installing `my_trackmania_agent`, run
`uv run trackmaniarl validate run.yaml`, then
`uv run trackmaniarl train run.yaml`.

**Primary paper:** [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905).

### REDQ / RandomizedEnsembleSAC

**Status and intuition.** `RandomizedEnsembleSAC` is a public REDQ-style SAC
learner. It trains every critic against the minimum of a randomly selected
target subset and updates the actor on the ensemble mean at a configurable
interval. It uses a fixed entropy coefficient. No first-party
`ENSEMBLE_ACTOR_CRITIC` model factory is bundled.

**Contract.** A custom model exposes `actor` and an `nn.ModuleList`-like
`critics`; every critic accepts continuous observation/action batches. It is
online off-policy, single-step only, uniform/PER, and local/distributed.
`training.updates_per_transition` controls the runtime update-to-data ratio;
that setting is separate from `policy_update_interval`.

**State, parameters, metrics, and failures.** Resume includes models,
optimizers, `update_count`, target-subset RNG, scaler, and learner RNG; no
partial warm start. Key knobs are ensemble size in the custom factory,
`target_subset_size`, `policy_update_interval`, `target_tau`, fixed
`entropy_coefficient`, learning rate, and runtime update ratio. Metrics are
`loss/critic` and `loss/actor`; actor loss is zero on skipped actor updates.
Setup rejects a subset larger than the ensemble. A high update ratio, correlated
critics, or an unsuitable fixed entropy coefficient can erase REDQ's intended
benefit. This class does not by itself reproduce the paper's benchmark
configuration.

**YAML fragment — REDQ with a required user model:**

```yaml
components:
  learner:
    class_path: trackmaniarl.algorithms.randomized_ensemble_sac:RandomizedEnsembleSAC
    kwargs:
      learning_rate: 3.0e-4
      target_tau: 0.005
      entropy_coefficient: 0.2
      target_subset_size: 2
      policy_update_interval: 20
  model_factory:
    # Project component; declare ModelContract.ENSEMBLE_ACTOR_CRITIC.
    class_path: my_trackmania_agent.models:RedqModelFactory
  sampler:
    class_path: trackmaniarl.core.replay:PrioritizedSampler
training:
  sequence_length: 1
  n_step: 1
  updates_per_transition: 4.0
  beta: 0.4
```

**Command:** after installing `my_trackmania_agent`, run
`uv run trackmaniarl validate run.yaml`, then
`uv run trackmaniarl train run.yaml`.

**Primary paper:** [Randomized Ensembled Double Q-Learning](https://arxiv.org/abs/2101.05982).

### Truncated Quantile Critic (TQC)

**Status and intuition.** `TruncatedQuantileCritic` is public and has a
first-party `TelemetryTqcModelFactory`. It concatenates and sorts target
quantiles from all target critics, removes the configured number of upper
quantiles per critic globally, and applies quantile Huber regression. The actor
uses the mean of current critic quantiles. Temperature learning is optional.

**Contract.** `CONTINUOUS_QUANTILE_ACTOR_CRITIC`; continuous actions. The
first-party model uses five critics, 25 quantiles each, and Trackmania's native
three control bounds. Training is online off-policy, `sequence_length: 1`,
uniform/PER, local/distributed.

**State, parameters, metrics, and failures.** Resume includes models, actor,
critic and alpha optimizers, log-alpha, scaler, and RNG; no partial warm start.
Key knobs are factory `critics`/`quantiles`, learner
`top_quantiles_to_drop_per_critic`, `target_tau`, learning rate, entropy
coefficient, and target entropy. Metrics are `loss/actor`, `loss/critic`,
`loss/alpha`, and `state/alpha`. At least two critics and two quantiles are
required, and truncation that removes every target quantile fails explicitly.
Large truncation can create severe underestimation.

**YAML fragment — first-party telemetry TQC:**

```yaml
components:
  learner:
    class_path: trackmaniarl.algorithms.truncated_quantile_critic:TruncatedQuantileCritic
    kwargs:
      learning_rate: 3.0e-4
      target_tau: 0.005
      top_quantiles_to_drop_per_critic: 2
      learn_entropy_coefficient: true
  model_factory:
    class_path: trackmaniarl.trackmania.baseline:TelemetryTqcModelFactory
    kwargs:
      config: {input_dim: 33, action_dim: 3, hidden_dim: 256, quantiles: 25, critics: 5}
  feature_pipeline:
    class_path: trackmaniarl.trackmania.features:TelemetryFeaturePipeline
  sampler:
    class_path: trackmaniarl.core.replay:PrioritizedSampler
training:
  sequence_length: 1
  n_step: 3
  gamma: 0.995
  beta: 0.4
```

**Command:** `uv run trackmaniarl validate run.yaml`, then
`uv run trackmaniarl train run.yaml`.

**Primary paper:** [Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics](https://arxiv.org/abs/2005.04269).

## On-policy learner

### Proximal Policy Optimization (PPO)

**Status and intuition.** `ProximalPolicyOptimization` is public and has a
first-party telemetry actor/value factory. It uses behavior-time latent actions,
log probabilities and values, GAE, normalized advantages, clipped policy and
value objectives, observation/reward normalization, linear learning-rate
annealing, minibatch epochs, and optional KL early stopping.

**Contract.** `CONTINUOUS_ACTOR_VALUE`; bounded continuous actions. The actor
must expose `sample_with_latent` and `evaluate_latent_actions`. PPO is routed by
`trackmaniarl train` to the in-process `Trainer`, not the asynchronous
coordinator. Use `OnPolicySequenceSampler`, `n_step: 1`, and a fixed rollout
`sequence_length`; `total_transitions` must be divisible by it. Distributed
learner/actor commands reject on-policy learners.

**State, parameters, metrics, and failures.** Resume stores model, optimizer,
observation and reward normalizers, processed-transition count, scaler, RNG,
and trainer counters. On-policy replay/sampler state is intentionally absent;
resume restarts at a recorded episode boundary. There is no partial warm start.
Key knobs are clip epsilons, `gae_lambda`, entropy/value coefficients, update
epochs, minibatch size, `target_kl`, normalization clips, and gradient norm.
Monitor `loss/policy`, `loss/value`, `state/entropy`, `state/approx_kl`,
`state/clip_fraction`, `state/early_stop`, and `state/learning_rate`. Replay
without behavior metadata, a non-on-policy sampler, partial/invalid
rollouts, or non-divisible transition budgets fail explicitly.

**YAML fragment — local telemetry PPO:**

```yaml
components:
  learner:
    class_path: trackmaniarl.algorithms.proximal_policy_optimization:ProximalPolicyOptimization
    kwargs:
      learning_rate: 3.0e-4
      clip_epsilon: 0.2
      gae_lambda: 0.95
      update_epochs: 10
      minibatch_size: 256
      target_kl: 0.02
  model_factory:
    class_path: trackmaniarl.trackmania.baseline:TelemetryPpoModelFactory
    kwargs: {input_dim: 33, hidden_dim: 256}
  replay_store:
    class_path: trackmaniarl.core.replay:InMemoryReplayStore
  sampler:
    class_path: trackmaniarl.core.replay:OnPolicySequenceSampler
  feature_pipeline:
    class_path: trackmaniarl.trackmania.features:TelemetryFeaturePipeline
training:
  total_transitions: 8192
  batch_size: 1
  sequence_length: 2048
  n_step: 1
  gamma: 0.995
  warmup_transitions: 0
```

**Command:** `uv run trackmaniarl validate run.yaml`, then run locally with
`uv run trackmaniarl train run.yaml`.

**Primary paper:** [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).

## Experimental discrete actor-critic

### StableDiscreteSoftActorCritic

**Status and intuition.** This is explicitly an **SD-SAC-inspired experimental**
learner, not a claim of full SD-SAC paper compliance. It uses a categorical
actor, two all-action critics, double-average target Q, Q-clipped critic losses,
an entropy-change penalty against the target actor, optional temperature
learning, and Polyak targets. It is publicly exported and registered under
`stable_discrete_soft_actor_critic`, but no first-party
`DISCRETE_ACTOR_CRITIC` factory or Trackmania baseline is supplied.

**Contract.** The custom model exposes `actor`, `q1`, and `q2`; the actor must
provide `probabilities(observation)` and categorical sampling, while each critic
returns `[batch, action_count]`. The policy returns the sampled/argmax index
directly and has no `policy_action_ids` mask, so a custom compact-action model
must align its output with the environment contract. Training is online
off-policy, single-step only, uniform/PER, and local/distributed.

**State, parameters, metrics, and failures.** Resume stores online/target
models, three optimizers when alpha is learned, alpha, scaler, and RNG; no
partial warm start. Key knobs are `q_clip_epsilon`,
`entropy_penalty_coefficient`, `target_entropy`, `entropy_coefficient`,
`target_tau`, and learning rate. Metrics are `loss/actor`, `loss/critic`,
`loss/entropy`, and `state/alpha`. Missing probability/all-action interfaces,
bad action mapping, sequence replay, or incompatible state fails. Paper-level
claims require a separate baseline-controlled experiment; the current tree
provides unit/runtime support, not the paper's Atari/MOBA evidence.

**YAML fragment — experimental custom-model SD-SAC:**

```yaml
components:
  learner:
    class_path: trackmaniarl.algorithms.stable_discrete_soft_actor_critic:StableDiscreteSoftActorCritic
    kwargs:
      learning_rate: 3.0e-4
      target_tau: 0.005
      q_clip_epsilon: 0.5
      entropy_penalty_coefficient: 0.5
      learn_entropy_coefficient: true
  model_factory:
    # Project component; declare ModelContract.DISCRETE_ACTOR_CRITIC.
    class_path: my_trackmania_agent.models:StableDiscreteSacModelFactory
  sampler:
    class_path: trackmaniarl.core.replay:PrioritizedSampler
training:
  sequence_length: 1
  n_step: 3
  beta: 0.4
```

**Command:** after installing `my_trackmania_agent`, run
`uv run trackmaniarl validate run.yaml`, then
`uv run trackmaniarl train run.yaml`.

**Primary paper:** [Revisiting Discrete Soft Actor-Critic](https://arxiv.org/abs/2209.10081).

## Imitation and demonstration-guided learning

### Behavior cloning

**Status and intuition.** `BehaviorCloningLearner` is the public Trackmania
offline-supervised lifecycle. It learns compact categorical actions from
complete human laps and optional recovery episodes. It shares lidar encoders
and temporal cores with RL models, but `bc-train` does not use the RL replay,
WAL, rollout actor, or distributed coordinator.

**Contract.** `CATEGORICAL_POLICY`; compact discrete actions. The model's
`action_ids` must exactly match `environment.config.compact_action_ids`, and
model feature/history dimensions must match `LidarFeaturePipeline`. Dataset
splitting is deterministic by complete lap/episode. History and optional GRU
burn-in are built by the offline collator, not a replay sampler. Training and
benchmarking are local.

**State, parameters, metrics, and failures.** `bc-latest.pt` stores exact
learner, optimizer, scheduler, scaler, RNG, dataset fingerprint, batch RNG and
selection state; use it for `--resume`. `bc-best-validation.pt` is the selected
open-loop candidate. Key knobs include learning rate/weight decay, label
smoothing, class weighting, transition weighting, focal gamma, steering loss,
history/burn-in, augmentation, validation interval, scheduler, and early
stopping. Monitor loss, exact/balanced/weighted accuracy, action-transition and
steering-transition accuracy, intervention/recovery metrics, gradient norm,
and the closed-loop finish/time/progress benchmark. Data contract mismatch,
fewer than three complete laps, incompatible augmentation schema, dataset
fingerprint mismatch, or action mismatch fails. High open-loop accuracy does
not remove covariate shift.

**YAML fragment — BC lifecycle:**

```yaml
components:
  learner:
    class_path: trackmaniarl.trackmania.imitation_learning:BehaviorCloningLearner
    kwargs:
      learning_rate: 3.0e-4
      validation_interval: 100
      early_stopping_patience: 30
  model_factory:
    class_path: trackmaniarl.trackmania.imitation_learning:LidarBehaviorCloningModelFactory
    kwargs:
      action_ids: [0, 1, 3, 39, 72, 73, 75]
      telemetry_dim: 17
      spatial_bins: 12
      history_length: 8
      previous_action_conditioning: false
  feature_pipeline:
    class_path: trackmaniarl.trackmania.features:LidarFeaturePipeline
    kwargs:
      config:
        geometry_path: assets/trackmaniarl-test.geometry.npz
        expected_map_uid: REPLACE_WITH_TEST_3_UID
        history_length: 8
        include_control_inputs: false
  replay_store:
    class_path: trackmaniarl.core.replay:InMemoryReplayStore
  sampler:
    class_path: trackmaniarl.core.replay:UniformSampler
training:
  batch_size: 256
  metrics_interval_updates: 50
```

The complete RunSpec must also configure the same compact IDs in the
environment.

**Command:**
`uv run trackmaniarl bc-train run.yaml --demo demonstrations`, then
`uv run trackmaniarl bc-benchmark run.yaml artifacts/<run-id>/checkpoints/bc-best-validation.pt --trials 30`.

**Primary paper:** [ALVINN: An Autonomous Land Vehicle in a Neural Network](https://proceedings.neurips.cc/paper/1988/hash/812b4ba287f5ee0bc9d43bbf5bbe87fb-Abstract.html).

### DAgger collection

**Status and intuition.** `dagger-collect` is a public data workflow, not a
standalone optimizer. It runs a deterministic BC student in Trackmania, labels
visited states with a trajectory-tracking demonstration policy, probabilistically
or error-threshold-selects teacher intervention, and records expert labels,
student actions, intervention flags, state error, and bounded sample weights.
The current workflow is DAgger-inspired: it performs one collection invocation;
the user explicitly retrains BC and repeats. The teacher is the configured
trajectory tracker rather than an arbitrary interactive human oracle.

**Contract, data, execution, and state.** It requires
`OpenPlanetEnvironmentFactory`, `LidarFeaturePipeline`, an evaluation map, a
valid teacher demonstration, compact actions, and a BC model with
`previous_action_conditioning: false`. It is local live collection. The output
recovery archive is split by episode when passed to `bc-train --recovery`; no RL
replay or DAgger optimizer checkpoint exists. Collection starts from a BC
checkpoint.

**Parameters, metrics, and failures.** Key CLI knobs are `--episodes`,
`--teacher-probability`, `--intervention-error`, and `--action-lead-ms`.
Collection reports per-episode finish and sample counts; the archive carries
weights/interventions/errors for BC recovery metrics. Invalid map/geometry/time
contracts, previous-action conditioning, missing evaluation map, or a teacher
outside its recorded trajectory contract fails. A trajectory teacher can
itself be wrong off-distribution, so interventions are not safety guarantees.

**YAML fragment — DAgger-required BC settings:**

```yaml
components:
  learner:
    class_path: trackmaniarl.trackmania.imitation_learning:BehaviorCloningLearner
  model_factory:
    class_path: trackmaniarl.trackmania.imitation_learning:LidarBehaviorCloningModelFactory
    kwargs:
      action_ids: [0, 1, 3, 39, 72, 73, 75]
      previous_action_conditioning: false
  evaluator:
    class_path: trackmaniarl.trackmania.evaluation:TrackmaniaEvaluator
evaluation:
  maps:
    - id: trackmaniarl-test
      map_path: maps/trackmaniarl-test.Map.Gbx
      geometry_path: assets/trackmaniarl-test.geometry.npz
      expected_map_uid: REPLACE_WITH_TEST_3_UID
```

**Command:**
`uv run trackmaniarl dagger-collect run.yaml artifacts/<run-id>/checkpoints/bc-best-validation.pt demonstrations/expert.npz recovery/dagger.npz --episodes 10 --teacher-probability 0.15 --intervention-error 0.8`,
then `uv run trackmaniarl bc-train run.yaml --demo demonstrations --recovery recovery/dagger.npz`.

**Primary paper:** [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://proceedings.mlr.press/v15/ross11a.html).

### DQfD-style objectives

**Status and intuition.** The unified learner exposes
`DemonstrationMarginObjective` and `DemonstrationCrossEntropyObjective`. Replay
marks demonstration sequences, and the objectives apply only to marked valid
positions. This supports TD plus large-margin and/or classification guidance,
but it is **DQfD-style, not a full DQfD implementation**: the exact paper loss,
regularization, priority constants, schedule, and benchmark recipe are not
packaged as one named learner. `PolicyAnchorObjective` is also public, but no
built-in sampler produces its required `policy_anchor_q_values`; it needs a
custom batch producer.

**Contract, replay, execution, and state.** Use any unified discrete-value
composition. Imported demonstrations are protected from ring eviction while
they occupy less than half the store. `PrioritizedSampler` supports demo flags
for single-step or sequence batches; `DemoMixSampler` enforces explicit demo
fractions but only with `sequence_length: 1`. `offline-pretrain` performs the
configured number of demo-only updates, after which the same learner can train
locally or through the distributed coordinator.

**Parameters, metrics, and failures.** Margin/CE weights and margin size are
objective parameters; `offline_pretrain_updates`, n-step/gamma, replay alpha/beta
and demo sampling control the data path. The unified learner reports their sum
as `loss/objectives`; replay reports demo sample fractions. There are not yet
separate per-objective metrics. Demonstration actions excluded by
`policy_action_ids`, insufficient batch footprint, missing demo flags, an
over-half-full protected demo set, or an impossible `DemoMixSampler` fraction
fails explicitly. Demonstrations do not validate reward correctness.

**YAML fragment — demo-guided unified IQN:**

```yaml
components:
  learner:
    class_path: trackmaniarl.algorithms.value_based:DiscreteValueLearner
    kwargs:
      objectives:
        - class_path: trackmaniarl.algorithms.value_based:DemonstrationMarginObjective
          kwargs: {margin: 0.8, weight: 1.0, steering_switch_weight: 1.0}
        - class_path: trackmaniarl.algorithms.value_based:DemonstrationCrossEntropyObjective
          kwargs: {weight: 0.1, steering_switch_weight: 1.0}
  sampler:
    class_path: trackmaniarl.core.replay:PrioritizedSampler
    kwargs: {alpha: 0.6, beta: 0.4}
training:
  sequence_length: 1
  n_step: 3
  gamma: 0.995
  offline_pretrain_updates: 10000
  beta: 0.4
```

**Command:**
`uv run trackmaniarl offline-pretrain run.yaml --demo demonstrations`, then
continue from the checkpoint path printed by that command with
`uv run trackmaniarl resume run.yaml artifacts/<offline-run-id>/checkpoints/distributed-update-00010000.pt`.

**Primary paper:** [Deep Q-learning from Demonstrations](https://arxiv.org/abs/1704.03732).

## Choosing a starting point

- Start with the generated unified IQN configuration for discrete Trackmania
  control. Change only head/strategy to compare scalar Q, QR-DQN, or FQF under
  the same seed, replay, update budget, and evaluation suite.
- Use TQC when a continuous three-control telemetry baseline is intentional.
  SAC and REDQ require a project-owned model bundle before they are viable.
- Use PPO only when synchronous local on-policy collection and its lower data
  reuse are acceptable; do not route it to distributed actor/learner commands.
- Treat SD-SAC as an opt-in experiment with an explicit baseline and failure
  report.
- Use BC as an initialization/data-quality tool, then require closed-loop
  benchmarking. Add DAgger recovery or DQfD-style objectives only for a measured
  distribution-shift problem.
