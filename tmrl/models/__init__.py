"""Canonical namespace for neural network models, organized by input modality.

This package is a re-export facade over ``tmrl.custom.models``.  All model
classes remain in their original locations — nothing is moved — so existing
imports are unaffected.

Modality sub-groups
-------------------
- **vector_input** — MLP, Residual-MLP and GRU actors/critics for boundary
  lidar or pure vector observations.
- **image_input** — Vanilla CNN, EfficientNet and IMPALA heads for
  camera/image observations (optionally with scalar side inputs).
- **hybrid_input** — Sophy family and GNN+EffNet+Sophy pipelines that combine
  track-boundary encodings, vehicle telemetry, and an optional image branch.
- **discrete_actions** — IQN Q-networks and DQN actors for discrete action
  spaces.
- **shared** — Reusable building blocks (residual MLP, EfficientNetV2,
  SimbaV2, cosine embeddings, observation utilities, …).

Plugin extension
----------------
Register a custom model class so that tmrl can discover it by name::

    from tmrl.registry import MODELS

    @MODELS.register("my_actor_critic")
    class MyActorCritic(nn.Module):
        ...

For third-party packages, declare an entry point in your ``pyproject.toml``
(or ``setup.cfg``) under the ``"tmrl.models"`` group::

    [project.entry-points."tmrl.models"]
    my_actor_critic = "mypackage.models:MyActorCritic"

tmrl will call ``importlib.metadata.entry_points(group="tmrl.models")`` at
startup and register each discovered class automatically.

Usage
-----
::

    from tmrl.models import MLPActorCritic, ResidualMLPActorCritic
    from tmrl.models import IQNQNetwork, DQNActor
    from tmrl.models import VanillaCNNActorCritic
    from tmrl.models import SophyResidualActorCritic
"""

from __future__ import annotations

__all__: list[str] = []

# ---------------------------------------------------------------------------
# Shared building blocks and constants
# ---------------------------------------------------------------------------
try:
    from tmrl.custom.models.shared.blocks import (
        EPSILON,
        LOG_STD_MAX,
        LOG_STD_MIN,
        EffNetV2,
        FrozenEfficientNetEncoder,
        HypersphericalLinear,
        MBConv,
        ResidualMLPBlock,
        SELayer,
        SiLU,
        SimbaV2Backbone,
        cat_obs,
        cat_obs_except_image,
        combined_shape,
        conv2d_out_dims,
        count_vars,
        effnetv2_l,
        effnetv2_m,
        effnetv2_s,
        effnetv2_xl,
        effnetv2_xs,
        ensure_float,
        mlp,
        num_flat_features,
        obs_dim,
        obs_spaces_list,
        residual_mlp_backbone,
        simba_v2_backbone,
        squashed_logprob,
        vector_dim_except,
    )

    __all__ += [
        "EPSILON",
        "LOG_STD_MAX",
        "LOG_STD_MIN",
        "EffNetV2",
        "FrozenEfficientNetEncoder",
        "HypersphericalLinear",
        "MBConv",
        "ResidualMLPBlock",
        "SELayer",
        "SiLU",
        "SimbaV2Backbone",
        "cat_obs",
        "cat_obs_except_image",
        "combined_shape",
        "conv2d_out_dims",
        "count_vars",
        "effnetv2_l",
        "effnetv2_m",
        "effnetv2_s",
        "effnetv2_xl",
        "effnetv2_xs",
        "ensure_float",
        "mlp",
        "num_flat_features",
        "obs_dim",
        "obs_spaces_list",
        "residual_mlp_backbone",
        "simba_v2_backbone",
        "squashed_logprob",
        "vector_dim_except",
    ]
except Exception:  # pragma: no cover
    pass

# ---------------------------------------------------------------------------
# Vector / boundary-lidar models
# ---------------------------------------------------------------------------
try:
    from tmrl.custom.models.vector_input.mlp_actor_critic import (
        MLPActor,
        MLPActorCritic,
        MLPQFunction,
        REDQMLPActorCritic,
    )

    __all__ += [
        "MLPActor",
        "MLPActorCritic",
        "MLPQFunction",
        "REDQMLPActorCritic",
    ]
except Exception:  # pragma: no cover
    pass

try:
    from tmrl.custom.models.vector_input.residual_mlp_actor_critic import (
        REDQResidualMLPActorCritic,
        ResidualMLPActor,
        ResidualMLPActorCritic,
        ResidualMLPQFunction,
    )

    __all__ += [
        "REDQResidualMLPActorCritic",
        "ResidualMLPActor",
        "ResidualMLPActorCritic",
        "ResidualMLPQFunction",
    ]
except Exception:  # pragma: no cover
    pass

try:
    from tmrl.custom.models.vector_input.gru_actor_critic import (
        GRUActor,
        GRUActorCritic,
        GRUQFunction,
        build_stacked_gru,
    )

    __all__ += [
        "GRUActor",
        "GRUActorCritic",
        "GRUQFunction",
        "build_stacked_gru",
    ]
except Exception:  # pragma: no cover
    pass

# ---------------------------------------------------------------------------
# Image-input models
# ---------------------------------------------------------------------------
try:
    from tmrl.custom.models.image_input.vanilla_cnn import (
        RGBVanillaCNNActor,
        RGBVanillaCNNActorCritic,
        RGBVanillaCNNQFunction,
        VanillaCNN,
        VanillaCNNActor,
        VanillaCNNActorCritic,
        VanillaCNNQFunction,
        rgb_to_grayscale,
    )

    __all__ += [
        "RGBVanillaCNNActor",
        "RGBVanillaCNNActorCritic",
        "RGBVanillaCNNQFunction",
        "VanillaCNN",
        "VanillaCNNActor",
        "VanillaCNNActorCritic",
        "VanillaCNNQFunction",
        "rgb_to_grayscale",
    ]
except Exception:  # pragma: no cover
    pass

try:
    from tmrl.custom.models.image_input.efficientnet import (
        EffNetActor,
        FrozenEffNetResidualActor,
        FrozenEffNetResidualActorCritic,
        FrozenEffNetResidualQFunction,
    )

    __all__ += [
        "EffNetActor",
        "FrozenEffNetResidualActor",
        "FrozenEffNetResidualActorCritic",
        "FrozenEffNetResidualQFunction",
    ]
except Exception:  # pragma: no cover
    pass

try:
    from tmrl.custom.models.image_input.impala import (
        CNNModule,
        QRCNNActorCritic,
        QRCNNQFunction,
        SquashedActorQRCNN,
    )

    __all__ += [
        "CNNModule",
        "QRCNNActorCritic",
        "QRCNNQFunction",
        "SquashedActorQRCNN",
    ]
except Exception:  # pragma: no cover
    pass

# ---------------------------------------------------------------------------
# Hybrid (track + telemetry + optional image) models
# ---------------------------------------------------------------------------
try:
    from tmrl.custom.models.hybrid_input.sophy import (
        QRCNNSophy,
        QRCNNSophyResidual,
        SophyActorCritic,
        SophyResidualActorCritic,
        SquashedActorSophy,
        SquashedActorSophyResidual,
    )

    __all__ += [
        "QRCNNSophy",
        "QRCNNSophyResidual",
        "SophyActorCritic",
        "SophyResidualActorCritic",
        "SquashedActorSophy",
        "SquashedActorSophyResidual",
    ]
except Exception:  # pragma: no cover
    pass

try:
    from tmrl.custom.models.hybrid_input.gnn_effnet_sophy import (
        GnnEffNetSophyResidualActorCritic,
        QRCNNGnnEffNetSophyResidual,
        SquashedActorGnnEffNetSophyResidual,
    )

    __all__ += [
        "GnnEffNetSophyResidualActorCritic",
        "QRCNNGnnEffNetSophyResidual",
        "SquashedActorGnnEffNetSophyResidual",
    ]
except Exception:  # pragma: no cover
    pass

# ---------------------------------------------------------------------------
# Discrete-action models (IQN / DQN)
# ---------------------------------------------------------------------------
try:
    from tmrl.custom.models.discrete_actions.iqn_discrete_q_network import (
        CosineEmbedding,
        DQNActor,
        DuelingHead,
        IQNFeatureBackbone,
        IQNQNetwork,
    )

    __all__ += [
        "CosineEmbedding",
        "DQNActor",
        "DuelingHead",
        "IQNFeatureBackbone",
        "IQNQNetwork",
    ]
except Exception:  # pragma: no cover
    pass
