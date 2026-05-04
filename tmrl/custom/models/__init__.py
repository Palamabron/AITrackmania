"""TrackMania RL models grouped by input modality.

Layout:
- ``shared``: reusable building blocks (mlp, residual MLP, EffNet, SimbaV2, …).
- ``vector_input``: vector/lidar policies (MLP, residual MLP, GRU).
- ``image_input``: image-first policies (vanilla CNN, frozen/trainable EfficientNet, IMPALA).
- ``hybrid_input``: mixed track+physics(+image) pipelines (Sophy family).
- ``discrete_actions``: IQN / discrete Q-networks.
"""

from tmrl.custom.models.hybrid_input.gnn_effnet_sophy import (
    GnnEffNetSophyResidualActorCritic,
    QRCNNGnnEffNetSophyResidual,
    SquashedActorGnnEffNetSophyResidual,
)
from tmrl.custom.models.image_input.efficientnet import (
    EffNetActor,
    FrozenEffNetResidualActor,
    FrozenEffNetResidualActorCritic,
    FrozenEffNetResidualQFunction,
)
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
from tmrl.custom.models.vector_input.gru_actor_critic import (
    GRUActor,
    GRUActorCritic,
    GRUQFunction,
    build_stacked_gru,
)
from tmrl.custom.models.vector_input.mlp_actor_critic import (
    MLPActor,
    MLPActorCritic,
    MLPQFunction,
    REDQMLPActorCritic,
)
from tmrl.custom.models.vector_input.residual_mlp_actor_critic import (
    REDQResidualMLPActorCritic,
    ResidualMLPActor,
    ResidualMLPActorCritic,
    ResidualMLPQFunction,
)

__all__ = [
    # Constants
    "EPSILON",
    "LOG_STD_MAX",
    "LOG_STD_MIN",
    "EffNetActor",
    # Shared building blocks
    "EffNetV2",
    "FrozenEffNetResidualActor",
    "FrozenEffNetResidualActorCritic",
    "FrozenEffNetResidualQFunction",
    "FrozenEfficientNetEncoder",
    # Vector-input actor-critics
    "GRUActor",
    "GRUActorCritic",
    "GRUQFunction",
    # Hybrid-input actor-critics
    "GnnEffNetSophyResidualActorCritic",
    "HypersphericalLinear",
    "MBConv",
    "MLPActor",
    "MLPActorCritic",
    "MLPQFunction",
    "QRCNNGnnEffNetSophyResidual",
    "REDQMLPActorCritic",
    "REDQResidualMLPActorCritic",
    # Image-input actor-critics
    "RGBVanillaCNNActor",
    "RGBVanillaCNNActorCritic",
    "RGBVanillaCNNQFunction",
    "ResidualMLPActor",
    "ResidualMLPActorCritic",
    "ResidualMLPBlock",
    "ResidualMLPQFunction",
    "SELayer",
    "SiLU",
    "SimbaV2Backbone",
    "SquashedActorGnnEffNetSophyResidual",
    "VanillaCNN",
    "VanillaCNNActor",
    "VanillaCNNActorCritic",
    "VanillaCNNQFunction",
    "build_stacked_gru",
    # Obs-space utilities
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
    "rgb_to_grayscale",
    "simba_v2_backbone",
    "squashed_logprob",
    "vector_dim_except",
]
