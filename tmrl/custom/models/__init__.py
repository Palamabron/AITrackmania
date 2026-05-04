"""TrackMania models grouped by accepted input modality.

Layout:
- ``shared``: reusable blocks/constants (`mlp`, residual backbones, EffNet stems).
- ``vector_input``: vector/lidar policies (MLP, residual MLP, GRU).
- ``image_input``: image-first policies (vanilla CNN, EfficientNet, IMPALA-like).
- ``hybrid_input``: mixed track+physics(+image) pipelines (Sophy family).
- ``discrete_actions``: IQN/discrete Q networks.

The package still re-exports common public classes for
`from tmrl.custom.models import ...`.
"""

from tmrl.custom.models.hybrid_input.gnn_effnet_sophy import (
    GnnEffNetSophyResidualActorCritic,
    QRCNNGnnEffNetSophyResidual,
    SquashedActorGnnEffNetSophyResidual,
    _build_track_gnn_branch,
    _obs_to_flat_tensor,
    _TrackGNN,
)
from tmrl.custom.models.image_input.efficientnet import (
    EffNetActorCritic,
    EffNetQFunction,
    FrozenEffNetResidualActorCritic,
    FrozenEffNetResidualQFunction,
    SquashedGaussianEffNetActor,
    SquashedGaussianFrozenEffNetResidualActor,
)
from tmrl.custom.models.image_input.vanilla_cnn_sac import (
    SquashedGaussianVanillaCNNActor,
    SquashedGaussianVanillaColorCNNActor,
    VanillaCNN,
    VanillaCNNActorCritic,
    VanillaCNNQFunction,
    VanillaColorCNNActorCritic,
    VanillaColorCNNQFunction,
    remove_colors,
)
from tmrl.custom.models.shared.base import (
    EPSILON,
    LOG_STD_MAX,
    LOG_STD_MIN,
    EffNetV2,
    MBConv,
    SELayer,
    SiLU,
    _cat_obs,
    _cat_obs_except_image,
    _ensure_float,
    _make_divisible,
    _obs_dim,
    _obs_spaces_list,
    _vector_dim_except,
    combined_shape,
    conv2d_out_dims,
    conv_1x1_bn,
    conv_3x3_bn,
    count_vars,
    effnetv2_l,
    effnetv2_m,
    effnetv2_s,
    effnetv2_xl,
    effnetv2_xs,
    mlp,
    num_flat_features,
)
from tmrl.custom.models.vector_input.sac_gru_actor_critic import (
    RNNActorCritic,
    RNNQFunction,
    SquashedGaussianRNNActor,
    build_stacked_gru,
)
from tmrl.custom.models.vector_input.sac_mlp_actor_critic import (
    MLPActorCritic,
    MLPQFunction,
    REDQMLPActorCritic,
    SquashedGaussianMLPActor,
)
from tmrl.custom.models.vector_input.sac_residual_mlp_actor_critic import (
    REDQResidualMLPActorCritic,
    ResidualMLPActorCritic,
    ResidualMLPQFunction,
    SquashedGaussianResidualMLPActor,
)

__all__ = [
    "EPSILON",
    "LOG_STD_MAX",
    "LOG_STD_MIN",
    "EffNetActorCritic",
    "EffNetQFunction",
    "EffNetV2",
    "FrozenEffNetResidualActorCritic",
    "FrozenEffNetResidualQFunction",
    "GnnEffNetSophyResidualActorCritic",
    "MBConv",
    "MLPActorCritic",
    "MLPQFunction",
    "QRCNNGnnEffNetSophyResidual",
    "REDQMLPActorCritic",
    "REDQResidualMLPActorCritic",
    "RNNActorCritic",
    "RNNQFunction",
    "ResidualMLPActorCritic",
    "ResidualMLPQFunction",
    "SELayer",
    "SiLU",
    "SquashedActorGnnEffNetSophyResidual",
    "SquashedGaussianEffNetActor",
    "SquashedGaussianFrozenEffNetResidualActor",
    "SquashedGaussianMLPActor",
    "SquashedGaussianRNNActor",
    "SquashedGaussianResidualMLPActor",
    "SquashedGaussianVanillaCNNActor",
    "SquashedGaussianVanillaColorCNNActor",
    "VanillaCNN",
    "VanillaCNNActorCritic",
    "VanillaCNNQFunction",
    "VanillaColorCNNActorCritic",
    "VanillaColorCNNQFunction",
    "_TrackGNN",
    "_build_track_gnn_branch",
    "_cat_obs",
    "_cat_obs_except_image",
    "_ensure_float",
    "_make_divisible",
    "_obs_dim",
    "_obs_spaces_list",
    "_obs_to_flat_tensor",
    "_vector_dim_except",
    "build_stacked_gru",
    "combined_shape",
    "conv2d_out_dims",
    "conv_1x1_bn",
    "conv_3x3_bn",
    "count_vars",
    "effnetv2_l",
    "effnetv2_m",
    "effnetv2_s",
    "effnetv2_xl",
    "effnetv2_xs",
    "mlp",
    "num_flat_features",
    "remove_colors",
]
