"""Shared neural network building blocks (re-export shim).

Content split into:
- nn_utils.py  — constants, basic utilities, obs-space helpers
- effnet.py    — EfficientNetV2 and FrozenEfficientNetEncoder
- backbones.py — ResidualMLP, SimbaV2, squashed_logprob
"""

from tmrl.custom.models.shared.backbones import (
    HypersphericalLinear as HypersphericalLinear,
)
from tmrl.custom.models.shared.backbones import (
    ResidualMLPBlock as ResidualMLPBlock,
)
from tmrl.custom.models.shared.backbones import (
    SimbaV2Backbone as SimbaV2Backbone,
)
from tmrl.custom.models.shared.backbones import (
    SimbaV2Block as SimbaV2Block,
)
from tmrl.custom.models.shared.backbones import (
    _l2_normalize as _l2_normalize,
)
from tmrl.custom.models.shared.backbones import (
    residual_mlp_backbone as residual_mlp_backbone,
)
from tmrl.custom.models.shared.backbones import (
    simba_v2_backbone as simba_v2_backbone,
)
from tmrl.custom.models.shared.backbones import (
    squashed_logprob as squashed_logprob,
)
from tmrl.custom.models.shared.effnet import (
    _EFFNET_VARIANTS as _EFFNET_VARIANTS,
)
from tmrl.custom.models.shared.effnet import (
    EffNetV2 as EffNetV2,
)
from tmrl.custom.models.shared.effnet import (
    FrozenEfficientNetEncoder as FrozenEfficientNetEncoder,
)
from tmrl.custom.models.shared.effnet import (
    MBConv as MBConv,
)
from tmrl.custom.models.shared.effnet import (
    SELayer as SELayer,
)
from tmrl.custom.models.shared.effnet import (
    conv_1x1_bn as conv_1x1_bn,
)
from tmrl.custom.models.shared.effnet import (
    conv_3x3_bn as conv_3x3_bn,
)
from tmrl.custom.models.shared.effnet import (
    conv_dw_3x3_bn as conv_dw_3x3_bn,
)
from tmrl.custom.models.shared.effnet import (
    effnetv2_l as effnetv2_l,
)
from tmrl.custom.models.shared.effnet import (
    effnetv2_m as effnetv2_m,
)
from tmrl.custom.models.shared.effnet import (
    effnetv2_s as effnetv2_s,
)
from tmrl.custom.models.shared.effnet import (
    effnetv2_xl as effnetv2_xl,
)
from tmrl.custom.models.shared.effnet import (
    effnetv2_xs as effnetv2_xs,
)
from tmrl.custom.models.shared.nn_utils import (
    EPSILON as EPSILON,
)
from tmrl.custom.models.shared.nn_utils import (
    LOG_STD_MAX as LOG_STD_MAX,
)
from tmrl.custom.models.shared.nn_utils import (
    LOG_STD_MIN as LOG_STD_MIN,
)
from tmrl.custom.models.shared.nn_utils import (
    SiLU as SiLU,
)
from tmrl.custom.models.shared.nn_utils import (
    _make_divisible as _make_divisible,
)
from tmrl.custom.models.shared.nn_utils import (
    cat_obs as cat_obs,
)
from tmrl.custom.models.shared.nn_utils import (
    cat_obs_except_image as cat_obs_except_image,
)
from tmrl.custom.models.shared.nn_utils import (
    combined_shape as combined_shape,
)
from tmrl.custom.models.shared.nn_utils import (
    conv2d_out_dims as conv2d_out_dims,
)
from tmrl.custom.models.shared.nn_utils import (
    count_vars as count_vars,
)
from tmrl.custom.models.shared.nn_utils import (
    ensure_float as ensure_float,
)
from tmrl.custom.models.shared.nn_utils import (
    mlp as mlp,
)
from tmrl.custom.models.shared.nn_utils import (
    num_flat_features as num_flat_features,
)
from tmrl.custom.models.shared.nn_utils import (
    obs_dim as obs_dim,
)
from tmrl.custom.models.shared.nn_utils import (
    obs_spaces_list as obs_spaces_list,
)
from tmrl.custom.models.shared.nn_utils import (
    vector_dim_except as vector_dim_except,
)
