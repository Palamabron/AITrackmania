"""IQN Q-network and DQN actor (re-export shim)."""

from tmrl.custom.models.discrete_actions.iqn_backbone import (
    _IQN_BACKBONE_KWARGS as _IQN_BACKBONE_KWARGS,
)
from tmrl.custom.models.discrete_actions.iqn_backbone import (
    _IQN_OUTPUT_INIT_GAIN as _IQN_OUTPUT_INIT_GAIN,
)
from tmrl.custom.models.discrete_actions.iqn_backbone import (
    CosineEmbedding as CosineEmbedding,
)
from tmrl.custom.models.discrete_actions.iqn_backbone import (
    IQNFeatureBackbone as IQNFeatureBackbone,
)
from tmrl.custom.models.discrete_actions.iqn_backbone import (
    _init_cosine_embedding as _init_cosine_embedding,
)
from tmrl.custom.models.discrete_actions.iqn_backbone import (
    _init_linear_small as _init_linear_small,
)
from tmrl.custom.models.discrete_actions.iqn_backbone import (
    _init_noisy_linear_small as _init_noisy_linear_small,
)
from tmrl.custom.models.discrete_actions.iqn_head_actor import (
    DQNActor as DQNActor,
)
from tmrl.custom.models.discrete_actions.iqn_head_actor import (
    DuelingHead as DuelingHead,
)
from tmrl.custom.models.discrete_actions.iqn_head_actor import (
    IQNQNetwork as IQNQNetwork,
)
from tmrl.custom.models.discrete_actions.iqn_head_actor import (
    _init_dueling_output_layers as _init_dueling_output_layers,
)
from tmrl.custom.models.discrete_actions.iqn_head_actor import (
    _init_iqn_q_head as _init_iqn_q_head,
)
