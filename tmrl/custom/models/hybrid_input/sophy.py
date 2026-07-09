"""Sophy model family (re-export shim).

Content split into:
- sophy_legacy.py           — QRCNNSophy, SquashedActorSophy, SophyActorCritic
- sophy_residual_actor.py   — backbone helpers + SquashedActorSophyResidual
- sophy_residual_critic.py  — QRCNNSophyResidual
- sophy_ac.py               — SophyResidualActorCritic, AsymmetricSophyResidualActorCritic
"""

from tmrl.custom.models.hybrid_input.sophy_ac import (
    AsymmetricSophyResidualActorCritic as AsymmetricSophyResidualActorCritic,
)
from tmrl.custom.models.hybrid_input.sophy_ac import (
    SophyResidualActorCritic as SophyResidualActorCritic,
)
from tmrl.custom.models.hybrid_input.sophy_ac import (
    _AsymmetricActorAdapter as _AsymmetricActorAdapter,
)
from tmrl.custom.models.hybrid_input.sophy_legacy import (
    QRCNNSophy as QRCNNSophy,
)
from tmrl.custom.models.hybrid_input.sophy_legacy import (
    SophyActorCritic as SophyActorCritic,
)
from tmrl.custom.models.hybrid_input.sophy_legacy import (
    SquashedActorSophy as SquashedActorSophy,
)
from tmrl.custom.models.hybrid_input.sophy_legacy import (
    mlp as mlp,
)
from tmrl.custom.models.hybrid_input.sophy_residual_actor import (
    SquashedActorSophyResidual as SquashedActorSophyResidual,
)
from tmrl.custom.models.hybrid_input.sophy_residual_actor import (
    _build_track_conv1d_branch as _build_track_conv1d_branch,
)
from tmrl.custom.models.hybrid_input.sophy_residual_actor import (
    _build_track_spline_mlp_branch as _build_track_spline_mlp_branch,
)
from tmrl.custom.models.hybrid_input.sophy_residual_actor import (
    _make_backbone as _make_backbone,
)
from tmrl.custom.models.hybrid_input.sophy_residual_actor import (
    _obs_to_flat_tensor as _obs_to_flat_tensor,
)
from tmrl.custom.models.hybrid_input.sophy_residual_critic import (
    QRCNNSophyResidual as QRCNNSophyResidual,
)
