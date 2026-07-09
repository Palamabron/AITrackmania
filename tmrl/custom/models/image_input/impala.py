"""IMPALA-style CNN + QR-CNN models (re-export shim)."""

from tmrl.custom.models.image_input._impala_utils import (
    gru as gru,
)
from tmrl.custom.models.image_input._impala_utils import (
    init_kaiming as init_kaiming,
)
from tmrl.custom.models.image_input._impala_utils import (
    lstm as lstm,
)
from tmrl.custom.models.image_input._impala_utils import (
    mlp as mlp,
)
from tmrl.custom.models.image_input.impala_actor_critic import (
    QRCNNActorCritic as QRCNNActorCritic,
)
from tmrl.custom.models.image_input.impala_actor_critic import (
    QRCNNQFunction as QRCNNQFunction,
)
from tmrl.custom.models.image_input.impala_actor_critic import (
    SquashedActorQRCNN as SquashedActorQRCNN,
)
from tmrl.custom.models.image_input.impala_encoder import CNNModule as CNNModule
