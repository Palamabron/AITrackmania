"""Public command-line facade for TrackmaniaRL."""

from trackmaniarl.commands.behavior_command import _bc_benchmark
from trackmaniarl.commands.common import (
    _inspect_config,
    _required_token,
    _resumed_attempt_spec,
    _validate,
)
from trackmaniarl.commands.demonstration_benchmark import _demo_benchmark
from trackmaniarl.commands.evaluation import (
    _bootstrap_median_interval,
    _wilson_interval,
)
from trackmaniarl.commands.helpers import _recovery_contract
from trackmaniarl.commands.parser import entrypoint
from trackmaniarl.commands.smoke import _restore_smoke_checkpoint, _smoke_training
from trackmaniarl.commands.training import _train
from trackmaniarl.commands.trajectory import _trajectory_optimize

__all__ = (
    "_bc_benchmark",
    "_bootstrap_median_interval",
    "_demo_benchmark",
    "_inspect_config",
    "_recovery_contract",
    "_required_token",
    "_restore_smoke_checkpoint",
    "_resumed_attempt_spec",
    "_smoke_training",
    "_train",
    "_trajectory_optimize",
    "_validate",
    "_wilson_interval",
    "entrypoint",
)
