"""TrackmaniaRL — extensible reinforcement-learning SDK for Trackmania."""

from trackmaniarl._version import __version__
from trackmaniarl.core import RunSpec, Trainer, resolve_run

__all__ = ["RunSpec", "Trainer", "__version__", "resolve_run"]
