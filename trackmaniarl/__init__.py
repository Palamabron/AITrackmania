"""TrackmaniaRL — extensible reinforcement-learning SDK for Trackmania."""

from importlib.metadata import version

__version__: str = version("trackmaniarl")

from trackmaniarl.core import RunSpec, Trainer, resolve_run

__all__ = ["RunSpec", "Trainer", "__version__", "resolve_run"]
