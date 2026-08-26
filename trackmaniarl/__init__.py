"""TrackmaniaRL — extensible reinforcement-learning SDK for Trackmania."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("trackmaniarl")
except PackageNotFoundError:
    __version__ = "2.0.0rc1"

from trackmaniarl.core import RunSpec, Trainer, resolve_run

__all__ = ["RunSpec", "Trainer", "__version__", "resolve_run"]
