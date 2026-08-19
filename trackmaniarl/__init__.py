"""TrackmaniaRL — extensible reinforcement-learning SDK for Trackmania."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("trackmaniarl")
except PackageNotFoundError:
    __version__ = "1.0.2"

from trackmaniarl.core import RunSpec, Trainer, resolve_run

__all__ = ["RunSpec", "Trainer", "__version__", "resolve_run"]
