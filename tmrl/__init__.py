"""TMRL — extensible reinforcement-learning SDK for Trackmania."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("tmrl")
except PackageNotFoundError:
    __version__ = "1.0.0.dev"

from tmrl.core import RunSpec, Trainer, resolve_run

__all__ = ["RunSpec", "Trainer", "__version__", "resolve_run"]
