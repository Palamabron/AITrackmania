"""TrackmaniaRL — extensible reinforcement-learning SDK for Trackmania."""

from importlib.metadata import PackageNotFoundError, version

from trackmaniarl.core import RunSpec, Trainer, resolve_run


def _resolve_version() -> str:
    try:
        return version("trackmaniarl")
    except PackageNotFoundError:
        return "0+unknown"


__version__ = _resolve_version()

__all__ = ["RunSpec", "Trainer", "__version__", "resolve_run"]
