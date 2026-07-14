"""Optional remote tracker adapters; the local JSONL tracker remains the source of truth."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class WandbTracker:
    """Minimal neutral-event adapter for the optional ``tmrl[wandb]`` extra."""

    def __init__(self, project: str, entity: str | None = None, run_dir: str | None = None) -> None:
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("Install tmrl[wandb] to configure WandbTracker") from exc
        self._wandb: Any = wandb
        self._wandb.init(project=project, entity=entity, dir=run_dir, reinit="finish_previous")

    def log(self, event: str, payload: Mapping[str, Any], *, step: int | None = None) -> None:
        self._wandb.log({event + "/" + key: value for key, value in payload.items()}, step=step)

    def close(self) -> None:
        self._wandb.finish()
