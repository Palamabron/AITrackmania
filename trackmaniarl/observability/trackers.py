"""Optional remote tracker adapters; the local JSONL tracker remains the source of truth."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from queue import Full, Queue
from threading import Thread
from typing import Any


def _wandb_metric_name(event: str, key: str) -> str:
    if event == "train/episode":
        return f"episode/{key}"
    if event == "train/update":
        learner_aliases = {
            "debug/gradient_norm_max": "gradient_norm_max",
            "debug/gradient_clipped_fraction": "clipped_fraction",
            "debug/q_selected_mean": "q_mean",
            "debug/q_selected_max": "q_max",
            "debug/q_selected_abs_max": "q_abs_max",
        }
        if key in learner_aliases:
            return f"learner/{learner_aliases[key]}"
        if key.startswith(("loss/", "debug/")):
            return f"learner/{key}"
        if key.startswith("timing/"):
            return f"performance/{key.removeprefix('timing/')}"
        if key in {"replay_size", "replay_fill_fraction", "per_beta"}:
            return f"replay/{key.removeprefix('replay_')}"
        return f"training/{key}"
    if event == "distributed/ingest":
        return f"actor/{key}"
    if event == "distributed/policy_published":
        return f"actor/policy/{key}"
    return f"{event}/{key}"


def _load_wandb_key_from_dotenv(run_dir: str | None) -> Path | None:
    """Load ``WANDB_API_KEY`` from the nearest project ``.env`` if necessary.

    W&B's standalone CLI does not load dotenv files.  Training does, while
    preserving an explicit environment variable as the higher-priority value.
    """

    if os.environ.get("WANDB_API_KEY"):
        return None
    starts = [Path.cwd()]
    if run_dir:
        starts.append(Path(run_dir))
    checked: set[Path] = set()
    for start in starts:
        for directory in (start, *start.parents):
            candidate = directory / ".env"
            if candidate in checked:
                continue
            checked.add(candidate)
            if not candidate.is_file():
                continue
            for line in candidate.read_text(encoding="utf-8").splitlines():
                text = line.strip()
                if not text or text.startswith("#"):
                    continue
                if text.startswith("export "):
                    text = text[7:].lstrip()
                key, separator, value = text.partition("=")
                if separator and key.strip() == "WANDB_API_KEY":
                    api_key = value.strip().strip('"').strip("'")
                    if api_key:
                        os.environ["WANDB_API_KEY"] = api_key
                        return candidate
    return None


class WandbTracker:
    """Asynchronous neutral-event adapter for the optional ``trackmaniarl[wandb]`` extra."""

    def __init__(
        self,
        project: str,
        entity: str | None = None,
        run_dir: str | None = None,
        run_id: str | None = None,
        config: Mapping[str, Any] | None = None,
        queue_size: int = 10_000,
    ) -> None:
        if queue_size < 1:
            raise ValueError("W&B queue_size must be positive")
        _load_wandb_key_from_dotenv(run_dir)
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("Install trackmaniarl[wandb] to configure WandbTracker") from exc
        self._wandb: Any = wandb
        settings = self._wandb.Settings(
            console="off",
            x_graphql_timeout_seconds=10,
            x_service_wait=5,
        )
        run = self._wandb.init(
            project=project,
            entity=entity,
            dir=run_dir,
            name=run_id,
            config=dict(config) if config is not None else None,
            reinit="finish_previous",
            settings=settings,
        )
        url = getattr(run, "url", None)
        if url:
            print(f"Weights & Biases run: {url}", flush=True)
        self._events: Queue[dict[str, Any] | None] = Queue(maxsize=queue_size)
        self._enabled = True
        self._dropped_events = 0
        self._worker = Thread(target=self._send_events, name="trackmaniarl-wandb", daemon=True)
        self._worker.start()

    def log(self, event: str, payload: Mapping[str, Any], *, step: int | None = None) -> None:
        del step
        if not self._enabled:
            return
        values = {_wandb_metric_name(event, key): value for key, value in payload.items()}
        try:
            self._events.put_nowait(values)
        except Full:
            self._dropped_events += 1
            if self._dropped_events == 1:
                print(
                    "W&B event queue is full; dropping remote metrics until it recovers.",
                    flush=True,
                )

    def close(self) -> None:
        try:
            self._events.put_nowait(None)
        except Full:
            print(
                "W&B event queue did not drain before shutdown; remote metrics may be incomplete.",
                flush=True,
            )
            return
        self._worker.join(timeout=5.0)
        if self._worker.is_alive():
            print(
                "W&B worker did not stop within 5 seconds; leaving it to process shutdown.",
                flush=True,
            )
            return
        self._wandb.finish(exit_code=0)

    def _send_events(self) -> None:
        while True:
            values = self._events.get()
            try:
                if values is None:
                    return
                if not self._enabled:
                    continue
                self._wandb.log(values)
            except Exception as exc:
                self._enabled = False
                print(f"W&B logging disabled after {type(exc).__name__}: {exc}", flush=True)
            finally:
                self._events.task_done()
