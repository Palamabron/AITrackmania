"""Distributed learner and actor commands."""

from __future__ import annotations

import argparse
import signal
from typing import Any

from trackmaniarl.commands.common import _configure_process_logging, _required_token
from trackmaniarl.core.spec import RunSpec
from trackmaniarl.distributed.actor_requests import ActorProcessRequest
from trackmaniarl.distributed.coordinator_types import (
    LearnerProcessRequest,
    ReplayRestoreMode,
)
from trackmaniarl.trackmania.demonstrations import resolve_demonstration_paths


def _learner(args: argparse.Namespace) -> None:
    config = args.config.resolve()
    spec = RunSpec.from_yaml(config)
    demos = resolve_demonstration_paths(args.demo)
    request = LearnerProcessRequest(
        str(config),
        args.bind or f"127.0.0.1:{spec.distributed.port}",
        _required_token(config),
        str(args.checkpoint) if args.checkpoint else None,
        ReplayRestoreMode.LEARNER_ONLY if args.reset_replay else ReplayRestoreMode.FULL,
        demo_paths=tuple(str(path) for path in demos),
    )
    _learner_process(request)


def _actor(args: argparse.Namespace) -> None:
    config = args.config.resolve()
    request = ActorProcessRequest(str(config), args.connect, args.actor_id, _required_token(config))
    _actor_process(request)


def _learner_process(request: LearnerProcessRequest) -> None:
    _configure_child_process(request.external_stop)
    try:
        from trackmaniarl.distributed.coordinator_runtime import learner_process_entry
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install trackmaniarl[distributed] to use distributed training") from exc
    learner_process_entry(request)


def _actor_process(request: ActorProcessRequest) -> None:
    _configure_child_process(request.external_stop)
    try:
        from trackmaniarl.distributed.actor import actor_process_entry
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install trackmaniarl[distributed] to use distributed training") from exc
    actor_process_entry(request)


def _configure_child_process(external_stop: Any | None) -> None:
    _configure_process_logging()
    if external_stop is not None:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
