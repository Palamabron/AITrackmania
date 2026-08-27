from __future__ import annotations

import argparse
from enum import Enum
from pathlib import Path

from trackmaniarl.commands.common import _init, _inspect_config, _validate
from trackmaniarl.commands.distributed import _actor, _learner
from trackmaniarl.commands.parser_types import CommandParsers
from trackmaniarl.commands.smoke import _smoke
from trackmaniarl.commands.training import _offline_pretrain, _train


class _DemonstrationRequirement(Enum):
    OPTIONAL = "optional"
    REQUIRED = "required"


def register_training_commands(commands: CommandParsers) -> None:
    _register_project_commands(commands)
    _register_train(commands)
    _register_offline_pretrain(commands)
    _register_resume(commands)
    _register_learner(commands)
    _register_actor(commands)
    _register_smoke(commands)


def _register_project_commands(commands: CommandParsers) -> None:
    init = commands.add_parser("init", help="create an installable local extension project")
    init.add_argument("directory")
    init.add_argument("--package", help="Python package name (defaults to directory name)")
    init.add_argument("--template", choices=("starter", "trackmania"), default="starter")
    init.set_defaults(handler=_init)
    validate = commands.add_parser(
        "validate", help="resolve components and run a synthetic smoke update"
    )
    validate.add_argument("config", type=Path)
    validate.set_defaults(handler=_validate)
    inspect_config = commands.add_parser(
        "inspect-config", help="list configured component paths without importing them"
    )
    inspect_config.add_argument("config", type=Path)
    inspect_config.set_defaults(handler=_inspect_config)


def _add_demonstrations(
    parser: argparse.ArgumentParser, requirement: _DemonstrationRequirement
) -> None:
    required = requirement is _DemonstrationRequirement.REQUIRED
    parser.add_argument(
        "--demo",
        action="append",
        type=Path,
        required=required,
        default=None if required else [],
        help="demonstration .npz file or directory of .npz files (repeatable)",
    )


def _register_train(commands: CommandParsers) -> None:
    parser = commands.add_parser("train", help="start a local asynchronous learner and actor")
    parser.add_argument("config", type=Path)
    _add_demonstrations(parser, _DemonstrationRequirement.OPTIONAL)
    parser.add_argument(
        "--model-initialization-checkpoint",
        type=Path,
        help="warm-start the configured learner model from this checkpoint",
    )
    parser.set_defaults(handler=_train)


def _register_offline_pretrain(commands: CommandParsers) -> None:
    parser = commands.add_parser(
        "offline-pretrain",
        help="run configured learner updates from demonstrations without starting TrackMania",
    )
    parser.add_argument("config", type=Path)
    _add_demonstrations(parser, _DemonstrationRequirement.REQUIRED)
    parser.add_argument(
        "--model-initialization-checkpoint",
        type=Path,
        help="warm-start the configured learner model from this checkpoint",
    )
    parser.set_defaults(handler=_offline_pretrain)


def _register_resume(commands: CommandParsers) -> None:
    parser = commands.add_parser("resume", help="resume a local asynchronous training run")
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--reset-replay",
        action="store_true",
        help="restore learner state while starting with an empty replay and sampler",
    )
    _add_demonstrations(parser, _DemonstrationRequirement.OPTIONAL)
    parser.set_defaults(handler=_train)


def _register_learner(commands: CommandParsers) -> None:
    parser = commands.add_parser("learner", help="run a distributed coordinator/learner")
    parser.add_argument("config", type=Path)
    parser.add_argument("--bind")
    parser.add_argument("--checkpoint", type=Path)
    _add_demonstrations(parser, _DemonstrationRequirement.OPTIONAL)
    parser.set_defaults(handler=_learner)


def _register_actor(commands: CommandParsers) -> None:
    parser = commands.add_parser("actor", help="run a remote continuous rollout actor")
    parser.add_argument("config", type=Path)
    parser.add_argument("--connect", required=True)
    parser.add_argument("--actor-id")
    parser.set_defaults(handler=_actor)


def _register_smoke(commands: CommandParsers) -> None:
    parser = commands.add_parser(
        "smoke", help="run a bounded local async TrackMania actor/learner release gate"
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--transitions", type=int, default=100)
    parser.set_defaults(handler=_smoke)
