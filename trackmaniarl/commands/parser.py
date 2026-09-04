from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any

from trackmaniarl.commands.parser_assets import register_asset_commands
from trackmaniarl.commands.parser_trackmania import register_trackmania_commands
from trackmaniarl.commands.parser_training import register_training_commands


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trackmaniarl", description="TrackmaniaRL project tooling"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    register_training_commands(commands)
    register_trackmania_commands(commands)
    register_asset_commands(commands)
    return parser


def entrypoint(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    handler = args.handler
    if not callable(handler):
        raise TypeError("parsed CLI handler must be callable")
    callable_handler = handler
    typed_handler: Callable[[argparse.Namespace], Any] = callable_handler
    typed_handler(args)
