from __future__ import annotations

import argparse
from typing import Any, Protocol


class CommandParsers(Protocol):
    def add_parser(self, name: str, **kwargs: Any) -> argparse.ArgumentParser: ...
