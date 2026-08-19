"""Extract lag-free (obs, action, reward, next_obs, done) tuples from a .Gbx ghost."""

from __future__ import annotations

import sys

from tmrl.cli import entrypoint


def main() -> None:
    entrypoint(["track", "extract-gbx", *sys.argv[1:]])


if __name__ == "__main__":
    main()
