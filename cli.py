"""Top-level CLI entry point.

Equivalent to ``python -m tmrl`` or the ``tmrl`` console script installed by
``pip install tmrl``.  Run with ``--help`` to see all available commands.

Examples:
    python cli.py --server
    python cli.py --trainer --no-wandb
    python cli.py --worker
    python cli.py --record-reward
    python cli.py --record-track --record-track-side right
"""

from tmrl.__main__ import entrypoint

if __name__ == "__main__":
    entrypoint()
