"""Canonical public namespace for tmrl built-in training algorithms.

This module re-exports all algorithm agent classes and shared training
utilities from their implementation modules under
``tmrl.custom.algorithms``.  The ``tmrl.custom.*`` paths remain
fully importable for backwards compatibility; this namespace is the
documented, stable API going forward.

Extending tmrl with new algorithms
-----------------------------------
Register custom algorithms via Python entry-points so they are
discovered automatically at import time::

    # pyproject.toml
    [project.entry-points."tmrl.algorithms"]
    my_algo = "my_package.my_algo:MyAgent"

Or use the registry decorator directly::

    from tmrl.registry import ALGORITHMS

    @ALGORITHMS.register("my_algo")
    class MyAgent:
        ...

Registered algorithms are then available through the orchestrator and
config system without modifying this file.

Available agents
-----------------
- :class:`SpinupSacAgent`  — Soft Actor-Critic (Haarnoja et al., 2018)
- :class:`REDQSACAgent`    — REDQ-SAC randomized ensemble (Chen et al., 2021)
- :class:`TQCAgent`        — Truncated Quantile Critics (Kuznetsov et al., 2020)
- :class:`IQNAgent`        — Implicit Quantile Networks + Double DQN (Dabney et al., 2018)
- :class:`SDSACAgent`      — Stable Discrete SAC (Zhou et al., TMLR 2024)
"""

from tmrl.custom.algorithms._common import amp_setup, sanitize_obs, set_seed
from tmrl.custom.algorithms.iqn import IQNAgent
from tmrl.custom.algorithms.redq_sac import REDQSACAgent
from tmrl.custom.algorithms.sac import SpinupSacAgent
from tmrl.custom.algorithms.sdsac import SDSACAgent
from tmrl.custom.algorithms.tqc import TQCAgent

__all__ = [
    # Agent classes
    "IQNAgent",
    "REDQSACAgent",
    "SDSACAgent",
    "SpinupSacAgent",
    "TQCAgent",
    # Shared utilities
    "amp_setup",
    "sanitize_obs",
    "set_seed",
]
