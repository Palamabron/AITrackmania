"""Canonical re-export facade for tmrl training algorithm agents."""

from tmrl.custom.algorithms._internal._common import amp_setup, sanitize_obs, set_seed
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
