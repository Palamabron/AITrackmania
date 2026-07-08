"""Backward-compatibility shim — use ``tmrl.custom.algorithms`` instead.

.. deprecated::
    ``tmrl.custom.custom_algorithms`` was renamed to ``tmrl.custom.algorithms``
    in v0.9.0.  This shim re-exports everything so existing imports keep
    working, but will be removed in a future release.
"""

import warnings

warnings.warn(
    "tmrl.custom.custom_algorithms is deprecated and will be removed in a future release. "
    "Use tmrl.custom.algorithms or the canonical tmrl.algorithms namespace instead.",
    DeprecationWarning,
    stacklevel=2,
)

from tmrl.custom.algorithms import (  # noqa: E402
    IQNAgent,
    REDQSACAgent,
    SDSACAgent,
    SpinupSacAgent,
    TQCAgent,
    amp_setup,
    set_seed,
)

__all__ = [
    "IQNAgent",
    "REDQSACAgent",
    "SDSACAgent",
    "SpinupSacAgent",
    "TQCAgent",
    "amp_setup",
    "set_seed",
]
