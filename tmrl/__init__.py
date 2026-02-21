"""TMRL: network-based framework for real-time robot learning (TrackMania 2020)."""

import platform
import sys

from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO")

if platform.system() == "Windows":
    try:
        import win32con  # noqa: F401
        import win32gui  # noqa: F401
        import win32ui  # noqa: F401
    except ImportError as e1:
        logger.info("pywin32 failed to import. Attempting to fix pywin32 installation...")
        from tmrl.tools.init_package.init_pywin32 import fix_pywin32

        try:
            fix_pywin32()
            import win32con  # noqa: F401
            import win32gui  # noqa: F401
            import win32ui  # noqa: F401
        except ImportError as e2:
            logger.error(
                "tmrl could not fix pywin32 on your system. The following exceptions were raised: "
                f"\n=== Exception 1 ===\n{e1!s}\n=== Exception 2 ===\n{e2!s}\n"
                "Please install pywin32 manually."
            )
            raise RuntimeError(
                "Please install pywin32 manually: https://github.com/mhammond/pywin32"
            ) from e2

# TMRL folder initialization (imports after platform-dependent block):
from tmrl.config.config_objects import CONFIG_DICT  # noqa: E402
from tmrl.envs import GenericGymEnv  # noqa: E402
from tmrl.tools.init_package.init_tmrl import TMRL_FOLDER  # noqa: E402, F401


def get_environment():
    """
    Default TMRL Gymnasium environment for TrackMania 2020.

    Returns:
        gymnasium.Env: An instance of the default TMRL Gymnasium environment
    """
    import tmrl.config.config_constants as cfg

    return GenericGymEnv(id=cfg.RTGYM_VERSION, gym_kwargs={"config": CONFIG_DICT})
