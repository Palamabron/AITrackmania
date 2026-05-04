"""CRC debug check for compressed/decompressed sample round-trips."""

import zlib
from typing import Any

from loguru import logger


def check_samples_crc(
    original_prev_obs: Any,
    original_action: Any,
    original_obs: Any,
    original_reward: Any,
    original_done: Any,
    original_truncated: Any,
    rebuilt_prev_obs: Any,
    rebuilt_action: Any,
    rebuilt_obs: Any,
    rebuilt_reward: Any,
    rebuilt_done: Any,
    rebuilt_truncated: Any,
    debug_timestep: int,
    debug_timestep_since_reset: int,
) -> None:
    """Assert that compressed-then-decompressed samples match originals (CRC debug).

    Args:
        original_prev_obs: Previous observation from source buffer.
        original_action: Action from source buffer.
        original_obs: Observation from source buffer.
        original_reward: Reward from source buffer.
        original_done: Terminated flag from source buffer.
        original_truncated: Truncated flag from source buffer.
        rebuilt_prev_obs: Previous observation after round-trip.
        rebuilt_action: Action after round-trip.
        rebuilt_obs: Observation after round-trip.
        rebuilt_reward: Reward after round-trip.
        rebuilt_done: Terminated after round-trip.
        rebuilt_truncated: Truncated after round-trip.
        debug_timestep: Global time step for error message.
        debug_timestep_since_reset: Steps since last reset for error message.
    """
    ts_msg = f"Time step: {debug_timestep}, since reset: {debug_timestep_since_reset}"
    assert original_prev_obs is None or str(original_prev_obs) == str(rebuilt_prev_obs), (
        f"previous observations don't match:\noriginal:\n{original_prev_obs}\n!= rebuilt:\n"
        f"{rebuilt_prev_obs}\n{ts_msg}"
    )
    assert str(original_action) == str(rebuilt_action), (
        f"actions don't match:\noriginal:\n{original_action}\n!= rebuilt:\n"
        f"{rebuilt_action}\n{ts_msg}"
    )
    assert str(original_obs) == str(rebuilt_obs), (
        f"observations don't match:\noriginal:\n{original_obs}\n!= rebuilt:\n"
        f"{rebuilt_obs}\n{ts_msg}"
    )
    assert str(original_reward) == str(rebuilt_reward), (
        f"rewards don't match:\noriginal:\n{original_reward}\n!= rebuilt:\n"
        f"{rebuilt_reward}\n{ts_msg}"
    )
    assert str(original_done) == str(rebuilt_done), (
        f"terminated don't match:\noriginal:\n{original_done}\n!= rebuilt:\n"
        f"{rebuilt_done}\n{ts_msg}"
    )
    assert str(original_truncated) == str(rebuilt_truncated), (
        f"truncated don't match:\noriginal:\n{original_truncated}\n!= rebuilt:\n"
        f"{rebuilt_truncated}\n{ts_msg}"
    )
    original_crc = zlib.crc32(
        str.encode(
            str(
                (
                    original_action,
                    original_obs,
                    original_reward,
                    original_done,
                    original_truncated,
                )
            )
        )
    )
    rebuilt_crc = zlib.crc32(
        str.encode(
            str(
                (
                    rebuilt_action,
                    rebuilt_obs,
                    rebuilt_reward,
                    rebuilt_done,
                    rebuilt_truncated,
                )
            )
        )
    )
    assert rebuilt_crc == original_crc, (
        f"CRC failed: new crc:{rebuilt_crc} != old crc:{original_crc}. "
        "Pipeline corrupted or crc_debug False. "
        f"original:\n{(original_action, original_obs, original_reward, original_done)}\n"
        f"!= rebuilt:\n{(rebuilt_action, rebuilt_obs, rebuilt_reward, rebuilt_done)}\n"
        f"{ts_msg}"
    )
    logger.debug(
        "CRC check passed. Time step: {}, since reset: {}",
        debug_timestep,
        debug_timestep_since_reset,
    )
