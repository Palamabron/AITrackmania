"""Shared utilities for the networking package."""

import datetime
import os
import socket

from loguru import logger
from requests import get  # type: ignore[import-untyped]

_WORKER_SEND_CHUNK_DEFAULT = 512
_WORKER_SEND_CHUNK_MAX = 65536


def print_with_timestamp(message: str) -> None:
    """Log message with current date/time prefix."""
    timestamp = datetime.datetime.now().strftime("%x %X ")
    logger.info("{}{}", timestamp, message)


def print_ip():
    """Log the public and local IP addresses of the current machine."""
    try:
        public_ip = get("http://api.ipify.org", timeout=5).text
    except Exception:
        public_ip = "unavailable"
    local_ip = socket.gethostbyname(socket.gethostname())
    print_with_timestamp(f"public IP: {public_ip}, local IP: {local_ip}")


def _parse_worker_send_chunk_size(raw: str | None) -> int:
    """Parse ``TMRL_WORKER_SEND_CHUNK_SIZE`` safely (fallback to default, clamp to sane range)."""
    if raw is None or not str(raw).strip():
        return _WORKER_SEND_CHUNK_DEFAULT
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        logger.warning(
            f"Invalid TMRL_WORKER_SEND_CHUNK_SIZE={raw!r}; "
            f"falling back to default {_WORKER_SEND_CHUNK_DEFAULT}."
        )
        return _WORKER_SEND_CHUNK_DEFAULT
    return max(1, min(value, _WORKER_SEND_CHUNK_MAX))


def log_environment_variables():
    """Return selected environment variables for logging to experiment trackers.

    Reads the ``LOG_VARIABLES`` environment variable for a whitespace-separated
    list of variable names to capture. Example usage::

        LOG_VARIABLES='HOME JOBID' python train.py

    Returns:
        dict[str, str]: Mapping from variable name to its value (empty string if unset).
    """
    return {k: os.environ.get(k, "") for k in os.environ.get("LOG_VARIABLES", "").strip().split()}


def _find_nested_reward_function(root: object, *, max_depth: int = 6) -> object | None:
    """Best-effort lookup of a TMRL ``RewardFunction`` living under a Gymnasium env stack.

    Real-time-gym stacks vary by version; this is intentionally defensive and bounded.
    """
    try:
        from tmrl.custom.tm.utils.compute_reward import RewardFunction
    except Exception:  # pragma: no cover - import guard for exotic installs
        RewardFunction = ()  # type: ignore[assignment,misc]  # noqa: N806

    if RewardFunction and isinstance(root, RewardFunction):  # type: ignore[truthy-function]
        return root

    visited: set[int] = set()
    queue: list[tuple[object, int]] = [(root, 0)]

    def _enqueue(obj: object, depth: int) -> None:
        """Add *obj* to the BFS queue if not already visited and within depth limit."""
        if depth > max_depth:
            return
        oid = id(obj)
        if oid in visited:
            return
        visited.add(oid)
        queue.append((obj, depth))

    while queue:
        obj, depth = queue.pop(0)
        if RewardFunction and isinstance(obj, RewardFunction):  # type: ignore[truthy-function]
            return obj

        rf = getattr(obj, "reward_function", None)
        if RewardFunction and isinstance(rf, RewardFunction):  # type: ignore[truthy-function]
            return rf

        # Common gymnasium nesting
        for attr in ("unwrapped", "env", "_env"):
            child = getattr(obj, attr, None)
            if child is not None and child is not obj:
                _enqueue(child, depth + 1)

        # Common rtgym naming (best effort)
        for attr in ("interface", "real_time_interface", "rt_interface"):
            child = getattr(obj, attr, None)
            if child is not None and child is not obj:
                _enqueue(child, depth + 1)

    if len(visited) > 50:
        logger.debug(
            "Reward function lookup visited {} objects before returning None. "
            "Check env wrapper stack for excessive nesting.",
            len(visited),
        )

    return None


def _maybe_log_reward_on_rollout_truncation(env: object, info: object) -> None:
    """Flush worker reward/W&B episode logs when rollout forces truncation."""
    if not isinstance(info, dict) or not info.get("env_truncated"):
        return
    rf = _find_nested_reward_function(env)
    if rf is None:
        return
    if getattr(rf, "_logged_run_this_episode", False):
        return
    end_of_track = bool(info.get("end_of_track", False))
    try:
        rf.log_model_run(terminated=False, end_of_track=end_of_track, truncated=True)  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Worker reward logging on truncation failed: {}", exc)
