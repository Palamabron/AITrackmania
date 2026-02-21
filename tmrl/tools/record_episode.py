"""Record an episode to store in the replay buffer."""

from loguru import logger


def record_episode():
    """Record an episode in TrackMania for replay buffer storage."""
    logger.warning(
        "record_episode is a placeholder. Implement this to record episodes for your replay buffer."
    )
    raise NotImplementedError(
        "record_episode is not yet implemented. "
        "Extend this module to add episode recording functionality."
    )
