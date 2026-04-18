"""Memory / replay buffer configuration."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class MemoryConfig(BaseModel):
    """Selects and configures the replay buffer implementation."""

    model_config = ConfigDict(extra="forbid")

    memory_type: str = Field(
        default="auto",
        description=(
            "Registry key for the memory class. "
            "'auto' derives the memory type from the active interface/environment flags."
        ),
    )
