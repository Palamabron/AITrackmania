"""Minimal smoke tests for config validation and routing."""

from __future__ import annotations

import pytest
from tmrl.config.active_config_explainer import model_policy_route
from tmrl.config.loader import MAIN_CONFIG, main_config_snapshot_redacted
from tmrl.config.schema.main import MainConfig


def _validate_with(**overrides: dict) -> MainConfig:
    d = MAIN_CONFIG.model_dump()
    for section, patch in overrides.items():
        d[section].update(patch)
    return MainConfig.model_validate(d)


def test_removed_interface_is_rejected():
    env = MAIN_CONFIG.environment.model_copy(
        update={"rtgym_interface": "TM20LIDARPROGRESS", "use_images": False}
    )
    with pytest.raises(ValueError, match="removed"):
        MainConfig.model_validate(MAIN_CONFIG.model_dump() | {"environment": env.model_dump()})


def test_model_route_is_not_unsupported_for_default_config():
    route = model_policy_route(_validate_with())
    assert route != "unsupported"


def test_redacted_snapshot_exposes_expected_top_level_keys():
    snapshot = main_config_snapshot_redacted()
    assert isinstance(snapshot, dict)
    assert "schema_version" in snapshot
    assert "algorithm" in snapshot
