"""Smoke tests for config validation and routing."""

from __future__ import annotations

import pytest
from tmrl.config.active_config_explainer import (
    ROUTE_ACTIVE_MODEL_FIELDS,
    active_model_field_names,
    explain_active_config_text,
    model_policy_route,
)
from tmrl.config.loader import MAIN_CONFIG, main_config_snapshot_redacted
from tmrl.config.schema.main import MainConfig


def _validate_with(**overrides: dict) -> MainConfig:
    d = MAIN_CONFIG.model_dump()
    for section, patch in overrides.items():
        d[section].update(patch)
    return MainConfig.model_validate(d)


# ---------------------------------------------------------------------------
# Existing smoke tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Route-matrix tests: supported algorithm + interface pairs
# ---------------------------------------------------------------------------


def test_lidar_iqn_route():
    """TM20LIDAR (default) + IQN selects the lidar_iqn route."""
    cfg = _validate_with(algorithm={"name": "IQN"})
    assert model_policy_route(cfg) == "lidar_iqn"


def test_lidar_sac_plain_mlp_route():
    """TM20LIDAR + SAC without residual MLP selects lidar_plain_mlp."""
    cfg = _validate_with(algorithm={"name": "SAC"}, model={"use_residual_mlp": False})
    assert model_policy_route(cfg) == "lidar_plain_mlp"


def test_lidar_residual_route():
    """TM20LIDAR + SAC with use_residual_mlp=True selects lidar_residual."""
    cfg = _validate_with(algorithm={"name": "SAC"}, model={"use_residual_mlp": True})
    assert model_policy_route(cfg) == "lidar_residual"


def test_vanilla_gray_route():
    """Non-lidar, non-advanced interface + SAC + img_grayscale=True → vanilla_gray."""
    cfg = _validate_with(algorithm={"name": "SAC"}, environment={"rtgym_interface": "TM20STANDARD"})
    assert model_policy_route(cfg) == "vanilla_gray"


# ---------------------------------------------------------------------------
# Route-matrix tests: unsupported algorithm + interface pairs
# ---------------------------------------------------------------------------


def test_lidar_tqc_is_unsupported():
    """TM20LIDAR + TQC has no runtime branch and should return unsupported."""
    cfg = _validate_with(algorithm={"name": "TQC"})
    assert model_policy_route(cfg) == "unsupported"


def test_vanilla_interface_non_sac_is_unsupported():
    """Non-lidar, non-advanced interface + REDQSAC → unsupported."""
    cfg = _validate_with(
        environment={"rtgym_interface": "TM20STANDARD"},
        algorithm={"name": "REDQSAC"},
    )
    assert model_policy_route(cfg) == "unsupported"


# ---------------------------------------------------------------------------
# Active-field regression tests
# ---------------------------------------------------------------------------


def test_active_fields_for_lidar_iqn_include_expected_keys():
    """lidar_iqn active fields include IQN-specific backbone knobs."""
    cfg = _validate_with(algorithm={"name": "IQN"})
    active = active_model_field_names(cfg)
    assert "residual_mlp_hidden_dim" in active
    assert "split_track_observation" in active
    assert "track_encoder" in active
    # frozen effnet fields are not read on the lidar_iqn route
    assert "use_frozen_effnet" not in active


def test_route_active_model_fields_covers_all_non_full_routes():
    """ROUTE_ACTIVE_MODEL_FIELDS maps every non-full (constrained) route."""
    expected_non_full = {
        "lidar_iqn",
        "lidar_sdsac",
        "lidar_sac_frozen_effnet",
        "lidar_residual",
        "lidar_plain_mlp",
        "adv_iqn",
        "adv_sdsac",
        "adv_sac_frozen_effnet",
    }
    assert expected_non_full.issubset(ROUTE_ACTIVE_MODEL_FIELDS.keys())


def test_unsupported_explain_text_contains_warning():
    """explain_active_config_text for an unsupported route emits a WARNING block."""
    cfg = _validate_with(algorithm={"name": "TQC"})
    text = explain_active_config_text(cfg)
    assert "WARNING" in text
    assert "unsupported" in text


def test_reward_normalize_scale_rejects_stale_divide_by_n_values():
    with pytest.raises(ValueError, match="reward_normalize_scale"):
        _validate_with(algorithm={"name": "IQN", "reward_normalize_scale": 200.0})


def test_iqn_lr_total_steps_must_exceed_warmup_when_cosine_decay_enabled():
    with pytest.raises(ValueError, match="iqn_lr_total_steps"):
        _validate_with(
            algorithm={
                "name": "IQN",
                "iqn_lr_cosine_decay": True,
                "iqn_lr_warmup_steps": 50_000,
                "iqn_lr_total_steps": 50_000,
            }
        )
