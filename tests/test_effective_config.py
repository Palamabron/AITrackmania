"""Tests for ``tmrl.config.effective_config`` routing and IQN footgun warnings."""

from __future__ import annotations

import warnings

import pytest
from tmrl.config.effective_config import (
    active_model_field_names,
    build_interface_context,
    model_policy_route,
)
from tmrl.config.loader import MAIN_CONFIG, main_config_snapshot_redacted
from tmrl.config.schema.main import MainConfig


def test_build_interface_context_lidar_progress_marks_lidar():
    env = MAIN_CONFIG.environment.model_copy(
        update={"rtgym_interface": "TM20LIDARPROGRESS", "use_images": False}
    )
    ctx = build_interface_context(env)
    assert ctx.use_lidar_observations is True


def test_model_policy_route_lidar_iqn():
    d = MAIN_CONFIG.model_dump()
    d["algorithm"]["name"] = "IQN"
    d["environment"]["rtgym_interface"] = "LIDAR"
    m = MainConfig.model_validate(d)
    assert model_policy_route(m) == "lidar_iqn"


def test_model_policy_route_lidar_sdsac():
    d = MAIN_CONFIG.model_dump()
    d["algorithm"]["name"] = "SDSAC"
    d["environment"]["rtgym_interface"] = "LIDAR"
    m = MainConfig.model_validate(d)
    assert model_policy_route(m) == "lidar_sdsac"


def test_active_model_fields_iqn_includes_trunk_not_actor_critic_split():
    d = MAIN_CONFIG.model_dump()
    d["algorithm"]["name"] = "IQN"
    d["environment"]["rtgym_interface"] = "LIDAR"
    m = MainConfig.model_validate(d)
    active = active_model_field_names(m)
    assert "residual_mlp_num_blocks" in active
    assert "residual_mlp_num_blocks_actor" not in active
    assert "split_track_observation" in active


def test_active_model_fields_sdsac_includes_actor_critic_blocks():
    d = MAIN_CONFIG.model_dump()
    d["algorithm"]["name"] = "SDSAC"
    d["environment"]["rtgym_interface"] = "LIDAR"
    m = MainConfig.model_validate(d)
    active = active_model_field_names(m)
    assert "residual_mlp_num_blocks_actor" in active
    assert "residual_mlp_num_blocks_critic" in active


def test_iqn_warns_when_actor_critic_depths_differ_from_num_blocks():
    d = MAIN_CONFIG.model_dump()
    d["algorithm"]["name"] = "IQN"
    d["model"]["residual_mlp_num_blocks"] = 4
    d["model"]["residual_mlp_num_blocks_actor"] = 2
    d["model"]["residual_mlp_num_blocks_critic"] = 4
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        MainConfig.model_validate(d)
    assert any("IQN uses only" in str(x.message) for x in w)


def test_iqn_no_warning_when_actor_critic_match_num_blocks():
    d = MAIN_CONFIG.model_dump()
    d["algorithm"]["name"] = "IQN"
    d["model"]["residual_mlp_num_blocks"] = 6
    d["model"]["residual_mlp_num_blocks_actor"] = 6
    d["model"]["residual_mlp_num_blocks_critic"] = 6
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        MainConfig.model_validate(d)
    assert not any("IQN uses only" in str(x.message) for x in w)


def test_iqn_rejects_vanilla_cnn_preset():
    d = MAIN_CONFIG.model_dump()
    d["algorithm"]["name"] = "IQN"
    d["environment"]["rtgym_interface"] = "LIDAR"
    d["model"]["type"] = "vanilla_cnn_actor_critic"
    with pytest.raises(ValueError, match="discrete-action-capable"):
        MainConfig.model_validate(d)


def test_iqn_accepts_mlp_preset():
    d = MAIN_CONFIG.model_dump()
    d["algorithm"]["name"] = "IQN"
    d["environment"]["rtgym_interface"] = "LIDAR"
    d["model"]["type"] = "mlp_actor_critic"
    MainConfig.model_validate(d)


def test_main_config_snapshot_redacted_is_jsonish_tree():
    s = main_config_snapshot_redacted()
    assert isinstance(s, dict)
    assert "schema_version" in s
    assert "algorithm" in s
    w = s.get("wandb")
    if isinstance(w, dict) and "api_key" in w and w["api_key"]:
        assert w["api_key"] == "<redacted>"
