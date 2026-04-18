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


# ---------------------------------------------------------------------------
# Route alignment: every supported (algorithm, interface) pair must produce
# a known route in model_policy_route and never "unsupported".
# ---------------------------------------------------------------------------

_LIDAR_IFACE = "LIDAR"
_ADVANCED_IFACE = "TQCGRAB_IMAGES"
_VANILLA_IFACE = "TM2020"

_SUPPORTED_LIDAR = [("SAC", _LIDAR_IFACE), ("REDQSAC", _LIDAR_IFACE),
                    ("IQN", _LIDAR_IFACE), ("SDSAC", _LIDAR_IFACE)]
_SUPPORTED_ADVANCED = [("TQC", _ADVANCED_IFACE), ("SAC", _ADVANCED_IFACE),
                       ("IQN", _ADVANCED_IFACE), ("SDSAC", _ADVANCED_IFACE)]
_SUPPORTED_VANILLA = [("SAC", _VANILLA_IFACE)]

_UNSUPPORTED = [
    ("TQC", _LIDAR_IFACE),
    ("REDQSAC", _ADVANCED_IFACE),
    ("TQC", _VANILLA_IFACE),
    ("IQN", _VANILLA_IFACE),
    ("SDSAC", _VANILLA_IFACE),
    ("REDQSAC", _VANILLA_IFACE),
]


@pytest.mark.parametrize("alg,iface", _SUPPORTED_LIDAR + _SUPPORTED_ADVANCED + _SUPPORTED_VANILLA)
def test_route_is_known_for_supported_combos(alg, iface):
    d = MAIN_CONFIG.model_dump()
    d["algorithm"]["name"] = alg
    d["environment"]["rtgym_interface"] = iface
    m = MainConfig.model_validate(d)
    route = model_policy_route(m)
    assert route != "unsupported", f"{alg}+{iface} should be supported but got 'unsupported'"


@pytest.mark.parametrize("alg,iface", _UNSUPPORTED)
def test_route_is_unsupported_for_invalid_combos(alg, iface):
    d = MAIN_CONFIG.model_dump()
    d["algorithm"]["name"] = alg
    d["environment"]["rtgym_interface"] = iface
    m = MainConfig.model_validate(d)
    route = model_policy_route(m)
    assert route == "unsupported", f"{alg}+{iface} should be unsupported but got {route!r}"


def test_explain_active_config_unsupported_does_not_crash():
    from tmrl.config.effective_config import explain_active_config_text
    d = MAIN_CONFIG.model_dump()
    d["algorithm"]["name"] = "TQC"
    d["environment"]["rtgym_interface"] = "LIDAR"
    m = MainConfig.model_validate(d)
    text = explain_active_config_text(m)
    assert "unsupported" in text.lower()
    assert "WARNING" in text


# ---------------------------------------------------------------------------
# run.name path-traversal validation
# ---------------------------------------------------------------------------

def test_run_name_rejects_path_separator():
    d = MAIN_CONFIG.model_dump()
    d["run"]["name"] = "../../etc/evil"
    with pytest.raises(ValueError, match="path separators"):
        MainConfig.model_validate(d)


def test_run_name_rejects_forward_slash():
    d = MAIN_CONFIG.model_dump()
    d["run"]["name"] = "sub/dir"
    with pytest.raises(ValueError, match="path separators"):
        MainConfig.model_validate(d)


def test_run_name_rejects_backslash():
    d = MAIN_CONFIG.model_dump()
    d["run"]["name"] = r"sub\dir"
    with pytest.raises(ValueError, match="path separators"):
        MainConfig.model_validate(d)


def test_run_name_accepts_safe_identifier():
    d = MAIN_CONFIG.model_dump()
    d["run"]["name"] = "my_experiment-01"
    m = MainConfig.model_validate(d)
    assert m.run.name == "my_experiment-01"
