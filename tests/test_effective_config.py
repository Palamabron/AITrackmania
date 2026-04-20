"""Tests for ``tmrl.config.effective_config`` routing and IQN footgun warnings."""

from __future__ import annotations

import warnings

import pytest
from tmrl.config.effective_config import (
    active_model_field_names,
    build_interface_context,
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


def test_build_interface_context_lidar_progress_marks_lidar():
    env = MAIN_CONFIG.environment.model_copy(
        update={"rtgym_interface": "TM20LIDARPROGRESS", "use_images": False}
    )
    assert build_interface_context(env).use_lidar_observations is True


@pytest.mark.parametrize(
    ("alg", "iface", "expected_route"),
    [
        ("IQN", "LIDAR", "lidar_iqn"),
        ("SDSAC", "LIDAR", "lidar_sdsac"),
    ],
)
def test_model_policy_route_named_routes(alg: str, iface: str, expected_route: str):
    m = _validate_with(algorithm={"name": alg}, environment={"rtgym_interface": iface})
    assert model_policy_route(m) == expected_route


@pytest.mark.parametrize(
    ("alg", "must_contain", "must_not_contain"),
    [
        (
            "IQN",
            ("residual_mlp_num_blocks", "split_track_observation"),
            ("residual_mlp_num_blocks_actor",),
        ),
        (
            "SDSAC",
            ("residual_mlp_num_blocks_actor", "residual_mlp_num_blocks_critic"),
            (),
        ),
    ],
)
def test_active_model_fields(
    alg: str, must_contain: tuple[str, ...], must_not_contain: tuple[str, ...]
):
    m = _validate_with(algorithm={"name": alg}, environment={"rtgym_interface": "LIDAR"})
    active = active_model_field_names(m)
    for name in must_contain:
        assert name in active
    for name in must_not_contain:
        assert name not in active


@pytest.mark.parametrize(
    ("actor_blocks", "critic_blocks", "expect_warning"),
    [
        (2, 4, True),
        (6, 6, False),
    ],
)
def test_iqn_warns_when_actor_critic_depths_disagree(
    actor_blocks: int, critic_blocks: int, expect_warning: bool
):
    d = MAIN_CONFIG.model_dump()
    d["algorithm"]["name"] = "IQN"
    d["model"]["residual_mlp_num_blocks"] = max(actor_blocks, critic_blocks)
    d["model"]["residual_mlp_num_blocks_actor"] = actor_blocks
    d["model"]["residual_mlp_num_blocks_critic"] = critic_blocks
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        MainConfig.model_validate(d)
    triggered = any("IQN uses only" in str(x.message) for x in w)
    assert triggered is expect_warning


@pytest.mark.parametrize(
    ("preset", "should_reject"),
    [
        ("vanilla_cnn_actor_critic", True),
        ("mlp_actor_critic", False),
    ],
)
def test_iqn_preset_compatibility(preset: str, should_reject: bool):
    d = MAIN_CONFIG.model_dump()
    d["algorithm"]["name"] = "IQN"
    d["environment"]["rtgym_interface"] = "LIDAR"
    d["model"]["type"] = preset
    if should_reject:
        with pytest.raises(ValueError, match="discrete-action-capable"):
            MainConfig.model_validate(d)
    else:
        MainConfig.model_validate(d)


def test_main_config_snapshot_redacted_is_jsonish_tree():
    s = main_config_snapshot_redacted()
    assert isinstance(s, dict)
    assert "schema_version" in s
    assert "algorithm" in s
    w = s.get("wandb")
    if isinstance(w, dict) and "api_key" in w and w["api_key"]:
        assert w["api_key"] == "<redacted>"


_LIDAR_IFACE = "LIDAR"
_ADVANCED_IFACE = "TQCGRAB_IMAGES"
_VANILLA_IFACE = "TM2020"

_SUPPORTED = [
    ("SAC", _LIDAR_IFACE),
    ("REDQSAC", _LIDAR_IFACE),
    ("IQN", _LIDAR_IFACE),
    ("SDSAC", _LIDAR_IFACE),
    ("TQC", _ADVANCED_IFACE),
    ("SAC", _ADVANCED_IFACE),
    ("IQN", _ADVANCED_IFACE),
    ("SDSAC", _ADVANCED_IFACE),
    ("SAC", _VANILLA_IFACE),
]

_UNSUPPORTED = [
    ("TQC", _LIDAR_IFACE),
    ("REDQSAC", _ADVANCED_IFACE),
    ("TQC", _VANILLA_IFACE),
    ("IQN", _VANILLA_IFACE),
    ("SDSAC", _VANILLA_IFACE),
    ("REDQSAC", _VANILLA_IFACE),
]


@pytest.mark.parametrize(
    ("alg", "iface", "supported"),
    [(a, i, True) for a, i in _SUPPORTED] + [(a, i, False) for a, i in _UNSUPPORTED],
)
def test_model_policy_route_supported_matrix(alg: str, iface: str, supported: bool):
    m = _validate_with(algorithm={"name": alg}, environment={"rtgym_interface": iface})
    route = model_policy_route(m)
    if supported:
        assert route != "unsupported", f"{alg}+{iface} should be supported, got 'unsupported'"
    else:
        assert route == "unsupported", f"{alg}+{iface} should be unsupported, got {route!r}"


def test_explain_active_config_unsupported_does_not_crash():
    m = _validate_with(algorithm={"name": "TQC"}, environment={"rtgym_interface": "LIDAR"})
    text = explain_active_config_text(m)
    assert "unsupported" in text.lower()
    assert "WARNING" in text


@pytest.mark.parametrize(
    ("name", "should_reject"),
    [
        ("../../etc/evil", True),
        ("sub/dir", True),
        (r"sub\dir", True),
        ("my_experiment-01", False),
    ],
)
def test_run_name_path_separator_validation(name: str, should_reject: bool):
    d = MAIN_CONFIG.model_dump()
    d["run"]["name"] = name
    if should_reject:
        with pytest.raises(ValueError, match="path separators"):
            MainConfig.model_validate(d)
    else:
        m = MainConfig.model_validate(d)
        assert m.run.name == name
