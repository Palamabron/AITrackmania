"""Explain which ``model.*`` keys are effective for the current runtime route.

Legacy module path kept for backward compatibility.
Prefer importing from ``tmrl.config.active_config_explainer`` in new code.

This module is the explainability layer for the config runtime selection logic:

- It mirrors the algorithm + interface routing used in
  ``tmrl.config.config_objects._train_model_and_policy`` via :func:`model_policy_route`.
- It exposes the subset of ``model`` fields that are actually read on that route
  via :func:`active_model_field_names`.
- It generates a human-readable report for ``python -m tmrl --explain-active-config``
  via :func:`explain_active_config_text`.

Why this file exists:
Hydra/Pydantic configs often include knobs that are valid globally but ignored for
a specific algorithm+interface path. This module prevents confusion by showing which
fields are active right now and why others are ignored.

Maintenance contract:
Whenever routing branches change in ``config_objects._train_model_and_policy``, keep
``model_policy_route`` and ``ROUTE_ACTIVE_MODEL_FIELDS`` in sync.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tmrl.config.rtgym_boundary_iface import (
    rtgym_discrete_boundary_lidar_family,
    rtgym_discrete_boundary_lidar_images,
)
from tmrl.config.schema.environment import EnvironmentConfig
from tmrl.config.schema.main import MainConfig


@dataclass(frozen=True)
class InterfaceContext:
    """Normalized interface flags derived from ``EnvironmentConfig`` only.

    This mirrors the interface booleans from ``tmrl.config.constants`` but avoids
    importing runtime constants directly, so route resolution stays deterministic from
    ``MainConfig`` input alone.
    """

    lidar_geometry_interface: bool
    lidar_includes_images: bool
    obs_includes_world_telemetry: bool
    images_mobilenet_pipeline: bool
    images_r2d2_sequence_buffer: bool
    use_images: bool
    img_grayscale: bool


def build_interface_context(env: EnvironmentConfig) -> InterfaceContext:
    """Build route-relevant interface flags from ``environment`` configuration."""
    rt = str(env.rtgym_interface).upper()
    lidar_images = rtgym_discrete_boundary_lidar_images(rt)
    lidar_geom = rtgym_discrete_boundary_lidar_family(rt)
    images_mobilenet_pipeline = (
        rt.endswith("MOBILEV3")
        or rt.endswith("CUSTOM")
        or rt.endswith("BEST")
        or rt.endswith("BEST_TQC")
    )
    obs_world_telemetry = "TQCGRAB" in rt
    images_r2d2_sequence_buffer = rt.endswith("MTQC") or obs_world_telemetry
    return InterfaceContext(
        lidar_geometry_interface=lidar_geom,
        lidar_includes_images=lidar_images,
        obs_includes_world_telemetry=obs_world_telemetry,
        images_mobilenet_pipeline=images_mobilenet_pipeline,
        images_r2d2_sequence_buffer=images_r2d2_sequence_buffer,
        use_images=env.use_images,
        img_grayscale=env.img_grayscale,
    )


def _advanced(ctx: InterfaceContext) -> bool:
    return ctx.images_mobilenet_pipeline or ctx.images_r2d2_sequence_buffer


_SUPPORTED_LIDAR_GEOMETRY_ALGORITHMS = frozenset({"SAC", "REDQSAC", "IQN", "SDSAC"})
_SUPPORTED_ADVANCED_ALGORITHMS = frozenset({"TQC", "SAC", "IQN", "SDSAC"})


def model_policy_route(m: MainConfig) -> str:
    """Return the stable route id selected by current algorithm + interface + model flags.

    When you add a branch there, add a route name here and map it in ``ROUTE_ACTIVE_MODEL_FIELDS``.
    Returns ``"unsupported"`` for algorithm+interface combinations with no runtime branch
    so ``--explain-active-config`` can report a useful warning instead of crashing.
    """
    alg = m.algorithm.name
    arch = m.model
    ctx = build_interface_context(m.environment)
    adv = _advanced(ctx)

    if ctx.lidar_geometry_interface:
        if alg not in _SUPPORTED_LIDAR_GEOMETRY_ALGORITHMS:
            return "unsupported"
        if alg in ("IQN", "SDSAC"):
            return "lidar_iqn" if alg == "IQN" else "lidar_sdsac"
        if ctx.lidar_includes_images and alg == "SAC":
            return "lidar_sac_frozen_effnet"
        if arch.use_residual_mlp and alg in ("SAC", "REDQSAC"):
            return "lidar_residual"
        if alg in ("SAC", "REDQSAC"):
            return "lidar_plain_mlp"
        return "unsupported"

    if adv:
        if alg not in _SUPPORTED_ADVANCED_ALGORITHMS:
            return "unsupported"
        if alg in ("IQN", "SDSAC"):
            return "adv_iqn" if alg == "IQN" else "adv_sdsac"
        if (
            ctx.use_images
            and not ctx.obs_includes_world_telemetry
            and arch.use_frozen_effnet
            and alg == "SAC"
        ):
            return "adv_sac_frozen_effnet"
        if ctx.use_images and not ctx.obs_includes_world_telemetry:
            return "adv_impala"
        if (
            ctx.obs_includes_world_telemetry
            and not ctx.use_images
            and arch.use_sophy_residual_actor
        ):
            return "adv_sophy_residual"
        if ctx.obs_includes_world_telemetry and not ctx.use_images:
            return "adv_sophy_classic"
        return "adv_sophy_images"

    if alg != "SAC":
        return "unsupported"
    return "vanilla_gray" if ctx.img_grayscale else "vanilla_color"


def _all_model_field_names(m: MainConfig) -> frozenset[str]:
    return frozenset(m.model.__class__.model_fields.keys())


# IQN / SDSAC worker policy: fields that reach IQNFeatureBackbone / IQNQNetwork for discrete
# boundary lidar observations.
# Keep aligned with ``_IQN_BACKBONE_KWARGS`` in
# ``tmrl/custom/models/discrete_actions/iqn_discrete_q_network.py`` plus
# ``residual_mlp_*`` and ``type``.
_LIDAR_DISCRETE_IQN: frozenset[str] = frozenset(
    {
        "type",
        "residual_mlp_hidden_dim",
        "residual_mlp_num_blocks",
        "split_track_observation",
        "track_encoder",
        "gnn_layers",
        "gnn_hidden",
        "use_simbav2",
        "api_layernorm",
        "use_rnn",
        "rnn_hidden_size",
    }
)

_LIDAR_DISCRETE_SDSAC: frozenset[str] = _LIDAR_DISCRETE_IQN | {
    "residual_mlp_num_blocks_actor",
    "residual_mlp_num_blocks_critic",
}

# Partial() only passes hidden_dim + num_blocks (+ frozen_* for EffNet wrapper).
_FROZEN_LIDAR_SAC: frozenset[str] = frozenset(
    {
        "type",
        "frozen_effnet_embed_dim",
        "frozen_effnet_width_mult",
        "frozen_effnet_variant",
        "frozen_effnet_use_dw_stem",
        "use_frozen_effnet",
        "residual_mlp_hidden_dim",
        "residual_mlp_num_blocks",
    }
)

# Residual / plain MLP actor-critic on boundary lidar vectors: no track encoder in these modules.
_LIDAR_RESIDUAL: frozenset[str] = frozenset(
    {
        "type",
        "use_residual_mlp",
        "residual_mlp_hidden_dim",
        "residual_mlp_num_blocks",
    }
)

_LIDAR_PLAIN_MLP: frozenset[str] = frozenset({"type", "use_residual_mlp"})


FIELD_IGNORE_HINTS: dict[str, str] = {
    "mlp_layernorm": (
        "Not passed to IQNFeatureBackbone (discrete IQN uses fixed LayerNorm in "
        "residual_mlp_backbone); used by some continuous backbones."
    ),
    "residual_mlp_num_blocks_actor": (
        "Used by SDSAC and continuous Sophy-style residuals, not by IQN "
        "(IQN has a single Q trunk: use model.residual_mlp_num_blocks only)."
    ),
    "residual_mlp_num_blocks_critic": (
        "Used by SDSAC and continuous Sophy-style residuals, not by IQN "
        "(IQN has a single Q trunk: use model.residual_mlp_num_blocks only)."
    ),
    "use_sophy_residual_actor": (
        "Only the world-telemetry (no screen) + MTQC-style vector branch with use_images=false; "
        "ignored on boundary lidar interfaces and IQN."
    ),
    "split_track_observation": (
        "Ignored on boundary lidar plain residual/MLP actors (tuple is flattened); "
        "used by IQN/SDSAC discrete backbones and Sophy when enabled."
    ),
    "track_encoder": (
        "Ignored when split_track_observation is false or on boundary lidar plain residual/MLP."
    ),
    "gnn_layers": (
        "Ignored when track_encoder is not gtn/gnn alias or track split is unused for this route."
    ),
    "gnn_hidden": (
        "Ignored when track_encoder is not gtn/gnn alias or track split is unused for this route."
    ),
}


# Routes that use most knobs (Sophy / IMPALA / vanilla presets).
_FULL_MODEL_ROUTES = frozenset(
    {
        "adv_impala",
        "adv_sophy_residual",
        "adv_sophy_classic",
        "adv_sophy_images",
        "vanilla_gray",
        "vanilla_color",
    }
)

ROUTE_ACTIVE_MODEL_FIELDS: dict[str, frozenset[str]] = {
    "lidar_iqn": _LIDAR_DISCRETE_IQN,
    "lidar_sdsac": _LIDAR_DISCRETE_SDSAC,
    "lidar_sac_frozen_effnet": _FROZEN_LIDAR_SAC,
    "lidar_residual": _LIDAR_RESIDUAL,
    "lidar_plain_mlp": _LIDAR_PLAIN_MLP,
    "adv_iqn": _LIDAR_DISCRETE_IQN,
    "adv_sdsac": _LIDAR_DISCRETE_SDSAC,
    "adv_sac_frozen_effnet": _FROZEN_LIDAR_SAC,
}


def active_model_field_names(m: MainConfig) -> frozenset[str]:
    """Return ``model`` field names that influence the resolved route for this run."""
    route = model_policy_route(m)
    if route in _FULL_MODEL_ROUTES:
        return _all_model_field_names(m)
    return ROUTE_ACTIVE_MODEL_FIELDS.get(route, _all_model_field_names(m))


def inactive_model_fields_report(m: MainConfig) -> list[tuple[str, Any, str]]:
    """Return ``(field_name, value, hint)`` rows for route-inactive ``model`` keys."""
    active = active_model_field_names(m)
    data = m.model.model_dump()
    rows: list[tuple[str, Any, str]] = []
    for key in sorted(data.keys()):
        if key in active:
            continue
        hint = FIELD_IGNORE_HINTS.get(
            key,
            (
                "Not read for this algorithm+interface route; "
                "preset may still set it for other presets."
            ),
        )
        rows.append((key, data[key], hint))
    return rows


def explain_active_config_text(m: MainConfig) -> str:
    """Build the CLI report used by ``--explain-active-config``."""
    route = model_policy_route(m)
    lines = [
        f"policy_route: {route}",
        f"algorithm.name: {m.algorithm.name}",
        f"environment.rtgym_interface: {m.environment.rtgym_interface}",
    ]
    if route == "unsupported":
        lines.append("")
        lines.append(
            "WARNING: this algorithm + interface combination has no runtime branch. "
            "Training will fail at startup. Change algorithm.name or "
            "environment.rtgym_interface to a supported pairing."
        )
        lines.append("")
        lines.append("All model fields (no route matched):")
        for k in sorted(_all_model_field_names(m)):
            lines.append(f"  - {k}")
        return "\n".join(lines) + "\n"

    active = sorted(active_model_field_names(m))
    lines.append("")
    lines.append("Active model fields (this run):")
    lines.extend(f"  - {k}" for k in active)
    inactive = inactive_model_fields_report(m)
    if inactive:
        lines.append("")
        lines.append("Model fields not used by this route (current values shown):")
        for name, val, hint in inactive:
            lines.append(f"  - {name}: {val!r}")
            lines.append(f"    ({hint})")
    return "\n".join(lines) + "\n"
