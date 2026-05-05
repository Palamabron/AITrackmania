"""Canonical import path for active-config explainability helpers.

This module provides a clearer name than ``effective_config`` for the same public
API used by ``python -m tmrl --explain-active-config``.

Backward compatibility note:
``tmrl.config.effective_config`` remains available for existing imports.
New code should prefer ``tmrl.config.active_config_explainer``.
"""

from tmrl.config.effective_config import (
    FIELD_IGNORE_HINTS,
    ROUTE_ACTIVE_MODEL_FIELDS,
    InterfaceContext,
    active_model_field_names,
    build_interface_context,
    explain_active_config_text,
    inactive_model_fields_report,
    model_policy_route,
)

__all__ = [
    "FIELD_IGNORE_HINTS",
    "ROUTE_ACTIVE_MODEL_FIELDS",
    "InterfaceContext",
    "active_model_field_names",
    "build_interface_context",
    "explain_active_config_text",
    "inactive_model_fields_report",
    "model_policy_route",
]
