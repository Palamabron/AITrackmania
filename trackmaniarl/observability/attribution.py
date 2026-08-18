"""Optional Captum helpers for image and vector-policy attribution."""

from __future__ import annotations

from typing import Any


def integrated_gradients(
    model: Any, inputs: Any, target: int | None = None, steps: int = 32
) -> Any:
    """Return Captum Integrated Gradients without making Captum a core dependency."""

    try:
        from captum.attr import IntegratedGradients
    except ImportError as exc:
        raise RuntimeError("Install trackmaniarl[explain] to use attribution helpers") from exc
    return IntegratedGradients(model).attribute(inputs, target=target, n_steps=steps)
