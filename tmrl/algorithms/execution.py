"""Hardware and numerical execution policy for bundled Torch learners."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, replace
from typing import Literal

import torch

type RequestedDevice = Literal["auto", "cuda", "rocm", "mps", "cpu"]
type RequestedPrecision = Literal["auto", "bfloat16", "float16", "float32"]
type CompileMode = Literal["default", "reduce-overhead", "max-autotune"]
type ResolvedBackend = Literal["cuda", "rocm", "mps", "cpu"]


class TorchExecutionError(ValueError):
    """Raised when requested Torch execution cannot be provided safely."""


@dataclass(frozen=True, slots=True)
class TorchExecutionConfig:
    device: RequestedDevice = "auto"
    precision: RequestedPrecision = "auto"
    compile: bool = False
    compile_mode: CompileMode = "default"

    def __post_init__(self) -> None:
        if self.device not in {"auto", "cuda", "rocm", "mps", "cpu"}:
            raise TorchExecutionError(f"Unsupported Torch device {self.device!r}")
        if self.precision not in {"auto", "bfloat16", "float16", "float32"}:
            raise TorchExecutionError(f"Unsupported Torch precision {self.precision!r}")
        if self.compile_mode not in {"default", "reduce-overhead", "max-autotune"}:
            raise TorchExecutionError(f"Unsupported torch.compile mode {self.compile_mode!r}")


@dataclass(frozen=True, slots=True)
class ResolvedTorchExecution:
    requested_device: RequestedDevice
    requested_precision: RequestedPrecision
    backend: ResolvedBackend
    torch_device: torch.device
    precision: Literal["bfloat16", "float16", "float32"]
    scaler_enabled: bool
    compile_requested: bool
    compile_effective: bool
    compile_mode: CompileMode
    fallback_reason: str | None = None

    def with_compile_result(
        self, *, effective: bool, fallback_reason: str | None = None
    ) -> ResolvedTorchExecution:
        return replace(
            self,
            compile_effective=effective,
            fallback_reason=fallback_reason,
        )

    def manifest(self) -> dict[str, object]:
        return {
            "requested_device": self.requested_device,
            "requested_precision": self.requested_precision,
            "backend": self.backend,
            "torch_device": str(self.torch_device),
            "precision": self.precision,
            "scaler_enabled": self.scaler_enabled,
            "compile_requested": self.compile_requested,
            "compile_effective": self.compile_effective,
            "compile_mode": self.compile_mode,
            "fallback_reason": self.fallback_reason,
        }


def visible_accelerators() -> set[str]:
    """Return accelerator vendors visible to the host driver tools."""

    visible: set[str] = set()
    if _probe_command(("nvidia-smi", "--list-gpus")):
        visible.add("cuda")
    if _probe_command(("rocm-smi", "--showproductname")):
        visible.add("rocm")
    return visible


def _probe_command(command: tuple[str, ...]) -> bool:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def resolve_torch_execution(config: TorchExecutionConfig) -> ResolvedTorchExecution:
    backend, device = _resolve_device(config.device)
    precision = _resolve_precision(config.precision, backend, device)
    return ResolvedTorchExecution(
        requested_device=config.device,
        requested_precision=config.precision,
        backend=backend,
        torch_device=device,
        precision=precision,
        scaler_enabled=precision == "float16" and backend in {"cuda", "rocm"},
        compile_requested=config.compile,
        compile_effective=False,
        compile_mode=config.compile_mode,
    )


def _resolve_device(requested: RequestedDevice) -> tuple[ResolvedBackend, torch.device]:
    hip_available = bool(torch.version.hip) and torch.cuda.is_available()
    cuda_available = bool(torch.version.cuda) and torch.cuda.is_available()
    mps_available = bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    )
    available: dict[str, bool] = {
        "rocm": hip_available,
        "cuda": cuda_available,
        "mps": mps_available,
        "cpu": True,
    }
    if requested != "auto":
        if not available[requested]:
            raise TorchExecutionError(
                f"Torch device {requested!r} was requested but is unavailable; "
                f"installed build is {torch.__version__}"
            )
        return _backend_device(requested)
    for backend in ("rocm", "cuda", "mps"):
        if available[backend]:
            return _backend_device(backend)
    visible = visible_accelerators()
    if visible:
        names = ", ".join(sorted(visible))
        raise TorchExecutionError(
            f"Host accelerator hardware ({names}) is visible, but Torch {torch.__version__} "
            "cannot use it. Install the matching accelerator-enabled Torch wheel."
        )
    return "cpu", torch.device("cpu")


def _backend_device(backend: str) -> tuple[ResolvedBackend, torch.device]:
    if backend == "rocm":
        return "rocm", torch.device("cuda")
    if backend == "cuda":
        return "cuda", torch.device("cuda")
    if backend == "mps":
        return "mps", torch.device("mps")
    return "cpu", torch.device("cpu")


def _resolve_precision(
    requested: RequestedPrecision,
    backend: ResolvedBackend,
    device: torch.device,
) -> Literal["bfloat16", "float16", "float32"]:
    supported = _supported_precisions(backend, device)
    if requested != "auto":
        if requested not in supported:
            raise TorchExecutionError(
                f"Precision {requested!r} is unsupported on resolved backend {backend!r}"
            )
        return requested
    for precision in ("bfloat16", "float16", "float32"):
        if precision in supported:
            return precision
    return "float32"


def _supported_precisions(
    backend: ResolvedBackend, device: torch.device
) -> set[Literal["bfloat16", "float16", "float32"]]:
    supported: set[Literal["bfloat16", "float16", "float32"]] = {"float32"}
    if backend == "cuda":
        capability = torch.cuda.get_device_capability(device)
        if capability >= (8, 0) and torch.cuda.is_bf16_supported():
            supported.add("bfloat16")
        if _precision_probe(device, torch.float16):
            supported.add("float16")
    elif backend == "rocm":
        if torch.cuda.is_bf16_supported() and _precision_probe(device, torch.bfloat16):
            supported.add("bfloat16")
        if _precision_probe(device, torch.float16):
            supported.add("float16")
    elif backend == "mps" and _precision_probe(device, torch.float16):
        supported.add("float16")
    return supported


def _precision_probe(device: torch.device, dtype: torch.dtype) -> bool:
    try:
        value = torch.ones((2, 2), device=device, requires_grad=True)
        with torch.autocast(device_type=device.type, dtype=dtype):
            loss = (value @ value).sum()
        loss.backward()
        return bool(torch.isfinite(loss).item())
    except (RuntimeError, TypeError):
        return False
