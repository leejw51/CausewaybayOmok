"""Backend selection: CUDA -> PyTorch, Apple Silicon -> MLX, else PyTorch CPU."""

from __future__ import annotations

import platform
from typing import Any

from ..netspec import NetSpec
from .base import Backend

__all__ = ["Backend", "available_backends", "select_backend", "make_backend"]


def _has_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _has_mlx() -> bool:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False
    try:
        import mlx.core  # noqa: F401

        return True
    except Exception:
        return False


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def _has_mps() -> bool:
    try:
        import torch

        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:
        return False


def available_backends() -> dict[str, Any]:
    return {"cuda": _has_cuda(), "mlx": _has_mlx(), "mps": _has_mps(), "torch": _has_torch()}


def select_backend(prefer: str = "auto") -> tuple[str, str | None]:
    """Return ``(backend_name, device)``.

    ``auto`` resolves to CUDA if a GPU is present, then MLX on Apple Silicon,
    then PyTorch MPS, and finally PyTorch on the CPU.
    """
    prefer = (prefer or "auto").lower()
    if prefer in ("torch-cpu", "cpu"):
        return "torch", "cpu"
    if prefer in ("cuda", "torch-cuda"):
        return "torch", "cuda"
    if prefer in ("mps", "torch-mps"):
        return "torch", "mps"
    if prefer == "mlx":
        return "mlx", None
    if prefer == "torch":
        return "torch", None
    if prefer != "auto":
        raise ValueError(f"unknown backend: {prefer}")
    if _has_cuda():
        return "torch", "cuda"
    if _has_mlx():
        return "mlx", None
    if _has_mps():
        return "torch", "mps"
    if _has_torch():
        return "torch", "cpu"
    raise RuntimeError("neither PyTorch nor MLX is installed -- run `make install`")


def make_backend(spec: NetSpec, prefer: str = "auto", lr: float = 2e-3,
                 weight_decay: float = 1e-4) -> Backend:
    name, device = select_backend(prefer)
    if name == "mlx":
        from .mlx_backend import MLXBackend

        return MLXBackend(spec, device=device, lr=lr, weight_decay=weight_decay)
    from .torch_backend import TorchBackend

    return TorchBackend(spec, device=device, lr=lr, weight_decay=weight_decay)
