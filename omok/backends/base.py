"""The contract every compute backend implements."""

from __future__ import annotations

import abc
from typing import Any

import numpy as np

from ..netspec import NetSpec


class Backend(abc.ABC):
    """Owns a network (and, when training, its optimiser).

    All array traffic across this boundary is numpy in NCHW float32 so the
    rest of the codebase never imports torch or mlx.
    """

    name: str = "base"
    device: str = "cpu"

    def __init__(self, spec: NetSpec) -> None:
        self.spec = spec

    # -- inference ---------------------------------------------------------
    @abc.abstractmethod
    def predict(self, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(B, C, H, W) float32 -> (policy probabilities (B, A), value (B,))."""

    # -- training ----------------------------------------------------------
    @abc.abstractmethod
    def train_step(self, planes: np.ndarray, pi: np.ndarray, z: np.ndarray,
                   lr: float, value_weight: float, grad_clip: float) -> dict[str, float]:
        """One optimiser step; returns loss metrics as plain floats."""

    # -- weights -----------------------------------------------------------
    @abc.abstractmethod
    def get_weights(self) -> dict[str, np.ndarray]:
        """Canonical (PyTorch-named, NCHW) float32 weights."""

    @abc.abstractmethod
    def set_weights(self, weights: dict[str, np.ndarray]) -> None:
        ...

    # -- optimiser state (best effort; weights are what really matters) ----
    def save_optimizer(self, path: str) -> bool:
        return False

    def load_optimizer(self, path: str) -> bool:
        return False

    # -- misc --------------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        return {"backend": self.name, "device": self.device,
                "params": self.spec.parameter_count()}

    def clone_for_inference(self) -> "Backend":
        """A separate copy holding the same weights (used by the arena)."""
        other = type(self)(self.spec)
        other.set_weights(self.get_weights())
        return other


def softmax_np(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - logits.max(axis=axis, keepdims=True)
    exp = np.exp(shifted, dtype=np.float32)
    return exp / exp.sum(axis=axis, keepdims=True)
