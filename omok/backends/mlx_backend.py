"""MLX backend -- the fast path on Apple Silicon (unified memory, Metal)."""

from __future__ import annotations

import os
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from ..netspec import NetSpec, check_weights
from .base import Backend

# MLX convolutions are NHWC while our canonical weights are PyTorch NCHW.
_CONV_TO_MLX = (0, 2, 3, 1)
_CONV_TO_TORCH = (0, 3, 1, 2)


class ConvBN(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False)
        self.bn = nn.BatchNorm(out_ch)

    def __call__(self, x: mx.array) -> mx.array:
        return self.bn(self.conv(x))


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm(channels)

    def __call__(self, x: mx.array) -> mx.array:
        y = nn.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return nn.relu(x + y)


def _flatten_nchw(y: mx.array) -> mx.array:
    """NHWC feature map -> flat vector in PyTorch's NCHW order (weight parity)."""
    return mx.transpose(y, _CONV_TO_TORCH).reshape(y.shape[0], -1)


class PolicyHead(nn.Module):
    def __init__(self, channels: int, head_channels: int, board_size: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, head_channels, 1, bias=False)
        self.bn = nn.BatchNorm(head_channels)
        self.fc = nn.Linear(head_channels * board_size * board_size, board_size * board_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc(_flatten_nchw(nn.relu(self.bn(self.conv(x)))))


class ValueHead(nn.Module):
    def __init__(self, channels: int, head_channels: int, board_size: int, hidden: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, head_channels, 1, bias=False)
        self.bn = nn.BatchNorm(head_channels)
        self.fc1 = nn.Linear(head_channels * board_size * board_size, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def __call__(self, x: mx.array) -> mx.array:
        y = nn.relu(self.bn(self.conv(x)))
        y = nn.relu(self.fc1(_flatten_nchw(y)))
        return mx.tanh(self.fc2(y)).squeeze(-1)


class OmokNet(nn.Module):
    def __init__(self, spec: NetSpec) -> None:
        super().__init__()
        self.stem = ConvBN(spec.in_planes, spec.channels, 3)
        self.blocks = [ResBlock(spec.channels) for _ in range(spec.blocks)]
        self.policy = PolicyHead(spec.channels, spec.policy_channels, spec.board_size)
        self.value = ValueHead(spec.channels, spec.value_channels, spec.board_size,
                               spec.value_hidden)

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        y = nn.relu(self.stem(x))
        for block in self.blocks:
            y = block(y)
        return self.policy(y), self.value(y)


def _loss_fn(model: OmokNet, x: mx.array, pi: mx.array, z: mx.array, value_weight: float):
    logits, value = model(x)
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    policy_loss = -(pi * logp).sum(axis=-1).mean()
    value_loss = ((value - z) ** 2).mean()
    return policy_loss + value_weight * value_loss, (policy_loss, value_loss)


class MLXBackend(Backend):
    name = "mlx"

    def __init__(self, spec: NetSpec, device: str | None = None,
                 lr: float = 2e-3, weight_decay: float = 1e-4) -> None:
        super().__init__(spec)
        self.device = device or "gpu"
        if self.device == "cpu":
            mx.set_default_device(mx.cpu)
        self.net = OmokNet(spec)
        mx.eval(self.net.parameters())
        self.opt = optim.AdamW(learning_rate=lr, weight_decay=weight_decay)
        self._value_and_grad = nn.value_and_grad(self.net, _loss_fn)

    # -- inference ---------------------------------------------------------
    def predict(self, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.net.eval()
        x = mx.array(np.ascontiguousarray(np.transpose(planes, (0, 2, 3, 1)), dtype=np.float32))
        logits, value = self.net(x)
        policy = mx.softmax(logits, axis=-1)
        mx.eval(policy, value)
        return np.array(policy, dtype=np.float32), np.array(value, dtype=np.float32)

    # -- training ----------------------------------------------------------
    def train_step(self, planes: np.ndarray, pi: np.ndarray, z: np.ndarray,
                   lr: float, value_weight: float, grad_clip: float) -> dict[str, float]:
        self.net.train()
        x = mx.array(np.ascontiguousarray(np.transpose(planes, (0, 2, 3, 1)), dtype=np.float32))
        target_pi = mx.array(np.ascontiguousarray(pi, dtype=np.float32))
        target_z = mx.array(np.ascontiguousarray(z, dtype=np.float32))

        self.opt.learning_rate = lr
        (loss, (policy_loss, value_loss)), grads = self._value_and_grad(
            self.net, x, target_pi, target_z, value_weight)
        grads, grad_norm = optim.clip_grad_norm(grads, grad_clip)
        self.opt.update(self.net, grads)
        mx.eval(self.net.parameters(), self.opt.state, loss)

        entropy = float(-(np.clip(pi, 1e-9, None) * np.log(np.clip(pi, 1e-9, None))).sum(-1).mean())
        return {
            "loss": float(loss),
            "policy_loss": float(policy_loss),
            "value_loss": float(value_loss),
            "target_entropy": entropy,
            "grad_norm": float(grad_norm),
        }

    # -- weights -----------------------------------------------------------
    def get_weights(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for key, value in tree_flatten(self.net.parameters()):
            array = np.array(value, dtype=np.float32)
            if key.endswith("conv.weight") or key.endswith("conv1.weight") or key.endswith("conv2.weight"):
                array = np.ascontiguousarray(np.transpose(array, _CONV_TO_TORCH))
            out[key] = array
        return out

    def set_weights(self, weights: dict[str, np.ndarray]) -> None:
        check_weights(self.spec, weights)
        flat = []
        for key, array in weights.items():
            array = np.ascontiguousarray(array, dtype=np.float32)
            if key.endswith("conv.weight") or key.endswith("conv1.weight") or key.endswith("conv2.weight"):
                array = np.ascontiguousarray(np.transpose(array, _CONV_TO_MLX))
            flat.append((key, mx.array(array)))
        self.net.update(tree_unflatten(flat))
        mx.eval(self.net.parameters())

    # -- optimiser ---------------------------------------------------------
    def save_optimizer(self, path: str) -> bool:
        from ..utils import atomic_write_with

        payload = {k.replace(".", "|"): v for k, v in tree_flatten(self.opt.state)
                   if isinstance(v, mx.array)}
        if not payload:
            return False
        atomic_write_with(path, lambda tmp: mx.savez(tmp, **payload), suffix=".npz")
        return True

    def load_optimizer(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        try:
            loaded = mx.load(path)
            flat = [(k.replace("|", "."), v) for k, v in loaded.items()]
            self.opt.state = tree_unflatten(flat)
            return True
        except Exception:
            return False

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info["mlx"] = getattr(mx, "__version__", "?")
        return info

    def clone_for_inference(self) -> "MLXBackend":
        other = MLXBackend(self.spec, device=self.device)
        other.set_weights(self.get_weights())
        return other
