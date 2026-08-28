"""Backend-independent description of the policy/value network.

The canonical parameter names below are exactly the PyTorch ``state_dict``
keys; the MLX backend translates to and from them.  That makes a checkpoint
portable: train on a CUDA box, resume on a Mac, export from either.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .encode import NUM_PLANES


@dataclass(frozen=True)
class NetSpec:
    board_size: int = 15
    in_planes: int = NUM_PLANES
    channels: int = 96
    blocks: int = 6
    policy_channels: int = 4
    value_channels: int = 2
    value_hidden: int = 128

    @property
    def action_size(self) -> int:
        return self.board_size * self.board_size

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NetSpec":
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in fields})

    @classmethod
    def from_config(cls, cfg) -> "NetSpec":
        return cls(
            board_size=cfg.game.board_size,
            in_planes=NUM_PLANES,
            channels=cfg.net.channels,
            blocks=cfg.net.blocks,
            policy_channels=cfg.net.policy_channels,
            value_channels=cfg.net.value_channels,
            value_hidden=cfg.net.value_hidden,
        )

    def parameter_count(self) -> int:
        return sum(int(np.prod(shape)) for _, shape in parameter_shapes(self))


def _bn_shapes(name: str, channels: int) -> list[tuple[str, tuple[int, ...]]]:
    return [
        (f"{name}.weight", (channels,)),
        (f"{name}.bias", (channels,)),
        (f"{name}.running_mean", (channels,)),
        (f"{name}.running_var", (channels,)),
    ]


def parameter_shapes(spec: NetSpec) -> list[tuple[str, tuple[int, ...]]]:
    c = spec.channels
    n = spec.board_size
    shapes: list[tuple[str, tuple[int, ...]]] = []
    shapes.append(("stem.conv.weight", (c, spec.in_planes, 3, 3)))
    shapes += _bn_shapes("stem.bn", c)
    for i in range(spec.blocks):
        shapes.append((f"blocks.{i}.conv1.weight", (c, c, 3, 3)))
        shapes += _bn_shapes(f"blocks.{i}.bn1", c)
        shapes.append((f"blocks.{i}.conv2.weight", (c, c, 3, 3)))
        shapes += _bn_shapes(f"blocks.{i}.bn2", c)
    pc, vc = spec.policy_channels, spec.value_channels
    shapes.append(("policy.conv.weight", (pc, c, 1, 1)))
    shapes += _bn_shapes("policy.bn", pc)
    shapes.append(("policy.fc.weight", (spec.action_size, pc * n * n)))
    shapes.append(("policy.fc.bias", (spec.action_size,)))
    shapes.append(("value.conv.weight", (vc, c, 1, 1)))
    shapes += _bn_shapes("value.bn", vc)
    shapes.append(("value.fc1.weight", (spec.value_hidden, vc * n * n)))
    shapes.append(("value.fc1.bias", (spec.value_hidden,)))
    shapes.append(("value.fc2.weight", (1, spec.value_hidden)))
    shapes.append(("value.fc2.bias", (1,)))
    return shapes


def init_weights(spec: NetSpec, seed: int = 0) -> dict[str, np.ndarray]:
    """Canonical initial weights, identical on every backend for a given seed."""
    rng = np.random.default_rng(seed)
    weights: dict[str, np.ndarray] = {}
    for name, shape in parameter_shapes(spec):
        module = name.rsplit(".", 2)[-2]
        leaf = name.rsplit(".", 1)[-1]
        if leaf == "running_mean":
            weights[name] = np.zeros(shape, dtype=np.float32)
        elif leaf == "running_var":
            weights[name] = np.ones(shape, dtype=np.float32)
        elif module.startswith("bn"):
            fill = np.ones if leaf == "weight" else np.zeros
            weights[name] = fill(shape, dtype=np.float32)
        elif module.startswith("conv"):
            fan_in = int(np.prod(shape[1:]))
            weights[name] = rng.normal(0.0, np.sqrt(2.0 / fan_in), size=shape).astype(np.float32)
        elif leaf == "weight":  # fully connected
            bound = 1.0 / np.sqrt(shape[1])
            weights[name] = rng.uniform(-bound, bound, size=shape).astype(np.float32)
        else:  # fully connected bias
            weights[name] = np.zeros(shape, dtype=np.float32)
    return weights


def check_weights(spec: NetSpec, weights: dict[str, np.ndarray]) -> None:
    expected = dict(parameter_shapes(spec))
    missing = sorted(set(expected) - set(weights))
    extra = sorted(set(weights) - set(expected))
    if missing or extra:
        raise ValueError(f"weight mismatch: missing={missing[:4]} extra={extra[:4]}")
    for name, shape in expected.items():
        if tuple(weights[name].shape) != tuple(shape):
            raise ValueError(f"{name}: expected {shape}, got {tuple(weights[name].shape)}")
