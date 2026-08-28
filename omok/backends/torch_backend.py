"""PyTorch backend -- CUDA when present, then MPS, otherwise CPU."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..netspec import NetSpec, check_weights
from .base import Backend


# TF32 matmul/conv on Ampere-and-later GPUs: ~2x throughput, and the model is
# far too small for the precision difference to matter.
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True


def pick_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ConvBN(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        return F.relu(x + y, inplace=True)


class PolicyHead(nn.Module):
    def __init__(self, channels: int, head_channels: int, board_size: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, head_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(head_channels)
        self.fc = nn.Linear(head_channels * board_size * board_size, board_size * board_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn(self.conv(x)), inplace=True)
        return self.fc(y.flatten(1))


class ValueHead(nn.Module):
    def __init__(self, channels: int, head_channels: int, board_size: int, hidden: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, head_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(head_channels)
        self.fc1 = nn.Linear(head_channels * board_size * board_size, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn(self.conv(x)), inplace=True)
        y = F.relu(self.fc1(y.flatten(1)), inplace=True)
        return torch.tanh(self.fc2(y)).squeeze(-1)


class OmokNet(nn.Module):
    """Module names are the canonical checkpoint keys -- do not rename."""

    def __init__(self, spec: NetSpec) -> None:
        super().__init__()
        self.spec = spec
        self.stem = ConvBN(spec.in_planes, spec.channels, 3)
        self.blocks = nn.ModuleList([ResBlock(spec.channels) for _ in range(spec.blocks)])
        self.policy = PolicyHead(spec.channels, spec.policy_channels, spec.board_size)
        self.value = ValueHead(spec.channels, spec.value_channels, spec.board_size,
                               spec.value_hidden)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = F.relu(self.stem(x), inplace=True)
        for block in self.blocks:
            y = block(y)
        return self.policy(y), self.value(y)


class ExportNet(nn.Module):
    """Inference wrapper emitting probabilities -- what ships to CoreML."""

    def __init__(self, net: OmokNet) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, value = self.net(x)
        return F.softmax(logits, dim=-1), value


class TorchBackend(Backend):
    name = "torch"

    def __init__(self, spec: NetSpec, device: str | None = None,
                 lr: float = 2e-3, weight_decay: float = 1e-4,
                 compile_model: bool = False) -> None:
        super().__init__(spec)
        self.torch_device = pick_device(device)
        self.device = str(self.torch_device)
        self.net = OmokNet(spec).to(self.torch_device)
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        self._infer = self.net
        if compile_model and hasattr(torch, "compile") and self.torch_device.type == "cuda":
            self._infer = torch.compile(self.net)  # pragma: no cover
        torch.manual_seed(0)

    # -- inference ---------------------------------------------------------
    @torch.no_grad()
    def predict(self, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.net.eval()
        x = torch.from_numpy(np.ascontiguousarray(planes, dtype=np.float32)).to(self.torch_device)
        logits, value = self.net(x)
        policy = torch.softmax(logits.float(), dim=-1)
        return (policy.detach().cpu().numpy().astype(np.float32),
                value.detach().float().cpu().numpy().astype(np.float32))

    # -- training ----------------------------------------------------------
    def train_step(self, planes: np.ndarray, pi: np.ndarray, z: np.ndarray,
                   lr: float, value_weight: float, grad_clip: float) -> dict[str, float]:
        self.net.train()
        device = self.torch_device
        x = torch.from_numpy(np.ascontiguousarray(planes, dtype=np.float32)).to(device)
        target_pi = torch.from_numpy(np.ascontiguousarray(pi, dtype=np.float32)).to(device)
        target_z = torch.from_numpy(np.ascontiguousarray(z, dtype=np.float32)).to(device)

        for group in self.opt.param_groups:
            group["lr"] = lr

        logits, value = self.net(x)
        logp = F.log_softmax(logits, dim=-1)
        policy_loss = -(target_pi * logp).sum(dim=-1).mean()
        value_loss = F.mse_loss(value, target_z)
        loss = policy_loss + value_weight * value_loss

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), grad_clip)
        self.opt.step()

        with torch.no_grad():
            entropy = -(target_pi.clamp_min(1e-9).log() * target_pi).sum(dim=-1).mean()
        return {
            "loss": float(loss.detach()),
            "policy_loss": float(policy_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "target_entropy": float(entropy),
            "grad_norm": float(grad_norm),
        }

    # -- weights -----------------------------------------------------------
    def get_weights(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for key, tensor in self.net.state_dict().items():
            if key.endswith("num_batches_tracked"):
                continue
            out[key] = tensor.detach().cpu().numpy().astype(np.float32)
        return out

    def set_weights(self, weights: dict[str, np.ndarray]) -> None:
        check_weights(self.spec, weights)
        state = self.net.state_dict()
        for key, array in weights.items():
            state[key].copy_(torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32)))
        self.net.load_state_dict(state)

    # -- optimiser ---------------------------------------------------------
    def save_optimizer(self, path: str) -> bool:
        from ..utils import atomic_write_with

        atomic_write_with(path, lambda tmp: torch.save(self.opt.state_dict(), tmp))
        return True

    def load_optimizer(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        try:
            state = torch.load(path, map_location=self.torch_device, weights_only=False)
            self.opt.load_state_dict(state)
            return True
        except Exception:  # a stale optimiser state must never block a resume
            return False

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info["torch"] = torch.__version__
        if self.torch_device.type == "cuda":
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = props.name
            info["gpu_memory"] = f"{props.total_memory / (1024 ** 3):.1f} GiB"
        elif self.torch_device.type == "mps":
            info["gpu"] = "Apple GPU (Metal)"
            recommended = getattr(getattr(torch, "mps", None), "recommended_max_memory", None)
            if recommended is not None:
                try:
                    info["gpu_memory"] = f"{recommended() / (1024 ** 3):.1f} GiB"
                except Exception:  # the call is unavailable on older torch
                    pass
        return info

    def clone_for_inference(self) -> "TorchBackend":
        other = TorchBackend(self.spec, device=self.device)
        other.set_weights(self.get_weights())
        return other

    def module_for_export(self) -> nn.Module:
        self.net.eval()
        return ExportNet(self.net).to("cpu").eval()
