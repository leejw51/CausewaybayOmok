"""Ship the trained network to the game: CoreML for iOS/macOS, ONNX, or raw npz.

The exported graph takes a (1, 5, N, N) float32 plane stack -- the same layout
:mod:`omok.encode` produces -- and returns move probabilities plus a value in
[-1, 1] from the side-to-move's point of view.  ``model_meta.json`` next to the
model documents that contract for the Swift side.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from .checkpoint import CheckpointManager, load_weights
from .config import Config
from .encode import NUM_PLANES
from .netspec import NetSpec
from .utils import log, write_json

PLANE_DOC = [
    "0: stones of the player to move",
    "1: stones of the opponent",
    "2: one-hot of the opponent's last move",
    "3: one-hot of the player-to-move's previous move",
    "4: all ones when the player to move is black",
]


def _torch_model(spec: NetSpec, weights: dict[str, np.ndarray]):
    from .backends.torch_backend import TorchBackend

    backend = TorchBackend(spec, device="cpu")
    backend.set_weights(weights)
    return backend.module_for_export()


def resolve_weights(cfg: Config, source: str | None) -> tuple[dict[str, np.ndarray], str, NetSpec]:
    """``source`` may be a path to a .npz, or None for the run's best model."""
    manager = CheckpointManager(cfg.paths().checkpoints)
    if source is None:
        checkpoint = manager.best() or manager.latest()
        if checkpoint is None:
            raise FileNotFoundError(f"no checkpoint found in {manager.dir}")
        spec = NetSpec.from_dict(checkpoint.meta.get("spec", {})) \
            if checkpoint.meta.get("spec") else NetSpec.from_config(cfg)
        return checkpoint.weights(), checkpoint.weights_path, spec
    weights = load_weights(source)
    meta_path = os.path.splitext(source)[0] + ".json"
    spec = NetSpec.from_config(cfg)
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            if isinstance(meta.get("spec"), dict):
                spec = NetSpec.from_dict(meta["spec"])
        except Exception:
            pass
    return weights, source, spec


def write_meta(directory: str, spec: NetSpec, cfg: Config, extra: dict[str, Any]) -> str:
    path = os.path.join(directory, "model_meta.json")
    write_json(path, {
        "board_size": spec.board_size,
        "win_length": cfg.game.win_length,
        "allow_overline": cfg.game.allow_overline,
        "input_planes": NUM_PLANES,
        "input_shape": [1, NUM_PLANES, spec.board_size, spec.board_size],
        "input_layout": "NCHW float32",
        "plane_meaning": PLANE_DOC,
        "outputs": {"policy": [1, spec.action_size], "value": [1]},
        "policy_index": "row * board_size + col, row 0 at the top",
        "value_convention": "+1 = side to move wins, -1 = side to move loses",
        "architecture": spec.to_dict(),
        "parameters": spec.parameter_count(),
        **extra,
    })
    return path


def export_coreml(cfg: Config, source: str | None = None, out_dir: str | None = None,
                  precision: str = "fp16", name: str = "OmokNet") -> str:
    try:
        import coremltools as ct
    except ImportError as exc:  # pragma: no cover - depends on the machine
        raise RuntimeError(
            "coremltools is not installed -- run `make install-export`") from exc
    import torch

    weights, src_path, spec = resolve_weights(cfg, source)
    out_dir = out_dir or cfg.paths().ensure().export
    os.makedirs(out_dir, exist_ok=True)
    model = _torch_model(spec, weights)
    example = torch.zeros(1, NUM_PLANES, spec.board_size, spec.board_size)
    traced = torch.jit.trace(model, example)

    precision_map = {"fp16": ct.precision.FLOAT16, "fp32": ct.precision.FLOAT32}
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="planes", shape=example.shape, dtype=np.float32)],
        outputs=[ct.TensorType(name="policy"), ct.TensorType(name="value")],
        convert_to="mlprogram",
        compute_precision=precision_map.get(precision, ct.precision.FLOAT16),
        minimum_deployment_target=ct.target.iOS16,
    )
    mlmodel.short_description = (
        f"Omok {spec.board_size}x{spec.board_size} policy/value network "
        f"({spec.blocks}x{spec.channels}, {spec.parameter_count():,} parameters)")
    path = os.path.join(out_dir, f"{name}.mlpackage")
    mlmodel.save(path)
    meta = write_meta(out_dir, spec, cfg, {"source_checkpoint": os.path.abspath(src_path),
                                           "format": "coreml", "precision": precision})
    log(f"CoreML model written to {os.path.abspath(path)}")
    log(f"metadata written to {os.path.abspath(meta)}")
    return path


def export_onnx(cfg: Config, source: str | None = None, out_dir: str | None = None,
                name: str = "OmokNet") -> str:
    import torch

    try:
        import onnx  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("the onnx package is not installed -- "
                           "run `make install-export`") from exc

    weights, src_path, spec = resolve_weights(cfg, source)
    out_dir = out_dir or cfg.paths().ensure().export
    os.makedirs(out_dir, exist_ok=True)
    model = _torch_model(spec, weights)
    example = torch.zeros(1, NUM_PLANES, spec.board_size, spec.board_size)
    path = os.path.join(out_dir, f"{name}.onnx")
    kwargs = dict(input_names=["planes"], output_names=["policy", "value"],
                  dynamic_axes={"planes": {0: "batch"}, "policy": {0: "batch"},
                                "value": {0: "batch"}}, opset_version=17)
    try:  # torch >= 2.9 defaults to the dynamo exporter, which needs onnxscript
        torch.onnx.export(model, example, path, dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(model, example, path, **kwargs)
    write_meta(out_dir, spec, cfg, {"source_checkpoint": os.path.abspath(src_path),
                                    "format": "onnx"})
    log(f"ONNX model written to {os.path.abspath(path)}")
    return path


def export_npz(cfg: Config, source: str | None = None, out_dir: str | None = None,
               name: str = "OmokNet") -> str:
    """Plain weights + metadata, for a hand-written Metal/Accelerate runtime."""
    from .checkpoint import save_weights

    weights, src_path, spec = resolve_weights(cfg, source)
    out_dir = out_dir or cfg.paths().ensure().export
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.npz")
    save_weights(path, weights)
    write_meta(out_dir, spec, cfg, {"source_checkpoint": os.path.abspath(src_path),
                                    "format": "npz"})
    log(f"weights written to {os.path.abspath(path)}")
    return path


EXPORTERS = {"coreml": export_coreml, "onnx": export_onnx, "npz": export_npz}


def export(cfg: Config, fmt: str = "coreml", **kwargs) -> str:
    if fmt not in EXPORTERS:
        raise ValueError(f"unknown export format {fmt!r}; choose from {sorted(EXPORTERS)}")
    return EXPORTERS[fmt](cfg, **kwargs)
