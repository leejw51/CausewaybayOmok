import importlib.util
import json
import os

import numpy as np
import pytest

from omok.checkpoint import CheckpointManager, load_weights
from omok.config import Config
from omok.encode import NUM_PLANES
from omok.export import export
from omok.netspec import NetSpec, init_weights

HAS_COREML = importlib.util.find_spec("coremltools") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_ONNX = importlib.util.find_spec("onnx") is not None


def prepared_run(tmp_path) -> tuple[Config, NetSpec]:
    cfg = Config()
    for key, value in {"run_dir": str(tmp_path / "run"), "game.board_size": "7",
                       "net.channels": "16", "net.blocks": "1",
                       "net.value_hidden": "32"}.items():
        cfg.override(key, value)
    paths = cfg.paths().ensure()
    spec = NetSpec.from_config(cfg)
    manager = CheckpointManager(paths.checkpoints)
    manager.save_best(init_weights(spec, 5), {"step": 1, "spec": spec.to_dict()})
    return cfg, spec


def test_npz_export_keeps_the_exact_weights(tmp_path):
    cfg, spec = prepared_run(tmp_path)
    path = export(cfg, "npz")
    exported = load_weights(path)
    for key, value in init_weights(spec, 5).items():
        assert np.allclose(exported[key], value)
    meta = json.loads(open(os.path.join(cfg.paths().export, "model_meta.json")).read())
    assert meta["input_planes"] == NUM_PLANES
    assert meta["parameters"] == spec.parameter_count()


@pytest.mark.skipif(not (HAS_TORCH and HAS_ONNX), reason="needs PyTorch + onnx")
def test_onnx_export_produces_a_file(tmp_path):
    cfg, _ = prepared_run(tmp_path)
    path = export(cfg, "onnx")
    assert os.path.getsize(path) > 1000


@pytest.mark.skipif(not (HAS_COREML and HAS_TORCH), reason="needs coremltools")
def test_coreml_output_matches_the_python_model(tmp_path):
    import coremltools as ct

    from omok.backends.torch_backend import TorchBackend

    cfg, spec = prepared_run(tmp_path)
    path = export(cfg, "coreml", precision="fp32")

    backend = TorchBackend(spec, device="cpu")
    backend.set_weights(init_weights(spec, 5))
    rng = np.random.default_rng(0)
    planes = (rng.random((1, NUM_PLANES, spec.board_size, spec.board_size)) < 0.2)
    planes = planes.astype(np.float32)
    policy_ref, value_ref = backend.predict(planes)

    model = ct.models.MLModel(path)
    out = model.predict({"planes": planes})
    policy = np.array(out["policy"]).reshape(-1)
    value = np.array(out["value"]).reshape(-1)
    assert np.abs(policy - policy_ref[0]).max() < 1e-3
    assert abs(float(value[0]) - float(value_ref[0])) < 1e-3
