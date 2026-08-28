import numpy as np
import pytest

from omok.backends import available_backends, select_backend
from omok.config import Config
from omok.encode import NUM_PLANES
from omok.netspec import NetSpec, init_weights

SPEC = NetSpec(board_size=7, channels=16, blocks=2, policy_channels=2,
               value_channels=2, value_hidden=32)
AVAILABLE = available_backends()


def make_torch():
    from omok.backends.torch_backend import TorchBackend

    return TorchBackend(SPEC, device="cpu")


def make_mlx():
    from omok.backends.mlx_backend import MLXBackend

    return MLXBackend(SPEC)


def sample_input(batch=4, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((batch, NUM_PLANES, SPEC.board_size, SPEC.board_size),
                      dtype=np.float32)


def backends():
    out = []
    if AVAILABLE["torch"]:
        out.append(("torch", make_torch))
    if AVAILABLE["mlx"]:
        out.append(("mlx", make_mlx))
    return out


@pytest.mark.parametrize("name,factory", backends())
def test_weight_roundtrip(name, factory):
    backend = factory()
    weights = init_weights(SPEC, seed=3)
    backend.set_weights(weights)
    restored = backend.get_weights()
    assert set(restored) == set(weights)
    for key, value in weights.items():
        assert np.allclose(restored[key], value, atol=1e-6), key


@pytest.mark.parametrize("name,factory", backends())
def test_predict_shapes_and_ranges(name, factory):
    backend = factory()
    backend.set_weights(init_weights(SPEC, seed=1))
    policy, value = backend.predict(sample_input(5))
    assert policy.shape == (5, SPEC.action_size)
    assert value.shape == (5,)
    assert np.allclose(policy.sum(axis=1), 1.0, atol=1e-4)
    assert np.all(np.abs(value) <= 1.0 + 1e-5)


@pytest.mark.parametrize("name,factory", backends())
def test_train_step_reduces_loss(name, factory):
    backend = factory()
    backend.set_weights(init_weights(SPEC, seed=2))
    planes = sample_input(8, seed=5)
    rng = np.random.default_rng(6)
    pi = rng.random((8, SPEC.action_size), dtype=np.float32)
    pi /= pi.sum(axis=1, keepdims=True)
    z = rng.random(8, dtype=np.float32) * 2 - 1
    first = backend.train_step(planes, pi, z, 1e-3, 1.0, 4.0)
    for _ in range(30):
        last = backend.train_step(planes, pi, z, 1e-3, 1.0, 4.0)
    assert last["loss"] < first["loss"]
    assert np.isfinite(last["loss"])


@pytest.mark.skipif(not (AVAILABLE["torch"] and AVAILABLE["mlx"]),
                    reason="needs both PyTorch and MLX")
def test_torch_and_mlx_agree_on_the_same_weights():
    """A checkpoint must mean the same thing on either backend."""
    weights = init_weights(SPEC, seed=11)
    torch_backend, mlx_backend = make_torch(), make_mlx()
    torch_backend.set_weights(weights)
    mlx_backend.set_weights(weights)
    planes = sample_input(6, seed=12)
    p_torch, v_torch = torch_backend.predict(planes)
    p_mlx, v_mlx = mlx_backend.predict(planes)
    assert np.allclose(p_torch, p_mlx, atol=2e-4)
    assert np.allclose(v_torch, v_mlx, atol=2e-4)


def test_auto_selection_returns_something_usable():
    name, device = select_backend("auto")
    assert name in ("torch", "mlx")
    assert select_backend("cpu") == ("torch", "cpu")
