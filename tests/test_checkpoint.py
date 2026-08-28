import os

import numpy as np

from omok.checkpoint import CheckpointManager, load_weights, save_weights
from omok.netspec import NetSpec, init_weights
from omok.utils import atomic_write_text, read_json

SPEC = NetSpec(board_size=7, channels=8, blocks=1, policy_channels=2,
               value_channels=2, value_hidden=16)


class FakeBackend:
    """Just enough surface for the checkpoint manager."""

    name = "fake"
    device = "cpu"

    def __init__(self, seed=0):
        self.spec = SPEC
        self._weights = init_weights(SPEC, seed)

    def get_weights(self):
        return self._weights

    def set_weights(self, weights):
        self._weights = weights

    def save_optimizer(self, path):
        save_weights(path, {"m": np.ones(3, dtype=np.float32)})
        return True

    def load_optimizer(self, path):
        return os.path.exists(path)


def test_save_and_restore_roundtrip(tmp_path):
    manager = CheckpointManager(str(tmp_path))
    backend = FakeBackend(seed=1)
    manager.save(backend, {"step": 10})
    other = FakeBackend(seed=2)
    meta = manager.restore(other)
    assert meta["step"] == 10
    for key, value in backend.get_weights().items():
        assert np.allclose(other.get_weights()[key], value)


def test_rotation_keeps_only_the_newest(tmp_path):
    manager = CheckpointManager(str(tmp_path), keep_last=2)
    backend = FakeBackend()
    for step in (1, 2, 3, 4):
        manager.save(backend, {"step": step})
    assert manager.list_tags() == ["step-00000003", "step-00000004"]
    assert manager.latest().step == 4


def test_latest_falls_back_when_the_newest_file_is_corrupt(tmp_path):
    manager = CheckpointManager(str(tmp_path), keep_last=5)
    backend = FakeBackend()
    manager.save(backend, {"step": 1})
    manager.save(backend, {"step": 2})
    # simulate a crash in the middle of writing the newest checkpoint
    atomic_write_text(manager.path_for("step-00000002"), "torn file")
    assert manager.latest().step == 1


def test_missing_pointer_is_rebuilt_from_disk(tmp_path):
    manager = CheckpointManager(str(tmp_path))
    manager.save(FakeBackend(), {"step": 7})
    os.unlink(manager.latest_pointer)
    assert manager.latest().step == 7


def test_best_model_is_separate_from_the_rolling_checkpoints(tmp_path):
    manager = CheckpointManager(str(tmp_path), keep_last=1)
    backend = FakeBackend(seed=3)
    manager.save_best(backend.get_weights(), {"step": 5, "score": 0.7})
    for step in (6, 7, 8):
        manager.save(FakeBackend(seed=step), {"step": step})
    best = manager.best()
    assert best is not None and best.meta["score"] == 0.7
    for key, value in backend.get_weights().items():
        assert np.allclose(best.weights()[key], value)


def test_weights_file_is_written_atomically(tmp_path):
    path = str(tmp_path / "w.npz")
    save_weights(path, {"a": np.arange(4, dtype=np.float32)})
    assert not [p for p in os.listdir(tmp_path) if p.startswith(".tmp-")]
    assert np.allclose(load_weights(path)["a"], np.arange(4))


def test_optimizer_state_is_recorded_in_the_metadata(tmp_path):
    manager = CheckpointManager(str(tmp_path))
    checkpoint = manager.save(FakeBackend(), {"step": 3})
    assert checkpoint.meta["optimizer"].endswith((".opt.pt", ".opt.npz"))
    meta = read_json(manager.path_for(checkpoint.tag, ".json"))
    assert meta["step"] == 3 and meta["bytes"] > 0
