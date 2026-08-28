"""Regression tests for the issues found in the 2026-08 code review."""

import os

import numpy as np
import pytest

from omok.board import Board, parse_move
from omok.checkpoint import CheckpointManager, save_weights
from omok.config import Config
from omok.netspec import NetSpec, init_weights
from omok.replay import GameRecord, ShardWriter, count_games, shard_paths
from omok.utils import atomic_write_text, read_json

SPEC = NetSpec(board_size=7, channels=8, blocks=1, policy_channels=2,
               value_channels=2, value_hidden=16)


class FakeBackend:
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
        return False

    def load_optimizer(self, path):
        return False


def tiny_config(run_dir) -> Config:
    cfg = Config()
    for key, value in {
        "run_dir": str(run_dir), "run_name": "test", "iterations": 1,
        "game.board_size": "7",
        "net.channels": "16", "net.blocks": "1", "net.value_hidden": "32",
        "mcts.simulations": "8", "mcts.temperature_moves": "4",
        "selfplay.games_per_iter": "4", "selfplay.parallel_games": "4",
        "train.batch_size": "16", "train.steps_per_iter": "5",
        "train.replay_min_positions": "10", "train.ckpt_every_steps": "2",
        "arena.games": "2", "arena.simulations": "8", "arena.parallel_games": "2",
    }.items():
        cfg.override(key, str(value))
    return cfg


class NoKiller:
    stop = False


# ---- benchmark must not touch the live network (report.py) -----------------
def test_bench_train_leaves_the_backend_unchanged():
    from omok.backends import make_backend
    from omok.report import bench_train

    cfg = Config()
    cfg.override("game.board_size", "7")
    cfg.override("train.batch_size", "8")
    backend = make_backend(SPEC, "auto")
    backend.set_weights(init_weights(SPEC, seed=9))
    before = {k: v.copy() for k, v in backend.get_weights().items()}
    bench_train(backend, cfg, SPEC, seconds=0.05)
    after = backend.get_weights()
    for key, value in before.items():
        assert np.array_equal(after[key], value), f"benchmark modified {key}"


# ---- torn best.npz must not block startup (checkpoint.py) ------------------
def test_torn_best_file_is_ignored(tmp_path):
    manager = CheckpointManager(str(tmp_path))
    atomic_write_text(manager.best_path, "definitely not an npz")
    assert manager.best() is None


def test_pipeline_recovers_from_a_torn_best_file(tmp_path):
    from omok.pipeline import Pipeline

    cfg = tiny_config(tmp_path / "run")
    paths = cfg.paths().ensure()
    atomic_write_text(os.path.join(paths.checkpoints, "best.npz"), "torn")
    pipeline = Pipeline(cfg, killer=NoKiller(), echo=False)  # must not raise
    pipeline.close()
    assert CheckpointManager(paths.checkpoints).best() is not None  # re-seeded


# ---- weights without meta .json still resume (checkpoint.py) ---------------
def test_checkpoint_without_meta_json_still_resumes(tmp_path):
    manager = CheckpointManager(str(tmp_path))
    backend = FakeBackend(seed=1)
    checkpoint = manager.save(backend, {"step": 42})
    os.unlink(manager.path_for(checkpoint.tag, ".json"))  # crash before meta
    os.unlink(manager.latest_pointer)
    other = FakeBackend(seed=2)
    meta = manager.restore(other)
    assert meta, "weights on disk were treated as 'no checkpoint'"
    assert meta["step"] == 42  # recovered from the tag
    for key, value in backend.get_weights().items():
        assert np.allclose(other.get_weights()[key], value)


# ---- a second writer never clobbers existing shards (replay.py) ------------
def test_shard_writer_skips_numbers_taken_by_another_process(tmp_path):
    def game(move):
        record = GameRecord(winner=1)
        policy = np.zeros(49, dtype=np.float32)
        policy[move] = 1.0
        record.add(move, policy)
        return record

    a = ShardWriter(str(tmp_path), flush_every_games=1)
    b = ShardWriter(str(tmp_path), flush_every_games=1)  # same starting seq
    a.add(game(0))
    b.add(game(1))  # must NOT overwrite a's shard
    assert len(shard_paths(str(tmp_path))) == 2
    assert count_games(str(tmp_path)) == 2


# ---- default run() finishes the configured total, not total-more -----------
def test_resume_of_a_finished_run_is_a_noop(tmp_path):
    from omok.pipeline import Pipeline

    cfg = tiny_config(tmp_path / "run")  # iterations = 1
    first = Pipeline(cfg, killer=NoKiller(), echo=False)
    try:
        assert first.run()["iteration"] == 1  # trains to the configured total
    finally:
        first.close()
    games_after = count_games(cfg.paths().replay)

    second = Pipeline(cfg, killer=NoKiller(), echo=False)
    try:
        summary = second.run()  # already complete -> nothing to do
    finally:
        second.close()
    assert summary["iteration"] == 1
    assert count_games(cfg.paths().replay) == games_after


def test_explicit_iteration_count_still_adds_more(tmp_path):
    from omok.pipeline import Pipeline

    cfg = tiny_config(tmp_path / "run")
    pipeline = Pipeline(cfg, killer=NoKiller(), echo=False)
    try:
        pipeline.run()               # finish the configured single iteration
        assert pipeline.run(1)["iteration"] == 2   # --iterations N: N more
    finally:
        pipeline.close()


# ---- a game decided during the random opening must not crash ---------------
def test_opening_that_finishes_the_game_is_handled(tmp_path):
    from omok.replay import ShardWriter as Writer
    from omok.selfplay import generate_games

    class UniformBackend:
        def predict(self, planes):
            n = planes.shape[0]
            actions = planes.shape[2] * planes.shape[3]
            return (np.full((n, actions), 1.0 / actions, dtype=np.float32),
                    np.zeros(n, dtype=np.float32))

    cfg = Config()
    for key, value in {"game.board_size": "3", "game.win_length": "5",
                       "mcts.simulations": "4", "selfplay.parallel_games": "2",
                       "selfplay.random_opening_plies": "9"}.items():
        cfg.override(key, value)
    # 3x3 board, 9 random opening plies: every game ends (draw) in the opening.
    writer = Writer(str(tmp_path), flush_every_games=1)
    stats = generate_games(UniformBackend(), cfg, writer, num_games=3,
                           rng=np.random.default_rng(0))
    writer.close()
    assert stats.games == 3
    assert count_games(str(tmp_path)) == 3


# ---- parse_move rejects off-board coordinates (board.py) -------------------
@pytest.mark.parametrize("text", ["7,20", "20,7", "q3", "a15", "-1,3"])
def test_parse_move_rejects_off_board_coordinates(text):
    with pytest.raises(ValueError):
        parse_move(text, 15)


def test_parse_move_still_accepts_valid_forms():
    assert parse_move("h8", 15) == 8 * 15 + 7
    assert parse_move("7,7", 15) == 7 * 15 + 7
    assert parse_move("0,0", 15) == 0
    assert parse_move("o14", 15) == 14 * 15 + 14


# ---- an architecture --set cannot poison a resumable run (pipeline.py) -----
def test_bad_architecture_override_does_not_poison_the_config(tmp_path):
    from omok.pipeline import Pipeline

    cfg = tiny_config(tmp_path / "run")
    pipeline = Pipeline(cfg, killer=NoKiller(), echo=False)
    try:
        pipeline.run()
    finally:
        pipeline.close()

    bad = tiny_config(tmp_path / "run")
    bad.override("net.blocks", "3")  # incompatible with the saved checkpoint
    with pytest.raises(ValueError):
        Pipeline(bad, killer=NoKiller(), echo=False)
    # the stored config must still describe the working architecture
    stored = read_json(cfg.paths().config)
    assert stored["net"]["blocks"] == 1

    resumed = Pipeline(tiny_config(tmp_path / "run"), killer=NoKiller(), echo=False)
    resumed.close()  # plain resume still works
