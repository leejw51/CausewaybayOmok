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


def test_orphan_checkpoint_beats_a_stale_latest_pointer(tmp_path):
    """save() writes weights -> optimiser -> meta -> pointer; a crash after the
    optimiser leaves the newest step with no meta and the pointer still naming
    the previous save.  The newest weights (and their optimiser) must win."""
    manager = CheckpointManager(str(tmp_path))
    manager.save(FakeBackend(seed=1), {"step": 10})
    newest = FakeBackend(seed=2)
    saved = manager.save(newest, {"step": 20})
    os.unlink(manager.path_for(saved.tag, ".json"))
    atomic_write_text(manager.path_for(saved.tag, ".opt.pt"), "optimiser state")
    atomic_write_text(manager.latest_pointer, '{"tag": "step-00000010", "step": 10}')

    latest = manager.latest()
    assert latest is not None and latest.tag == saved.tag
    assert latest.meta["step"] == 20 and latest.meta["meta_missing"]
    assert latest.meta["optimizer"] == os.path.basename(manager.path_for(saved.tag, ".opt.pt"))
    for key, value in newest.get_weights().items():
        assert np.allclose(latest.weights()[key], value)


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


def test_shard_writer_survives_losing_the_race(tmp_path, monkeypatch):
    """Both writers pass the exists() check before either has written: the
    loser must not overwrite the winner's shard (os.replace would)."""
    import omok.replay as replay

    def game(move):
        record = GameRecord(winner=1)
        policy = np.zeros(49, dtype=np.float32)
        policy[move] = 1.0
        record.add(move, policy)
        return record

    monkeypatch.setattr(replay.os.path, "exists", lambda path: False)  # stale view
    a = ShardWriter(str(tmp_path), flush_every_games=1)
    b = ShardWriter(str(tmp_path), flush_every_games=1)
    a.add(game(0))
    b.add(game(1))
    monkeypatch.undo()
    paths = shard_paths(str(tmp_path))
    assert [os.path.basename(p) for p in paths] == ["shard-00000000.npz", "shard-00000001.npz"]
    assert count_games(str(tmp_path)) == 2
    assert not [n for n in os.listdir(tmp_path) if n.startswith(".tmp-")]  # no litter


def test_atomic_write_exclusive_never_overwrites(tmp_path):
    from omok.utils import atomic_write_with

    target = tmp_path / "claimed.txt"
    target.write_text("winner")
    with pytest.raises(FileExistsError):
        atomic_write_with(str(target), lambda tmp: open(tmp, "w").write("loser"),
                          exclusive=True)
    assert target.read_text() == "winner"
    assert sorted(os.listdir(tmp_path)) == ["claimed.txt"]


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

    checkpoints = cfg.paths().checkpoints
    before = {name: os.stat(os.path.join(checkpoints, name)).st_mtime_ns
              for name in os.listdir(checkpoints)}

    second = Pipeline(cfg, killer=NoKiller(), echo=False)
    try:
        assert second.target_iteration() == 1  # what cmd_train checks first
        summary = second.run()  # already complete -> nothing to do
    finally:
        second.close()
    assert summary["iteration"] == 1 and summary["complete"]
    assert count_games(cfg.paths().replay) == games_after
    after = {name: os.stat(os.path.join(checkpoints, name)).st_mtime_ns
             for name in os.listdir(checkpoints)}
    assert after == before, "a no-op resume rewrote checkpoint files"
    events = [line for line in open(os.path.join(cfg.paths().logs, "run.jsonl"))]
    assert not any('"pipeline.start"' in line for line in events[-3:])


def test_interrupted_run_resumes_to_its_own_target(tmp_path):
    """`make train` (to N) crashes -> `make resume` must finish at N, not at
    the preset total and not become a no-op."""
    from omok.pipeline import Pipeline

    cfg = tiny_config(tmp_path / "run")  # configured total: 1

    class StopAfterFirstArena(NoKiller):
        stop = False

    killer = StopAfterFirstArena()
    first = Pipeline(cfg, killer=killer, echo=False)
    real_arena = first._phase_arena

    def arena_then_stop(iteration):
        real_arena(iteration)
        killer.stop = True

    first._phase_arena = arena_then_stop
    try:
        first.run(3)  # asked for 3 more, interrupted during the first
    finally:
        first.close()
    state = read_json(cfg.paths().state)
    assert state["iteration"] < 3 and state["target_iteration"] == 3

    resumed = Pipeline(cfg, killer=NoKiller(), echo=False)
    try:
        assert resumed.target_iteration() == 3  # not cfg.iterations (1)
        assert resumed.run()["iteration"] == 3
    finally:
        resumed.close()


def test_stale_relative_iterations_in_config_do_not_add_a_fresh_budget(tmp_path):
    """Runs made before the semantics change hold the old relative --iterations
    value in config.json: a bare resume must not treat it as 'that many more'."""
    from omok.pipeline import Pipeline

    cfg = tiny_config(tmp_path / "run")
    pipeline = Pipeline(cfg, killer=NoKiller(), echo=False)
    try:
        pipeline.run()
    finally:
        pipeline.close()
    state = read_json(cfg.paths().state)
    del state["target_iteration"]  # an old run never recorded one
    atomic_write_text(cfg.paths().state, __import__("json").dumps(state))

    old = Pipeline(cfg, killer=NoKiller(), echo=False)  # config.iterations == 1
    try:
        assert old.target_iteration() == 1
        assert old.target_iteration(2) == 3  # explicit count still adds
    finally:
        old.close()


def test_explicit_iteration_count_still_adds_more(tmp_path):
    from omok.pipeline import Pipeline

    cfg = tiny_config(tmp_path / "run")
    pipeline = Pipeline(cfg, killer=NoKiller(), echo=False)
    try:
        pipeline.run()               # finish the configured single iteration
        assert pipeline.run(1)["iteration"] == 2   # --iterations N: N more
    finally:
        pipeline.close()


# ---- `omok selfplay` adds games instead of pre-filling the next quota ------
def test_manual_selfplay_games_are_added_not_absorbed(tmp_path, capsys):
    from omok.cli import main
    from omok.pipeline import Pipeline

    cfg = tiny_config(tmp_path / "run")
    pipeline = Pipeline(cfg, killer=NoKiller(), echo=False)
    try:
        pipeline.run()  # iteration 0 done; state now points at iteration 1
    finally:
        pipeline.close()
    games_before = count_games(cfg.paths().replay)
    cfg.save(cfg.paths().config)

    assert main(["selfplay", "--run-dir", str(tmp_path / "run"), "--games", "2"]) == 0
    assert count_games(cfg.paths().replay) == games_before + 2
    assert count_games(cfg.paths().replay, iteration=1) == 0  # not iteration 1's quota

    again = Pipeline(cfg, killer=NoKiller(), echo=False)
    try:
        again.run(1)  # iteration 1 still self-plays its full games_per_iter
    finally:
        again.close()
    assert count_games(cfg.paths().replay, iteration=1) == cfg.selfplay.games_per_iter


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
