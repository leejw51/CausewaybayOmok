import json
import os
import signal
import subprocess
import sys
import time

import numpy as np
import pytest

from omok.checkpoint import CheckpointManager
from omok.config import Config
from omok.pipeline import Pipeline
from omok.replay import count_games, dataset_stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class NoKiller:
    stop = False


def tiny_config(run_dir) -> Config:
    cfg = Config()
    for key, value in {
        "run_dir": str(run_dir), "run_name": "test", "backend": "auto", "iterations": 1,
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


def test_full_iteration_produces_data_checkpoints_and_state(tmp_path):
    cfg = tiny_config(tmp_path / "run")
    pipeline = Pipeline(cfg, killer=NoKiller(), echo=False)
    try:
        summary = pipeline.run(1)
    finally:
        pipeline.close()
    paths = cfg.paths()
    assert summary["iteration"] == 1
    assert count_games(paths.replay) >= 4
    assert CheckpointManager(paths.checkpoints).latest() is not None
    assert os.path.exists(os.path.join(paths.checkpoints, "best.npz"))
    state = json.loads(open(paths.state).read())
    assert state["iteration"] == 1 and state["phase"] == "selfplay"
    assert os.path.exists(os.path.join(paths.logs, "run.jsonl"))


def test_second_run_resumes_instead_of_restarting(tmp_path):
    cfg = tiny_config(tmp_path / "run")
    first = Pipeline(cfg, killer=NoKiller(), echo=False)
    try:
        first.run(1)
        step_after_first = first.trainer.global_step
    finally:
        first.close()

    second = Pipeline(cfg, killer=NoKiller(), echo=False)
    try:
        assert second.trainer.global_step == step_after_first  # weights + step restored
        summary = second.run(1)
    finally:
        second.close()
    assert summary["iteration"] == 2
    assert summary["global_step"] > step_after_first


def test_selfplay_phase_tops_up_existing_games(tmp_path):
    from omok.backends import make_backend
    from omok.netspec import NetSpec, init_weights
    from omok.replay import ShardWriter
    from omok.selfplay import run_selfplay

    cfg = tiny_config(tmp_path / "run")
    cfg.paths().ensure()
    spec = NetSpec.from_config(cfg)
    backend = make_backend(spec, cfg.backend)
    backend.set_weights(init_weights(spec, 0))

    first = run_selfplay(cfg, backend, iteration=0, target_games=3, killer=NoKiller())
    assert first.games == 3
    second = run_selfplay(cfg, backend, iteration=0, target_games=3, killer=NoKiller())
    assert second.games == 0            # already on disk, nothing to do
    third = run_selfplay(cfg, backend, iteration=1, target_games=2, killer=NoKiller())
    assert third.games == 2             # a new iteration generates fresh games
    assert count_games(cfg.paths().replay) == 5


def test_data_survives_a_hard_kill(tmp_path):
    """SIGKILL mid-run must leave usable shards and a resumable state."""
    run_dir = tmp_path / "run"
    cfg = tiny_config(run_dir)
    cfg.override("selfplay.games_per_iter", "200")   # long enough to interrupt
    cfg.override("iterations", "5")
    cfg.paths().ensure()
    cfg.save(cfg.paths().config)

    cmd = [sys.executable, "-m", "omok", "train", "--config", cfg.paths().config,
           "--no-bench", "--quiet"]
    process = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + 90
        while time.time() < deadline and count_games(cfg.paths().replay) < 4:
            if process.poll() is not None:
                pytest.fail("training exited early")
            time.sleep(0.5)
        assert count_games(cfg.paths().replay) >= 4, "no data was flushed before the kill"
        process.send_signal(signal.SIGKILL)
    finally:
        process.wait(timeout=30)

    games_before = count_games(cfg.paths().replay)
    assert dataset_stats(cfg.paths().replay)["positions"] > 0

    # Restarting must keep the old games and continue from the saved state.
    cfg.override("selfplay.games_per_iter", str(games_before + 2))
    cfg.save(cfg.paths().config)
    resumed = Pipeline(cfg, killer=NoKiller(), echo=False)
    try:
        summary = resumed.run(1)
    finally:
        resumed.close()
    assert summary["games"] >= games_before
    assert summary["iteration"] == 1


def test_export_npz_and_metadata(tmp_path):
    from omok.export import export

    cfg = tiny_config(tmp_path / "run")
    pipeline = Pipeline(cfg, killer=NoKiller(), echo=False)
    pipeline.close()
    path = export(cfg, "npz")
    assert os.path.exists(path)
    meta = json.loads(open(os.path.join(cfg.paths().export, "model_meta.json")).read())
    assert meta["board_size"] == 7
    assert meta["input_shape"] == [1, 5, 7, 7]
    assert meta["outputs"]["policy"] == [1, 49]
