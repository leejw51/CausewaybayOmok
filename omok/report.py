"""Model-size accounting, speed benchmarks and training-time estimates."""

from __future__ import annotations

import os
import platform
import time
from typing import Any

import numpy as np

from .board import Board
from .config import Config
from .encode import NUM_PLANES
from .mcts import NetworkEvaluator, Tree, run_search
from .netspec import NetSpec, parameter_shapes
from .replay import dataset_stats
from .utils import human_time

# Fallback used before any self-play data exists.
DEFAULT_GAME_LENGTH = 50.0


# --------------------------------------------------------------- model size
def model_report(spec: NetSpec) -> dict[str, Any]:
    params = spec.parameter_count()
    by_part: dict[str, int] = {}
    for name, shape in parameter_shapes(spec):
        part = name.split(".")[0]
        by_part[part] = by_part.get(part, 0) + int(np.prod(shape))
    return {
        "params": params,
        "fp32_mb": params * 4 / 1e6,
        "fp16_mb": params * 2 / 1e6,
        "int8_mb": params / 1e6,
        "by_part": by_part,
        "input_shape": [NUM_PLANES, spec.board_size, spec.board_size],
        "blocks": spec.blocks,
        "channels": spec.channels,
    }


# ---------------------------------------------------------------- benchmarks
def bench_network(backend, spec: NetSpec, batch: int = 64,
                  seconds: float = 1.5) -> dict[str, float]:
    planes = np.zeros((batch, NUM_PLANES, spec.board_size, spec.board_size), dtype=np.float32)
    backend.predict(planes)  # warm up kernels / lazy compilation
    start = time.perf_counter()
    iterations = 0
    while time.perf_counter() - start < seconds:
        backend.predict(planes)
        iterations += 1
    elapsed = time.perf_counter() - start
    return {"eval_batch": batch, "eval_positions_per_s": iterations * batch / elapsed,
            "eval_batches_per_s": iterations / elapsed}


def bench_train(backend, cfg: Config, spec: NetSpec, seconds: float = 2.0) -> dict[str, float]:
    batch = cfg.train.batch_size
    size = spec.board_size
    rng = np.random.default_rng(0)
    planes = rng.random((batch, NUM_PLANES, size, size), dtype=np.float32)
    pi = rng.random((batch, spec.action_size), dtype=np.float32)
    pi /= pi.sum(axis=1, keepdims=True)
    z = rng.random(batch, dtype=np.float32) * 2 - 1
    backend.train_step(planes, pi, z, cfg.train.lr, cfg.train.value_loss_weight,
                       cfg.train.grad_clip)
    start = time.perf_counter()
    steps = 0
    while time.perf_counter() - start < seconds:
        backend.train_step(planes, pi, z, cfg.train.lr, cfg.train.value_loss_weight,
                           cfg.train.grad_clip)
        steps += 1
    elapsed = time.perf_counter() - start
    return {"train_batch": batch, "train_steps_per_s": steps / elapsed,
            "train_samples_per_s": steps * batch / elapsed}


def bench_selfplay(backend, cfg: Config, seconds: float = 4.0) -> dict[str, float]:
    """Measure real MCTS throughput at the configured simulation count."""
    parallel = max(1, cfg.selfplay.parallel_games)
    evaluator = NetworkEvaluator(backend, batch_size=max(32, parallel))
    rng = np.random.default_rng(12345)
    trees = [Tree(Board(cfg.game.board_size, cfg.game.win_length, cfg.game.allow_overline),
                  cfg.mcts) for _ in range(parallel)]
    start = time.perf_counter()
    moves = 0
    while time.perf_counter() - start < seconds:
        run_search(trees, evaluator, cfg.mcts.simulations, rng)
        for i, tree in enumerate(trees):
            if tree.board.over:
                trees[i] = Tree(Board(cfg.game.board_size, cfg.game.win_length,
                                      cfg.game.allow_overline), cfg.mcts)
                continue
            tree.advance(tree.pick_move(rng, cfg.mcts.temperature))
            moves += 1
    elapsed = time.perf_counter() - start
    return {"selfplay_moves_per_s": moves / elapsed,
            "selfplay_parallel": parallel,
            "selfplay_sims": cfg.mcts.simulations}


def benchmark(backend, cfg: Config, spec: NetSpec, quick: bool = True) -> dict[str, float]:
    scale = 1.0 if quick else 3.0
    out: dict[str, float] = {}
    out.update(bench_network(backend, spec, batch=max(32, cfg.selfplay.parallel_games),
                             seconds=1.0 * scale))
    out.update(bench_train(backend, cfg, spec, seconds=1.5 * scale))
    out.update(bench_selfplay(backend, cfg, seconds=3.0 * scale))
    return out


# ----------------------------------------------------------------- estimates
def estimate_schedule(cfg: Config, measured: dict[str, float],
                      mean_game_length: float | None = None) -> dict[str, Any]:
    """Turn measured throughput into per-iteration and total time estimates."""
    # On small boards games cannot run as long as the 15x15 default assumes.
    length = mean_game_length or min(DEFAULT_GAME_LENGTH, cfg.action_size * 0.6)
    moves_per_s = max(1e-6, measured.get("selfplay_moves_per_s", 1.0))
    steps_per_s = max(1e-6, measured.get("train_steps_per_s", 1.0))

    selfplay_seconds = cfg.selfplay.games_per_iter * length / moves_per_s
    train_seconds = cfg.train.steps_per_iter / steps_per_s

    # Arena moves cost less when it searches fewer simulations, and scale with
    # how many games run concurrently (bigger batches -> better GPU use).
    sim_ratio = cfg.mcts.simulations / max(1, cfg.arena.simulations)
    parallel_ratio = max(1, cfg.arena.parallel_games) / max(1, cfg.selfplay.parallel_games)
    arena_moves_per_s = moves_per_s * sim_ratio * (parallel_ratio ** 0.5)
    arena_seconds = (cfg.arena.games * length / arena_moves_per_s) if cfg.arena.games else 0.0

    per_iteration = selfplay_seconds + train_seconds + arena_seconds
    return {
        "mean_game_length": length,
        "selfplay_seconds": selfplay_seconds,
        "train_seconds": train_seconds,
        "arena_seconds": arena_seconds,
        "iteration_seconds": per_iteration,
        "iterations": cfg.iterations,
        "total_seconds": per_iteration * cfg.iterations,
        "positions_per_iteration": cfg.selfplay.games_per_iter * length,
        "samples_per_iteration": cfg.train.steps_per_iter * cfg.train.batch_size,
    }


# -------------------------------------------------------------------- output
def format_report(cfg: Config, spec: NetSpec, backend_info: dict[str, Any],
                  model: dict[str, Any], measured: dict[str, float] | None,
                  schedule: dict[str, Any] | None) -> str:
    paths = cfg.paths()
    lines: list[str] = []
    add = lines.append
    add("=" * 68)
    add(f"  Omok trainer -- run '{cfg.run_name}'")
    add("=" * 68)
    add("")
    add("Model")
    add(f"  architecture      {spec.blocks} residual blocks x {spec.channels} channels")
    add(f"  input             {model['input_shape'][0]} planes x "
        f"{spec.board_size}x{spec.board_size}  ->  policy {spec.action_size} + value 1")
    add(f"  parameters        {model['params']:,}")
    add(f"  size on disk      {model['fp32_mb']:.2f} MB float32   "
        f"({model['fp16_mb']:.2f} MB float16, ~{model['int8_mb']:.2f} MB int8)")
    parts = ", ".join(f"{k}={v:,}" for k, v in model["by_part"].items())
    add(f"  parameters by part {parts}")
    add("")
    add("Compute")
    add(f"  backend           {backend_info.get('backend')} on {backend_info.get('device')}")
    for key in ("torch", "mlx", "gpu"):
        if key in backend_info:
            add(f"  {key:<17} {backend_info[key]}")
    add(f"  host              {platform.system()} {platform.machine()}, "
        f"python {platform.python_version()}")
    if measured:
        add("")
        add("Measured throughput")
        add(f"  network eval      {measured['eval_positions_per_s']:,.0f} positions/s "
            f"(batch {int(measured['eval_batch'])})")
        add(f"  training          {measured['train_steps_per_s']:.2f} steps/s "
            f"= {measured['train_samples_per_s']:,.0f} samples/s (batch {int(measured['train_batch'])})")
        add(f"  self-play         {measured['selfplay_moves_per_s']:.2f} moves/s "
            f"({int(measured['selfplay_parallel'])} games in parallel, "
            f"{int(measured['selfplay_sims'])} sims/move)")
    if schedule:
        add("")
        add("Expected training time  (estimate from the numbers above)")
        add(f"  self-play         {human_time(schedule['selfplay_seconds'])} per iteration "
            f"({cfg.selfplay.games_per_iter} games, ~{schedule['mean_game_length']:.0f} moves each)")
        add(f"  training          {human_time(schedule['train_seconds'])} per iteration "
            f"({cfg.train.steps_per_iter} steps x batch {cfg.train.batch_size})")
        if schedule["arena_seconds"]:
            add(f"  arena             {human_time(schedule['arena_seconds'])} per iteration "
                f"({cfg.arena.games} games)")
        add(f"  one iteration     {human_time(schedule['iteration_seconds'])}")
        add(f"  {cfg.iterations} iterations".ljust(20)
            + f"{human_time(schedule['total_seconds'])}"
            + f"   ({schedule['total_seconds'] / 3600:.1f} hours of compute)")
        milestones = [(label, n) for label, n in
                      (("plays legally, blocks obvious threats", 10),
                       ("beats a casual human", 60),
                       ("hard to beat without study", 200))
                      if n <= cfg.iterations]
        if milestones:
            add("")
            add("  Rough milestones (self-play strength grows roughly with iterations)")
            for label, n in milestones:
                add(f"    {n:>4} iterations  "
                    f"{human_time(schedule['iteration_seconds'] * n):>9}   {label}")
    add("")
    add("Files")
    add(f"  run directory     {os.path.abspath(paths.root)}")
    add(f"  config            {os.path.abspath(paths.config)}")
    add(f"  state (resume)    {os.path.abspath(paths.state)}")
    add(f"  checkpoints       {os.path.abspath(paths.checkpoints)}/step-XXXXXXXX.npz")
    add(f"  best model        {os.path.abspath(os.path.join(paths.checkpoints, 'best.npz'))}")
    add(f"  self-play data    {os.path.abspath(paths.replay)}/shard-XXXXXXXX.npz")
    add(f"  logs              {os.path.abspath(os.path.join(paths.logs, 'run.jsonl'))}")
    add(f"  exports (CoreML)  {os.path.abspath(paths.export)}/")
    add("=" * 68)
    return "\n".join(lines)


def run_report(cfg: Config, backend=None, do_benchmark: bool = True,
               quick: bool = True) -> str:
    from .backends import make_backend

    spec = NetSpec.from_config(cfg)
    if backend is None:
        backend = make_backend(spec, cfg.backend, lr=cfg.train.lr,
                               weight_decay=cfg.train.weight_decay)
    model = model_report(spec)
    measured = benchmark(backend, cfg, spec, quick=quick) if do_benchmark else None
    data = dataset_stats(cfg.paths().replay) if os.path.isdir(cfg.paths().replay) else None
    length = data["mean_length"] if data and data["games"] >= 5 else None
    schedule = estimate_schedule(cfg, measured, length) if measured else None
    return format_report(cfg, spec, backend.describe(), model, measured, schedule)
