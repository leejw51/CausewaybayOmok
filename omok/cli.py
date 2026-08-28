"""Command line entry point -- see ``make help`` for the usual invocations."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from .config import Config
from .presets import PRESETS, make_config
from .utils import human_time, read_json


# ------------------------------------------------------------------ config
def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preset", default="base", choices=sorted(PRESETS),
                        help="starting configuration (default: base)")
    parser.add_argument("--config", default=None, help="explicit config.json to load")
    parser.add_argument("--run-dir", default=None, help="where the run lives (runs/<name>)")
    parser.add_argument("--backend", default=None,
                        choices=["auto", "torch", "mlx", "cuda", "mps", "cpu"],
                        help="auto: CUDA if present, else MLX on Apple Silicon")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override any config field, e.g. --set train.batch_size=256")


def resolve_config(args: argparse.Namespace) -> Config:
    if args.config:
        cfg = Config.load(args.config)
    else:
        cfg = make_config(args.preset)
        if args.run_dir:
            cfg.run_dir = args.run_dir
            cfg.run_name = os.path.basename(os.path.normpath(args.run_dir))
        stored = os.path.join(cfg.run_dir, "config.json")
        if os.path.exists(stored):  # resuming: the run's own config wins
            cfg = Config.load(stored)
    if args.run_dir:
        cfg.run_dir = args.run_dir
        cfg.run_name = os.path.basename(os.path.normpath(args.run_dir))
    if args.backend:
        cfg.backend = args.backend
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        cfg.override(key.strip(), value.strip())
    return cfg


# ---------------------------------------------------------------- commands
def cmd_info(args: argparse.Namespace) -> int:
    from .report import run_report

    cfg = resolve_config(args)
    print(run_report(cfg, do_benchmark=not args.no_bench, quick=not args.full))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from .pipeline import Pipeline
    from .report import model_report, run_report
    from .utils import GracefulKiller

    cfg = resolve_config(args)
    killer = GracefulKiller()
    pipeline = Pipeline(cfg, killer=killer)
    try:
        if not args.quiet:
            print(run_report(cfg, backend=pipeline.backend, do_benchmark=not args.no_bench))
        summary = pipeline.run(args.iterations)
    finally:
        pipeline.close()
        killer.restore()
    print_summary(summary)
    return 0


def cmd_selfplay(args: argparse.Namespace) -> int:
    from .backends import make_backend
    from .checkpoint import CheckpointManager
    from .netspec import NetSpec, init_weights
    from .selfplay import run_selfplay
    from .utils import GracefulKiller, JsonlLogger

    cfg = resolve_config(args)
    paths = cfg.paths().ensure()
    spec = NetSpec.from_config(cfg)
    backend = make_backend(spec, cfg.backend)
    manager = CheckpointManager(paths.checkpoints)
    checkpoint = manager.best() or manager.latest()
    backend.set_weights(checkpoint.weights() if checkpoint else init_weights(spec, cfg.seed))
    logger = JsonlLogger(os.path.join(paths.logs, "selfplay.jsonl"))
    killer = GracefulKiller()
    if args.iteration is None:
        # Default: add --games NEW games to the run's current iteration.
        # (With an explicit --iteration, --games is the top-up target instead,
        # matching how the pipeline resumes an interrupted self-play phase.)
        from .replay import count_games

        state = read_json(paths.state, default={}) or {}
        iteration = int(state.get("iteration", 0))
        target = count_games(paths.replay, iteration=iteration) + args.games
    else:
        iteration, target = args.iteration, args.games
    try:
        stats = run_selfplay(cfg, backend, iteration, target, killer=killer,
                             logger=logger)
    finally:
        logger.close()
        killer.restore()
    print(f"self-play finished: {stats.as_dict()}")
    print(f"data -> {os.path.abspath(paths.replay)}")
    return 0


def cmd_fit(args: argparse.Namespace) -> int:
    from .backends import make_backend
    from .checkpoint import CheckpointManager
    from .netspec import NetSpec, init_weights
    from .replay import ReplayBuffer
    from .train import Trainer
    from .utils import GracefulKiller, JsonlLogger

    cfg = resolve_config(args)
    paths = cfg.paths().ensure()
    spec = NetSpec.from_config(cfg)
    backend = make_backend(spec, cfg.backend, lr=cfg.train.lr,
                           weight_decay=cfg.train.weight_decay)
    manager = CheckpointManager(paths.checkpoints, cfg.train.keep_last_ckpts)
    meta = manager.restore(backend)
    if not meta:
        backend.set_weights(init_weights(spec, cfg.seed))
    buffer = ReplayBuffer(cfg.game.board_size, cfg.game.win_length,
                          cfg.game.allow_overline, cfg.train.replay_max_positions)
    buffer.load_dir(paths.replay)
    if buffer.size == 0:
        print("no self-play data yet -- run `make selfplay` first")
        return 1
    logger = JsonlLogger(os.path.join(paths.logs, "fit.jsonl"))
    killer = GracefulKiller()
    trainer = Trainer(cfg, backend, manager, logger, global_step=int(meta.get("step", 0)))
    try:
        stats = trainer.train(buffer, args.steps, killer=killer)
    finally:
        logger.close()
        killer.restore()
    print(f"trained {stats.steps} steps in {human_time(stats.seconds)}")
    return 0


def cmd_arena(args: argparse.Namespace) -> int:
    from .arena import play_match
    from .backends import make_backend
    from .checkpoint import CheckpointManager, load_weights
    from .netspec import NetSpec, init_weights
    from .utils import GracefulKiller

    cfg = resolve_config(args)
    spec = NetSpec.from_config(cfg)
    manager = CheckpointManager(cfg.paths().checkpoints)

    def weights_for(path: str | None, fallback: str):
        if path:
            return load_weights(path), path
        checkpoint = manager.best() if fallback == "best" else manager.latest()
        if checkpoint is None:
            return init_weights(spec, cfg.seed), f"{fallback} (random init)"
        return checkpoint.weights(), checkpoint.weights_path

    weights_a, name_a = weights_for(args.a, "latest")
    weights_b, name_b = weights_for(args.b, "best")
    backend_a = make_backend(spec, cfg.backend)
    backend_a.set_weights(weights_a)
    backend_b = backend_a.clone_for_inference()
    backend_b.set_weights(weights_b)
    killer = GracefulKiller()
    try:
        result = play_match(cfg, backend_a, backend_b, args.games,
                            args.simulations or cfg.arena.simulations, killer=killer)
    finally:
        killer.restore()
    print(f"A = {name_a}")
    print(f"B = {name_b}")
    print(f"result: {result.as_dict()}  ->  A scores {result.score_a * 100:.1f}%")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from .export import export

    cfg = resolve_config(args)
    kwargs: dict[str, Any] = {"source": args.model, "out_dir": args.out, "name": args.name}
    if args.format == "coreml":
        kwargs["precision"] = args.precision
    path = export(cfg, args.format, **kwargs)
    print(f"exported -> {os.path.abspath(path)}")
    return 0


def cmd_play(args: argparse.Namespace) -> int:
    from .play import play

    cfg = resolve_config(args)
    play(cfg, model_path=args.model, simulations=args.simulations, human=args.color)
    return 0


def cmd_gui(args: argparse.Namespace) -> int:
    cfg = resolve_config(args)
    try:
        from .gui import run_gui
    except ImportError as exc:
        print(exc)
        print("install it with `make install-gui` (or `pip install arcade`)")
        return 1
    run_gui(cfg, model_path=args.model, simulations=args.simulations, human=args.color,
            opening_plies=args.opening_plies, effects=not args.no_effects)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    from .checkpoint import CheckpointManager
    from .replay import dataset_stats
    from .report import model_report
    from .netspec import NetSpec

    cfg = resolve_config(args)
    paths = cfg.paths()
    state = read_json(paths.state, default={}) or {}
    data = dataset_stats(paths.replay) if os.path.isdir(paths.replay) else {}
    manager = CheckpointManager(paths.checkpoints)
    latest, best = manager.latest(), manager.best()
    spec = NetSpec.from_config(cfg)
    model = model_report(spec)

    print(f"run            {os.path.abspath(paths.root)}")
    print(f"iteration      {state.get('iteration', 0)}  (phase: {state.get('phase', '-')})")
    print(f"global step    {state.get('global_step', 0)}")
    print(f"promotions     {state.get('promotions', 0)}")
    print(f"model          {spec.blocks}x{spec.channels}, {model['params']:,} params, "
          f"{model['fp32_mb']:.2f} MB")
    if latest:
        print(f"latest ckpt    {latest.weights_path}  (step {latest.step})")
    if best:
        print(f"best model     {manager.best_path}  (step {best.meta.get('step', 0)}, "
              f"iteration {best.meta.get('iteration', 0)})")
    if data:
        print(f"self-play      {data['games']} games / {data['positions']} positions "
              f"in {data['shards']} shards ({data['bytes'] / 1e6:.1f} MB)")
        print(f"               mean length {data['mean_length']:.1f}, "
              f"black {data['black_win_rate'] * 100:.0f}% / "
              f"white {data['white_win_rate'] * 100:.0f}% / "
              f"draw {data['draw_rate'] * 100:.0f}%")
    if state.get("last_iteration_seconds"):
        print(f"last iteration {human_time(state['last_iteration_seconds'])}")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from .backends import make_backend
    from .netspec import NetSpec
    from .report import benchmark, estimate_schedule

    cfg = resolve_config(args)
    spec = NetSpec.from_config(cfg)
    backend = make_backend(spec, cfg.backend)
    measured = benchmark(backend, cfg, spec, quick=not args.full)
    schedule = estimate_schedule(cfg, measured)
    for key, value in measured.items():
        print(f"{key:<28} {value:,.2f}")
    print()
    for key in ("selfplay_seconds", "train_seconds", "arena_seconds", "iteration_seconds",
                "total_seconds"):
        print(f"{key:<28} {human_time(schedule[key])}")
    return 0


def cmd_backends(args: argparse.Namespace) -> int:
    from .backends import available_backends, select_backend

    info = available_backends()
    for name, present in info.items():
        print(f"{name:<8} {'yes' if present else 'no'}")
    name, device = select_backend("auto")
    print(f"\nauto -> {name} ({device or 'default device'})")
    return 0


def print_summary(summary: dict[str, Any]) -> None:
    print()
    print("-" * 60)
    print(f"run          {summary['run_dir']}")
    print(f"iteration    {summary['iteration']}   step {summary['global_step']}   "
          f"promotions {summary['promotions']}")
    print(f"self-play    {summary['games']} games / {summary['positions']} positions")
    print(f"best model   {summary['best_path']}")
    print("-" * 60)


# ------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omok", description="Omok (gomoku) AlphaZero-style trainer")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("info", help="model size, backend, speed and time estimates")
    add_common(p)
    p.add_argument("--no-bench", action="store_true", help="skip the speed measurement")
    p.add_argument("--full", action="store_true", help="longer, more accurate benchmark")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("train", help="run the full loop (resumes automatically)")
    add_common(p)
    p.add_argument("--iterations", type=int, default=None,
                   help="run this many MORE iterations (default: continue "
                        "to the run's configured total, then stop)")
    p.add_argument("--no-bench", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("selfplay", help="generate self-play games only")
    add_common(p)
    p.add_argument("--games", type=int, default=32,
                   help="how many NEW games to generate")
    p.add_argument("--iteration", type=int, default=None,
                   help="tag games with this iteration and treat --games as "
                        "the top-up target for it (default: current iteration, "
                        "--games added on top of what exists)")
    p.set_defaults(func=cmd_selfplay)

    p = sub.add_parser("fit", help="train on existing self-play data only")
    add_common(p)
    p.add_argument("--steps", type=int, default=500)
    p.set_defaults(func=cmd_fit)

    p = sub.add_parser("arena", help="play two checkpoints against each other")
    add_common(p)
    p.add_argument("--a", default=None, help="weights .npz for side A (default: latest)")
    p.add_argument("--b", default=None, help="weights .npz for side B (default: best)")
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--simulations", type=int, default=None)
    p.set_defaults(func=cmd_arena)

    p = sub.add_parser("export", help="export for the iPhone / Mac game")
    add_common(p)
    p.add_argument("--format", default="coreml", choices=["coreml", "onnx", "npz"])
    p.add_argument("--model", default=None, help="weights .npz (default: the run's best)")
    p.add_argument("--out", default=None, help="output directory")
    p.add_argument("--name", default="OmokNet")
    p.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("play", help="play against the trained model in the terminal")
    add_common(p)
    p.add_argument("--model", default=None)
    p.add_argument("--simulations", type=int, default=None)
    p.add_argument("--color", default="black", choices=["black", "white", "none"])
    p.set_defaults(func=cmd_play)

    p = sub.add_parser("gui", help="play against the trained model in a window (Arcade)")
    add_common(p)
    p.add_argument("--model", default=None, help="weights .npz (default: the run's best)")
    p.add_argument("--simulations", type=int, default=None)
    p.add_argument("--color", default="black", choices=["black", "white", "none", "both"],
                   help="which colour you play ('none' watches the engine play itself)")
    p.add_argument("--opening-plies", type=int, default=2,
                   help="plies the engine plays with some randomness, for variety")
    p.add_argument("--no-effects", action="store_true",
                   help="start with the animations and particles off (toggle with 'f')")
    p.set_defaults(func=cmd_gui)

    p = sub.add_parser("status", help="what is on disk for this run")
    add_common(p)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("bench", help="measure throughput and estimate training time")
    add_common(p)
    p.add_argument("--full", action="store_true")
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("backends", help="show which compute backends are available")
    add_common(p)
    p.set_defaults(func=cmd_backends)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
