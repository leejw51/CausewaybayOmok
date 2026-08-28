"""Play against the trained network in the terminal (a quick sanity check)."""

from __future__ import annotations

import numpy as np

from .backends import make_backend
from .board import BLACK, Board, WHITE, format_move, parse_move
from .checkpoint import CheckpointManager, load_weights
from .config import Config
from .mcts import NetworkEvaluator, search_position
from .netspec import NetSpec, init_weights


def load_play_backend(cfg: Config, model_path: str | None = None):
    spec = NetSpec.from_config(cfg)
    manager = CheckpointManager(cfg.paths().checkpoints)
    weights = None
    source = "random init"
    if model_path:
        weights, source = load_weights(model_path), model_path
    else:
        checkpoint = manager.best() or manager.latest()
        if checkpoint is not None:
            if checkpoint.meta.get("spec"):
                spec = NetSpec.from_dict(checkpoint.meta["spec"])
            weights, source = checkpoint.weights(), checkpoint.weights_path
    backend = make_backend(spec, cfg.backend)
    backend.set_weights(weights if weights is not None else init_weights(spec, cfg.seed))
    return backend, spec, source


def analyse(cfg: Config, backend, board: Board, simulations: int,
            rng: np.random.Generator, top: int = 5):
    evaluator = NetworkEvaluator(backend, batch_size=64)
    tree = search_position(board, evaluator, cfg.mcts, simulations, rng)
    probs = tree.visit_distribution()
    order = np.argsort(-probs)[:top]
    lines = [f"{format_move(int(i), board.size)} {probs[i] * 100:5.1f}%" for i in order
             if probs[i] > 0]
    return tree, lines


def play(cfg: Config, model_path: str | None = None, simulations: int | None = None,
         human: str = "black", show_analysis: bool = True) -> None:
    backend, spec, source = load_play_backend(cfg, model_path)
    simulations = simulations or cfg.mcts.simulations
    rng = np.random.default_rng()
    board = Board(cfg.game.board_size, cfg.game.win_length, cfg.game.allow_overline)
    human_colour = {"black": BLACK, "white": WHITE, "none": 0}[human.lower()]

    print(f"model: {source}")
    print(f"{spec.blocks}x{spec.channels} net, {spec.parameter_count():,} parameters, "
          f"{simulations} simulations/move")
    print("enter moves like 'h7' (column letter + row number), or 'quit'\n")

    while not board.over:
        print(board.to_ascii(highlight=board.moves[-1] if board.moves else None))
        side = "black (X)" if board.to_move == BLACK else "white (O)"
        if board.to_move == human_colour:
            try:
                text = input(f"your move as {side}: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nbye")
                return
            if text.lower() in ("quit", "exit", "q"):
                return
            try:
                move = parse_move(text, board.size)
            except ValueError as exc:
                print(f"  ? {exc}")
                continue
            if not board.is_legal(move):
                print("  ? illegal move")
                continue
        else:
            tree, lines = analyse(cfg, backend, board, simulations, rng)
            move = tree.pick_move(rng, 0.0)
            value = tree.root_value()
            if show_analysis:
                print(f"  engine ({side}) eval {value:+.2f}   " + "  ".join(lines))
        board.play(move)
        print(f"  -> {format_move(move, board.size)}\n")

    print(board.to_ascii(highlight=board.moves[-1] if board.moves else None))
    if board.winner == 0:
        print("draw")
    else:
        print(f"{'black (X)' if board.winner == BLACK else 'white (O)'} wins "
              f"in {board.move_number} moves")
