"""Self-play game generation.

``parallel_games`` games advance in lockstep so every MCTS iteration produces
one big network batch.  Finished games are handed to a :class:`ShardWriter`
which flushes to disk every couple of games -- crash recovery costs seconds.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .board import BLACK, Board, WHITE
from .config import Config
from .mcts import NetworkEvaluator, Tree, run_search
from .replay import GameRecord, ShardWriter
from .utils import GracefulKiller, Timer


@dataclass
class SelfPlayStats:
    games: int = 0
    moves: int = 0
    black_wins: int = 0
    white_wins: int = 0
    draws: int = 0
    seconds: float = 0.0
    nn_positions: int = 0

    @property
    def moves_per_second(self) -> float:
        return self.moves / self.seconds if self.seconds > 0 else 0.0

    @property
    def games_per_second(self) -> float:
        return self.games / self.seconds if self.seconds > 0 else 0.0

    @property
    def mean_length(self) -> float:
        return self.moves / self.games if self.games else 0.0

    def as_dict(self) -> dict[str, float]:
        return {"games": self.games, "moves": self.moves, "black_wins": self.black_wins,
                "white_wins": self.white_wins, "draws": self.draws,
                "seconds": round(self.seconds, 2),
                "games_per_s": round(self.games_per_second, 3),
                "moves_per_s": round(self.moves_per_second, 2),
                "mean_len": round(self.mean_length, 1)}


@dataclass
class _ActiveGame:
    tree: Tree
    record: GameRecord
    opening_left: int = 0


def new_board(cfg: Config) -> Board:
    return Board(cfg.game.board_size, cfg.game.win_length, cfg.game.allow_overline)


def generate_games(backend, cfg: Config, writer: ShardWriter, num_games: int,
                   iteration: int = 0, rng: np.random.Generator | None = None,
                   killer: GracefulKiller | None = None,
                   on_game=None) -> SelfPlayStats:
    """Play ``num_games`` self-play games, streaming them to ``writer``."""
    rng = rng or np.random.default_rng(cfg.seed + iteration)
    sp = cfg.selfplay
    mcts_cfg = cfg.mcts
    max_moves = sp.max_moves or cfg.action_size
    evaluator = NetworkEvaluator(backend, batch_size=max(32, sp.parallel_games))
    stats = SelfPlayStats()
    timer = Timer()

    def start_game() -> _ActiveGame | None:
        """Returns None when the random opening already finished the game."""
        board = new_board(cfg)
        game = _ActiveGame(Tree(board, mcts_cfg), GameRecord(iteration=iteration),
                           opening_left=sp.random_opening_plies)
        _play_opening(game, rng, cfg)
        if game.tree.board.over:  # decided before a single searched move
            _finalise(game, stats)
            writer.add(game.record)
            stats.seconds = timer.elapsed()
            if on_game is not None:
                on_game(game.record, stats)
            return None
        return game

    active: list[_ActiveGame] = []
    started = 0
    slots = min(sp.parallel_games, num_games)
    while started < num_games and len(active) < slots:
        game = start_game()
        started += 1
        if game is not None:
            active.append(game)

    while active:
        if killer is not None and killer.stop:
            break
        # Positions with a one-ply forced answer (take the win / block the
        # opponent's) need no search: the move is played outright and recorded
        # as a one-hot target, which teaches the net the tactic directly.
        forced = {id(g): g.tree.board.forced_move() for g in active}
        run_search([g.tree for g in active if forced[id(g)] is None],
                   evaluator, mcts_cfg.simulations, rng)
        finished: list[_ActiveGame] = []
        for game in active:
            tree = game.tree
            move = forced[id(game)]
            if move is not None:
                probs = np.zeros(cfg.action_size, dtype=np.float32)
                probs[move] = 1.0
            else:
                probs = tree.visit_distribution()
                temperature = (mcts_cfg.temperature
                               if tree.board.move_number < mcts_cfg.temperature_moves else 0.0)
                move = tree.pick_move(rng, temperature)
            game.record.add(move, probs, trainable=True)
            tree.advance(move)
            stats.moves += 1
            if tree.board.over or tree.board.move_number >= max_moves:
                finished.append(game)
        for game in finished:
            active.remove(game)
            _finalise(game, stats)
            writer.add(game.record)
            stats.seconds = timer.elapsed()
            if on_game is not None:
                on_game(game.record, stats)
            while started < num_games and len(active) < slots \
                    and not (killer is not None and killer.stop):
                replacement = start_game()
                started += 1
                if replacement is not None:
                    active.append(replacement)
                    break  # one slot freed, one replacement seated

    writer.flush()
    stats.seconds = timer.elapsed()
    stats.nn_positions = evaluator.positions
    return stats


def _play_opening(game: _ActiveGame, rng: np.random.Generator, cfg: Config) -> None:
    """Optional uniformly random opening plies, excluded from training targets."""
    action_size = cfg.action_size
    for _ in range(game.opening_left):
        board = game.tree.board
        if board.over:
            break
        legal = np.nonzero(board.legal_mask())[0]
        if len(legal) == 0:
            break
        move = int(rng.choice(legal))
        onehot = np.zeros(action_size, dtype=np.float32)
        onehot[move] = 1.0
        game.record.add(move, onehot, trainable=False)
        game.tree.advance(move)


def _finalise(game: _ActiveGame, stats: SelfPlayStats) -> None:
    board = game.tree.board
    game.record.winner = board.winner if board.over else 0
    stats.games += 1
    if game.record.winner == BLACK:
        stats.black_wins += 1
    elif game.record.winner == WHITE:
        stats.white_wins += 1
    else:
        stats.draws += 1


def run_selfplay(cfg: Config, backend, iteration: int, target_games: int,
                 killer: GracefulKiller | None = None, logger=None,
                 rng: np.random.Generator | None = None) -> SelfPlayStats:
    """Top-level self-play phase: resumes by counting what is already on disk."""
    from .replay import count_games

    paths = cfg.paths().ensure()
    have = count_games(paths.replay, iteration=iteration)
    todo = max(0, target_games - have)
    if logger is not None:
        logger.log("selfplay.start", iteration=iteration, have=have, todo=todo,
                   sims=cfg.mcts.simulations, parallel=cfg.selfplay.parallel_games)
    if todo == 0:
        return SelfPlayStats()
    writer = ShardWriter(paths.replay, cfg.selfplay.flush_every_games,
                         cfg.selfplay.flush_every_seconds)
    last_report = [time.time()]

    def report(record: GameRecord, stats: SelfPlayStats) -> None:
        if logger is not None and time.time() - last_report[0] > 20.0:
            last_report[0] = time.time()
            logger.log("selfplay.progress", iteration=iteration,
                       games=have + stats.games, target=target_games,
                       games_per_s=round(stats.games_per_second, 3),
                       moves_per_s=round(stats.moves_per_second, 1))

    try:
        stats = generate_games(backend, cfg, writer, todo, iteration=iteration,
                               rng=rng, killer=killer, on_game=report)
    finally:
        writer.close()
    if logger is not None:
        logger.log("selfplay.done", iteration=iteration, **stats.as_dict())
    return stats
