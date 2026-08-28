"""Head-to-head evaluation used to gate promotions.

A candidate only becomes the self-play network if it beats the current best by
``promote_winrate``.  Each side searches with its own network, so trees are
built fresh per move rather than reused across a colour change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .board import BLACK, Board
from .config import Config
from .mcts import NetworkEvaluator, Tree, run_search
from .utils import GracefulKiller, Timer


@dataclass
class MatchResult:
    wins_a: int = 0
    wins_b: int = 0
    draws: int = 0
    seconds: float = 0.0
    games_played: int = 0

    @property
    def score_a(self) -> float:
        """Win rate for A with draws counted as half a point."""
        total = self.games_played
        return (self.wins_a + 0.5 * self.draws) / total if total else 0.0

    def as_dict(self) -> dict[str, float]:
        return {"games": self.games_played, "wins_a": self.wins_a, "wins_b": self.wins_b,
                "draws": self.draws, "score_a": round(self.score_a, 4),
                "seconds": round(self.seconds, 1)}


def play_match(cfg: Config, backend_a, backend_b, games: int, simulations: int,
               rng: np.random.Generator | None = None,
               killer: GracefulKiller | None = None,
               parallel: int | None = None,
               temperature_moves: int | None = None) -> MatchResult:
    """Play ``games`` games, alternating who has black."""
    rng = rng or np.random.default_rng(cfg.seed + 4242)
    parallel = parallel or cfg.arena.parallel_games
    temperature_moves = (cfg.arena.temperature_moves if temperature_moves is None
                         else temperature_moves)
    mcts_cfg = _match_config(cfg)
    evaluators = {
        "a": NetworkEvaluator(backend_a, batch_size=max(32, parallel)),
        "b": NetworkEvaluator(backend_b, batch_size=max(32, parallel)),
    }
    result = MatchResult()
    timer = Timer()
    max_moves = cfg.action_size

    for start in range(0, games, parallel):
        if killer is not None and killer.stop:
            break
        batch = list(range(start, min(games, start + parallel)))
        boards = [Board(cfg.game.board_size, cfg.game.win_length, cfg.game.allow_overline)
                  for _ in batch]
        # In even-numbered games A plays black, in odd ones B does.
        black_is_a = [(index % 2 == 0) for index in batch]
        live = list(range(len(batch)))
        while live:
            if killer is not None and killer.stop:
                break
            for side in ("a", "b"):
                group = [i for i in live
                         if not boards[i].over
                         and _side_to_move(boards[i], black_is_a[i]) == side]
                if not group:
                    continue
                trees = [Tree(boards[i].copy(), mcts_cfg) for i in group]
                for tree in trees:
                    tree.root_noise_applied = True  # deterministic evaluation
                run_search(trees, evaluators[side], simulations, rng)
                for i, tree in zip(group, trees):
                    temperature = (mcts_cfg.temperature
                                   if boards[i].move_number < temperature_moves else 0.0)
                    boards[i].play(tree.pick_move(rng, temperature))
            live = [i for i in live if not boards[i].over and boards[i].move_number < max_moves]
        for i, board in enumerate(boards):
            if not board.over and board.move_number < max_moves:
                continue  # interrupted mid-game: do not score it
            result.games_played += 1
            if board.winner == 0:
                result.draws += 1
            elif (board.winner == BLACK) == black_is_a[i]:
                result.wins_a += 1
            else:
                result.wins_b += 1
    result.seconds = timer.elapsed()
    return result


def _side_to_move(board: Board, black_is_a: bool) -> str:
    if board.to_move == BLACK:
        return "a" if black_is_a else "b"
    return "b" if black_is_a else "a"


def _match_config(cfg: Config):
    from dataclasses import replace

    return replace(cfg.mcts, dirichlet_weight=0.0, reuse_tree=False,
                   temperature_moves=cfg.arena.temperature_moves)


def evaluate_candidate(cfg: Config, candidate_backend, best_backend, logger=None,
                       killer: GracefulKiller | None = None) -> tuple[bool, MatchResult]:
    """Returns ``(promote?, result)``."""
    result = play_match(cfg, candidate_backend, best_backend, cfg.arena.games,
                        cfg.arena.simulations, killer=killer)
    promote = result.games_played > 0 and result.score_a >= cfg.arena.promote_winrate
    if logger is not None:
        logger.log("arena", promote=promote, threshold=cfg.arena.promote_winrate,
                   **result.as_dict())
    return promote, result
