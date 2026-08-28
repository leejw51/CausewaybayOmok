"""Self-play data on disk, and the in-memory replay buffer built from it.

Games -- not encoded tensors -- are what get stored: a game is just its move
list plus one MCTS policy vector per ply, which is ~20x smaller than storing
input planes and lets the plane encoding change without invalidating the data.
Shards are written atomically every few games so a crash loses seconds of
self-play, never the run.
"""

from __future__ import annotations

import glob
import os
import re
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .board import BLACK, WHITE
from .encode import NUM_PLANES, transform_planes, transform_policy
from .utils import atomic_write_with, log


@dataclass
class GameRecord:
    """One finished self-play game."""

    moves: list[int] = field(default_factory=list)
    policies: list[np.ndarray] = field(default_factory=list)  # float32 (A,) per ply
    trainable: list[bool] = field(default_factory=list)
    winner: int = 0  # 0 draw, 1 black, 2 white
    iteration: int = 0

    def add(self, move: int, policy: np.ndarray, trainable: bool = True) -> None:
        self.moves.append(int(move))
        self.policies.append(np.asarray(policy, dtype=np.float32))
        self.trainable.append(bool(trainable))

    def __len__(self) -> int:
        return len(self.moves)


# ----------------------------------------------------------------- writing
class ShardWriter:
    """Buffers finished games and flushes them to ``.npz`` shards."""

    def __init__(self, directory: str, flush_every_games: int = 2,
                 flush_every_seconds: float = 60.0) -> None:
        self.dir = directory
        os.makedirs(self.dir, exist_ok=True)
        self.flush_every_games = max(1, flush_every_games)
        self.flush_every_seconds = flush_every_seconds
        self._pending: list[GameRecord] = []
        self._last_flush = time.time()
        self._seq = _next_shard_seq(self.dir)

    def add(self, game: GameRecord) -> bool:
        self._pending.append(game)
        due = (len(self._pending) >= self.flush_every_games
               or time.time() - self._last_flush >= self.flush_every_seconds)
        return self.flush() if due else False

    def flush(self) -> bool:
        if not self._pending:
            return False
        path = os.path.join(self.dir, f"shard-{self._seq:08d}.npz")
        # Another process may share this directory (a second `make train-bg`,
        # or `make selfplay` next to it): skip past any shard that appeared
        # since we chose our number instead of silently clobbering it.
        while os.path.exists(path):
            self._seq += 1
            path = os.path.join(self.dir, f"shard-{self._seq:08d}.npz")
        write_shard(path, self._pending)
        self._seq += 1
        self._pending.clear()
        self._last_flush = time.time()
        return True

    def close(self) -> None:
        self.flush()


def _next_shard_seq(directory: str) -> int:
    best = -1
    for path in glob.glob(os.path.join(directory, "shard-*.npz")):
        match = re.search(r"shard-(\d+)\.npz$", os.path.basename(path))
        if match:
            best = max(best, int(match.group(1)))
    return best + 1


def write_shard(path: str, games: Sequence[GameRecord]) -> None:
    action_size = len(games[0].policies[0])
    offsets = np.zeros(len(games) + 1, dtype=np.int64)
    for i, game in enumerate(games):
        offsets[i + 1] = offsets[i] + len(game)
    total = int(offsets[-1])
    moves = np.zeros(total, dtype=np.int16)
    policies = np.zeros((total, action_size), dtype=np.float16)
    trainable = np.zeros(total, dtype=np.uint8)
    winners = np.zeros(len(games), dtype=np.uint8)
    iterations = np.zeros(len(games), dtype=np.int32)
    for i, game in enumerate(games):
        start = int(offsets[i])
        moves[start:start + len(game)] = np.asarray(game.moves, dtype=np.int16)
        policies[start:start + len(game)] = np.asarray(game.policies, dtype=np.float16)
        trainable[start:start + len(game)] = np.asarray(game.trainable, dtype=np.uint8)
        winners[i] = game.winner
        iterations[i] = game.iteration
    payload = dict(moves=moves, offsets=offsets, policies=policies,
                   trainable=trainable, winners=winners, iterations=iterations)
    atomic_write_with(path, lambda tmp: np.savez_compressed(tmp, **payload), suffix=".npz")


def shard_paths(directory: str) -> list[str]:
    return sorted(glob.glob(os.path.join(directory, "shard-*.npz")))


def shard_positions(path: str) -> int:
    try:
        with np.load(path) as data:
            return int(data["offsets"][-1])
    except Exception:
        return 0


def read_shard(path: str) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def count_games(directory: str, iteration: int | None = None) -> int:
    """How many games are on disk (optionally only for one iteration).

    This is what makes self-play resumable: on restart we simply count what is
    already there and generate the remainder.
    """
    total = 0
    for path in shard_paths(directory):
        try:
            with np.load(path) as data:
                iterations = data["iterations"]
        except Exception:
            log(f"warning: skipping unreadable shard {os.path.basename(path)}")
            continue
        total += int(len(iterations) if iteration is None else (iterations == iteration).sum())
    return total


# ----------------------------------------------------------------- reading
class ReplayBuffer:
    """Positions expanded from shards, ready for vectorised encoding.

    Positions are stored as raw board cells rather than input planes: 225 bytes
    per position instead of 1125, and the encoding stays changeable.
    """

    def __init__(self, board_size: int = 15, win_length: int = 5,
                 allow_overline: bool = True, max_positions: int = 400_000) -> None:
        self.board_size = board_size
        self.win_length = win_length
        self.allow_overline = allow_overline
        self.max_positions = max_positions
        self.action_size = board_size * board_size
        self._chunks: list[dict[str, np.ndarray]] = []  # one per loaded shard
        self._loaded: list[str] = []
        self._cache: dict[str, np.ndarray] | None = None

    # -- construction ------------------------------------------------------
    def load_dir(self, directory: str) -> "ReplayBuffer":
        """Load the most recent shards that fit in ``max_positions``."""
        chosen: list[str] = []
        budget = 0
        for path in reversed(shard_paths(directory)):
            if budget >= self.max_positions:
                break
            budget += shard_positions(path)
            chosen.append(path)
        for path in reversed(chosen):  # keep chunks in chronological order
            self.add_shard(path)
        return self

    def add_shard(self, path: str) -> int:
        if path in self._loaded:
            return 0
        try:
            shard = read_shard(path)
        except Exception:
            log(f"warning: skipping unreadable shard {os.path.basename(path)}")
            return 0
        chunk = self._expand(shard)
        if chunk is None:
            return 0
        self._chunks.append(chunk)
        self._loaded.append(path)
        self._cache = None
        self._evict()
        return len(chunk["z"])

    def _expand(self, shard: dict[str, np.ndarray]) -> dict[str, np.ndarray] | None:
        moves = shard["moves"].astype(np.int32)
        offsets = shard["offsets"].astype(np.int64)
        policies = shard["policies"]
        trainable = shard["trainable"].astype(bool)
        winners = shard["winners"].astype(np.int32)
        total = int(offsets[-1])
        if total == 0:
            return None
        size = self.board_size
        cells = np.zeros((total, self.action_size), dtype=np.uint8)
        to_move = np.zeros(total, dtype=np.uint8)
        last1 = np.full(total, -1, dtype=np.int16)
        last2 = np.full(total, -1, dtype=np.int16)
        z = np.zeros(total, dtype=np.float32)
        for g in range(len(winners)):
            start, end = int(offsets[g]), int(offsets[g + 1])
            board = np.zeros(self.action_size, dtype=np.uint8)
            player = BLACK
            for t in range(start, end):
                cells[t] = board
                to_move[t] = player
                if t > start:
                    last1[t] = moves[t - 1]
                if t > start + 1:
                    last2[t] = moves[t - 2]
                winner = winners[g]
                z[t] = 0.0 if winner == 0 else (1.0 if winner == player else -1.0)
                board[moves[t]] = player
                player = WHITE if player == BLACK else BLACK
        return {"cells": cells, "to_move": to_move, "last1": last1, "last2": last2,
                "pi": policies.astype(np.float16), "z": z,
                "trainable": trainable}

    def _evict(self) -> None:
        while self.size > self.max_positions and len(self._chunks) > 1:
            self._chunks.pop(0)  # chunks are chronological; drop the oldest
            self._loaded.pop(0)
            self._cache = None

    # -- properties --------------------------------------------------------
    @property
    def size(self) -> int:
        return sum(len(c["z"]) for c in self._chunks)

    @property
    def trainable_size(self) -> int:
        return int(sum(c["trainable"].sum() for c in self._chunks))

    def merged(self) -> dict[str, np.ndarray]:
        if self._cache is None:
            if not self._chunks:
                raise ValueError("replay buffer is empty")
            keys = self._chunks[0].keys()
            self._cache = {k: np.concatenate([c[k] for c in self._chunks]) for k in keys}
        return self._cache

    # -- sampling ----------------------------------------------------------
    def sample(self, batch_size: int, rng: np.random.Generator,
               augment: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        data = self.merged()
        usable = np.nonzero(data["trainable"])[0]
        if len(usable) == 0:
            raise ValueError("no trainable positions in replay buffer")
        idx = usable[rng.integers(0, len(usable), size=batch_size)]
        planes = self.encode_indices(data, idx)
        pi = data["pi"][idx].astype(np.float32)
        z = data["z"][idx].astype(np.float32)
        if augment:
            planes, pi = augment_batch(planes, pi, self.board_size, rng)
        return planes, pi, z

    def encode_indices(self, data: dict[str, np.ndarray], idx: np.ndarray) -> np.ndarray:
        size = self.board_size
        cells = data["cells"][idx]
        me = data["to_move"][idx][:, None]
        opponent = np.where(me == BLACK, WHITE, BLACK)
        batch = len(idx)
        planes = np.zeros((batch, NUM_PLANES, self.action_size), dtype=np.float32)
        planes[:, 0] = (cells == me)
        planes[:, 1] = (cells == opponent)
        rows = np.arange(batch)
        last1 = data["last1"][idx]
        has1 = last1 >= 0
        planes[rows[has1], 2, last1[has1]] = 1.0
        last2 = data["last2"][idx]
        has2 = last2 >= 0
        planes[rows[has2], 3, last2[has2]] = 1.0
        planes[:, 4] = (me == BLACK)
        return planes.reshape(batch, NUM_PLANES, size, size)


def augment_batch(planes: np.ndarray, pi: np.ndarray, board_size: int,
                  rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Apply an independent random dihedral symmetry to each sample."""
    ks = rng.integers(0, 8, size=len(planes))
    out_planes = planes.copy()
    out_pi = pi.copy()
    for k in range(1, 8):
        sel = np.nonzero(ks == k)[0]
        if len(sel) == 0:
            continue
        out_planes[sel] = transform_planes(planes[sel], k)
        out_pi[sel] = transform_policy(pi[sel], k, board_size)
    return out_planes, out_pi


def dataset_stats(directory: str) -> dict[str, float]:
    games = 0
    positions = 0
    black_wins = white_wins = draws = 0
    lengths: list[int] = []
    for path in shard_paths(directory):
        try:
            with np.load(path) as data:
                offsets = data["offsets"]
                winners = data["winners"]
        except Exception:
            continue
        games += len(winners)
        positions += int(offsets[-1])
        lengths.extend(np.diff(offsets).tolist())
        black_wins += int((winners == BLACK).sum())
        white_wins += int((winners == WHITE).sum())
        draws += int((winners == 0).sum())
    return {
        "games": games,
        "positions": positions,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "black_win_rate": black_wins / games if games else 0.0,
        "white_win_rate": white_wins / games if games else 0.0,
        "draw_rate": draws / games if games else 0.0,
        "shards": len(shard_paths(directory)),
        "bytes": int(sum(os.path.getsize(p) for p in shard_paths(directory))),
    }
