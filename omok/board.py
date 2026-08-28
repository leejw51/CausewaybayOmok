"""Omok / gomoku rules.

Deliberately dependency-light and fast: the board is a ``bytearray`` because
MCTS copies and mutates positions millions of times and per-element numpy
access is an order of magnitude slower than plain Python indexing.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

EMPTY = 0
BLACK = 1
WHITE = 2

DRAW = 0  # value used in `winner` when the board filled up

# (dr, dc) for the four axes a line can run along.
DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


class Board:
    """A gomoku position.  ``to_move`` is BLACK (1) or WHITE (2)."""

    __slots__ = ("size", "win_length", "allow_overline", "cells", "to_move",
                 "moves", "winner", "over")

    def __init__(self, size: int = 15, win_length: int = 5,
                 allow_overline: bool = True) -> None:
        self.size = size
        self.win_length = win_length
        self.allow_overline = allow_overline
        self.cells = bytearray(size * size)
        self.to_move = BLACK
        self.moves: list[int] = []
        self.winner = DRAW
        self.over = False

    # ------------------------------------------------------------ basics
    @property
    def action_size(self) -> int:
        return self.size * self.size

    @property
    def move_number(self) -> int:
        return len(self.moves)

    def copy(self) -> "Board":
        other = Board.__new__(Board)
        other.size = self.size
        other.win_length = self.win_length
        other.allow_overline = self.allow_overline
        other.cells = bytearray(self.cells)
        other.to_move = self.to_move
        other.moves = list(self.moves)
        other.winner = self.winner
        other.over = self.over
        return other

    def index(self, row: int, col: int) -> int:
        return row * self.size + col

    def coords(self, index: int) -> tuple[int, int]:
        return divmod(index, self.size)

    # ------------------------------------------------------------ moves
    def legal_moves(self) -> list[int]:
        if self.over:
            return []
        cells = self.cells
        return [i for i in range(len(cells)) if cells[i] == EMPTY]

    def legal_mask(self) -> np.ndarray:
        mask = np.frombuffer(bytes(self.cells), dtype=np.uint8) == EMPTY
        if self.over:
            mask = np.zeros_like(mask)
        return mask

    def is_legal(self, index: int) -> bool:
        return (not self.over) and 0 <= index < len(self.cells) and self.cells[index] == EMPTY

    def play(self, index: int) -> None:
        if self.over:
            raise ValueError("game already finished")
        if not (0 <= index < len(self.cells)) or self.cells[index] != EMPTY:
            raise ValueError(f"illegal move: {index}")
        player = self.to_move
        self.cells[index] = player
        self.moves.append(index)
        if self.wins_at(index, player):
            self.winner = player
            self.over = True
        elif len(self.moves) == len(self.cells):
            self.winner = DRAW
            self.over = True
        # The side to move always flips, terminal or not: search code relies on
        # `result_for(to_move)` being the value of a finished position from the
        # perspective of the player who would have moved next.
        self.to_move = WHITE if player == BLACK else BLACK

    def play_many(self, indices: Iterable[int]) -> "Board":
        for index in indices:
            self.play(index)
        return self

    # ------------------------------------------------------------ win test
    def wins_at(self, index: int, player: int) -> bool:
        """Does the stone just placed at ``index`` complete a winning line?"""
        size = self.size
        cells = self.cells
        need = self.win_length
        row, col = divmod(index, size)
        for dr, dc in DIRECTIONS:
            count = 1
            # forward
            r, c = row + dr, col + dc
            while 0 <= r < size and 0 <= c < size and cells[r * size + c] == player:
                count += 1
                r += dr
                c += dc
            # backward
            r, c = row - dr, col - dc
            while 0 <= r < size and 0 <= c < size and cells[r * size + c] == player:
                count += 1
                r -= dr
                c -= dc
            if count == need or (count > need and self.allow_overline):
                return True
        return False

    def winning_squares(self, player: int, candidates: Iterable[int] | None = None) -> list[int]:
        """Empty squares where ``player`` would complete a winning line now.

        ``wins_at`` never reads the square being tested, so the stone does not
        need to be placed first.
        """
        if self.over:
            return []
        cells = self.cells
        if candidates is None:
            candidates = range(len(cells))
        return [i for i in candidates
                if cells[i] == EMPTY and self.wins_at(i, player)]

    def forced_move(self) -> int | None:
        """One-ply tactics: take an immediate win, else block the opponent's.

        Search at casual simulation budgets can miss both, and missing either
        one is what makes an engine look broken to a human.
        """
        if self.over:
            return None
        # A five-completing square always touches the stone next to the gap,
        # so scanning the radius-2 neighbourhood of the stones is exhaustive.
        candidates = np.nonzero(self.neighbourhood(2))[0]
        mine = self.winning_squares(self.to_move, candidates)
        if mine:
            return mine[0]
        opponent = WHITE if self.to_move == BLACK else BLACK
        theirs = self.winning_squares(opponent, candidates)
        if theirs:
            return theirs[0]  # several at once is lost anyway; block one
        return None

    def result_for(self, player: int) -> float:
        """+1 win / -1 loss / 0 draw or unfinished, from ``player``'s view."""
        if not self.over or self.winner == DRAW:
            return 0.0
        return 1.0 if self.winner == player else -1.0

    # ------------------------------------------------------------ helpers
    def neighbourhood(self, radius: int = 2) -> np.ndarray:
        """Mask of empty points within ``radius`` of an existing stone.

        Handy as a cheap move filter for fast play-outs and shallow search.
        Falls back to the centre point on an empty board.
        """
        size = self.size
        occupied = np.frombuffer(bytes(self.cells), dtype=np.uint8).reshape(size, size) != EMPTY
        if not occupied.any():
            mask = np.zeros((size, size), dtype=bool)
            mask[size // 2, size // 2] = True
            return mask.reshape(-1)
        near = np.zeros((size, size), dtype=bool)
        rows, cols = np.nonzero(occupied)
        for r, c in zip(rows.tolist(), cols.tolist()):
            r0, r1 = max(0, r - radius), min(size, r + radius + 1)
            c0, c1 = max(0, c - radius), min(size, c + radius + 1)
            near[r0:r1, c0:c1] = True
        return (near & ~occupied).reshape(-1)

    def to_ascii(self, highlight: int | None = None) -> str:
        size = self.size
        glyphs = {EMPTY: ".", BLACK: "X", WHITE: "O"}
        header = "    " + " ".join(f"{c % 10}" for c in range(size))
        lines = [header]
        for r in range(size):
            row = []
            for c in range(size):
                i = r * size + c
                ch = glyphs[self.cells[i]]
                if highlight is not None and i == highlight and ch != ".":
                    ch = ch.lower()
                row.append(ch)
            lines.append(f"{r:>3} " + " ".join(row))
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (f"Board(size={self.size}, moves={len(self.moves)}, "
                f"to_move={'X' if self.to_move == BLACK else 'O'}, over={self.over})")


def board_from_moves(moves: Iterable[int], size: int = 15, win_length: int = 5,
                     allow_overline: bool = True) -> Board:
    return Board(size, win_length, allow_overline).play_many(moves)


def parse_move(text: str, size: int) -> int:
    """Accept ``h8``/``H8`` or ``7,7`` or a raw index."""
    text = text.strip().lower().replace(" ", "")
    if not text:
        raise ValueError("empty move")
    if "," in text:
        row, col = text.split(",", 1)
        return int(row) * size + int(col)
    if text[0].isalpha():
        col = ord(text[0]) - ord("a")
        row = int(text[1:])
        return row * size + col
    value = int(text)
    if value >= size * size:
        raise ValueError("index out of range")
    return value


def format_move(index: int, size: int) -> str:
    row, col = divmod(index, size)
    return f"{chr(ord('a') + col)}{row}"
