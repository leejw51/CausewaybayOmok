"""Board -> neural-network input planes, plus the 8-fold dihedral symmetry.

Plane layout (all NCHW, float32, always from the side-to-move's point of view):

    0  stones of the player to move
    1  stones of the opponent
    2  one-hot of the opponent's last move
    3  one-hot of the player-to-move's previous move
    4  all ones when the player to move is black

Keeping the encoding perspective-relative means the network never needs a
"whose turn is it" branch and self-play data can be re-used for both colours.
"""

from __future__ import annotations

import numpy as np

from .board import BLACK, Board, WHITE

NUM_PLANES = 5


def encode_board(board: Board, out: np.ndarray | None = None) -> np.ndarray:
    size = board.size
    if out is None:
        out = np.zeros((NUM_PLANES, size, size), dtype=np.float32)
    else:
        out.fill(0.0)
    cells = np.frombuffer(bytes(board.cells), dtype=np.uint8).reshape(size, size)
    me = board.to_move
    opponent = WHITE if me == BLACK else BLACK
    out[0] = (cells == me)
    out[1] = (cells == opponent)
    moves = board.moves
    if moves:
        r, c = divmod(moves[-1], size)
        out[2, r, c] = 1.0
    if len(moves) > 1:
        r, c = divmod(moves[-2], size)
        out[3, r, c] = 1.0
    if me == BLACK:
        out[4] = 1.0
    return out


def encode_batch(boards, out: np.ndarray | None = None) -> np.ndarray:
    boards = list(boards)
    size = boards[0].size if boards else 0
    if out is None or out.shape[0] < len(boards):
        out = np.zeros((len(boards), NUM_PLANES, size, size), dtype=np.float32)
    for i, board in enumerate(boards):
        encode_board(board, out[i])
    return out[: len(boards)]


# ------------------------------------------------------------- symmetries
def transform_planes(planes: np.ndarray, k: int) -> np.ndarray:
    """Apply dihedral transform ``k`` in [0, 8) to a (..., H, W) array."""
    if k == 0:
        return planes
    rot = k % 4
    out = np.rot90(planes, rot, axes=(-2, -1)) if rot else planes
    if k >= 4:
        out = out[..., ::-1]
    return np.ascontiguousarray(out)


def transform_policy(policy: np.ndarray, k: int, size: int) -> np.ndarray:
    """Apply the same transform to a flat (..., size*size) policy vector."""
    if k == 0:
        return policy
    shaped = policy.reshape(*policy.shape[:-1], size, size)
    return transform_planes(shaped, k).reshape(*policy.shape[:-1], size * size)


def inverse_index(k: int) -> int:
    """The transform that undoes ``k`` (reflections are their own inverse)."""
    if k >= 4:
        return k
    return (4 - k) % 4


def all_symmetries(planes: np.ndarray, policy: np.ndarray, size: int):
    """Yield the 8 (planes, policy) variants used for training augmentation."""
    for k in range(8):
        yield transform_planes(planes, k), transform_policy(policy, k, size)
