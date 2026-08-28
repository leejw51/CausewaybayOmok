import numpy as np

from omok.board import BLACK, Board, WHITE
from omok.encode import (NUM_PLANES, encode_batch, encode_board, inverse_index,
                         transform_planes, transform_policy)


def test_planes_are_from_the_movers_point_of_view():
    b = Board(9, 5)
    b.play(40)   # black
    b.play(0)    # white
    planes = encode_board(b)          # black to move again
    assert planes.shape == (NUM_PLANES, 9, 9)
    assert planes[0, 4, 4] == 1.0     # our stone
    assert planes[1, 0, 0] == 1.0     # their stone
    assert planes[2, 0, 0] == 1.0     # their last move
    assert planes[3, 4, 4] == 1.0     # our previous move
    assert planes[4].all()            # black to move

    b.play(41)
    planes = encode_board(b)          # white to move
    assert planes[0, 0, 0] == 1.0     # white's own stone is now plane 0
    assert planes[1, 4, 4] == 1.0
    assert not planes[4].any()


def test_encode_batch_matches_single():
    boards = []
    b = Board(9, 5)
    for move in (40, 0, 41, 1, 42):
        b = b.copy()
        b.play(move)
        boards.append(b.copy())
    batch = encode_batch(boards)
    for i, board in enumerate(boards):
        assert np.array_equal(batch[i], encode_board(board))


def test_symmetry_moves_planes_and_policy_together():
    rng = np.random.default_rng(0)
    planes = rng.random((NUM_PLANES, 9, 9), dtype=np.float32)
    policy = np.zeros(81, dtype=np.float32)
    policy[3 * 9 + 1] = 1.0
    for k in range(8):
        tp = transform_planes(planes, k)
        tpi = transform_policy(policy, k, 9)
        # the marked point must land where the same transform sends plane 0
        marker = np.zeros((9, 9), dtype=np.float32)
        marker[3, 1] = 1.0
        assert np.array_equal(transform_planes(marker, k).reshape(-1), tpi)
        assert tp.shape == planes.shape


def test_symmetry_inverse_restores_original():
    rng = np.random.default_rng(1)
    planes = rng.random((NUM_PLANES, 9, 9), dtype=np.float32)
    for k in range(8):
        restored = transform_planes(transform_planes(planes, k), inverse_index(k))
        assert np.allclose(restored, planes)
