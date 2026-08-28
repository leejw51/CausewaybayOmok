import numpy as np
import pytest

from omok.board import BLACK, Board, WHITE, format_move, parse_move


# Replies scattered so the opponent never forms a line of its own.
FILLERS = (8, 26, 44, 62, 80)


def line(board, start, step, count):
    """Play ``count`` stones of one colour on a line, opponent stones elsewhere."""
    for i in range(count):
        board.play(start + i * step)
        if not board.over:
            board.play(FILLERS[i])


def test_horizontal_win():
    b = Board(9, 5)
    line(b, 0, 1, 5)
    assert b.over and b.winner == BLACK


def test_vertical_and_diagonal_win():
    b = Board(9, 5)
    line(b, 0, 9, 5)
    assert b.winner == BLACK

    b = Board(9, 5)
    line(b, 0, 10, 5)
    assert b.winner == BLACK

    b = Board(9, 5)
    line(b, 4, 8, 5)
    assert b.winner == BLACK


def test_four_is_not_a_win():
    b = Board(9, 5)
    line(b, 0, 1, 4)
    assert not b.over


def test_overline_rules():
    freestyle = Board(11, 5, allow_overline=True)
    line(freestyle, 0, 1, 5)
    assert freestyle.winner == BLACK

    strict = Board(11, 5, allow_overline=False)
    # Build six in a row by filling the ends last so no five ever stands alone.
    for index in [1, 2, 3, 4]:
        strict.play(index)
        strict.play(100 - index)
    strict.play(5)          # five in a row from 1..5 -> would win in freestyle
    assert strict.over and strict.winner == BLACK

    strict = Board(11, 5, allow_overline=False)
    for index in [1, 2, 4, 5]:
        strict.play(index)
        strict.play(100 - index)
    strict.play(0)          # 0,1,2 _ 4,5 : no five yet
    assert not strict.over
    strict.play(90)
    strict.play(3)          # closes 0..5 -> six in a row, not a win under strict rules
    assert not strict.over


def test_to_move_flips_even_on_terminal_positions():
    b = Board(9, 5)
    line(b, 0, 1, 5)
    assert b.winner == BLACK
    assert b.to_move == WHITE
    assert b.result_for(b.to_move) == -1.0


def test_draw_fills_board():
    b = Board(3, 5)  # no line of five fits on a 3x3 board
    for i in range(9):
        b.play(i)
    assert b.over and b.winner == 0
    assert b.legal_moves() == []


def test_legal_mask_and_copy_independence():
    b = Board(9, 5)
    b.play(40)
    mask = b.legal_mask()
    assert mask.sum() == 80 and not mask[40]
    clone = b.copy()
    clone.play(0)
    assert b.move_number == 1 and clone.move_number == 2


def test_illegal_moves_raise():
    b = Board(9, 5)
    b.play(0)
    with pytest.raises(ValueError):
        b.play(0)
    with pytest.raises(ValueError):
        b.play(999)


def test_move_parsing_roundtrip():
    for index in (0, 17, 80):
        assert parse_move(format_move(index, 9), 9) == index
    assert parse_move("4,4", 9) == 40
    assert parse_move("40", 9) == 40


def test_neighbourhood_mask():
    b = Board(9, 5)
    assert b.neighbourhood().sum() == 1  # empty board -> centre only
    b.play(40)
    near = b.neighbourhood(radius=1)
    assert near.sum() == 8 and not near[40]


def test_winning_squares_and_forced_move():
    b = Board(9, 5)
    # black: 4 in a row on row 4, cols 1-4; white: scattered on row 0
    b.play_many([b.index(4, 1), b.index(0, 0), b.index(4, 2), b.index(0, 1),
                 b.index(4, 3), b.index(0, 2), b.index(4, 4), b.index(0, 3)])
    # black to move: completing squares on either end
    assert sorted(b.winning_squares(BLACK)) == [b.index(4, 0), b.index(4, 5)]
    assert b.forced_move() in (b.index(4, 0), b.index(4, 5))
    # white to move against the same four must block one of the ends
    b2 = Board(9, 5)
    b2.play_many([b2.index(4, 1), b2.index(0, 0), b2.index(4, 2), b2.index(0, 1),
                  b2.index(4, 3), b2.index(0, 2), b2.index(4, 4)])
    assert b2.to_move == WHITE
    assert b2.winning_squares(WHITE) == []
    assert b2.forced_move() in (b2.index(4, 0), b2.index(4, 5))
    # no threats -> no forced move
    b3 = Board(9, 5)
    b3.play(40)
    assert b3.forced_move() is None
