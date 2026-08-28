import pytest

from omok.board import BLACK, Board, WHITE
from omok.engine import Engine
from omok.presets import make_config

arcade = pytest.importorskip("arcade", reason="the GUI needs `make install-gui`")

from omok.gui import Geometry, human_colours, winning_line  # noqa: E402


# ------------------------------------------------------------------ engine
def test_engine_plays_the_winning_move(tmp_path):
    """End to end through the worker thread, on a random-init tiny network."""
    cfg = make_config("tiny")
    cfg.run_dir = str(tmp_path)  # empty run: the engine falls back to random weights
    cfg.mcts.dirichlet_weight = 0.0
    board = Board(9, 5)
    for i, filler in enumerate((8, 26, 44, 62)):  # black four in a row, both ends open
        board.play(30 + i)
        board.play(filler)

    engine = Engine(cfg, batch_size=8)
    engine.start()
    try:
        job = engine.submit(board, simulations=240)
        result = _wait_for_result(engine, job)
    finally:
        engine.shutdown()
    assert result.move in (29, 34)
    assert result.value > 0.3
    assert result.simulations == 240
    assert result.top and result.top[0][1] >= result.top[-1][1]


def test_engine_drops_a_cancelled_search(tmp_path):
    cfg = make_config("tiny")
    cfg.run_dir = str(tmp_path)
    engine = Engine(cfg, batch_size=8)
    engine.start()
    try:
        job = engine.submit(Board(9, 5), simulations=4_000)
        engine.cancel()
        second = engine.submit(Board(9, 5), simulations=16)
        result = _wait_for_result(engine, second)
    finally:
        engine.shutdown()
    assert result.job == second != job


def _wait_for_result(engine, job, timeout=120.0):
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        for event in engine.poll():
            if event[0] == "error":
                pytest.fail(event[1])
            if event[0] == "result" and event[1].job == job:
                return event[1]
        time.sleep(0.02)
    pytest.fail(f"engine produced no result for job {job} within {timeout}s")


# ---------------------------------------------------------------- geometry
def test_geometry_round_trips_every_intersection():
    geo = Geometry.fit(15, left=0, bottom=0, width=600, height=700)
    for index in range(15 * 15):
        x, y = geo.point_of(index)
        assert geo.hit(x, y) == index


def test_geometry_fits_and_centres_the_board():
    geo = Geometry.fit(9, left=10, bottom=0, width=400, height=300)
    assert geo.extent <= 300 + 1e-6
    assert geo.bottom == pytest.approx((300 - geo.extent) / 2)
    assert geo.left == pytest.approx(10 + (400 - geo.extent) / 2)
    assert geo.hit(geo.left - 5, geo.bottom - 5) is None
    # Row 0 is the top row, as in the ASCII board.
    assert geo.point(0, 0)[1] > geo.point(8, 0)[1]


def test_geometry_ignores_clicks_between_intersections():
    geo = Geometry.fit(15, 0, 0, 600, 600)
    x, y = geo.point(7, 7)
    assert geo.hit(x + geo.cell * 0.49, y) == 7 * 15 + 7
    assert geo.hit(x + geo.cell * 0.5, y + geo.cell * 0.5) is None


# ------------------------------------------------------------------ helpers
def test_winning_line_marks_the_five_stones():
    board = Board(9, 5)
    for i in range(5):
        board.play(30 + i)
        if not board.over:
            board.play(8 + i * 9)
    assert board.winner == BLACK
    assert winning_line(board) == [30, 31, 32, 33, 34]


def test_winning_line_is_empty_for_an_unfinished_game():
    board = Board(9, 5).play_many([40, 41])
    assert winning_line(board) == []


def test_human_colours_cover_every_mode():
    assert human_colours("black") == {BLACK}
    assert human_colours("white") == {WHITE}
    assert human_colours("none") == set()
    assert human_colours("both") == {BLACK, WHITE}
