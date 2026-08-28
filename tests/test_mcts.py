import numpy as np

from omok.board import BLACK, Board
from omok.config import MCTSConfig
from omok.mcts import Tree, run_search, search_position


def uniform_evaluator(boards):
    """A network stand-in: uniform policy, zero value."""
    action_size = boards[0].action_size
    policy = np.full((len(boards), action_size), 1.0 / action_size, dtype=np.float32)
    return policy, np.zeros(len(boards), dtype=np.float32)


# Scattered white replies -- they must not accidentally form a line themselves.
FILLERS = (8, 26, 44, 62)


def four_in_a_row():
    """Black has four in a row with both ends open; black to move."""
    b = Board(9, 5)
    for i in range(4):
        b.play(30 + i)
        b.play(FILLERS[i])
    return b


def test_search_finds_the_immediate_win():
    board = four_in_a_row()
    cfg = MCTSConfig(simulations=200, dirichlet_weight=0.0)
    rng = np.random.default_rng(0)
    tree = search_position(board, uniform_evaluator, cfg, 300, rng)
    best = int(np.argmax(tree.visit_distribution()))
    assert best in (29, 34)
    assert tree.root_value() > 0.3


def test_search_blocks_the_opponents_win():
    """Seeing a threat needs deeper search than making one -- smaller board."""
    b = Board(7, 5)
    b.play(22)
    b.play(26)  # white closes one end straight away
    for i, filler in enumerate((0, 2, 4)):
        b.play(23 + i)
        b.play(filler)
    b.play(45)  # black wastes a move; white must now block the open end at 21
    cfg = MCTSConfig(simulations=2000, dirichlet_weight=0.0)
    tree = search_position(b, uniform_evaluator, cfg, 2000, np.random.default_rng(1))
    assert int(np.argmax(tree.visit_distribution())) == 21


def test_visits_match_simulation_count():
    cfg = MCTSConfig(simulations=32, dirichlet_weight=0.0)
    tree = Tree(Board(9, 5), cfg)
    tree.root_noise_applied = True
    run_search([tree], uniform_evaluator, 32, np.random.default_rng(0))
    # one simulation is spent expanding the root itself
    assert tree.root.N.sum() == 31
    assert abs(tree.visit_distribution().sum() - 1.0) < 1e-6


def test_batched_search_matches_single_tree():
    cfg = MCTSConfig(simulations=64, dirichlet_weight=0.0)
    boards = [four_in_a_row(), four_in_a_row()]
    trees = [Tree(b, cfg) for b in boards]
    for tree in trees:
        tree.root_noise_applied = True
    run_search(trees, uniform_evaluator, 64, np.random.default_rng(0))
    a, b = (t.visit_distribution() for t in trees)
    assert np.allclose(a, b)  # identical positions, deterministic search


def test_tree_reuse_keeps_subtree_statistics():
    cfg = MCTSConfig(simulations=64, dirichlet_weight=0.0, reuse_tree=True)
    tree = Tree(Board(9, 5), cfg)
    tree.root_noise_applied = True
    run_search([tree], uniform_evaluator, 64, np.random.default_rng(0))
    move = int(np.argmax(tree.visit_distribution()))
    visits_before = tree.root.children[move].visits
    tree.advance(move)
    assert tree.root.visits == visits_before
    assert tree.board.move_number == 1


def test_search_on_finished_game_is_a_noop():
    board = four_in_a_row()
    board.play(29)  # black wins
    cfg = MCTSConfig(simulations=16, dirichlet_weight=0.0)
    tree = Tree(board, cfg)
    run_search([tree], uniform_evaluator, 16, np.random.default_rng(0))
    assert tree.root.N is None
    assert abs(tree.visit_distribution().sum()) < 1e-6 or tree.board.over


def test_dirichlet_noise_changes_root_priors():
    cfg = MCTSConfig(simulations=8, dirichlet_weight=0.5, dirichlet_alpha=0.3)
    tree = Tree(Board(9, 5), cfg)
    run_search([tree], uniform_evaluator, 8, np.random.default_rng(3))
    assert tree.root_noise_applied
    assert tree.root.P.std() > 1e-4  # uniform priors would have zero spread
