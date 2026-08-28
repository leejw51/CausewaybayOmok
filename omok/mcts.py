"""Batched PUCT Monte-Carlo tree search.

The trick that makes this usable in Python: many games are searched at once and
every game contributes one leaf per iteration, so each neural-network call sees
a batch of `len(trees)` positions instead of one.  Nothing here is
multi-threaded -- the parallelism is in the batch.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np

from .board import Board
from .config import MCTSConfig
from .encode import encode_batch


class Node:
    # `legal`/`bias`/`nf`/`unvisited` and the two scratch buffers exist purely to
    # keep `Tree.select_action` cheap: it runs tens of thousands of times per
    # search and every avoided array allocation shows up in the profile.
    __slots__ = ("P", "N", "W", "children", "expanded", "terminal", "visits", "value",
                 "legal", "bias", "nf", "unvisited", "q_buf", "u_buf",
                 "w_sum", "p_explored")

    def __init__(self) -> None:
        self.P: np.ndarray | None = None
        self.N: np.ndarray | None = None
        self.W: np.ndarray | None = None
        self.children: dict[int, "Node"] = {}
        self.expanded = False
        self.terminal = False
        self.visits = 0
        self.value = 0.0
        self.legal: np.ndarray | None = None
        self.bias: np.ndarray | None = None
        self.nf: np.ndarray | None = None
        self.unvisited: np.ndarray | None = None
        self.q_buf: np.ndarray | None = None
        self.u_buf: np.ndarray | None = None
        self.w_sum = 0.0
        self.p_explored = 0.0

    def expand(self, priors: np.ndarray, legal: np.ndarray) -> None:
        action_size = len(priors)
        masked = np.where(legal, priors, 0.0).astype(np.float32)
        total = float(masked.sum())
        if total <= 1e-8:  # network gave all its mass to illegal moves
            masked = legal.astype(np.float32)
            total = float(masked.sum())
        self.P = masked / max(total, 1e-8)
        self.N = np.zeros(action_size, dtype=np.int32)
        self.W = np.zeros(action_size, dtype=np.float32)
        # A node's position never changes, so its legal moves never change:
        # cache the mask instead of recomputing it on every descent, and keep
        # it as an additive 0 / -inf bias so scoring stays one vectorised add.
        self.legal = legal
        self.bias = np.where(legal, 0.0, -np.inf)
        self.nf = np.zeros(action_size, dtype=np.float64)
        self.unvisited = np.ones(action_size, dtype=np.float64)
        self.q_buf = np.empty(action_size, dtype=np.float64)
        self.u_buf = np.empty(action_size, dtype=np.float64)
        self.w_sum = 0.0
        self.p_explored = 0.0
        self.expanded = True

    def q_values(self) -> np.ndarray:
        assert self.N is not None and self.W is not None
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.N > 0, self.W / np.maximum(self.N, 1), 0.0)


class Tree:
    """A search tree rooted at a position, reused across the moves of a game."""

    def __init__(self, board: Board, cfg: MCTSConfig) -> None:
        self.cfg = cfg
        self.board = board
        self.root = Node()
        self.root_noise_applied = False
        self._pending_path: list[tuple[Node, int]] | None = None
        self._pending_board: Board | None = None
        self._pending_node: Node | None = None

    # -- tree walking ------------------------------------------------------
    def select_action(self, node: Node) -> int:
        """PUCT: argmax over Q + U, with unvisited children held at the FPU.

        Identical arithmetic to the obvious formulation, but `w_sum` and
        `p_explored` are maintained by `backup` rather than re-reduced here,
        and the intermediates land in the node's own scratch buffers.
        """
        assert node.nf is not None and node.W is not None and node.P is not None
        visits = node.visits
        total = visits if visits > 0 else 1
        parent_q = node.w_sum / visits if visits > 0 else 0.0
        fpu = parent_q - self.cfg.fpu_reduction * math.sqrt(max(node.p_explored, 0.0))

        nf, q, u = node.nf, node.q_buf, node.u_buf
        np.maximum(nf, 1.0, out=q)
        np.divide(node.W, q, out=q)          # Q, and exactly 0 where unvisited
        np.add(nf, 1.0, out=u)
        np.divide(node.P, u, out=u)
        u *= self.cfg.c_puct * math.sqrt(total)
        q += u
        if fpu != 0.0:                        # unvisited children sit at the FPU
            q += node.unvisited * fpu
        q += node.bias                        # -inf on illegal moves
        return int(np.argmax(q))

    def descend(self) -> tuple[Node, Board, list[tuple[Node, int]]]:
        node = self.root
        board = self.board.copy()
        path: list[tuple[Node, int]] = []
        while node.expanded and not node.terminal:
            action = self.select_action(node)
            path.append((node, action))
            board.play(action)
            child = node.children.get(action)
            if child is None:
                child = Node()
                node.children[action] = child
            node = child
            if board.over:
                node.terminal = True
                node.value = board.result_for(board.to_move)
                break
        return node, board, path

    @staticmethod
    def backup(path: Sequence[tuple[Node, int]], value: float) -> None:
        sign = -1.0  # the leaf value is from the leaf mover's point of view
        for node, action in reversed(path):
            assert node.N is not None and node.W is not None
            if node.N[action] == 0:           # first visit: this child is now explored
                node.unvisited[action] = 0.0
                node.p_explored += float(node.P[action])
            node.N[action] += 1
            node.nf[action] += 1.0
            delta = sign * value
            node.W[action] += delta
            node.w_sum += delta
            node.visits += 1
            sign = -sign

    # -- one search iteration, split around the network call ---------------
    def begin_iteration(self) -> Board | None:
        """Descend to a leaf.  Returns the position needing evaluation, if any."""
        if self.board.over:
            return None
        node, board, path = self.descend()
        if node.terminal:
            self.backup(path, node.value)
            return None
        self._pending_node, self._pending_board, self._pending_path = node, board, path
        return board

    def finish_iteration(self, priors: np.ndarray, value: float,
                         rng: np.random.Generator | None = None) -> None:
        node, board, path = self._pending_node, self._pending_board, self._pending_path
        self._pending_node = self._pending_board = self._pending_path = None
        if node is None or board is None or path is None:
            return
        radius = self.cfg.prior_local_radius
        if radius > 0:
            local = np.where(board.neighbourhood(radius), priors, 0.0)
            if float(local.sum()) > 1e-6:
                priors = local
        node.expand(priors, board.legal_mask())
        node.value = float(value)
        if node is self.root and rng is not None:
            self.apply_root_noise(rng)
        self.backup(path, float(value))

    # -- root handling -----------------------------------------------------
    def apply_root_noise(self, rng: np.random.Generator) -> None:
        cfg = self.cfg
        if cfg.dirichlet_weight <= 0.0 or self.root.P is None or self.root_noise_applied:
            return
        legal = self.board.legal_mask()
        if cfg.prior_local_radius > 0:
            near = legal & self.board.neighbourhood(cfg.prior_local_radius)
            if near.any():
                legal = near
        indices = np.nonzero(legal)[0]
        if len(indices) == 0:
            return
        noise = rng.dirichlet([cfg.dirichlet_alpha] * len(indices)).astype(np.float32)
        priors = self.root.P.copy()
        priors[indices] = (1.0 - cfg.dirichlet_weight) * priors[indices] + cfg.dirichlet_weight * noise
        total = float(priors.sum())
        self.root.P = priors / max(total, 1e-8)
        self.root_noise_applied = True

    def visit_distribution(self) -> np.ndarray:
        if self.root.N is None:
            legal = self.board.legal_mask().astype(np.float32)
            return legal / max(legal.sum(), 1.0)
        counts = self.root.N.astype(np.float32)
        total = counts.sum()
        if total <= 0:
            legal = self.board.legal_mask().astype(np.float32)
            return legal / max(legal.sum(), 1.0)
        return counts / total

    def root_value(self) -> float:
        if self.root.N is None or self.root.visits == 0:
            return float(self.root.value)
        assert self.root.W is not None
        return float(self.root.W.sum() / max(self.root.visits, 1))

    def pick_move(self, rng: np.random.Generator, temperature: float) -> int:
        probs = self.visit_distribution()
        if temperature <= 1e-3:
            return int(np.argmax(probs))
        scaled = np.power(probs, 1.0 / temperature)
        total = scaled.sum()
        if total <= 0:
            return int(np.argmax(probs))
        return int(rng.choice(len(scaled), p=scaled / total))

    def advance(self, move: int) -> None:
        """Play ``move`` and keep the corresponding sub-tree."""
        child = self.root.children.get(move) if self.cfg.reuse_tree else None
        self.board = self.board.copy()
        self.board.play(move)
        self.root = child if child is not None else Node()
        self.root_noise_applied = False


Evaluator = Callable[[Sequence[Board]], tuple[np.ndarray, np.ndarray]]


class NetworkEvaluator:
    """Encodes positions and runs them through a backend, with a small cache."""

    def __init__(self, backend, batch_size: int = 256) -> None:
        self.backend = backend
        self.batch_size = batch_size
        self.calls = 0
        self.positions = 0

    def __call__(self, boards: Sequence[Board]) -> tuple[np.ndarray, np.ndarray]:
        if not boards:
            action_size = 0
            return np.zeros((0, action_size), np.float32), np.zeros((0,), np.float32)
        policies: list[np.ndarray] = []
        values: list[np.ndarray] = []
        for start in range(0, len(boards), self.batch_size):
            chunk = boards[start:start + self.batch_size]
            planes = encode_batch(chunk)
            policy, value = self.backend.predict(planes)
            policies.append(policy)
            values.append(value)
            self.calls += 1
            self.positions += len(chunk)
        return np.concatenate(policies), np.concatenate(values)


def run_search(trees: Sequence[Tree], evaluator: Evaluator, simulations: int,
               rng: np.random.Generator) -> None:
    """Run ``simulations`` iterations over all trees, batching the network calls."""
    active = [t for t in trees if not t.board.over]
    if not active:
        return
    # A reused sub-tree arrives with its root already expanded, so the noise
    # normally added on expansion must be injected here instead -- otherwise
    # self-play explores only on the first move of each game.
    for tree in active:
        if tree.root.expanded and not tree.root_noise_applied:
            tree.apply_root_noise(rng)
    for _ in range(simulations):
        pending: list[Tree] = []
        boards: list[Board] = []
        for tree in active:
            board = tree.begin_iteration()
            if board is not None:
                pending.append(tree)
                boards.append(board)
        if not pending:
            continue
        policies, values = evaluator(boards)
        for i, tree in enumerate(pending):
            tree.finish_iteration(policies[i], float(values[i]), rng)


def search_position(board: Board, evaluator: Evaluator, cfg: MCTSConfig,
                    simulations: int, rng: np.random.Generator,
                    add_noise: bool = False) -> Tree:
    """Convenience wrapper for searching a single position (play / analysis)."""
    tree = Tree(board.copy(), cfg)
    if not add_noise:
        tree.root_noise_applied = True  # suppress exploration noise
    run_search([tree], evaluator, simulations, rng)
    return tree
