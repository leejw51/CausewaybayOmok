"""Configuration objects for the Omok trainer.

Everything is a plain dataclass so a run's configuration can be serialised to
JSON, stored next to the checkpoints and reloaded verbatim after a crash.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GameConfig:
    board_size: int = 15
    win_length: int = 5
    # Freestyle gomoku: six or more in a row also wins.  Set to False for the
    # "exact five" rule used by many tournament rule sets.
    allow_overline: bool = True


@dataclass
class NetConfig:
    channels: int = 96
    blocks: int = 6
    policy_channels: int = 4
    value_channels: int = 2
    value_hidden: int = 128


@dataclass
class MCTSConfig:
    simulations: int = 160
    c_puct: float = 1.6
    dirichlet_alpha: float = 0.15
    dirichlet_weight: float = 0.25
    # Moves played with temperature 1.0 before switching to (near) argmax.
    temperature_moves: int = 12
    temperature: float = 1.0
    # First play urgency reduction: value used for unvisited children.
    fpu_reduction: float = 0.25
    # Re-use the sub-tree of the played move between moves of a game.
    reuse_tree: bool = True


@dataclass
class SelfPlayConfig:
    games_per_iter: int = 64
    # Number of games stepped through MCTS simultaneously.  This is what makes
    # the neural-network batches large enough to be worth a GPU.
    parallel_games: int = 32
    # Flush finished games to disk after this many games (crash safety).
    flush_every_games: int = 2
    # ... or after this many seconds, whichever comes first.
    flush_every_seconds: float = 60.0
    # Random plies played uniformly at the start of a game to diversify data.
    random_opening_plies: int = 0
    max_moves: int = 0  # 0 => board_size**2


@dataclass
class TrainConfig:
    batch_size: int = 512
    steps_per_iter: int = 400
    lr: float = 2e-3
    lr_min: float = 2e-4
    lr_warmup_steps: int = 200
    lr_decay_steps: int = 200_000
    weight_decay: float = 1e-4
    value_loss_weight: float = 1.0
    grad_clip: float = 4.0
    # Checkpoint cadence -- keep it tight, a crash should cost seconds.
    ckpt_every_steps: int = 100
    ckpt_every_seconds: float = 120.0
    keep_last_ckpts: int = 5
    replay_max_positions: int = 400_000
    replay_min_positions: int = 2_000
    augment: bool = True


@dataclass
class ArenaConfig:
    games: int = 24
    simulations: int = 100
    promote_winrate: float = 0.55
    temperature_moves: int = 4
    parallel_games: int = 12


@dataclass
class Config:
    run_name: str = "omok"
    run_dir: str = "runs/omok"
    seed: int = 1234
    backend: str = "auto"  # auto | torch | mlx | torch-cpu
    iterations: int = 1000
    game: GameConfig = field(default_factory=GameConfig)
    net: NetConfig = field(default_factory=NetConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    arena: ArenaConfig = field(default_factory=ArenaConfig)

    # -- serialisation ---------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str) -> None:
        from .utils import atomic_write_text

        atomic_write_text(path, json.dumps(self.to_dict(), indent=2, sort_keys=True))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        data = copy.deepcopy(data)
        kwargs: dict[str, Any] = {}
        for f in dataclasses.fields(cls):
            if f.name not in data:
                continue
            value = data.pop(f.name)
            if dataclasses.is_dataclass(f.type) or isinstance(value, dict):
                sub = _SUB_TYPES.get(f.name)
                kwargs[f.name] = sub(**value) if sub else value
            else:
                kwargs[f.name] = value
        if data:
            raise ValueError(f"unknown config keys: {sorted(data)}")
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    # -- dotted overrides -------------------------------------------------
    def override(self, dotted: str, raw: str) -> None:
        """Apply ``--set train.batch_size=256`` style overrides in place."""
        parts = dotted.split(".")
        target: Any = self
        for part in parts[:-1]:
            if not hasattr(target, part):
                raise ValueError(f"unknown config section: {dotted}")
            target = getattr(target, part)
        leaf = parts[-1]
        if not hasattr(target, leaf):
            raise ValueError(f"unknown config key: {dotted}")
        current = getattr(target, leaf)
        setattr(target, leaf, _coerce(raw, current))

    # -- derived ----------------------------------------------------------
    @property
    def action_size(self) -> int:
        return self.game.board_size ** 2

    def paths(self) -> "RunPaths":
        return RunPaths(self.run_dir)


_SUB_TYPES = {
    "game": GameConfig,
    "net": NetConfig,
    "mcts": MCTSConfig,
    "selfplay": SelfPlayConfig,
    "train": TrainConfig,
    "arena": ArenaConfig,
}


def _coerce(raw: str, current: Any) -> Any:
    if isinstance(current, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(current, int):
        return int(float(raw))
    if isinstance(current, float):
        return float(raw)
    return raw


@dataclass
class RunPaths:
    """Every on-disk location a run touches, in one place."""

    root: str

    @property
    def config(self) -> str:
        return os.path.join(self.root, "config.json")

    @property
    def state(self) -> str:
        return os.path.join(self.root, "state.json")

    @property
    def checkpoints(self) -> str:
        return os.path.join(self.root, "checkpoints")

    @property
    def replay(self) -> str:
        return os.path.join(self.root, "replay")

    @property
    def logs(self) -> str:
        return os.path.join(self.root, "logs")

    @property
    def export(self) -> str:
        return os.path.join(self.root, "export")

    def ensure(self) -> "RunPaths":
        for p in (self.root, self.checkpoints, self.replay, self.logs, self.export):
            os.makedirs(p, exist_ok=True)
        return self
