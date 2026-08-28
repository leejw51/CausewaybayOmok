"""Ready-made configurations, from a 30-second smoke test to a long run."""

from __future__ import annotations

from .config import Config

PRESETS = {}


def _preset(name: str, **overrides):
    def build() -> Config:
        cfg = Config()
        for dotted, value in overrides.items():
            cfg.override(dotted.replace("__", "."), str(value))
        cfg.run_name = name
        cfg.run_dir = f"runs/{name}"
        return cfg

    PRESETS[name] = build
    return build


# Small board, tiny net -- used by the tests and by `make smoke`.
_preset(
    "tiny",
    game__board_size=9,
    net__channels=32, net__blocks=2, net__value_hidden=64,
    mcts__simulations=24, mcts__temperature_moves=8,
    selfplay__games_per_iter=8, selfplay__parallel_games=8,
    train__batch_size=64, train__steps_per_iter=30, train__replay_min_positions=100,
    train__ckpt_every_steps=10, train__replay_max_positions=50_000,
    arena__games=4, arena__simulations=24, arena__parallel_games=4,
    iterations=3,
)

# A real but modest 15x15 run that fits comfortably on a laptop.
_preset(
    "small",
    net__channels=64, net__blocks=4,
    mcts__simulations=100,
    selfplay__games_per_iter=32, selfplay__parallel_games=32,
    train__batch_size=256, train__steps_per_iter=200,
    arena__games=12, arena__simulations=80,
    iterations=200,
)

# The default: 6x96, 160 simulations.
_preset("base")

# For a CUDA box you are happy to leave running.
_preset(
    "strong",
    net__channels=128, net__blocks=10,
    mcts__simulations=400,
    selfplay__games_per_iter=256, selfplay__parallel_games=128,
    train__batch_size=1024, train__steps_per_iter=1000,
    train__replay_max_positions=1_500_000,
    arena__games=40, arena__simulations=200, arena__parallel_games=40,
    iterations=1000,
)


def make_config(preset: str = "base") -> Config:
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    return PRESETS[preset]()
