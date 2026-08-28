# CausewaybayOmok — Omok (gomoku) AI trainer

An AlphaZero-style self-play trainer for Omok, written from scratch: game rules,
MCTS, network, training loop, crash recovery and CoreML export. The result is a
small policy/value network you can drop into an iPhone or Mac game.

* **PyTorch on CUDA** when an NVIDIA GPU is present.
* **MLX on Apple Silicon** — the default on this Mac.
* PyTorch MPS / CPU as fallbacks. The same checkpoint loads on any of them.
* **Saves constantly.** Self-play games are flushed to disk every couple of
  games, checkpoints every 100 steps, and every file is written atomically.
  Kill the process at any moment and `make resume` continues where it stopped.

```bash
make install      # torch + numpy (+ mlx on Apple Silicon)
make info         # model size, backend, measured speed, expected training time
make train        # run the loop; Ctrl-C is safe, `make resume` picks it up
make export       # CoreML .mlpackage for the iOS / macOS game
make gui          # play against it in a window (Arcade)
make play         # ... or in the terminal
```

There is also a **Rust AI core on Apple MLX** that plays the model the trainer
produced, with a terminal client and a LÖVE game on top of it:

```bash
make ai           # build the Rust/MLX engine as a shared library
make game         # play "Causewaybay Omok" (LÖVE)
make tui          # ... or in the terminal (LuaJIT)
```

## What `make info` tells you

It benchmarks *this* machine and prints the model size, where every file lives,
and how long training will actually take. On the M-series Mac this was written
on (MLX backend, `base` preset, 15×15 board):

```
Model
  architecture      6 residual blocks x 96 channels
  parameters        1,265,822
  size on disk      5.06 MB float32   (2.53 MB float16, ~1.27 MB int8)

Measured throughput
  network eval      8,541 positions/s (batch 32)
  training          4.07 steps/s = 2,083 samples/s (batch 512)
  self-play         28.64 moves/s (32 games in parallel, 160 sims/move)

Expected training time
  self-play         1m51s per iteration (64 games, ~50 moves each)
  training          1m38s per iteration (400 steps x batch 512)
  arena             42.8s per iteration (24 games)
  one iteration     4m12s
  1000 iterations   70h13m
```

So: **~4 minutes per iteration, a playable net in under an hour, a genuinely
strong one overnight-to-a-few-days.** A CUDA box is several times faster; run
`make info PRESET=strong` there to see its own numbers. The 2.5 MB float16
CoreML model runs in about a millisecond per position on an A-series Neural
Engine, so the phone can afford a few hundred MCTS simulations per move.

## Where everything is saved

Everything for one run lives under `runs/<preset>/` (override with `RUN=...`):

| Path | What |
| --- | --- |
| `runs/base/config.json` | the exact configuration of this run |
| `runs/base/state.json` | iteration + phase — this is what makes it resumable |
| `runs/base/replay/shard-XXXXXXXX.npz` | self-play games (moves + MCTS policies) |
| `runs/base/checkpoints/step-XXXXXXXX.npz` | rolling checkpoints (last 5) |
| `runs/base/checkpoints/best.npz` | the current best model — what self-play uses |
| `runs/base/logs/run.jsonl` | structured log of every phase |
| `runs/base/export/OmokNet.mlpackage` | the exported CoreML model |
| `runs/base/export/model_meta.json` | input/output contract for the app |

`make status` summarises all of it at a glance.

## How the training loop works

Each iteration runs three phases, and each phase is independently resumable:

1. **Self-play** — `games_per_iter` games with the *best* network. Games run
   `parallel_games` at a time so every MCTS iteration produces one big network
   batch. Dirichlet noise at the root and temperature sampling for the first
   `temperature_moves` plies keep the data diverse. Resuming counts the games
   already on disk for this iteration and generates only the remainder.
2. **Train** — `steps_per_iter` AdamW steps on positions sampled from the replay
   buffer, with random dihedral symmetries (the board has 8). Targets are the
   MCTS visit distribution (policy, cross-entropy) and the game result from the
   side-to-move's point of view (value, MSE). Resuming continues to the step
   count recorded in `state.json`.
3. **Arena** — the new network plays `arena.games` games against the current
   best, colours alternating. It is promoted to `best.npz` only if it scores
   `promote_winrate` (0.55) or better. Set `arena.games=0` to always promote.

## Presets

| Preset | Board | Net | Sims | Use |
| --- | --- | --- | --- | --- |
| `tiny` | 9×9 | 2×32 | 24 | 30-second smoke test (`make smoke`) |
| `small` | 15×15 | 4×64 | 100 | laptop-friendly real run |
| `base` | 15×15 | 6×96 | 160 | the default |
| `strong` | 15×15 | 10×128 | 400 | for a CUDA box you leave running |

Any field can be overridden without editing files:

```bash
make train PRESET=small ITERS=20
python3 -m omok train --preset base --set mcts.simulations=400 \
                                    --set selfplay.parallel_games=128
```

Useful knobs: `mcts.simulations` (strength per move vs. speed),
`selfplay.parallel_games` (bigger batches — raise it on a GPU),
`train.batch_size`, `train.replay_max_positions` (how much history to keep),
`game.allow_overline` (`false` for the exact-five rule).

## Crash safety

* Every write goes to a temp file and is `os.replace`d into place, so a
  half-written file can never be observed.
* Self-play flushes finished games every 2 games or 60 seconds, whichever first.
* Checkpoints are written every 100 steps or 2 minutes, plus at the end of each
  phase, and include the optimiser state.
* `Ctrl-C` sets a stop flag: the current unit of work finishes, everything is
  flushed and checkpointed, and the process exits. A second `Ctrl-C` exits now.
* An unreadable checkpoint or shard is skipped, not fatal — the run falls back
  to the previous good checkpoint.
* `tests/test_pipeline.py::test_data_survives_a_hard_kill` actually `SIGKILL`s a
  training process and asserts the run resumes.

Background training:

```bash
make train-bg     # detached, logs to runs/<preset>/logs/train.out
make tail         # follow the log
make stop         # SIGTERM -> it checkpoints, then exits
```

## Playing against it (`make gui`)

```bash
make install-gui                 # arcade, only needed for this
make gui                         # you are black, the run's best model is white
make gui COLOR=white SIMS=400    # stronger opponent, you play white
make gui COLOR=none              # watch the engine play itself
make gui PRESET=tiny RUN=runs/smoke
```

The window loads the same checkpoint everything else uses
(`runs/<name>/checkpoints/best.npz`, or `--model some.npz`) through the usual
backend selection, so it runs on CUDA, MLX, MPS or CPU unchanged. Search runs on
a worker thread and is cancellable, so the board stays responsive while the
engine thinks and an undo takes effect immediately.

| Key | |
| --- | --- |
| click | place a stone |
| `space` | make the engine move now for the side to play |
| `h` | hint: search the current position without playing it |
| `a` | toggle the candidate-move overlay |
| `f` | toggle the animations and particles |
| `u` | undo (your move and the reply) |
| `n` | new game |
| `s` | cycle black / white / engine-vs-engine / two humans |
| `p` | pause the engine |
| `1` `2` `3` `4` | difficulty: 24 / 100 / 320 / 900 simulations per move |
| `q` `esc` | quit |

The side panel shows the loaded model and backend, an evaluation bar from
black's point of view, the search's top candidate moves with their visit
shares, and how fast the last search ran.

### Look and feel

Stones land with an `ease_out_back` overshoot that squashes on impact, ring the
board with a shockwave and throw a burst of sparks — warm for black, cool for
white. The evaluation bar and the search progress ease toward their new values
rather than jumping, candidate moves fade in staggered and the best one
breathes, and the last move keeps a slow pulse. Winning sweeps a springy
highlight down the five stones, sets off confetti, flashes the screen and
rattles the board. Undo and new game puff the stones away.

`f` turns all of it off (`--no-effects` to start that way) — the game plays
identically, it just stops moving. The easing curves and the particle pool are
in `omok/effects.py`, kept free of arcade imports so they are unit-tested
without a display.

### The art

The board, stones, spark and panel textures in `omok/assets/` were generated
with **Grok** (`grok-imagine-image`) and are committed, so playing needs no API
key. `tools/make_assets.py` holds the prompts and the post-processing that
turns an opaque rectangle into a game sprite — the stones are found by contrast
against their backdrop and given an anti-aliased circular alpha, and the spark
becomes alpha-from-luminance so its black background reads as transparency
under additive blending.

```bash
export XAI_API_KEY=...
make assets              # reuse any raw generations in assets_raw/
make assets FORCE=1      # ask Grok for fresh ones
```

If the assets are deleted the window falls back to flat colours and keeps
working.

## The Rust core, and the two games on top of it

The Python side trains the network; this side plays it. `rust/` is an Omok
engine on [Apple MLX](https://github.com/ml-explore/mlx) that loads the
trainer's own `.npz` checkpoints and exposes a C ABI, and `lua/` and `love2d/`
are two clients that share one set of LuaJIT bindings to it.

```
runs/<name>/checkpoints/best.npz     the trainer's output
        │
        ▼
rust/crates/omok-core   rules, plane encoding, npz reader, batched PUCT search
rust/crates/omok-mlx    the network on MLX (Metal)
rust/crates/omok-ffi    C ABI  ->  libomok_ai.dylib
        │
        ▼
lua/omok/init.lua       LuaJIT FFI bindings, shared by both clients
        ├── lua/main.lua      terminal client
        └── love2d/main.lua   the LÖVE game
```

**It plays the model, not an approximation of it.** The forward pass is checked
against `omok/backends/mlx_backend.py` on the same checkpoint and agrees to
every digit printed; `cargo run --release -p omok-mlx --example parity` prints
the numbers to compare. The rules are a port of `omok/board.py` down to the
one-ply `forced_move` safety net, so the engine never misses a win or fails to
block one however small its search budget is.

**Why it is quicker than the Python engine.** The trainer batches its network
calls across *games* — many self-play games search at once. A game against a
person has one position to think about, so the Rust search batches across
*leaves* instead: it descends 16 times before calling the network, holding a
virtual loss on each path so the descents pick different leaves. Same
arithmetic, same priors, but MLX sees batches of 16 rather than a stream of
single positions. On an M-series Mac with the shipped 4×64 network:

| | 160 simulations |
| --- | --- |
| `omok/mcts.py` (Python, MLX) | ~1300 simulations/s |
| `rust/` (Rust, MLX) | ~5100-9800 simulations/s |

Two more things happen once, at load time, rather than on every forward pass:
batch-norm folds into the preceding convolution, and the layout differences
between PyTorch's `OIHW` weights and MLX's `OHWI` — including the NCHW flatten
the heads do before their fully connected layer — are baked into the weights.
What is left at run time is convolutions, matmuls and two activations.

### Building it

MLX compiles its own Metal kernels, and the `metal` compiler ships with Xcode
rather than with the Command Line Tools. `make ai` points `DEVELOPER_DIR` at
`/Applications/Xcode.app` for the build, which — unlike `xcode-select -s` —
needs no `sudo` and changes nothing outside the build. Override with
`make ai XCODE=/path/to/Developer` if Xcode is somewhere else.

```bash
make ai           # build   -> rust/target/release/libomok_ai.dylib
make ai-test      # the Rust unit tests
make test-lua     # the same core through its Lua bindings
make ai-bench     # load the trained model and time a few searches
```

The library finds its model on its own: `OMOK_MODEL`, or the most recently
written `runs/*/checkpoints/best.npz`. Train a new run and the games pick it
up next time they start. `OMOK_LIB` overrides where the library itself is.

### The terminal client (`make tui`)

```bash
make tui                     # play black at the default difficulty
make tui LEVEL=4 COLOR=white # let the model open, and make it think
luajit lua/main.lua --watch  # the model plays itself
```

Arrow keys or `hjkl` move the cursor, space places a stone, `u` takes a move
back, `?` asks the model what it would play, `1`-`5` set the difficulty and `w`
watches it play itself. The engine thinks on its own thread inside the Rust
core, so the spinner keeps turning and keys keep working while it searches, and
taking a move back abandons the search instead of waiting for it.

### The LÖVE game (`make game`)

"Causewaybay Omok" — Prague's Old Town Square at dusk, in 16-bit pixel art.

```bash
make game            # a window
make game-portrait   # started in a portrait window
```

Click or use the arrow keys, `SPACE` to place, `U` undo, `H` hint, `N` new
game, `1`-`5` difficulty, `TAB` to swap colours, `W` to watch, `F11` for
fullscreen. The window is resizable and the layout follows it: the panel sits
beside the board in landscape and below it in portrait, and the board is always
the largest whole-pixel grid that leaves the panel its room.

Everything that moves goes through `love2d/effects.lua` — the Penner easing
curves, a tween type, particle bursts, expanding shockwaves, floating text,
decaying screen shake and the ambient motes drifting over the square. Stones
land with an overshoot and a spark burst, the last move keeps a pulsing halo,
the winning line lights up one stone at a time under a fountain of particles,
the board dims under a slow sweep while the model thinks, and the backdrop
drifts against the pointer.

### The art

`tools/make_love2d_assets.py` holds the prompts and the processing, so the art
is reproducible rather than a chat log:

```bash
export XAI_API_KEY=...
make love-assets FORCE=1
```

An image model draws at 1024px with soft anti-aliased edges, which is not the
same thing as pixel art, so every asset is downsampled onto its true pixel grid
with a box filter and then quantised to a small palette — that is what turns
smooth shading into flat blocks and gives the edges their staircase. The stones
are cut out of their flat backdrop with a hard circular alpha (an anti-aliased
one would undo the effect), and the glows become alpha-from-luminance so their
black backgrounds turn into transparency. Everything is PNG: these are
composited over lit backgrounds with additive blending, where JPEG's ringing
shows as haloes around the lamps. The text is a real bitmap font, drawn
locally rather than generated, so it stays crisp at whole-number scales.

## Using the model in the game (iPhone / Mac)

`make export` writes `OmokNet.mlpackage` plus `model_meta.json` describing the
contract. Input is a `(1, 5, N, N)` float32 tensor, NCHW, always from the point
of view of the player to move:

| Plane | Meaning |
| --- | --- |
| 0 | stones of the player to move |
| 1 | stones of the opponent |
| 2 | one-hot of the opponent's last move |
| 3 | one-hot of the player-to-move's previous move |
| 4 | all ones when the player to move is black |

Outputs are `policy` — 225 probabilities indexed `row * board_size + col` — and
`value` in [-1, 1], where +1 means the side to move is winning.

```swift
let model = try OmokNet(configuration: MLModelConfiguration())
let planes = try MLMultiArray(shape: [1, 5, 15, 15], dataType: .float32)
// ... fill planes as described above ...
let out = try model.prediction(planes: planes)

// Mask occupied points before picking a move.
var best = -1, bestP: Float = -1
for i in 0..<225 where board[i] == .empty {
    let p = out.policy[i].floatValue
    if p > bestP { bestP = p; best = i }
}
```

Playing the raw policy argmax is already a decent opponent. For a stronger one,
run the same PUCT search as `omok/mcts.py` in Swift on top of the model — a few
hundred simulations per move is comfortable on a modern iPhone. Keep the value
sign convention: it is always from the side-to-move's perspective.

`make export FORMAT=npz` writes plain weights + metadata instead, if you would
rather write your own Metal/Accelerate inference.

## Code map

| File | Role |
| --- | --- |
| `omok/board.py` | rules: moves, win detection, overline handling |
| `omok/encode.py` | board → input planes, 8-fold dihedral symmetry |
| `omok/netspec.py` | backend-independent architecture + canonical weight names |
| `omok/backends/` | `torch_backend.py` (CUDA/MPS/CPU), `mlx_backend.py` (Apple) |
| `omok/mcts.py` | batched PUCT search |
| `omok/selfplay.py` | parallel game generation |
| `omok/replay.py` | shard files on disk + in-memory sampling |
| `omok/train.py` | optimisation loop, LR schedule, checkpoint cadence |
| `omok/arena.py` | candidate vs. best gating matches |
| `omok/pipeline.py` | the resumable orchestrator |
| `omok/checkpoint.py` | atomic, backend-portable checkpoints |
| `omok/export.py` | CoreML / ONNX / npz export |
| `omok/report.py` | model size, benchmarks, time estimates |
| `omok/engine.py` | the search on a worker thread, for interactive play |
| `omok/gui.py` | the Arcade window: board, side panel, input |
| `omok/effects.py` | easing curves, tweened values, the particle pool |
| `omok/assets/` | Grok-generated art (see `tools/make_assets.py`) |
| `omok/play.py` | terminal play + model loading shared with the GUI |
| `omok/cli.py` | the `python3 -m omok ...` command line |

The Rust core and the two clients that play the trained model:

| File | Role |
| --- | --- |
| `rust/crates/omok-core/src/board.rs` | the rules, ported from `omok/board.py` |
| `rust/crates/omok-core/src/encode.rs` | board → input planes |
| `rust/crates/omok-core/src/npz.rs` | reads the trainer's `.npz` checkpoints |
| `rust/crates/omok-core/src/mcts.rs` | PUCT search, batched across leaves |
| `rust/crates/omok-mlx/src/weights.rs` | batch-norm folding and layout fixes, done once |
| `rust/crates/omok-mlx/src/lib.rs` | the network on MLX |
| `rust/crates/omok-ffi/src/lib.rs` | the C ABI and the engine thread |
| `rust/include/omok.h` | the same interface as a C header |
| `lua/omok/init.lua` | LuaJIT bindings, shared by both clients |
| `lua/omok/term.lua` | raw mode, keys and buffered ANSI output |
| `lua/main.lua` | the terminal client |
| `lua/test.lua` | tests for the bindings and, through them, the core |
| `love2d/main.lua` | the LÖVE game |
| `love2d/effects.lua` | easing, tweens, particles, shockwaves, shake |
| `love2d/layout.lua` | the landscape / portrait layouts |
| `love2d/assets/` | Grok-generated pixel art (see `tools/make_love2d_assets.py`) |

Checkpoints are plain `.npz` files of PyTorch-named NCHW arrays, so a model
trained on a CUDA machine resumes on a Mac and vice versa — there is a test for
that (`test_torch_and_mlx_agree_on_the_same_weights`).

## Tests

```bash
make install-dev
make test        # 88 tests, ~8 seconds
make smoke       # full loop end to end on a 9x9 board

make ai-test     # 16 Rust tests: rules, encoding, checkpoint loading
make test-lua    # the core through its Lua bindings, including a full
                 # engine-against-engine game
```
