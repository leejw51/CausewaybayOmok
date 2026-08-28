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
game, `1`-`5` difficulty, `TAB` to swap colours, `W` to watch, `V` to turn the
board, `S` for the text size, `M` to mute and `F11` for fullscreen. Every key
has a button in the panel beside it.

The game opens fullscreen — it is a board that wants every pixel it can have
and an interface drawn at console sizes, and neither is served by a window
somebody has to enlarge first. `F11`, or the button that is on screen from the
first frame, comes back out.

The last row of the panel is the window rather than the game: **TALL/WIDE**
turns the board, **FULL/WINDOW** fills the screen, and **SIZE** steps the whole
interface through four sizes. The first two are plain toggles — press, and it
stays pressed, through a drag, a fullscreen and the next time the game opens.
The size is a preference rather than an instruction: a window too small to hold
the size asked for gets the largest one it *can* hold, and the one asked for
comes back on its own when the window grows.

### What it remembers, and where

```
~/.causewaybayomok/settings.jsonl
```

Which way up the board is, how big the text is, whether the window fills the
screen, and whether the sound is on. Four values somebody might reasonably want
to look at, edit or delete — which is the argument against LÖVE's own save
directory, a path nobody can guess buried under `~/Library/Application Support`
on a Mac.

JSON Lines, and a line is only ever appended; the file is read from the top with
each line overwriting the keys it names, so the last word on any setting wins:

```json
{"orientation":"landscape","text_size":2,"window":"full"}
{"text_size":3}
{"sound":"off"}
```

Worth the small strangeness of keeping four values in a growing file for two
reasons. Appending never reads first, so two things changing a setting cannot
lose each other's write the way a read-modify-write pair can. And a write cut
off half way through damages one line at the end rather than the whole file —
the reader takes only lines that are complete objects, so a torn record costs
the setting it was carrying and nothing else. It is compacted back to a single
line once it gets long. If the home directory cannot be written to, the game
opens at its defaults, which is the right way for a preference to fail.

The window is resizable and the layout follows it: the panel sits beside the
board in landscape and below it in portrait, and the board is always the largest
whole-pixel grid that leaves the panel its room. `V` chooses the arrangement
without dragging anything — it turns the window over when there is one, and on a
fullscreen 16:9 display, where there is no window to reshape, it still stacks the
panel under the board, which is the only way to give the board every pixel of the
height.

The panel is measured in characters rather than in pixels, so the text size
moves everything in step: turn it up and the panel widens, the buttons and bars
grow with it and the board takes what is left. `layout.fit` is where that
settles — it steps the size down until the panel can hold the blocks that cannot
stand down, which is why the controls never fall off the bottom of it.

Everything that moves goes through `love2d/effects.lua` — the Penner easing
curves, a tween type, particle bursts and trails, expanding shockwaves, floating
text, decaying screen shake and the ambient motes drifting over the square.

A stone is thrown at its point rather than faded onto it. It comes in from off
the board as a comet — three ghosts of the sprite strung out behind it, read off
the same curve a few hundredths of a second earlier so they string out while it
travels and pile into it as it slows, over a trail of sparks laid along the
ground it actually covered rather than one puff a frame — and lands under two
shockwaves, a directional spray thrown on down the line it came in on, and a
kick of screen shake. Which edge it arrives from is the palette's own division:
indigo falls out of the night sky, amber rises from the lamplight, which is the
same thing the two stone sounds say a fourth apart.

The landing is the moment everything else hangs off. The rules are settled when
the move is played, but the fountain and the fanfare wait for the stone that
caused them, and so does the model: it holds its reply until the board is still,
because at 96 simulations the search comes back in a hundredth of a second and a
watched game otherwise answers a stone that is still crossing the screen.

After that the last move keeps a pulsing halo, the winning line lights up one
stone at a time under a fountain of particles, the board dims under a slow sweep
while the model thinks, and the backdrop drifts against the pointer.

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
shows as haloes around the lamps.

### The font

```bash
make love-font
```

`tools/make_love2d_font.py` draws all ninety-five glyphs by hand on a 7×9 grid,
as `#` and `.` in a table, and writes the atlas with nothing but the standard
library — a PNG is a zlib stream with a length and a checksum around it, and a
font that needs Pillow installed to edit one letter is a font nobody edits.

It used to be Pillow's built-in default, which is a fine terminal face and the
wrong thing entirely: light, narrow, and drawn for a nine-pixel line of code
rather than for a menu on a console. A UI font from this era is not a typeface
that happens to be small — it is designed on the grid. Every stem here is one
pixel, so scaled up it is a solid two- or three-pixel bar; the alpha is 0 or 255
and nothing between, so it stays hard-edged under the nearest-neighbour filter;
capitals fill seven of the nine rows with the last two left for descenders; and
every curve is a staircase drawn as one, because at three times the size an
attempt at a smooth arc is what looks wrong.

The interface is set in capitals throughout, which is what these machines did
and is not only a style choice: a 7×9 cell has no room for the ascenders and
descenders that make mixed case worth having at this size.

Around it the panel is a console menu of about 1991 — a flat fill, a hard black
edge, and a one-pixel bevel that is light along the top and left and dark along
the bottom and right, which is the entire 3D effect. Headings are knocked out of
filled bars, buttons invert when they are switched on rather than growing a
third rendering for the pointer, the evaluation is sixteen lamps filling out
from the middle rather than a smooth bar the hardware could not have drawn, and
a scanline every fourth row takes the flatness off a large area of one colour.

### The sound

Fourteen effects, synthesised rather than downloaded:

```bash
make love-sfx                       # all of them
make love-sfx SOUND="indigo win"    # just these two
```

`tools/make_love2d_sfx.py` is a PSG in a hundred lines of arithmetic — two
square waves, a stepped triangle, the NES's 15-bit noise register and a
four-bit volume — so a stone's clack is a number in that file rather than a
waveform in a binary nobody can open. There is no sample pack, no licence to
honour and nothing to download; the whole set is 105 KB of 8-bit mono WAV and is
committed, so `make love-sfx` only needs running to *change* a sound.

Three constraints keep it a chip rather than a soft synth, and all three are
deliberate: the envelopes are quantised to sixteen levels, because that
stair-step on a decay tail is most of the sound; the waveforms are the ones the
hardware had, with no sine anywhere; and the noise really is a shift register
clocked by a divider, which is why the stones sound like wood and not like
static. Indigo and amber get the same stone a fourth apart — one sound per side
rather than one per player, so a move is heard as one colour or the other
without looking up from the board, and watch mode needs no third sound.

`love2d/sound.lua` is the part that plays them, and it exists because playing a
sound naively goes wrong three ways: a Source is one voice and re-playing it
restarts it, so every effect keeps a small pool of clones; the game fires far
more often than an ear wants, so every effect has a minimum gap and the ones
that fire most have the longest; and the same sample twice running sounds like a
stuck machine, so the ones that repeat get a few percent of random detune. None
of it is load-bearing — a checkout without `assets/sfx` and a machine with no
audio device both play on in silence, and the game says so once at startup.

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
| `love2d/sound.lua` | voice pools, throttles and detune for the effects |
| `love2d/store.lua` | the preferences, as a JSON Lines log in `~/.causewaybayomok` |
| `love2d/assets/font.png` | the bitmap font (see `tools/make_love2d_font.py`) |
| `love2d/assets/` | Grok-generated pixel art (see `tools/make_love2d_assets.py`) |
| `love2d/assets/sfx/` | 8-bit sound effects (see `tools/make_love2d_sfx.py`) |

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
