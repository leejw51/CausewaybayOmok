# ---------------------------------------------------------------------------
# Omok AI trainer -- `make help` lists everything.
#
# Common flow:   make install && make info && make train
# After a crash: make resume        (picks up from runs/<name>/state.json)
# ---------------------------------------------------------------------------

# Dependencies live in a conda environment that `make install` creates.
CONDA ?= conda
CONDA_ENV ?= omok
CONDA_BASE := $(shell $(CONDA) info --base 2>/dev/null)
ENV_PY := $(CONDA_BASE)/envs/$(CONDA_ENV)/bin/python

# A python that has torch (and mlx on Apple Silicon).  Defaults to the conda
# env above once it exists, so no `conda activate` is needed.  Override PY=...
PY ?= $(if $(wildcard $(ENV_PY)),$(ENV_PY),$(shell command -v python3.13 2>/dev/null || command -v python3))

# conda-forge ships a CUDA build as pytorch-gpu; on Apple Silicon the plain
# package is the Metal one, and mlx only exists on PyPI.
IS_MAC := $(filter Darwin-arm64,$(shell uname -s)-$(shell uname -m))
CONDA_TORCH ?= $(if $(IS_MAC),pytorch,pytorch-gpu)
ENV_PIP := $(CONDA) run -n $(CONDA_ENV) python -m pip install
INSTALL_MLX := $(if $(IS_MAC),$(ENV_PIP) "mlx>=0.20",true)
PRESET ?= base
RUN ?= runs/$(PRESET)
ITERS ?=
GAMES ?= 32
STEPS ?= 500
FORMAT ?= coreml
COLOR ?= black
SIMS ?=
OMOK := $(PY) -m omok
COMMON := --preset $(PRESET) --run-dir $(RUN)
PIDFILE := $(RUN)/train.pid
OUTLOG := $(RUN)/logs/train.out

.DEFAULT_GOAL := help
.PHONY: help env install install-dev install-export install-gui test smoke info bench \
        backends train resume train-bg stop tail status selfplay fit arena export \
        export-onnx export-npz play gui assets watch clear clean-run clean distclean \
        train-to train-fast train-average train-basic train-casual train-strong train-full \
        ai ai-test ai-bench tui game game-portrait love-assets love-sfx love-font love-icon \
        test-lua clean-ai version app package dist notarize gatekeeper clean-app

help: ## Show this help
	@echo "Omok trainer"
	@echo "  python : $(PY)"
	@echo "  preset : $(PRESET)   (tiny | blitz | small | fast | base | strong)"
	@echo "  conda  : $(CONDA_ENV)"
	@echo "  run dir: $(RUN)"
	@echo ""
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Variables (append VAR=value to any target):"
	@echo "  PRESET=$(PRESET)     tiny | blitz | small | fast | base | strong"
	@echo "  RUN=$(RUN)   where this run's data/checkpoints live"
	@echo "  ITERS=          how many MORE iterations to run, counted from the"
	@echo "                  iteration saved in state.json; empty = train to the"
	@echo "                  target's milestone (make train stops at iteration 200)."
	@echo "                  Ctrl-C / make stop is always safe: make resume continues."
	@echo "  GAMES=$(GAMES)        games for selfplay / arena"
	@echo "  STEPS=$(STEPS)       optimisation steps for fit"
	@echo "  SIMS=           MCTS simulations per move for arena / play"
	@echo "  PY=$(notdir $(PY))     python interpreter to use"
	@echo "  CONDA_ENV=$(CONDA_ENV)   conda env that make install builds and make train uses"
	@echo ""
	@echo "Examples:"
	@echo "  make train-fast                   # ~10 min: check the whole flow works"
	@echo "  make train-average                # ~30 min: a real but quick run"
	@echo "  make train-casual                 # train until it beats a casual human"
	@echo "  make train                        # ~3.5h: beats casual players every game"
	@echo "  make train-strong                 # ~8.5h: much stronger than make train"
	@echo "  make train-to TO=300              # train until iteration 300 (absolute)"
	@echo "  make train ITERS=200              # 200 MORE iterations, then stop"
	@echo "  make train PRESET=small ITERS=20  # quick run with the small net"
	@echo "  make train-bg ITERS=1000          # long run in the background"
	@echo "  make gui COLOR=white SIMS=320     # play the trained model in a window"
	@echo ""
	@echo "The Rust core and its two Lua clients:"
	@echo "  make ai                           # build the MLX engine as a shared library"
	@echo "  make tui                          # play in the terminal (LuaJIT)"
	@echo "  make game                         # play the LÖVE game"
	@echo "  make test-lua                     # test the core through its Lua bindings"
	@echo ""
	@echo "Shipping the game:"
	@echo "  make package                      # release build, signed .app, zip + SHA256SUMS in dist/"
	@echo "  make notarize                     # send it to Apple (needs APPLE_ID etc.), then make dist"
	@echo "  git tag v\$$(make version) && git push --tags   # CI builds, notarises, releases"

# ----------------------------------------------------------------- setup
env: ## Show interpreter and available compute backends
	@$(PY) -c "import sys; print('python', sys.version.split()[0], sys.executable)"
	@$(OMOK) backends

install: ## Create the conda env $(CONDA_ENV) with everything needed to train and play
	@$(CONDA) env list | awk '{print $$1}' | grep -qx '$(CONDA_ENV)' \
	  || $(CONDA) create -y -n $(CONDA_ENV) -c conda-forge python=3.13
	$(CONDA) install -y -n $(CONDA_ENV) -c conda-forge numpy $(CONDA_TORCH)
	@$(INSTALL_MLX)
	$(ENV_PIP) "arcade>=3.0"
	@$(MAKE) --no-print-directory env

# The extras go on top of the conda env; they deliberately do NOT reinstall
# torch from PyPI, which would shadow the CUDA build conda just placed.
install-dev: ## Install test dependencies into $(CONDA_ENV)
	$(CONDA) install -y -n $(CONDA_ENV) -c conda-forge pytest

install-export: ## Install coremltools (needed only for `make export`)
	$(ENV_PIP) -r requirements-export.txt

install-gui: ## Reinstall just the GUI dependency (`make install` already includes it)
	$(ENV_PIP) "arcade>=3.0"

assets: ## Regenerate the GUI art with Grok (needs XAI_API_KEY; art is committed)
	$(PY) tools/make_assets.py $(if $(FORCE),--force,)

# ----------------------------------------------------------------- rust core
# MLX compiles its own Metal kernels, and the `metal` compiler ships with Xcode
# rather than with the Command Line Tools.  Pointing DEVELOPER_DIR at Xcode is
# enough -- unlike `xcode-select -s`, it needs no sudo and changes nothing
# outside this build.
XCODE ?= /Applications/Xcode.app/Contents/Developer
CARGO_ENV := $(if $(wildcard $(XCODE)),DEVELOPER_DIR=$(XCODE),)
CARGO ?= cargo
LIB := rust/target/release/libomok_ai.dylib
LUAJIT ?= luajit
LOVE ?= love

ai: ## Build the Rust/MLX AI core as a shared library for the Lua clients
	cd rust && $(CARGO_ENV) $(CARGO) build --release
	@echo "built $(LIB)"

ai-test: ## Run the Rust unit tests (rules, encoding, checkpoint loading)
	cd rust && $(CARGO_ENV) $(CARGO) test

ai-bench: ai ## Load the trained model in Rust and time a few searches
	cd rust && $(CARGO_ENV) $(CARGO) run --release -p omok-mlx --example bench -- \
	  ../$(RUN)/checkpoints/best.npz

test-lua: ai ## Test the AI core through its LuaJIT bindings
	$(LUAJIT) lua/test.lua

tui: ai ## Play the trained model in the terminal (LEVEL=1-5, COLOR=black|white)
	$(LUAJIT) lua/main.lua $(if $(LEVEL),--level $(LEVEL),) \
	  $(if $(filter white,$(COLOR)),--white,)

game: ai ## Play "Causewaybay Omok" in a window (LÖVE)
	$(LOVE) love2d

game-portrait: ai ## The same game started in a portrait window
	OMOK_WINDOW=820x1180 $(LOVE) love2d

love-assets: ## Regenerate the game's pixel art with Grok (needs XAI_API_KEY)
	$(PY) tools/make_love2d_assets.py $(if $(FORCE),--force,)

love-sfx: ## Regenerate the game's 8-bit sound effects (no API key, no samples)
	$(PY) tools/make_love2d_sfx.py $(SOUND)

love-font: ## Redraw the game's bitmap font (no API key, no dependencies)
	$(PY) tools/make_love2d_font.py

clean-ai: ## Remove the Rust build directory
	rm -rf rust/target

# ----------------------------------------------------------------- shipping the game
# The recipe lives in love2d/Makefile so that `make app` on a laptop and a
# tagged release in CI do the same thing.  A tag `v<version>` is what starts
# the release workflow, and it refuses a tag that disagrees with `make version`.
RS_VERSION := $(shell sed -n 's/^version = "\(.*\)"/\1/p' rust/Cargo.toml | head -1)
PY_VERSION := $(shell sed -n 's/^__version__ = "\(.*\)"/\1/p' omok/__init__.py | head -1)

version: ## Print the version rust/Cargo.toml and omok/__init__.py agree on
	@test -n "$(RS_VERSION)" || { echo "ERROR: no version in rust/Cargo.toml" >&2; exit 1; }
	@test "$(RS_VERSION)" = "$(PY_VERSION)" || { \
		echo "ERROR: rust/Cargo.toml is $(RS_VERSION), omok/__init__.py is $(PY_VERSION)" >&2; \
		exit 1; }
	@echo "$(RS_VERSION)"

app: ## Build a double-clickable macOS .app of the LÖVE game (love2d/build)
	@$(MAKE) --no-print-directory -C love2d app

# What a release is made of: the release core (`make ai`, via love2d's `app`),
# the signed bundle, and the zip under the name the GitHub release uses.
# `make notarize` re-zips the stapled bundle, so `make dist` alone refreshes
# dist/ afterwards without building anything again.
DIST := dist
ARCH := $(shell uname -m)
PACKAGE_ZIP := $(DIST)/CausewaybayOmok-$(RS_VERSION)-$(ARCH)-apple-darwin.zip

package: version app dist ## Build the release core and the signed .app; zip + SHA256SUMS in dist/

dist: ## Copy the built .app zip into dist/ under its release name, with SHA256SUMS
	@test -f love2d/build/CausewaybayOmok-macos.zip \
	  || { echo "no love2d/build/CausewaybayOmok-macos.zip -- run 'make package'" >&2; exit 1; }
	@mkdir -p $(DIST)
	@cp -f love2d/build/CausewaybayOmok-macos.zip $(PACKAGE_ZIP)
	@cd $(DIST) && shasum -a 256 *.zip > SHA256SUMS
	@echo "  $(PACKAGE_ZIP)"
	@cat $(DIST)/SHA256SUMS | sed 's/^/    /'

notarize: ## Notarize and staple the .app (needs APPLE_ID, APPLE_PASSWORD, APPLE_TEAM_ID)
	@$(MAKE) --no-print-directory -C love2d notarize

gatekeeper: ## Assess the built .app the way Finder will
	@$(MAKE) --no-print-directory -C love2d gatekeeper

love-icon: ## Redraw the app icon from the game's art (no API key, no dependencies)
	$(PY) tools/make_love2d_icon.py

clean-app: ## Remove the built .app, everything staged for it, and dist/
	@$(MAKE) --no-print-directory -C love2d clean
	rm -rf $(DIST)

# ----------------------------------------------------------------- checks
test: ## Run the test suite
	$(PY) -m pytest tests -q

smoke: ## 30-second end-to-end run on a 9x9 board (sanity check)
	rm -rf runs/smoke
	$(OMOK) train --preset tiny --run-dir runs/smoke --no-bench
	$(OMOK) export --preset tiny --run-dir runs/smoke --format npz

# ----------------------------------------------------------------- info
info: ## Model size, backend, measured speed, estimated training time, file paths
	$(OMOK) info $(COMMON)

bench: ## Measure throughput only
	$(OMOK) bench $(COMMON)

backends: ## Which compute backends this machine has
	$(OMOK) backends

status: ## What is on disk for this run (iteration, data, checkpoints)
	$(OMOK) status $(COMMON)

# ----------------------------------------------------------------- training
# `make train` is a strength milestone: iteration 200 is well past the casual
# level (iteration 60) and reliably beats human players in a normal game.
# ITERS= keeps the old additive behaviour (N more iterations, then stop).
train: ## Train to iteration 200 (~3.5h on RTX 5070): beats casual players every game
ifeq ($(ITERS),)
	@$(MAKE) --no-print-directory train-to TO=200
else
	$(OMOK) train $(COMMON) --iterations $(ITERS)
endif

resume: ## Continue this run to the target it was started with (ITERS= to add more)
	$(OMOK) train $(COMMON) $(if $(ITERS),--iterations $(ITERS),)

# Milestone targets train UP TO an absolute iteration (idempotent: already
# past it means nothing to do), unlike ITERS= which always adds more.
train-to: ## Train until iteration TO (absolute), e.g. make train-to TO=200
	@if [ -z "$(TO)" ]; then echo "usage: make train-to TO=<iteration>"; exit 2; fi
	@cur=$$($(PY) -c "import json,sys; \
	  print(json.load(open('$(RUN)/state.json')).get('iteration', 0))" 2>/dev/null || echo 0); \
	rem=$$(( $(TO) - cur )); \
	if [ $$rem -le 0 ]; then echo "already at iteration $$cur (>= $(TO)) -- nothing to do"; \
	else echo "iteration $$cur -> $(TO) ($$rem to go)"; \
	  $(OMOK) train $(COMMON) --iterations $$rem; fi

# These two are sized by wall-clock budget rather than by strength, and use
# the cheaper presets to get there.  Iteration counts come from measured
# per-iteration times on an RTX 5070, not from the `make info` estimate.
train-fast: ## ~10 minutes (blitz preset): end-to-end check that the whole flow works
	@$(MAKE) --no-print-directory train-to TO=30 PRESET=blitz

train-average: ## ~30 minutes (fast preset): a real but quick run
	@$(MAKE) --no-print-directory train-to TO=45 PRESET=fast

train-basic: ## Train to iteration 10 (~10m on RTX 5070): plays legally, blocks threats
	@$(MAKE) --no-print-directory train-to TO=10

train-casual: ## Train to iteration 60 (~1h on RTX 5070): a real game of omok
	@$(MAKE) --no-print-directory train-to TO=60

train-strong: ## Train to iteration 500 (~8.5h on RTX 5070): much stronger than make train
	@$(MAKE) --no-print-directory train-to TO=500

train-full: ## Train to iteration 1000 (~17h on RTX 5070): the full run
	@$(MAKE) --no-print-directory train-to TO=1000

train-bg: ## Train in the background (ITERS= or the run's saved target), logging to $(OUTLOG)
	@mkdir -p $(RUN)/logs
	@if [ -f $(PIDFILE) ] && kill -0 $$(cat $(PIDFILE)) 2>/dev/null; then \
	  echo "already training (pid $$(cat $(PIDFILE))) -- make stop first"; exit 1; fi
	@nohup $(OMOK) train $(COMMON) $(if $(ITERS),--iterations $(ITERS),) \
	  >> $(OUTLOG) 2>&1 & echo $$! > $(PIDFILE)
	@echo "training started (pid $$(cat $(PIDFILE))) -- log: $(OUTLOG)"

stop: ## Stop a background run cleanly (it checkpoints before exiting)
	@if [ -f $(PIDFILE) ]; then \
	  kill -TERM $$(cat $(PIDFILE)) 2>/dev/null && echo "asked pid $$(cat $(PIDFILE)) to stop"; \
	  rm -f $(PIDFILE); \
	else echo "no pidfile at $(PIDFILE)"; fi

tail: ## Follow the background training log
	@tail -f $(OUTLOG)

watch: ## Print the run status every 30 seconds
	@while true; do clear; $(MAKE) --no-print-directory status; sleep 30; done

# ----------------------------------------------------------------- phases
selfplay: ## Generate GAMES self-play games with the current best model
	$(OMOK) selfplay $(COMMON) --games $(GAMES)

fit: ## Train STEPS optimisation steps on the data already on disk
	$(OMOK) fit $(COMMON) --steps $(STEPS)

arena: ## Play the latest checkpoint against the current best model
	$(OMOK) arena $(COMMON) --games $(GAMES) $(if $(SIMS),--simulations $(SIMS),)

# ----------------------------------------------------------------- shipping
export: ## Export the best model to CoreML for the iPhone / Mac game
	$(OMOK) export $(COMMON) --format $(FORMAT)

export-onnx: ## Export the best model to ONNX
	$(OMOK) export $(COMMON) --format onnx

export-npz: ## Export raw weights + metadata (no coremltools needed)
	$(OMOK) export $(COMMON) --format npz

play: ## Play against the trained model in the terminal (COLOR=black|white)
	$(OMOK) play $(COMMON) --color $(COLOR) $(if $(SIMS),--simulations $(SIMS),)

gui: ## Play against the trained model in a window (COLOR=black|white|none|both)
	$(OMOK) gui $(COMMON) --color $(COLOR) $(if $(SIMS),--simulations $(SIMS),)

# ----------------------------------------------------------------- cleaning
clear: ## Remove the trained AI model and its data ($(RUN)) for a fresh start
	rm -rf $(RUN)
	@echo "removed $(RUN) -- the next make train starts from scratch"

clean-run: ## Delete this run's directory ($(RUN)) -- data and checkpoints included
	rm -rf $(RUN)

clean: ## Remove caches and smoke-test output
	rm -rf runs/smoke .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

distclean: clean ## Also delete every run under runs/
	rm -rf runs
