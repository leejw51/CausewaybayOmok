# ---------------------------------------------------------------------------
# Omok AI trainer -- `make help` lists everything.
#
# Common flow:   make install && make info && make train
# After a crash: make resume        (picks up from runs/<name>/state.json)
# ---------------------------------------------------------------------------

# A python that has torch (and mlx on Apple Silicon).  Override with PY=...
PY ?= $(shell command -v python3.13 2>/dev/null || command -v python3)
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
        export-onnx export-npz play gui assets watch clean-run clean distclean \
        train-to train-basic train-casual train-strong train-full

help: ## Show this help
	@echo "Omok trainer"
	@echo "  python : $(PY)"
	@echo "  preset : $(PRESET)   (tiny | small | base | strong)"
	@echo "  run dir: $(RUN)"
	@echo ""
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Variables (append VAR=value to any target):"
	@echo "  PRESET=$(PRESET)     tiny | small | base | strong"
	@echo "  RUN=$(RUN)   where this run's data/checkpoints live"
	@echo "  ITERS=          how many MORE iterations to run, counted from the"
	@echo "                  iteration saved in state.json; empty = preset default."
	@echo "                  Ctrl-C / make stop is always safe: make resume continues."
	@echo "  GAMES=$(GAMES)        games for selfplay / arena"
	@echo "  STEPS=$(STEPS)       optimisation steps for fit"
	@echo "  SIMS=           MCTS simulations per move for arena / play"
	@echo "  PY=$(notdir $(PY))     python interpreter to use"
	@echo ""
	@echo "Examples:"
	@echo "  make train-casual                 # train until it beats a casual human"
	@echo "  make train-to TO=300              # train until iteration 300 (absolute)"
	@echo "  make train ITERS=200              # 200 MORE iterations, then stop"
	@echo "  make train PRESET=small ITERS=20  # quick run with the small net"
	@echo "  make train-bg ITERS=1000          # long run in the background"
	@echo "  make gui COLOR=white SIMS=320     # play the trained model in a window"

# ----------------------------------------------------------------- setup
env: ## Show interpreter and available compute backends
	@$(PY) -c "import sys; print('python', sys.version.split()[0], sys.executable)"
	@$(OMOK) backends

install: ## Install runtime dependencies (torch, numpy, mlx on Apple Silicon)
	$(PY) -m pip install -r requirements.txt

install-dev: ## Install runtime + test dependencies
	$(PY) -m pip install -r requirements-dev.txt

install-export: ## Install coremltools (needed only for `make export`)
	$(PY) -m pip install -r requirements-export.txt

install-gui: ## Install arcade (needed only for `make gui`)
	$(PY) -m pip install -r requirements-gui.txt

assets: ## Regenerate the GUI art with Grok (needs XAI_API_KEY; art is committed)
	$(PY) tools/make_assets.py $(if $(FORCE),--force,)

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
train: ## Run the self-play -> train -> gate loop (resumes automatically)
	$(OMOK) train $(COMMON) $(if $(ITERS),--iterations $(ITERS),)

resume: train ## Alias for train: every run continues from the saved state

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

train-basic: ## Train to iteration 10 (~35m on M4 Max): plays legally, blocks threats
	@$(MAKE) --no-print-directory train-to TO=10

train-casual: ## Train to iteration 60 (~3.5h): beats a casual human
	@$(MAKE) --no-print-directory train-to TO=60

train-strong: ## Train to iteration 200 (~12h): hard to beat without study
	@$(MAKE) --no-print-directory train-to TO=200

train-full: ## Train to iteration 1000 (~60h): the full run
	@$(MAKE) --no-print-directory train-to TO=1000

train-bg: ## Same as `train` but in the background, logging to $(OUTLOG)
	@mkdir -p $(RUN)/logs
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
clean-run: ## Delete this run's directory ($(RUN)) -- data and checkpoints included
	rm -rf $(RUN)

clean: ## Remove caches and smoke-test output
	rm -rf runs/smoke .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

distclean: clean ## Also delete every run under runs/
	rm -rf runs
