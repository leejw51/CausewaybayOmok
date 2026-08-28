# Causewaybay Omok — packaging the LÖVE game as a macOS app.
#
#   make -C love2d app         build build/CausewaybayOmok.app (and its zip)
#   make -C love2d notarize    send it to Apple and staple the ticket
#   make -C love2d gatekeeper  assess it the way Finder does
#
# Or from the repository root: `make app`, `make notarize`.  Running the game
# from a checkout stays `make game` in the root Makefile; this file is only
# about producing something a person without a checkout can double-click.
#
# What goes in the bundle, and why each piece is where it is:
#
#   Contents/Resources/game.love     the game and the Lua binding, zipped.  LÖVE
#                                    finds a .love in Resources and runs fused.
#   Contents/Frameworks/OmokAI.framework
#                                    the Rust/MLX core as a framework:
#                                    Versions/A/OmokAI is the dylib and
#                                    Versions/A/Resources/mlx.metallib the
#                                    Metal kernels.  The dylib has the path of
#                                    the build tree's metallib compiled in,
#                                    which exists on the machine that built it
#                                    and nowhere else; MLX then looks beside
#                                    the dylib and in `Resources/` next to it,
#                                    which is a framework's layout.  A framework
#                                    rather than two loose files because
#                                    codesign treats every file in Frameworks
#                                    as code, and an 88 MB metallib is not.
#   Contents/Resources/model/best.npz (+ best.json)
#                                    the trained network.  `lua/omok/init.lua`
#                                    looks here first when the game is fused.
#
# The binding is copied *into* the archive rather than referenced at ../lua,
# because a bundle that needs a checkout beside it is not a bundle.

LOVE         ?= love
LOVE_VERSION ?= 11.5
LOVE_URL     := https://github.com/love2d/love/releases/download/$(LOVE_VERSION)/love-$(LOVE_VERSION)-macos.zip

APP_NAME     := CausewaybayOmok
DISPLAY_NAME := Causewaybay Omok
BUNDLE_ID    := com.causewaybay.omok

ROOT     := $(abspath $(CURDIR)/..)
RUST_DIR := $(ROOT)/rust
LUA_DIR  := $(ROOT)/lua
LIB      := $(RUST_DIR)/target/release/libomok_ai.dylib
FW_NAME  := OmokAI
# The shipped model.  MODEL=path/to/best.npz packages another; its sidecar
# .json is taken from beside it when there is one.
MODEL    ?= $(ROOT)/runs/base/checkpoints/best.npz
VERSION  := $(shell sed -n 's/^version = "\(.*\)"/\1/p' $(RUST_DIR)/Cargo.toml | head -1)

BUILD := build
APP   := $(BUILD)/$(APP_NAME).app
ZIP   := $(BUILD)/$(APP_NAME)-macos.zip

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help
	@echo "Causewaybay Omok — macOS app ($(VERSION))"
	@echo
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "  MODEL=$(MODEL)"
	@echo "  SKIP_SMOKE=1 skips starting the bundle (for a machine with no display)."
	@echo

# ---------------------------------------------------------------------- build

.PHONY: build
build: ## Build the release shared library the game loads
	@$(MAKE) --no-print-directory -C $(ROOT) ai

# The game as one archive: every .lua beside main.lua, the art and sounds, and
# the binding from ../lua/omok so `require("omok")` resolves inside it.
.PHONY: archive
archive:
	@rm -rf "$(BUILD)/stage" "$(BUILD)/game.love" && mkdir -p "$(BUILD)/stage"
	@cp -R *.lua assets "$(BUILD)/stage/"
	@cp -R "$(LUA_DIR)/omok" "$(BUILD)/stage/"
	@cd "$(BUILD)/stage" && zip -qr9 "../game.love" . -x '.DS_Store'
	@rm -rf "$(BUILD)/stage"
	@$(MAKE) --no-print-directory verify-archive ARCHIVE="$(BUILD)/game.love"

# Every `.lua` beside main.lua has to be in the archive, and so does the
# binding.  A missing module is not a subtle failure — the bundle dies on the
# first `require` — but it dies on *somebody else's* machine, because the
# checkout it was built from has the file right there.
.PHONY: verify-archive
verify-archive:
	@missing=""; \
	for file in *.lua omok/init.lua omok/term.lua; do \
		unzip -l "$(ARCHIVE)" 2>/dev/null | grep -qE " $$file$$" || missing="$$missing $$file"; \
	done; \
	if [ -n "$$missing" ]; then \
		echo "  the archive is missing:$$missing" >&2; \
		exit 1; \
	fi; \
	echo "  archive carries every module ($$(ls *.lua | wc -l | tr -d ' ') files + omok/)"

# ------------------------------------------------------------------ macOS app

# A real double-clickable .app, with LÖVE inside it.
#
# Two things about the signing are not optional and both are LuaJIT's fault:
# the hardened runtime needs `allow-jit` and `allow-unsigned-executable-memory`
# or the app dies the moment LuaJIT compiles anything, and
# `disable-library-validation` because the core's .dylib is not signed by
# whoever signed LÖVE, and without it dlopen refuses to load it.
#
# Signing uses APPLE_SIGNING_IDENTITY when set (the release workflow exports
# it), then a Developer ID from the keychain, then ad-hoc — which is enough to
# run locally and not enough to hand to somebody else without a right-click.
.PHONY: app
app: build ## Build a double-clickable macOS .app with LÖVE embedded
	@test "$$(uname -s)" = "Darwin" || { echo "  'make app' is macOS only." >&2; exit 1; }
	@test -f "$(MODEL)" || { echo "  no model at $(MODEL); run 'make train' or set MODEL=" >&2; exit 1; }
	@rm -rf "$(APP)" "$(BUILD)/love-macos" "$(ZIP)"
	@mkdir -p "$(BUILD)"
	@$(MAKE) --no-print-directory archive

	@echo "  fetching LÖVE $(LOVE_VERSION)"
	@test -f "$(BUILD)/love-$(LOVE_VERSION)-macos.zip" \
		|| curl -fsSL -o "$(BUILD)/love-$(LOVE_VERSION)-macos.zip" "$(LOVE_URL)"
	@unzip -q -o "$(BUILD)/love-$(LOVE_VERSION)-macos.zip" -d "$(BUILD)/love-macos"
	@mv "$(BUILD)/love-macos/love.app" "$(APP)"

	@cp "$(BUILD)/game.love" "$(APP)/Contents/Resources/game.love"
	@$(MAKE) --no-print-directory framework FRAMEWORK="$(APP)/Contents/Frameworks/$(FW_NAME).framework"
	@mkdir -p "$(APP)/Contents/Resources/model"
	@cp -f "$(MODEL)" "$(APP)/Contents/Resources/model/best.npz"
	@if [ -f "$(MODEL:.npz=.json)" ]; then \
		cp -f "$(MODEL:.npz=.json)" "$(APP)/Contents/Resources/model/best.json"; fi

	@# Without these the bundle is still called "LÖVE" everywhere macOS shows a
	@# name, and two LÖVE games installed at once collide on the bundle id.
	@plutil -replace CFBundleName -string "$(DISPLAY_NAME)" "$(APP)/Contents/Info.plist"
	@plutil -replace CFBundleDisplayName -string "$(DISPLAY_NAME)" "$(APP)/Contents/Info.plist"
	@plutil -replace CFBundleIdentifier -string "$(BUNDLE_ID)" "$(APP)/Contents/Info.plist"
	@plutil -replace CFBundleShortVersionString -string "$(VERSION)" "$(APP)/Contents/Info.plist"
	@plutil -replace CFBundleVersion -string "$(VERSION)" "$(APP)/Contents/Info.plist"
	@# LÖVE ships as an editor for .love documents.  This bundle *is* one game
	@# and must not open somebody else's.
	@plutil -remove CFBundleDocumentTypes "$(APP)/Contents/Info.plist" 2>/dev/null || true
	@plutil -remove UTExportedTypeDeclarations "$(APP)/Contents/Info.plist" 2>/dev/null || true
	@plutil -replace CFBundleExecutable -string "love" "$(APP)/Contents/Info.plist"
	@plutil -replace NSHighResolutionCapable -bool true "$(APP)/Contents/Info.plist"

	@$(MAKE) --no-print-directory app-icon
	@$(MAKE) --no-print-directory app-sign

	@# Skipped where there is no window server to draw into — a CI runner has
	@# no display.  `verify-archive` still ran there and catches a module left
	@# out of the staging; what it cannot catch is a library or model the
	@# bundle fails to find, which is what this checks.
	@if [ -n "$(SKIP_SMOKE)" ]; then \
		echo "  smoke test skipped (SKIP_SMOKE set)"; \
	else \
		$(MAKE) --no-print-directory app-smoke; \
	fi

	@cd "$(BUILD)" && ditto -c -k --sequesterRsrc --keepParent "$(APP_NAME).app" "$(APP_NAME)-macos.zip"
	@rm -rf "$(BUILD)/love-macos" "$(BUILD)/game.love"
	@echo
	@echo "  $(APP)"
	@du -sh "$(APP)" "$(ZIP)" | sed 's/^/    /'
	@echo
	@echo "  Open it with: open \"$(APP)\""

# The core as a framework, in the layout `codesign` and MLX both expect.
#
#   OmokAI.framework/Versions/A/OmokAI                 the dylib
#   OmokAI.framework/Versions/A/Resources/Info.plist   what makes it a bundle
#   OmokAI.framework/Versions/A/Resources/mlx.metallib the Metal kernels
#   OmokAI.framework/Versions/Current -> A, and the two top-level symlinks
#
# The metallib is the newest one under rust/target: cargo names the build
# directory after a hash of the crate's inputs, so there can be several.
.PHONY: framework
framework:
	@test -n "$(FRAMEWORK)" || { echo "  FRAMEWORK= is required" >&2; exit 1; }
	@metallib=$$(ls -t "$(RUST_DIR)"/target/release/build/mlx-sys-*/out/build/lib/mlx.metallib 2>/dev/null | head -1); \
	test -n "$$metallib" || { echo "  no mlx.metallib under rust/target; was the core built with Metal?" >&2; exit 1; }; \
	rm -rf "$(FRAMEWORK)"; \
	mkdir -p "$(FRAMEWORK)/Versions/A/Resources"; \
	cp -f "$(LIB)" "$(FRAMEWORK)/Versions/A/$(FW_NAME)"; \
	install_name_tool -id "@rpath/$(FW_NAME).framework/Versions/A/$(FW_NAME)" "$(FRAMEWORK)/Versions/A/$(FW_NAME)"; \
	cp -f "$$metallib" "$(FRAMEWORK)/Versions/A/Resources/mlx.metallib"; \
	printf '%s\n' \
		'<?xml version="1.0" encoding="UTF-8"?>' \
		'<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">' \
		'<plist version="1.0"><dict>' \
		'  <key>CFBundleDevelopmentRegion</key><string>en</string>' \
		'  <key>CFBundleExecutable</key><string>$(FW_NAME)</string>' \
		'  <key>CFBundleIdentifier</key><string>$(BUNDLE_ID).ai</string>' \
		'  <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>' \
		'  <key>CFBundleName</key><string>$(FW_NAME)</string>' \
		'  <key>CFBundlePackageType</key><string>FMWK</string>' \
		'  <key>CFBundleShortVersionString</key><string>$(VERSION)</string>' \
		'  <key>CFBundleVersion</key><string>$(VERSION)</string>' \
		'  <key>LSMinimumSystemVersion</key><string>14.0</string>' \
		'</dict></plist>' \
		> "$(FRAMEWORK)/Versions/A/Resources/Info.plist"; \
	ln -s A "$(FRAMEWORK)/Versions/Current"; \
	ln -s "Versions/Current/$(FW_NAME)" "$(FRAMEWORK)/$(FW_NAME)"; \
	ln -s "Versions/Current/Resources" "$(FRAMEWORK)/Resources"; \
	echo "  $(FW_NAME).framework: $$(du -sh "$(FRAMEWORK)" | cut -f1) (kernels $$(du -h "$$metallib" | cut -f1))"

# The icon, from love2d/icon.png (drawn by tools/make_love2d_icon.py from the
# game's own art).  `sips` downscales cleanly; the source is the largest size.
.PHONY: app-icon
app-icon:
	@test -f icon.png || { echo "  no icon.png; keeping LÖVE's icon (make love-icon draws one)"; exit 0; }
	@rm -rf "$(BUILD)/icon.iconset" && mkdir -p "$(BUILD)/icon.iconset"
	@for size in 16 32 64 128 256 512 1024; do \
		sips -s format png -z $$size $$size icon.png \
			--out "$(BUILD)/icon.iconset/icon_$${size}x$${size}.png" > /dev/null 2>&1; \
	done
	@for size in 16 32 128 256 512; do \
		double=$$((size * 2)); \
		cp "$(BUILD)/icon.iconset/icon_$${double}x$${double}.png" \
			"$(BUILD)/icon.iconset/icon_$${size}x$${size}@2x.png" 2>/dev/null || true; \
	done
	@rm -f "$(BUILD)/icon.iconset/icon_1024x1024.png"
	@iconutil -c icns "$(BUILD)/icon.iconset" -o "$(BUILD)/icon.icns"
	@cp "$(BUILD)/icon.icns" "$(APP)/Contents/Resources/OS X AppIcon.icns" 2>/dev/null || true
	@cp "$(BUILD)/icon.icns" "$(APP)/Contents/Resources/Love.icns" 2>/dev/null || true
	@rm -rf "$(BUILD)/icon.iconset" "$(BUILD)/icon.icns"
	@echo "  icon from icon.png"

# Signed inside-out: nested binaries first, then the executable, then the
# bundle.  Signing the outer bundle first invalidates itself the moment
# anything inside it is signed afterwards.
.PHONY: app-sign
app-sign:
	@test -d "$(APP)" || { echo "  no $(APP); run 'make app'" >&2; exit 1; }
	@rm -f "$(BUILD)/entitlements.plist"
	@printf '%s\n' \
		'<?xml version="1.0" encoding="UTF-8"?>' \
		'<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">' \
		'<plist version="1.0"><dict>' \
		'  <key>com.apple.security.cs.allow-jit</key><true/>' \
		'  <key>com.apple.security.cs.allow-unsigned-executable-memory</key><true/>' \
		'  <key>com.apple.security.cs.disable-library-validation</key><true/>' \
		'</dict></plist>' \
		> "$(BUILD)/entitlements.plist"
	@identity="$${APPLE_SIGNING_IDENTITY:-}"; \
	if [ -z "$$identity" ] || [ "$$identity" = "-" ]; then \
		identity=$$(security find-identity -v -p codesigning 2>/dev/null \
			| grep "Developer ID Application" | head -1 | awk -F'"' '{print $$2}'); \
	fi; \
	if [ -z "$$identity" ]; then \
		identity="-"; \
		echo "  no Developer ID found - signing ad-hoc (runs here, not elsewhere)"; \
	else \
		echo "  signing as $$identity"; \
	fi; \
	stamp=--timestamp; \
	[ "$$identity" = "-" ] && stamp=""; \
	find "$(APP)/Contents/Frameworks" -type f 2>/dev/null | while read -r file; do \
		if file "$$file" | grep -q "Mach-O"; then \
			codesign --force --options runtime --sign "$$identity" \
				--entitlements "$(BUILD)/entitlements.plist" $$stamp "$$file" || exit 1; \
		fi; \
	done; \
	find "$(APP)/Contents/Frameworks" -maxdepth 1 -name "*.framework" 2>/dev/null | while read -r bundle; do \
		codesign --force --options runtime --sign "$$identity" \
			--entitlements "$(BUILD)/entitlements.plist" $$stamp "$$bundle" || exit 1; \
	done; \
	codesign --force --options runtime --sign "$$identity" \
		--entitlements "$(BUILD)/entitlements.plist" $$stamp \
		"$(APP)/Contents/MacOS/love" || exit 1; \
	codesign --force --options runtime --sign "$$identity" \
		--entitlements "$(BUILD)/entitlements.plist" $$stamp "$(APP)" || exit 1
	@rm -f "$(BUILD)/entitlements.plist"
	@codesign --verify --deep --strict "$(APP)" && echo "  signature verifies"

# Start the thing that was built and see whether it plays.
#
# The archive check proves every file is present; this proves they add up to
# a program that finds its library and its model *inside the bundle*.  It is
# the only test here that runs the bundle rather than the checkout, and the
# difference is the whole point: `make game` finds everything in the working
# tree, where it always exists.
#
# `OMOK_SCREENSHOT` puts the game in watch mode — the engine plays itself —
# and captures a frame after `OMOK_SHOT_AFTER` seconds, then quits.  A frame
# only gets captured once the engine is up, so a bundle that cannot find its
# dylib, its metallib or its model never writes the file.  HOME is pointed at
# a throwaway so the run does not touch ~/.causewaybayomok.
.PHONY: app-smoke
app-smoke:
	@test -d "$(APP)" || { echo "  no $(APP); run 'make app'" >&2; exit 1; }
	@rm -rf "$(BUILD)/smoke.png" "$(BUILD)/smoke-home" && mkdir -p "$(BUILD)/smoke-home"
	@HOME="$(abspath $(BUILD))/smoke-home" OMOK_WINDOW=1000x700 \
		OMOK_SCREENSHOT="$(abspath $(BUILD))/smoke.png" OMOK_SHOT_AFTER=2 \
		"$(APP)/Contents/MacOS/love" > "$(BUILD)/smoke.log" 2>&1 & \
	pid=$$!; \
	for _ in $$(seq 1 60); do \
		[ -f "$(BUILD)/smoke.png" ] && break; \
		sleep 0.5; \
	done; \
	kill $$pid 2>/dev/null || true; \
	wait $$pid 2>/dev/null || true; \
	if [ ! -f "$(BUILD)/smoke.png" ]; then \
		echo "  the bundle did not draw a frame:" >&2; \
		head -20 "$(BUILD)/smoke.log" >&2; \
		exit 1; \
	fi
	@rm -rf "$(BUILD)/smoke-home" "$(BUILD)/smoke.log"
	@echo "  the bundle starts, finds its engine and plays ($(BUILD)/smoke.png)"

# ------------------------------------------------------------------ notarize

# Signing is not enough on its own.  Since macOS 10.15 anything distributed
# outside the App Store must also be notarised, or a *downloaded* copy is
# refused with "Apple could not verify…".  Not part of `make app`: it needs
# credentials this repository must not carry, and it uploads to Apple and
# waits, which a build should not do without being asked.
#
#     export APPLE_ID=you@example.com
#     export APPLE_PASSWORD=abcd-efgh-ijkl-mnop   # app-specific
#     export APPLE_TEAM_ID=ABCDE12345
#     make notarize
.PHONY: notarize
notarize: ## Notarize and staple the built .app (needs Apple credentials)
	@test -d "$(APP)" || { echo "  no $(APP); run 'make app' first" >&2; exit 1; }
	@for name in APPLE_ID APPLE_PASSWORD APPLE_TEAM_ID; do \
		eval "value=\$$$$name"; \
		[ -n "$$value" ] || { \
			echo "  $$name is not set." >&2; \
			echo "  Notarising needs an Apple ID, an app-specific password" >&2; \
			echo "  from appleid.apple.com, and the team id. None is kept here." >&2; \
			exit 1; }; \
	done
	@# Ad-hoc cannot be notarised, and finding that out from Apple takes
	@# minutes.  The signature says which it is, so ask it first.
	@codesign -dv --verbose=2 "$(APP)" 2>&1 | grep -q "Authority=Developer ID Application" || { \
		echo "  $(APP) is not signed with a Developer ID." >&2; \
		echo "  Apple will refuse an ad-hoc signature; there is nothing to submit." >&2; \
		exit 1; }
	@echo "  submitting to Apple - this takes minutes, not seconds"
	@rm -f "$(BUILD)/notarize.zip"
	@ditto -c -k --sequesterRsrc --keepParent "$(APP)" "$(BUILD)/notarize.zip"
	@xcrun notarytool submit "$(BUILD)/notarize.zip" \
		--apple-id "$$APPLE_ID" \
		--password "$$APPLE_PASSWORD" \
		--team-id "$$APPLE_TEAM_ID" \
		--wait --timeout 30m
	@rm -f "$(BUILD)/notarize.zip"
	@# Stapled, so the ticket travels with the bundle and Gatekeeper can
	@# check it with no network at all.
	@xcrun stapler staple "$(APP)"
	@xcrun stapler validate "$(APP)"
	@# The distribution zip was made before any of this and carries an
	@# unstapled copy.  Rebuilt, or the download is exactly what it was.
	@rm -f "$(ZIP)"
	@cd "$(BUILD)" && ditto -c -k --sequesterRsrc --keepParent "$(APP_NAME).app" "$(APP_NAME)-macos.zip"
	@$(MAKE) --no-print-directory gatekeeper

# What a person who downloads it will actually get.  `--context
# context:primary-signature` is the one that matters: plain `spctl -a` accepts
# an un-notarised Developer ID bundle, and Finder does not.
.PHONY: gatekeeper
gatekeeper: ## Assess the bundle the way Finder does
	@test -d "$(APP)" || { echo "  no $(APP); run 'make app' first" >&2; exit 1; }
	@codesign --verify --deep --strict "$(APP)" && echo "  signature: valid"
	@if xcrun stapler validate "$(APP)" > /dev/null 2>&1; then \
		echo "  notarised: yes, and stapled"; \
	else \
		echo "  notarised: NO - a downloaded copy will be blocked"; \
	fi
	@spctl -a -vvv -t open --context context:primary-signature "$(APP)" 2>&1 \
		| sed 's/^/  /' || true

.PHONY: clean
clean: ## Remove the app bundle and everything staged for it
	rm -rf "$(BUILD)"
