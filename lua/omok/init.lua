--- LuaJIT bindings for the Omok AI core.
--
-- Shared verbatim by the terminal client in `lua/` and the LÖVE game in
-- `love2d/` (which reaches this file through a symlink), so there is one
-- description of the C ABI and one place to fix it.
--
-- Board squares are **0-based flat indices**, `row * size + col`, row 0 at the
-- top.  That is the trainer's own indexing -- it is what the network's policy
-- vector is indexed by and what `omok.name()` formats -- so keeping it means
-- no conversion sits between the model and the screen.  Use `omok.rc()` and
-- `omok.index()` to move between an index and a 0-based row/column pair.
--
--     local omok = require("omok")
--     local game = omok.Game(15)
--     local engine = omok.Engine(omok.findModel())
--     game:play(omok.index(7, 7))
--     local result = engine:think(game)      -- blocking
--     game:play(result.move)

local ffi = require("ffi")

local omok = {}

omok.EMPTY, omok.BLACK, omok.WHITE = 0, 1, 2

ffi.cdef[[
typedef struct OmokEngine OmokEngine;
typedef struct OmokGame OmokGame;

typedef struct {
    int   simulations;
    float c_puct;
    float fpu_reduction;
    int   prior_local_radius;
    float temperature;
    int   batch;
} OmokOptions;

typedef struct {
    int    best_move;
    float  value;
    int    simulations;
    double seconds;
    int    forced;
    int    top_count;
    int    top_moves[8];
    float  top_probs[8];
} OmokResult;

const char *omok_last_error(void);
const char *omok_version(void);

OmokGame *omok_game_new(int size, int win_length, int allow_overline);
void      omok_game_free(OmokGame *game);
OmokGame *omok_game_clone(const OmokGame *game);
void      omok_game_reset(OmokGame *game);
int       omok_game_play(OmokGame *game, int index);
int       omok_game_undo(OmokGame *game);
int       omok_game_is_legal(const OmokGame *game, int index);
int       omok_game_size(const OmokGame *game);
int       omok_game_to_move(const OmokGame *game);
int       omok_game_winner(const OmokGame *game);
int       omok_game_is_over(const OmokGame *game);
int       omok_game_move_count(const OmokGame *game);
int       omok_game_move_at(const OmokGame *game, int i);
int       omok_game_last_move(const OmokGame *game);
const unsigned char *omok_game_cells(const OmokGame *game);
int       omok_game_win_line(const OmokGame *game, int *out, int cap);
int       omok_game_forced_move(const OmokGame *game);

void      omok_options_default(OmokOptions *out);
OmokEngine *omok_engine_new(const char *model_path, uint64_t seed);
void      omok_engine_free(OmokEngine *engine);
const char *omok_engine_describe(const OmokEngine *engine);
int       omok_engine_board_size(const OmokEngine *engine);

int       omok_search_begin(OmokEngine *engine, const OmokGame *game, const OmokOptions *options);
int       omok_search_poll(OmokEngine *engine, OmokResult *out);
void      omok_search_cancel(OmokEngine *engine);
int       omok_search_is_running(const OmokEngine *engine);
int       omok_search(OmokEngine *engine, const OmokGame *game, const OmokOptions *options, OmokResult *out);

char *getcwd(char *buf, size_t size);
int   usleep(unsigned int usec);
]]

-- --------------------------------------------------------------- locating things

local function exists(path)
    if not path then return false end
    local handle = io.open(path, "rb")
    if handle then handle:close() return true end
    return false
end

--- The process's working directory, so relative paths can be made absolute.
local function cwd()
    local buffer = ffi.new("char[?]", 4096)
    if ffi.C.getcwd(buffer, 4096) == nil then return "." end
    return ffi.string(buffer)
end

--- The directory this file lives in, as an absolute path.
--
-- `debug.getinfo` reports the path the file was *found* at, which is relative
-- whenever `package.path` was; walking up from a relative path stops at the
-- first component, so it is anchored here instead.
local function here()
    local source = debug.getinfo(1, "S").source:sub(2)
    local dir = source:match("^(.*)[/\\]") or "."
    if dir:sub(1, 1) ~= "/" then dir = cwd() .. "/" .. dir end
    return dir
end

--- Walk up from `start` looking for a directory that contains `marker`.
local function findUp(start, marker)
    local dir = start
    for _ = 1, 8 do
        if exists(dir .. "/" .. marker) then return dir end
        local parent = dir:match("^(.*)/[^/]+$")
        if not parent or parent == dir then break end
        dir = parent
    end
    return nil
end

--- Absolute path of the project root, found by walking up from this file.
-- Inside LÖVE `debug.getinfo` reports a path relative to the game directory,
-- so the mount point is asked for first.
function omok.root()
    -- Inside LÖVE the game directory is the mount point, and `debug.getinfo`
    -- reports paths relative to it rather than to the process's cwd.
    local base = here()
    if love and love.filesystem and love.filesystem.getSource then
        local source = love.filesystem.getSource()
        if source and source:sub(1, 1) == "/" then base = source end
    end
    return findUp(base, "rust/Cargo.toml") or findUp(base, "runs") or base
end

--- The `Contents` directory of the macOS app this game is running inside, or
-- nil when it is not inside one.
--
-- `make app` fuses the game into a LÖVE bundle: the archive sits at
-- `Contents/Resources/game.love`, the compiled core beside the runtime's own
-- libraries as `Contents/Frameworks/OmokAI.framework` (the dylib, with the
-- Metal kernels in its Resources), and the trained model under
-- `Contents/Resources/model`.  A double-clicked app inherits no environment,
-- so `OMOK_LIB` and `OMOK_MODEL` cannot reach it -- and the checkout it was
-- built from is not there either.  So the bundle is asked first, and only
-- when fused: `love love2d` in a checkout is not a bundle and must keep
-- finding what `make ai` and `make train` produced.
local function bundleContents()
    if not (love and love.filesystem and love.filesystem.isFused
            and love.filesystem.isFused()) then
        return nil
    end
    local source = love.filesystem.getSource() or ""
    return source:match("^(.*/Contents)/Resources/[^/]+$")
end

local LIB_NAMES = {
    "rust/target/release/libomok_ai.dylib",
    "rust/target/debug/libomok_ai.dylib",
    "rust/target/release/libomok_ai.so",
    "rust/target/debug/libomok_ai.so",
}

--- Where the compiled core is, or nil.
function omok.findLibrary()
    if exists(os.getenv("OMOK_LIB")) then return os.getenv("OMOK_LIB") end
    local contents = bundleContents()
    if contents and exists(contents .. "/Frameworks/OmokAI.framework/OmokAI") then
        return contents .. "/Frameworks/OmokAI.framework/OmokAI"
    end
    local root = omok.root()
    for _, name in ipairs(LIB_NAMES) do
        local path = root .. "/" .. name
        if exists(path) then return path end
    end
    return nil
end

local MODEL_NAMES = {
    "runs/blitz/checkpoints/best.npz",
    "runs/base/checkpoints/best.npz",
    "runs/omok/checkpoints/best.npz",
    "export/OmokNet.npz",
}

--- The trained network to play against, or nil.  `OMOK_MODEL` wins; otherwise
-- the newest `best.npz` under `runs/` is used, so training a fresh run and
-- restarting the game is all it takes to face the new model.
function omok.findModel()
    if exists(os.getenv("OMOK_MODEL")) then return os.getenv("OMOK_MODEL") end
    local contents = bundleContents()
    if contents and exists(contents .. "/Resources/model/best.npz") then
        return contents .. "/Resources/model/best.npz"
    end
    local root = omok.root()
    -- `ls -t` puts the most recently written checkpoint first, so a fresh run
    -- is picked up without touching anything here.
    local pipe = io.popen('ls -t "' .. root .. '"/runs/*/checkpoints/best.npz 2>/dev/null')
    if pipe then
        local newest = pipe:read("*l")
        pipe:close()
        if exists(newest) then return newest end
    end
    for _, name in ipairs(MODEL_NAMES) do
        local path = root .. "/" .. name
        if exists(path) then return path end
    end
    return nil
end

-- ------------------------------------------------------------------- loading

local C = nil

--- Load the shared library.  Safe to call repeatedly; returns the FFI
-- namespace, or raises with a message that says how to build it.
function omok.load(path)
    if C then return C end
    path = path or omok.findLibrary()
    if not path then
        error("cannot find libomok_ai -- build it with `make ai` (or set OMOK_LIB)", 2)
    end
    local ok, lib = pcall(ffi.load, path)
    if not ok then
        error(("cannot load %s: %s"):format(path, lib), 2)
    end
    C = lib
    omok.C = C
    omok.libraryPath = path
    return C
end

local function lib()
    return C or omok.load()
end

--- Wait for `seconds`, so a caller polling an asynchronous search can yield
-- the CPU instead of spinning on it.
function omok.sleep(seconds)
    ffi.C.usleep(math.max(0, math.floor(seconds * 1e6)))
end

function omok.version()
    return ffi.string(lib().omok_version())
end

local function lastError()
    return ffi.string(lib().omok_last_error())
end

-- ------------------------------------------------------------- index helpers

--- 0-based row and column of a flat index.
function omok.rc(index, size)
    return math.floor(index / size), index % size
end

--- Flat index of a 0-based row and column.
function omok.index(row, col, size)
    return row * (size or 15) + col
end

--- "h7" style name, matching `omok.board.format_move` in the Python trainer.
function omok.name(index, size)
    local row, col = omok.rc(index, size)
    return string.char(string.byte("a") + col) .. tostring(row)
end

--- Parse "h7", "7,7" or a raw index.  Returns nil if it is not on the board.
function omok.parse(text, size)
    text = (text or ""):gsub("%s", ""):lower()
    if text == "" then return nil end
    local row, col = text:match("^(%d+),(%d+)$")
    if row then row, col = tonumber(row), tonumber(col)
    else
        local letter, number = text:match("^(%a)(%d+)$")
        if letter then
            col = string.byte(letter) - string.byte("a")
            row = tonumber(number)
        elseif text:match("^%d+$") then
            local flat = tonumber(text)
            if flat < size * size then return flat end
            return nil
        else
            return nil
        end
    end
    if row < 0 or row >= size or col < 0 or col >= size then return nil end
    return row * size + col
end

-- ---------------------------------------------------------------------- Game

local Game = {}
Game.__index = Game
omok.Game = setmetatable(Game, {
    __call = function(_, size, winLength, allowOverline)
        return Game.new(size, winLength, allowOverline)
    end,
})

--- A position with full move history.  Pure Rust, no engine required, so the
-- rules are exactly the ones the network was trained against.
function Game.new(size, winLength, allowOverline)
    size = size or 15
    local handle = lib().omok_game_new(size, winLength or 5,
                                       allowOverline == false and 0 or 1)
    if handle == nil then error(lastError(), 2) end
    return setmetatable({
        handle = ffi.gc(handle, lib().omok_game_free),
        size = size,
    }, Game)
end

function Game:clone()
    local handle = lib().omok_game_clone(self.handle)
    if handle == nil then error(lastError(), 2) end
    return setmetatable({
        handle = ffi.gc(handle, lib().omok_game_free),
        size = self.size,
    }, Game)
end

--- Play at a flat index.  Returns true, or false if the move was illegal.
function Game:play(index)
    return lib().omok_game_play(self.handle, index) ~= 0
end

--- Take back the last move.  Returns false at the start of the game.
function Game:undo()
    return lib().omok_game_undo(self.handle) ~= 0
end

function Game:reset()
    lib().omok_game_reset(self.handle)
end

function Game:legal(index)
    return lib().omok_game_is_legal(self.handle, index) ~= 0
end

--- `omok.BLACK` or `omok.WHITE`.
function Game:toMove()
    return lib().omok_game_to_move(self.handle)
end

--- The winner, or `omok.EMPTY` for a draw or an unfinished game.
function Game:winner()
    return lib().omok_game_winner(self.handle)
end

function Game:isOver()
    return lib().omok_game_is_over(self.handle) ~= 0
end

function Game:moveCount()
    return lib().omok_game_move_count(self.handle)
end

--- The `i`-th move of the game, counting from 1 as Lua does.
function Game:moveAt(i)
    local index = lib().omok_game_move_at(self.handle, i - 1)
    if index < 0 then return nil end
    return index
end

function Game:lastMove()
    local index = lib().omok_game_last_move(self.handle)
    if index < 0 then return nil end
    return index
end

--- The stone at a flat index: `omok.EMPTY`, `omok.BLACK` or `omok.WHITE`.
function Game:get(index)
    return lib().omok_game_cells(self.handle)[index]
end

--- The stone at a 0-based row and column.
function Game:at(row, col)
    if row < 0 or col < 0 or row >= self.size or col >= self.size then
        return omok.EMPTY
    end
    return self:get(row * self.size + col)
end

--- The whole board as a flat Lua table of `size * size` cell values, indexed
-- from 1 -- handy for iterating without touching cdata.
function Game:cells()
    local cells = lib().omok_game_cells(self.handle)
    local out = {}
    for i = 0, self.size * self.size - 1 do out[i + 1] = cells[i] end
    return out
end

--- The stones of the winning line, as flat indices.  Empty until someone wins.
function Game:winLine()
    local buffer = ffi.new("int[?]", self.size)
    local n = lib().omok_game_win_line(self.handle, buffer, self.size)
    local out = {}
    for i = 0, n - 1 do out[i + 1] = buffer[i] end
    return out
end

--- An immediate win for the side to move, else a square that must be blocked,
-- else nil.  Instant -- no search, no network.
function Game:forcedMove()
    local index = lib().omok_game_forced_move(self.handle)
    if index < 0 then return nil end
    return index
end

function Game:ascii()
    local rows = {}
    local header = {"   "}
    for c = 0, self.size - 1 do header[#header + 1] = tostring(c % 10) end
    rows[1] = table.concat(header, " ")
    local glyph = {[omok.EMPTY] = ".", [omok.BLACK] = "X", [omok.WHITE] = "O"}
    for r = 0, self.size - 1 do
        local line = {("%2d "):format(r)}
        for c = 0, self.size - 1 do line[#line + 1] = glyph[self:at(r, c)] end
        rows[#rows + 1] = table.concat(line, " ")
    end
    return table.concat(rows, "\n")
end

-- -------------------------------------------------------------------- Engine

local Engine = {}
Engine.__index = Engine
omok.Engine = setmetatable(Engine, {
    __call = function(_, path, seed) return Engine.new(path, seed) end,
})

--- Search settings.  `nil` fields keep the values the shipped checkpoint was
-- trained with.
local function options(opts)
    local out = ffi.new("OmokOptions")
    lib().omok_options_default(out)
    if opts then
        if opts.simulations then out.simulations = opts.simulations end
        if opts.cPuct then out.c_puct = opts.cPuct end
        if opts.fpuReduction then out.fpu_reduction = opts.fpuReduction end
        if opts.priorLocalRadius then out.prior_local_radius = opts.priorLocalRadius end
        if opts.temperature then out.temperature = opts.temperature end
        if opts.batch then out.batch = opts.batch end
    end
    return out
end
omok.options = options

--- Load a trained `.npz` checkpoint and start the engine thread.
-- `seed` of 0 (the default) seeds move sampling from the clock.
function Engine.new(path, seed)
    if not path then
        error("no model to load -- train one with `make train`, or set OMOK_MODEL", 2)
    end
    local handle = lib().omok_engine_new(path, seed or 0)
    if handle == nil then error(lastError(), 2) end
    return setmetatable({
        handle = ffi.gc(handle, lib().omok_engine_free),
        path = path,
        result = ffi.new("OmokResult"),
    }, Engine)
end

--- Backend, architecture and checkpoint, for a status line.
function Engine:describe()
    return ffi.string(lib().omok_engine_describe(self.handle))
end

--- The board size the network was trained for.
function Engine:boardSize()
    return lib().omok_engine_board_size(self.handle)
end

local function readResult(raw, size)
    local top = {}
    for i = 0, raw.top_count - 1 do
        top[i + 1] = {move = raw.top_moves[i], prob = raw.top_probs[i]}
    end
    return {
        move = raw.best_move,
        name = omok.name(raw.best_move, size),
        value = raw.value,
        simulations = raw.simulations,
        seconds = raw.seconds,
        forced = raw.forced ~= 0,
        top = top,
    }
end

--- Start thinking, cancelling any search already running.  Returns true, or
-- false plus a message.  Poll with `Engine:poll()`.
function Engine:begin(game, opts)
    self.size = game.size
    if lib().omok_search_begin(self.handle, game.handle, options(opts)) == 0 then
        return false, lastError()
    end
    return true
end

--- Collect a finished search without blocking: the result table when one is
-- ready, nil while the engine is still thinking, or nil plus a message.
function Engine:poll()
    local status = lib().omok_search_poll(self.handle, self.result)
    if status == 1 then return readResult(self.result, self.size or 15) end
    if status < 0 then return nil, lastError() end
    return nil
end

function Engine:cancel()
    lib().omok_search_cancel(self.handle)
end

function Engine:isRunning()
    return lib().omok_search_is_running(self.handle) ~= 0
end

--- Think and wait.  For callers with nothing to draw meanwhile.
function Engine:think(game, opts)
    if lib().omok_search(self.handle, game.handle, options(opts), self.result) == 0 then
        return nil, lastError()
    end
    return readResult(self.result, game.size)
end

return omok
