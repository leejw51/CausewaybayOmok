--- Tests for the Lua bindings and, through them, the Rust core.
--
--     luajit lua/test.lua
--
-- The engine tests need a trained checkpoint; they are skipped, not failed,
-- when there is not one, so the rules tests still run on a fresh clone.

package.path = (debug.getinfo(1, "S").source:sub(2):match("^(.*)/") or ".")
    .. "/?.lua;" .. (debug.getinfo(1, "S").source:sub(2):match("^(.*)/") or ".")
    .. "/?/init.lua;" .. package.path

local omok = require("omok")

local passed, failed, skipped = 0, 0, 0
local currentName

local function check(condition, message)
    if condition then
        passed = passed + 1
    else
        failed = failed + 1
        print(("  FAIL  %s: %s"):format(currentName, message or "assertion failed"))
    end
end

local function equal(actual, expected, message)
    check(actual == expected,
          ("%s (expected %s, got %s)"):format(message or "values differ",
                                              tostring(expected), tostring(actual)))
end

local function near(actual, expected, tolerance, message)
    check(math.abs(actual - expected) <= tolerance,
          ("%s (expected %s +/- %s, got %s)"):format(message or "values differ",
                                                     expected, tolerance, actual))
end

local function test(name, fn)
    currentName = name
    local ok, err = pcall(fn)
    if not ok then
        failed = failed + 1
        print(("  ERROR %s: %s"):format(name, err))
    end
end

local function skip(name, why)
    skipped = skipped + 1
    print(("  SKIP  %s (%s)"):format(name, why))
end

-- ------------------------------------------------------------------ library

print("library")
local libraryPath = omok.findLibrary()
if not libraryPath then
    print("  the core is not built -- run `make ai` first")
    os.exit(1)
end
omok.load(libraryPath)
print(("  loaded %s (v%s)"):format(libraryPath, omok.version()))

-- ------------------------------------------------------------- index helpers

print("index helpers")

test("rc and index round-trip", function()
    for _, index in ipairs({0, 1, 14, 15, 112, 224}) do
        local row, col = omok.rc(index, 15)
        equal(omok.index(row, col, 15), index, "round-trip of " .. index)
    end
end)

test("name matches the trainer's format_move", function()
    equal(omok.name(0, 15), "a0", "index 0")
    equal(omok.name(112, 15), "h7", "the centre of a 15x15 board")
    equal(omok.name(224, 15), "o14", "the far corner")
end)

test("parse accepts every documented spelling", function()
    equal(omok.parse("h7", 15), 112, "letter/number")
    equal(omok.parse("H7", 15), 112, "upper case")
    equal(omok.parse("7,7", 15), 112, "row,col")
    equal(omok.parse("112", 15), 112, "raw index")
    equal(omok.parse("z9", 15), nil, "off the board")
    equal(omok.parse("7,99", 15), nil, "column off the board")
    equal(omok.parse("", 15), nil, "empty")
end)

-- --------------------------------------------------------------------- rules

print("rules")

test("a new game is empty with black to move", function()
    local game = omok.Game(15)
    equal(game:toMove(), omok.BLACK, "black moves first")
    equal(game:moveCount(), 0, "no moves yet")
    equal(game:isOver(), false, "not over")
    equal(game:lastMove(), nil, "no last move")
    for i = 0, 224 do
        if game:get(i) ~= omok.EMPTY then check(false, "square " .. i .. " is not empty") end
    end
    equal(#game:cells(), 225, "cells() returns the whole board")
end)

test("playing alternates colours and records history", function()
    local game = omok.Game(15)
    check(game:play(112), "centre is playable")
    equal(game:get(112), omok.BLACK, "black stone landed")
    equal(game:toMove(), omok.WHITE, "white to move")
    check(game:play(113), "next square is playable")
    equal(game:get(113), omok.WHITE, "white stone landed")
    equal(game:moveCount(), 2, "two moves")
    equal(game:moveAt(1), 112, "first move")
    equal(game:moveAt(2), 113, "second move")
    equal(game:moveAt(3), nil, "no third move")
    equal(game:lastMove(), 113, "last move")
end)

test("occupied and off-board moves are rejected", function()
    local game = omok.Game(15)
    game:play(112)
    equal(game:play(112), false, "cannot play on a stone")
    equal(game:legal(112), false, "occupied square is not legal")
    equal(game:play(-1), false, "negative index")
    equal(game:play(225), false, "past the end of the board")
    equal(game:moveCount(), 1, "no rejected move was recorded")
end)

test("undo restores the previous position", function()
    local game = omok.Game(15)
    game:play(112); game:play(113)
    check(game:undo(), "undo the white move")
    equal(game:get(113), omok.EMPTY, "square is empty again")
    equal(game:toMove(), omok.WHITE, "white to move again")
    equal(game:moveCount(), 1, "one move left")
    check(game:undo(), "undo the black move")
    equal(game:undo(), false, "nothing left to undo")
end)

test("five in a row wins", function()
    local game = omok.Game(15)
    -- Black builds a row along row 7; white answers on row 0, out of the way.
    for i = 0, 4 do
        check(game:play(7 * 15 + i), "black " .. i)
        if i < 4 then check(game:play(0 * 15 + i), "white " .. i) end
    end
    equal(game:isOver(), true, "the game is over")
    equal(game:winner(), omok.BLACK, "black won")
    equal(#game:winLine(), 5, "the winning line has five stones")
    equal(game:play(200), false, "no moves after the game ends")
end)

test("undo un-wins the game", function()
    local game = omok.Game(15)
    for i = 0, 4 do
        game:play(7 * 15 + i)
        if i < 4 then game:play(0 * 15 + i) end
    end
    equal(game:winner(), omok.BLACK, "black won")
    game:undo()
    equal(game:isOver(), false, "the game is live again")
    equal(game:winner(), omok.EMPTY, "nobody has won")
    check(game:play(7 * 15 + 4), "the winning square is playable again")
end)

test("forcedMove takes a win before blocking one", function()
    local game = omok.Game(15)
    -- Black has four in a row and so does white; black to move must win, not block.
    for i = 0, 3 do game:play(7 * 15 + i); game:play(9 * 15 + i) end
    equal(game:toMove(), omok.BLACK, "black to move")
    equal(game:forcedMove(), 7 * 15 + 4, "complete black's five")
end)

test("forcedMove blocks when there is nothing to win", function()
    local game = omok.Game(15)
    -- White gets four in a row; black has scattered stones with no threat.
    game:play(0);  game:play(9 * 15 + 0)
    game:play(2);  game:play(9 * 15 + 1)
    game:play(4);  game:play(9 * 15 + 2)
    game:play(6);  game:play(9 * 15 + 3)
    equal(game:toMove(), omok.BLACK, "black to move")
    local forced = game:forcedMove()
    check(forced == 9 * 15 + 4 or forced == 9 * 15 - 1,
          "block one end of white's four, got " .. tostring(forced))
end)

test("clone is independent of the original", function()
    local game = omok.Game(15)
    game:play(112)
    local copy = game:clone()
    copy:play(113)
    equal(game:moveCount(), 1, "the original is untouched")
    equal(copy:moveCount(), 2, "the copy moved on")
    equal(game:get(113), omok.EMPTY, "the original square is still empty")
end)

test("reset empties the board", function()
    local game = omok.Game(15)
    game:play(112); game:play(113)
    game:reset()
    equal(game:moveCount(), 0, "no moves")
    equal(game:toMove(), omok.BLACK, "black to move")
    equal(game:get(112), omok.EMPTY, "board is clear")
end)

test("a smaller board with its own win length works", function()
    local game = omok.Game(9, 3, true)
    equal(game.size, 9, "size")
    for i = 0, 2 do
        game:play(4 * 9 + i)
        if i < 2 then game:play(0 * 9 + i) end
    end
    equal(game:winner(), omok.BLACK, "three in a row wins here")
end)

test("ascii renders the board", function()
    local game = omok.Game(15)
    game:play(112)
    local lines = {}
    for line in game:ascii():gmatch("[^\n]+") do lines[#lines + 1] = line end
    equal(#lines, 16, "a header and fifteen rows")
    check(lines[9]:find("X"), "the black stone shows on row 7")
end)

-- -------------------------------------------------------------------- engine

print("engine")
local modelPath = omok.findModel()

if not modelPath then
    skip("engine tests", "no trained checkpoint found; run `make train`")
else
    print("  model " .. modelPath)
    local engine = omok.Engine(modelPath, 1234)
    print("  " .. engine:describe())

    test("the engine reports the board size it was trained for", function()
        check(engine:boardSize() >= 5, "a sensible board size")
    end)

    test("a blocking search returns a legal move", function()
        local game = omok.Game(engine:boardSize())
        local result, err = engine:think(game, {simulations = 32})
        check(result ~= nil, "search succeeded: " .. tostring(err))
        if result then
            check(game:legal(result.move), "the move is legal")
            check(result.simulations > 0, "it ran simulations")
            check(result.value >= -1 and result.value <= 1, "value is in range")
            check(#result.top > 0, "it reported its top moves")
            equal(result.name, omok.name(result.move, game.size), "name matches move")
        end
    end)

    test("an asynchronous search returns the same kind of answer", function()
        local game = omok.Game(engine:boardSize())
        game:play(112)
        check(engine:begin(game, {simulations = 64}), "the search started")
        -- Poll the way a game loop would, rather than spinning.
        local result, err
        local deadline = os.time() + 30
        while os.time() < deadline do
            result, err = engine:poll()
            if result or err then break end
            omok.sleep(0.002)
        end
        check(result ~= nil, "a result arrived: " .. tostring(err))
        if result then check(game:legal(result.move), "the move is legal") end
        equal(engine:isRunning(), false, "the engine is idle afterwards")
    end)

    test("cancelling drops the answer", function()
        local game = omok.Game(engine:boardSize())
        engine:begin(game, {simulations = 4096})
        engine:cancel()
        local result = engine:poll()
        equal(result, nil, "no result is delivered for a cancelled search")
    end)

    test("the engine takes an immediate win", function()
        local game = omok.Game(engine:boardSize())
        local n = game.size
        -- Black four in a row on row 7, white scattered harmlessly on row 0.
        for i = 0, 3 do game:play(7 * n + i); game:play(0 * n + 2 * i) end
        local result = engine:think(game, {simulations = 32})
        check(result ~= nil, "search succeeded")
        if result then
            equal(result.move, 7 * n + 4, "it completed the five")
            equal(result.forced, true, "and knew it was forced")
        end
    end)

    test("the engine blocks an immediate loss", function()
        local game = omok.Game(engine:boardSize())
        local n = game.size
        -- White builds four; black's stones make no threat of their own.
        game:play(0);  game:play(9 * n + 0)
        game:play(2);  game:play(9 * n + 1)
        game:play(4);  game:play(9 * n + 2)
        game:play(6);  game:play(9 * n + 3)
        local result = engine:think(game, {simulations = 32})
        check(result ~= nil, "search succeeded")
        if result then
            check(result.move == 9 * n + 4 or result.move == 9 * n - 1,
                  "it blocked, got " .. omok.name(result.move, n))
        end
    end)

    test("searching a finished game is an error, not a crash", function()
        local game = omok.Game(engine:boardSize())
        local n = game.size
        for i = 0, 4 do
            game:play(7 * n + i)
            if i < 4 then game:play(0 * n + i) end
        end
        equal(game:isOver(), true, "the game is over")
        local result, err = engine:think(game, {simulations = 8})
        equal(result, nil, "no move is returned")
        check(err and #err > 0, "and there is a message")
    end)

    test("a seeded engine is reproducible", function()
        local a = omok.Engine(modelPath, 99)
        local b = omok.Engine(modelPath, 99)
        local game = omok.Game(engine:boardSize())
        game:play(112)
        local first = a:think(game, {simulations = 48, temperature = 1.0})
        local second = b:think(game, {simulations = 48, temperature = 1.0})
        check(first and second, "both searched")
        if first and second then
            equal(first.move, second.move, "the same seed gives the same move")
            near(first.value, second.value, 1e-6, "and the same value")
        end
    end)

    test("more simulations means a more considered answer", function()
        local game = omok.Game(engine:boardSize())
        game:play(112); game:play(113)
        local small = engine:think(game, {simulations = 16})
        local large = engine:think(game, {simulations = 128})
        check(small and large, "both searched")
        if small and large then
            check(large.simulations > small.simulations, "it ran the budget it was given")
            check(game:legal(large.move), "the move is legal")
        end
    end)

    test("a full game between two engines finishes legally", function()
        local game = omok.Game(engine:boardSize())
        local plies = 0
        while not game:isOver() and plies < game.size * game.size do
            local result = engine:think(game, {simulations = 16, temperature = 0.8})
            check(result ~= nil, "search succeeded at ply " .. plies)
            if not result then break end
            check(game:play(result.move), "ply " .. plies .. " was legal")
            plies = plies + 1
        end
        check(game:isOver(), "the game reached a conclusion in " .. plies .. " plies")
        if game:winner() ~= omok.EMPTY then
            equal(#game:winLine() >= 5, true, "the winner has a line of five or more")
        end
    end)
end

-- -------------------------------------------------------------------- report

print()
print(("%d passed, %d failed, %d skipped"):format(passed, failed, skipped))
os.exit(failed == 0 and 0 or 1)
