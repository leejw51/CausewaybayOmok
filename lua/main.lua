#!/usr/bin/env luajit
--- Causewaybay Omok -- the terminal client.
--
--     luajit lua/main.lua                  play black against the trained model
--     luajit lua/main.lua --white          let the model open
--     luajit lua/main.lua --level 4        pick a difficulty (1-5)
--     luajit lua/main.lua --watch          the model plays itself
--     luajit lua/main.lua --help
--
-- The engine runs on its own thread inside the Rust core, so the search is
-- started and then polled: the spinner keeps turning and keys keep working
-- while the model thinks, and pressing `u` abandons the search rather than
-- waiting for it.

local root = debug.getinfo(1, "S").source:sub(2):match("^(.*)/") or "."
package.path = root .. "/?.lua;" .. root .. "/?/init.lua;" .. package.path

local omok = require("omok")
local term = require("omok.term")

-- ------------------------------------------------------------------ settings

-- Simulations, and how loosely the model picks among its best moves.  The easy
-- levels are not just slower to think: a higher temperature makes them stray
-- from the top move often enough to be beatable.
local LEVELS = {
    {name = "Stroll",   simulations = 24,   temperature = 1.2},
    {name = "Casual",   simulations = 96,   temperature = 0.7},
    {name = "Sharp",    simulations = 320,  temperature = 0.25},
    {name = "Serious",  simulations = 900,  temperature = 0.0},
    {name = "Merciless", simulations = 2400, temperature = 0.0},
}

local options = {level = 2, human = omok.BLACK, watch = false}

local function usage()
    print("Causewaybay Omok -- terminal client\n")
    print("  --white        play white; the model opens")
    print("  --level N      difficulty 1-5 (default 2)")
    print("  --watch        the model plays both sides")
    print("  --size N       board size (default: whatever the model was trained for)")
    print("  --model PATH   a .npz checkpoint to play against")
    print("  --help")
    os.exit(0)
end

do
    local i = 1
    while i <= #arg do
        local a = arg[i]
        if a == "--help" or a == "-h" then usage()
        elseif a == "--white" then options.human = omok.WHITE
        elseif a == "--watch" then options.watch = true
        elseif a == "--level" then i = i + 1; options.level = tonumber(arg[i]) or 2
        elseif a == "--size" then i = i + 1; options.size = tonumber(arg[i])
        elseif a == "--model" then i = i + 1; options.model = arg[i]
        else print("unknown option: " .. a); usage() end
        i = i + 1
    end
    options.level = math.max(1, math.min(#LEVELS, options.level))
end

-- -------------------------------------------------------------------- colours

local C = {
    frame     = 238,
    grid      = 240,
    label     = 245,
    black     = 39,    -- the dark stone, drawn bright enough to see
    white     = 231,
    lastRing  = 214,
    win       = 46,
    hint      = 141,
    cursor    = 220,
    title     = 214,
    accent    = 208,
    good      = 78,
    bad       = 203,
    dim       = 242,
    text      = 252,
}

local SPINNER = {"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}

-- ----------------------------------------------------------------- the engine

local modelPath = options.model or omok.findModel()
if not modelPath then
    io.stderr:write("no trained model found.\n")
    io.stderr:write("train one with `make train`, or pass --model path/to/best.npz\n")
    os.exit(1)
end

local ok, engine = pcall(omok.Engine, modelPath)
if not ok then
    io.stderr:write("could not start the engine: " .. tostring(engine) .. "\n")
    os.exit(1)
end

local size = options.size or engine:boardSize()
local game = omok.Game(size)

-- ------------------------------------------------------------------ ui state

local ui = {
    cursor = math.floor(size / 2) * size + math.floor(size / 2),
    thinking = false,
    frame = 0,
    message = "arrows or hjkl to move, space to place",
    lastResult = nil,
    hint = nil,
    thinkingSince = 0,
    history = {},          -- one line per move, newest last
    quit = false,
}

local function levelOptions(kind)
    local level = LEVELS[options.level]
    -- A hint is advice, so it always names the move the search actually liked
    -- best; the easy levels' sampling would otherwise suggest something the
    -- model does not rate.
    return {
        simulations = level.simulations,
        temperature = (kind == "hint") and 0.0 or level.temperature,
    }
end

local function humanTurn()
    if options.watch then return false end
    return game:toMove() == options.human
end

local function note(text)
    ui.message = text
end

local function record(who, index, result)
    local line = ("%2d. %s %s"):format(game:moveCount(), who, omok.name(index, size))
    if result then
        line = line .. ("  %+.2f  %d sims"):format(result.value, result.simulations)
        if result.forced then line = line .. " !" end
    end
    ui.history[#ui.history + 1] = line
end

--- Ask the engine for a move (or, for a hint, just for its opinion).
local function startThinking(kind)
    if game:isOver() then return end
    ui.hint = nil
    local started, err = engine:begin(game, levelOptions(kind))
    if not started then
        note("engine: " .. tostring(err))
        return
    end
    ui.thinking = kind
    ui.thinkingSince = os.clock()
end

local function stopThinking()
    if ui.thinking then
        engine:cancel()
        ui.thinking = false
    end
end

-- ------------------------------------------------------------------- drawing

local GLYPH_EMPTY = "·"
local GLYPH_STONE = "●"

--- The board occupies `2 * size + 1` columns; everything else is laid out
-- around it, so the whole frame is centred in whatever terminal it is given.
local function draw()
    local rows, cols = term.size()
    local boardWidth = size * 2
    local left = math.max(2, math.floor((cols - boardWidth - 26) / 2))
    local top = math.max(2, math.floor((rows - size - 8) / 2))

    local winning = {}
    for _, index in ipairs(game:winLine()) do winning[index] = true end
    local top3 = {}
    if ui.hint then
        for rank, entry in ipairs(ui.hint.top) do
            if rank <= 3 then top3[entry.move] = rank end
        end
    end
    local last = game:lastMove()

    local s = term.screen()
    s:clear()

    -- title
    s:at(top - 1, left):bold():fg(C.title):put("CAUSEWAYBAY OMOK"):reset()
    s:fg(C.dim):put("   " .. LEVELS[options.level].name):reset()

    -- column labels
    s:at(top + 1, left + 4):fg(C.label)
    for c = 0, size - 1 do
        s:put(string.char(string.byte("a") + c) .. " ")
    end
    s:reset()

    -- the board
    for r = 0, size - 1 do
        s:at(top + 2 + r, left):fg(C.label):put(("%2d "):format(r)):reset()
        for c = 0, size - 1 do
            local index = r * size + c
            local cell = game:at(r, c)
            local isCursor = (index == ui.cursor) and humanTurn() and not game:isOver()

            if isCursor then s:bg(C.cursor > 0 and 236 or nil) end
            if cell == omok.EMPTY then
                local rank = top3[index]
                if rank then
                    s:fg(C.hint):put(tostring(rank))
                elseif isCursor then
                    s:fg(C.cursor):bold():put("+"):reset()
                    if isCursor then s:bg(236) end
                else
                    s:fg(C.grid):put(GLYPH_EMPTY)
                end
            else
                local colour = (cell == omok.BLACK) and C.black or C.white
                if winning[index] then colour = C.win end
                s:fg(colour)
                if winning[index] or index == last then s:bold() end
                s:put(GLYPH_STONE)
                s:reset()
                if isCursor then s:bg(236) end
            end
            -- the gap between columns, which also carries the last-move ring
            if index == last and not winning[index] then
                s:fg(C.lastRing):put("<"):reset()
            else
                s:put(" ")
            end
            s:reset()
        end
    end

    -- ---------------------------------------------------------- side panel
    local px = left + boardWidth + 4
    local py = top + 1
    local function line(text, colour, bold)
        s:at(py, px)
        if colour then s:fg(colour) end
        if bold then s:bold() end
        s:put(text):reset()
        py = py + 1
    end

    line("ENGINE", C.accent, true)
    local description = engine:describe():gsub("^mlx | ", "")
    line("mlx  " .. description:match("^[^|]+"):gsub("%s+$", ""), C.dim)
    line(("level %d/%d  %s"):format(options.level, #LEVELS, LEVELS[options.level].name), C.text)
    line(("%d sims  temp %.2f"):format(LEVELS[options.level].simulations,
                                       LEVELS[options.level].temperature), C.dim)
    py = py + 1

    line("POSITION", C.accent, true)
    local mover = game:toMove() == omok.BLACK and "black" or "white"
    if game:isOver() then
        local winner = game:winner()
        if winner == omok.EMPTY then
            line("a draw", C.text, true)
        else
            local who = winner == omok.BLACK and "black" or "white"
            local mine = (not options.watch) and winner == options.human
            line(who .. " wins" .. (options.watch and "" or (mine and " -- you!" or "")),
                 mine and C.good or C.bad, true)
        end
    elseif ui.thinking then
        local spin = SPINNER[(ui.frame % #SPINNER) + 1]
        line(spin .. " thinking (" .. mover .. ")", C.accent, true)
    else
        line(mover .. " to move", C.text, true)
    end
    line(("move %d"):format(game:moveCount()), C.dim)

    -- evaluation bar, from the point of view of the side to move
    local result = ui.lastResult
    if result then
        local value = result.value
        local width = 20
        local filled = math.floor((value + 1) / 2 * width + 0.5)
        s:at(py, px):fg(C.dim):put("eval "):reset()
        s:fg(value >= 0 and C.good or C.bad)
        s:put(string.rep("█", filled) .. string.rep("░", width - filled))
        s:reset():fg(C.dim):put((" %+.2f"):format(value)):reset()
        py = py + 2
    else
        py = py + 1
    end

    if ui.hint then
        line("THE MODEL LIKES", C.accent, true)
        for rank, entry in ipairs(ui.hint.top) do
            if rank > 3 then break end
            line(("%d. %-4s %4.0f%%"):format(rank, omok.name(entry.move, size),
                                             entry.prob * 100), C.hint)
        end
        py = py + 1
    end

    if #ui.history > 0 then
        line("MOVES", C.accent, true)
        local first = math.max(1, #ui.history - 7)
        for i = first, #ui.history do
            line(ui.history[i], i == #ui.history and C.text or C.dim)
        end
    end

    -- ------------------------------------------------------------- footer
    local footer = top + size + 3
    s:at(footer, left):fg(C.text):put(ui.message):reset()
    s:at(footer + 1, left):fg(C.dim)
     :put("space place   u undo   ? hint   n new   1-5 level   w watch   q quit")
     :reset()
    s:at(rows, 1)
    s:flush()
end

-- --------------------------------------------------------------------- input

local function moveCursor(dr, dc)
    local r, c = omok.rc(ui.cursor, size)
    r = math.max(0, math.min(size - 1, r + dr))
    c = math.max(0, math.min(size - 1, c + dc))
    ui.cursor = r * size + c
end

local function place()
    if game:isOver() then note("the game is over -- press n for a new one") return end
    if not humanTurn() then note("not your turn") return end
    if not game:play(ui.cursor) then note("that square is taken") return end
    record(options.human == omok.BLACK and "you" or "you", ui.cursor)
    ui.hint = nil
    ui.lastResult = nil
    note("")
end

local function newGame()
    stopThinking()
    game:reset()
    ui.history = {}
    ui.lastResult = nil
    ui.hint = nil
    ui.cursor = math.floor(size / 2) * size + math.floor(size / 2)
    note("new game -- you are " .. (options.human == omok.BLACK and "black" or "white"))
end

local function undo()
    stopThinking()
    -- Take back the pair, so it is the human's turn again.
    game:undo()
    if not options.watch and game:toMove() ~= options.human and game:moveCount() > 0 then
        game:undo()
    end
    while #ui.history > game:moveCount() do table.remove(ui.history) end
    ui.lastResult = nil
    ui.hint = nil
    note("took it back")
end

local function handle(key)
    if key == "q" or key == "escape" then ui.quit = true
    elseif key == "up" or key == "k" then moveCursor(-1, 0)
    elseif key == "down" or key == "j" then moveCursor(1, 0)
    elseif key == "left" or key == "h" then moveCursor(0, -1)
    elseif key == "right" or key == "l" then moveCursor(0, 1)
    elseif key == " " or key == "\r" or key == "\n" then place()
    elseif key == "u" then undo()
    elseif key == "n" then newGame()
    elseif key == "w" then
        options.watch = not options.watch
        stopThinking()
        note(options.watch and "watching the model play itself" or "your move again")
    elseif key == "?" or key == "e" then
        if game:isOver() then note("the game is over") else
            startThinking("hint")
            note("asking the model...")
        end
    elseif key and key:match("^[1-5]$") then
        options.level = tonumber(key)
        note("level " .. options.level .. " -- " .. LEVELS[options.level].name)
    end
end

-- ---------------------------------------------------------------- the loop

local function applyResult(result)
    if ui.thinking == "hint" then
        ui.hint = result
        ui.lastResult = result
        ui.cursor = result.move
        note(("the model would play %s"):format(result.name))
    else
        ui.lastResult = result
        local who = game:toMove() == omok.BLACK and "black" or "white"
        game:play(result.move)
        record(options.watch and who or "ai", result.move, result)
        ui.cursor = result.move
        note(("model plays %s in %.2fs (%.0f sims/s)"):format(
             result.name, result.seconds,
             result.simulations / math.max(result.seconds, 1e-6)))
    end
    ui.thinking = false
end

local function step()
    ui.frame = ui.frame + 1

    if ui.thinking then
        local result, err = engine:poll()
        if err then
            note("engine: " .. err)
            ui.thinking = false
        elseif result then
            applyResult(result)
        end
    elseif not game:isOver() and not humanTurn() then
        startThinking("move")
    end

    draw()

    local key = term.key()
    while key do
        handle(key)
        if ui.quit then return end
        key = term.key()
    end
end

term.raw()
local running, err = pcall(function()
    note(options.watch and "watching the model play itself"
                        or ("you are " .. (options.human == omok.BLACK and "black" or "white")
                            .. " -- arrows to move, space to place"))
    while not ui.quit do step() end
end)
stopThinking()
term.restore()

if not running then
    io.stderr:write("error: " .. tostring(err) .. "\n")
    os.exit(1)
end
print("thanks for playing Causewaybay Omok")
