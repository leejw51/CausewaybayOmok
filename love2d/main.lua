--- Causewaybay Omok -- a LÖVE game against the trained network.
--
--     love love2d
--
-- The board, the rules and the search all live in the Rust core; this file is
-- the part you can see.  The engine thinks on its own thread, so `love.update`
-- polls it instead of waiting: the window keeps running at full speed while
-- the model works, and undoing a move abandons the search rather than queuing
-- behind it.
--
-- Controls: click or arrow keys to move, space to place, U undo, H hint,
-- N new game, 1-5 difficulty, TAB swap colours, W watch, F11 fullscreen.

-- The Lua bindings are shared with the terminal client, so they are loaded
-- from `lua/` on the real filesystem rather than copied in here.  LÖVE keeps
-- the standard Lua file loader alongside its own, so `package.path` works.
local source = love.filesystem.getSource()
package.path = table.concat({
    source .. "/../lua/?.lua",
    source .. "/../lua/?/init.lua",
    source .. "/?.lua",
    package.path,
}, ";")

local omok = require("omok")
local effects = require("effects")
local layout = require("layout")
local ease = effects.ease

-- ------------------------------------------------------------------ palette

-- Prague at dusk: amber lamplight against an indigo sky, which is also what
-- separates the two players -- black stones take the indigo, white the amber.
local P = {
    ink        = {0.05, 0.06, 0.11},
    panel      = {0.07, 0.08, 0.15, 0.90},
    panelEdge  = {0.85, 0.65, 0.30, 0.55},
    gold       = {1.00, 0.78, 0.35},
    brightGold = {1.00, 0.90, 0.60},
    cream      = {1.00, 0.96, 0.88},
    indigo     = {0.42, 0.50, 0.88},
    deepIndigo = {0.20, 0.24, 0.52},
    grid       = {0.20, 0.14, 0.08, 0.85},
    good       = {0.45, 0.90, 0.55},
    bad        = {0.95, 0.45, 0.45},
    dim        = {0.62, 0.62, 0.72},
    hint       = {0.70, 0.55, 1.00},
}

local function stoneColour(player)
    return player == omok.BLACK and P.indigo or P.gold
end

-- Simulations, and how loosely the model picks among its best moves.  The easy
-- levels are not merely faster: the higher temperature makes them stray from
-- the top move often enough to be beatable.
local LEVELS = {
    {name = "STROLL",    simulations = 24,   temperature = 1.2},
    {name = "CASUAL",    simulations = 96,   temperature = 0.7},
    {name = "SHARP",     simulations = 320,  temperature = 0.25},
    {name = "SERIOUS",   simulations = 900,  temperature = 0.0},
    {name = "MERCILESS", simulations = 2400, temperature = 0.0},
}

-- --------------------------------------------------------------------- state

local art = {}
local font
local S = {
    level = 2,
    human = omok.BLACK,
    watch = false,
    time = 0,
    stones = {},           -- flat index -> {player, tween}
    hover = nil,
    cursor = nil,
    thinking = nil,        -- "move" | "hint"
    hint = nil,
    evalShown = 0,
    evalTarget = 0,
    history = {},
    message = "",
    messageAge = 99,
    buttons = {},
    fatal = nil,
    parallaxX = 0, parallaxY = 0,
    winTween = nil,
    introTween = nil,
    bannerTween = nil,
}

-- ------------------------------------------------------------------- helpers

local function say(text)
    S.message = text
    S.messageAge = 0
end

local function level()
    return LEVELS[S.level]
end

local function humanTurn()
    if S.watch then return false end
    return S.game:toMove() == S.human
end

local function boardCentre(index)
    local row, col = omok.rc(index, S.size)
    return layout.cellCentre(S.frame.board, row, col)
end

local function text(str, x, y, scale, colour, align, width)
    scale = scale or 1
    love.graphics.setColor(colour or P.cream)
    local w = font:getWidth(str) * scale
    if align == "center" then x = x + (width - w) / 2
    elseif align == "right" then x = x + width - w end
    love.graphics.print(str, math.floor(x), math.floor(y), 0, scale, scale)
    love.graphics.setColor(1, 1, 1, 1)
    return w
end

local function textWidth(str, scale)
    return font:getWidth(str) * (scale or 1)
end

-- ------------------------------------------------------------------ the game

local function refreshLayout()
    local w, h = love.graphics.getDimensions()
    S.frame = layout.compute(w, h, S.size)
    if S.motes then S.motes:resize(w, h) end
    -- The board is a repeating texture; the quad only changes when the board
    -- does, so it is built here rather than in the draw loop.
    art.board:setWrap("repeat", "repeat")
    local tw, th = art.board:getDimensions()
    S.boardQuad = love.graphics.newQuad(0, 0, S.frame.board.size / 2,
                                        S.frame.board.size / 2, tw, th)
end

local function newGame(keepColours)
    if S.engine then S.engine:cancel() end
    S.thinking = nil
    S.game:reset()
    S.stones = {}
    S.history = {}
    S.hint = nil
    S.winTween = nil
    S.bannerTween = nil
    S.evalTarget, S.evalShown = 0, 0
    S.cursor = math.floor(S.size / 2) * S.size + math.floor(S.size / 2)
    S.particles:clear()
    S.waves:clear()
    S.floaters:clear()
    -- The board fades in from the centre outwards, so a new game reads as a
    -- fresh start rather than as stones vanishing.
    S.introTween = effects.tween(0.75, ease.outCubic)
    if not keepColours then
        say(S.watch and "the model plays itself" or
            ("you are " .. (S.human == omok.BLACK and "indigo" or "amber")))
    end
end

--- Drop a stone with all the trimmings: the landing animation, a shockwave,
-- a burst of sparks and a floating label.
local function placeStone(index, player, label)
    S.stones[index] = {
        player = player,
        tween = effects.tween(0.42, ease.outBack),
    }
    local x, y = boardCentre(index)
    local colour = stoneColour(player)
    local cell = S.frame.board.cell

    S.waves:add(x, y, {
        from = cell * 0.25, to = cell * 2.1, life = 0.55,
        colour = colour, curve = ease.outQuart,
    })
    S.particles:burst(x, y, 22, {
        speed = cell * 5.5, life = 0.55, size = cell * 0.4,
        colour = colour, gravity = cell * 3.5, drag = 2.6,
    })
    S.shake:kick(2.5)
    if label then
        S.floaters:add(label, x, y - cell * 0.7,
                       {colour = colour, scale = 2, life = 1.0})
    end
end

local function pushHistory(who, index, result)
    local line = ("%2d %s %s"):format(S.game:moveCount(), who, omok.name(index, S.size))
    if result then line = line .. (" %+.2f"):format(result.value) end
    S.history[#S.history + 1] = line
end

local function checkFinished()
    if not S.game:isOver() then return end
    S.winTween = effects.tween(1.4, ease.outCubic)
    S.bannerTween = effects.tween(0.9, ease.outElastic)
    S.shake:kick(14)

    local winner = S.game:winner()
    local line = S.game:winLine()
    if winner == omok.EMPTY then
        say("a draw -- the board is full")
        return
    end
    local colour = stoneColour(winner)
    -- A fountain from each stone of the winning line, walked along the line so
    -- the eye follows it from one end to the other.
    for i, index in ipairs(line) do
        local x, y = boardCentre(index)
        S.waves:add(x, y, {
            from = S.frame.board.cell * 0.4, to = S.frame.board.cell * 3.4,
            life = 0.9, colour = colour,
        })
        S.particles:burst(x, y, 40, {
            speed = 320, life = 1.5, size = S.frame.board.cell * 0.55,
            colour = colour, gravity = 260, drag = 1.1, angle = -math.pi / 2,
            spread = math.pi * 1.2,
        })
        S.floaters:add(tostring(i), x, y, {colour = colour, scale = 2, life = 1.4})
    end
    say(winner == S.human and not S.watch and "you win!" or
        (winner == omok.BLACK and "indigo wins" or "amber wins"))
end

local function startThinking(kind)
    if S.game:isOver() then return end
    local opts = {
        simulations = level().simulations,
        -- A hint is advice, so it names the move the search actually rated
        -- best; the easy levels' sampling would otherwise suggest something
        -- the model does not believe in.
        temperature = (kind == "hint") and 0.0 or level().temperature,
    }
    local started, err = S.engine:begin(S.game, opts)
    if not started then
        say(tostring(err))
        return
    end
    S.thinking = kind
    S.hint = nil
end

local function humanPlay(index)
    if S.game:isOver() then say("press N for a new game") return end
    if not humanTurn() then say("the model is thinking") return end
    if not S.game:play(index) then say("that point is taken") return end
    placeStone(index, S.human, omok.name(index, S.size))
    pushHistory("you", index)
    S.hint = nil
    checkFinished()
end

local function undo()
    if S.game:moveCount() == 0 then say("nothing to take back") return end
    S.engine:cancel()
    S.thinking = nil
    S.winTween, S.bannerTween = nil, nil

    local function pop()
        local index = S.game:lastMove()
        if not index then return end
        local x, y = boardCentre(index)
        local stone = S.stones[index]
        S.particles:implode(x, y, 18, {
            colour = stone and stoneColour(stone.player) or P.dim,
            radius = S.frame.board.cell * 1.4, life = 0.4,
            size = S.frame.board.cell * 0.3,
        })
        S.stones[index] = nil
        S.game:undo()
        if #S.history > 0 then table.remove(S.history) end
    end

    pop()
    -- Take back the pair, so it is the human's turn again.
    if not S.watch and S.game:toMove() ~= S.human and S.game:moveCount() > 0 then
        pop()
    end
    S.hint = nil
    S.shake:kick(3)
    say("taken back")
end

local function applyResult(result)
    if S.thinking == "hint" then
        S.hint = result
        S.evalTarget = result.value
        S.cursor = result.move
        local x, y = boardCentre(result.move)
        S.waves:add(x, y, {from = 4, to = S.frame.board.cell * 1.8,
                           life = 0.7, colour = P.hint})
        say("the model likes " .. result.name)
    else
        local player = S.game:toMove()
        S.evalTarget = result.value
        S.game:play(result.move)
        placeStone(result.move, player, result.name)
        pushHistory(S.watch and (player == omok.BLACK and "ind" or "amb") or "ai", result.move, result)
        S.cursor = result.move
        say(("model plays %s  %.2fs  %d sims"):format(
            result.name, result.seconds, result.simulations))
        checkFinished()
    end
    S.thinking = nil
end

-- ------------------------------------------------------------------- buttons

--- Buttons are rebuilt every frame from the current layout, so they follow the
-- window through a resize or a flip to portrait without any bookkeeping.
local function button(id, label, x, y, w, h, action, active)
    local mx, my = love.mouse.getPosition()
    local hot = mx >= x and mx <= x + w and my >= y and my <= y + h
    S.buttons[#S.buttons + 1] = {
        id = id, label = label, x = x, y = y, w = w, h = h,
        action = action, hot = hot, active = active,
    }
    return hot
end

local function drawButtons()
    for _, b in ipairs(S.buttons) do
        local lift = b.hot and 2 or 0
        local glow = b.hot and (0.35 + 0.25 * effects.pulse(S.time, 1.2)) or 0
        local y = b.y - lift

        if b.active then
            love.graphics.setColor(P.gold[1], P.gold[2], P.gold[3], 0.28)
        else
            love.graphics.setColor(0.10, 0.11, 0.20, 0.85)
        end
        love.graphics.rectangle("fill", b.x, y, b.w, b.h)

        local edge = b.active and P.gold or P.panelEdge
        love.graphics.setColor(edge[1], edge[2], edge[3], 0.5 + glow)
        love.graphics.setLineWidth(1)
        love.graphics.rectangle("line", b.x + 0.5, y + 0.5, b.w - 1, b.h - 1)

        local colour = b.active and P.brightGold or (b.hot and P.cream or P.dim)
        text(b.label, b.x, y + (b.h - font:getHeight()) / 2, 1, colour, "center", b.w)
    end
    love.graphics.setColor(1, 1, 1, 1)
end

-- ------------------------------------------------------------------- drawing

local function drawBackdrop()
    local w, h = love.graphics.getDimensions()
    local iw, ih = art.backdrop:getDimensions()
    -- Cover the window, with a little slack for the parallax to move into.
    local scale = math.max(w / iw, h / ih) * 1.06
    local x = (w - iw * scale) / 2 + S.parallaxX
    local y = (h - ih * scale) / 2 + S.parallaxY
    love.graphics.setColor(1, 1, 1, 1)
    love.graphics.draw(art.backdrop, x, y, 0, scale, scale)

    -- A vignette, so the panel and the board sit on something darker than the
    -- lit square behind them.
    love.graphics.setColor(P.ink[1], P.ink[2], P.ink[3], 0.30)
    love.graphics.rectangle("fill", 0, 0, w, h)
    love.graphics.setColor(1, 1, 1, 1)
end

local STAR_POINTS = {3, 7, 11}

local function drawBoard()
    local b = S.frame.board
    local intro = S.introTween and S.introTween:at() or 1

    -- shadow
    love.graphics.setColor(0, 0, 0, 0.55 * intro)
    love.graphics.rectangle("fill", b.x + 6, b.y + 8, b.size, b.size)

    -- the wooden surface, tiled at 2x so the grain keeps its chunky pixels
    art.board:setWrap("repeat", "repeat")
    local tw, th = art.board:getDimensions()
    local quad = love.graphics.newQuad(0, 0, b.size / 2, b.size / 2, tw, th)
    -- Knocked back and cooled a shade so the board sits in the same dusk as
    -- the square behind it instead of glowing off the screen.
    love.graphics.setColor(0.74, 0.68, 0.62, intro)
    love.graphics.draw(art.board, quad, b.x, b.y, 0, 2, 2)

    -- border
    love.graphics.setColor(P.gold[1], P.gold[2], P.gold[3], 0.55 * intro)
    love.graphics.setLineWidth(2)
    love.graphics.rectangle("line", b.x + 1, b.y + 1, b.size - 2, b.size - 2)

    -- grid, drawn as one-pixel rectangles so every line is exactly as wide as
    -- every other one whatever the cell size works out at
    love.graphics.setColor(P.grid[1], P.grid[2], P.grid[3], P.grid[4] * intro)
    local first = b.cell / 2
    local span = (b.cells - 1) * b.cell
    for i = 0, b.cells - 1 do
        local at = math.floor(first + i * b.cell)
        love.graphics.rectangle("fill", b.x + math.floor(first), b.y + at, span + 1, 1)
        love.graphics.rectangle("fill", b.x + at, b.y + math.floor(first), 1, span + 1)
    end

    -- star points
    for _, r in ipairs(STAR_POINTS) do
        for _, c in ipairs(STAR_POINTS) do
            if r < b.cells and c < b.cells then
                local x, y = layout.cellCentre(b, r, c)
                love.graphics.circle("fill", x, y, math.max(2, b.cell * 0.09))
            end
        end
    end

    -- coordinates
    for i = 0, b.cells - 1 do
        local x, y = layout.cellCentre(b, i, i)
        text(string.char(string.byte("a") + i), x - 3, b.y - font:getHeight() - 3, 1,
             {P.gold[1], P.gold[2], P.gold[3], 0.5 * intro})
        text(tostring(i), b.x - 16, y - font:getHeight() / 2, 1,
             {P.gold[1], P.gold[2], P.gold[3], 0.5 * intro})
    end
    love.graphics.setColor(1, 1, 1, 1)
end

local function drawStone(index, player, scale, alpha)
    local x, y = boardCentre(index)
    local sprite = player == omok.BLACK and art.stone_black or art.stone_white
    local sw, sh = sprite:getDimensions()
    local radius = S.frame.board.cell * 0.44
    local draw = radius * 2 / sw * scale

    -- The shadow grows with the stone, which is what sells the drop.
    love.graphics.setColor(0, 0, 0, 0.45 * alpha)
    love.graphics.draw(sprite, x + 2, y + 3, 0, draw, draw, sw / 2, sh / 2)
    love.graphics.setColor(1, 1, 1, alpha)
    love.graphics.draw(sprite, x, y, 0, draw, draw, sw / 2, sh / 2)
end

local function drawStones()
    local b = S.frame.board
    local last = S.game:lastMove()
    local winning = {}
    if S.game:isOver() then
        for _, index in ipairs(S.game:winLine()) do winning[index] = true end
    end

    -- the marker under the last move, pulsing so the eye finds it
    if last and S.stones[last] and S.stones[last].tween:done() and not S.game:isOver() then
        local x, y = boardCentre(last)
        local pulse = 0.55 + 0.45 * effects.pulse(S.time, 1.6)
        local colour = stoneColour(S.stones[last].player)
        local sw = art.halo:getWidth()
        love.graphics.setBlendMode("add", "alphamultiply")
        love.graphics.setColor(colour[1], colour[2], colour[3], 0.30 * pulse)
        local ring = b.cell * (1.15 + 0.10 * pulse)
        love.graphics.draw(art.halo, x, y, 0, ring / sw, ring / sw, sw / 2, sw / 2)
        love.graphics.setBlendMode("alpha", "alphamultiply")
    end

    for index, stone in pairs(S.stones) do
        local t = stone.tween:at()
        local scale = 1
        if not stone.tween:done() then
            -- Falls in from above its square, overshooting on the way down.
            scale = effects.lerp(2.3, 1.0, stone.tween:raw(), ease.outBack)
        end
        if winning[index] and S.winTween then
            -- The winning stones breathe, each a beat behind the last.
            local order = 0
            for i, w in ipairs(S.game:winLine()) do if w == index then order = i end end
            local phase = S.time * 2.4 - order * 0.35
            scale = scale * (1 + 0.14 * (0.5 + 0.5 * math.sin(phase)))

            local x, y = boardCentre(index)
            local colour = stoneColour(S.game:winner())
            local sw = art.halo:getWidth()
            love.graphics.setBlendMode("add", "alphamultiply")
            love.graphics.setColor(colour[1], colour[2], colour[3],
                                   0.5 * (0.5 + 0.5 * math.sin(phase)))
            local ring = b.cell * 1.6
            love.graphics.draw(art.halo, x, y, 0, ring / sw, ring / sw, sw / 2, sw / 2)
            love.graphics.setBlendMode("alpha", "alphamultiply")
        end
        drawStone(index, stone.player, scale, math.min(1, t * 3))
    end
    love.graphics.setColor(1, 1, 1, 1)
end

local function drawOverlay()
    local b = S.frame.board
    local target = S.hover or S.cursor

    -- what the model suggested, ranked
    if S.hint then
        for rank, entry in ipairs(S.hint.top) do
            if rank > 3 then break end
            local x, y = boardCentre(entry.move)
            local pulse = 0.5 + 0.5 * effects.pulse(S.time + rank * 0.2, 1.4)
            love.graphics.setColor(P.hint[1], P.hint[2], P.hint[3], 0.25 + 0.30 * pulse)
            love.graphics.setLineWidth(2)
            love.graphics.circle("line", x, y, b.cell * (0.42 - rank * 0.04))
            text(tostring(rank), x - 2, y - font:getHeight() / 2, 1, P.hint)
        end
    end

    if S.game:isOver() or not humanTurn() or not target then return end

    local x, y = boardCentre(target)
    local free = S.game:legal(target)

    -- crosshair guides, so a click on a big board is easy to line up
    love.graphics.setColor(P.gold[1], P.gold[2], P.gold[3], 0.16)
    love.graphics.rectangle("fill", b.x, y, b.size, 1)
    love.graphics.rectangle("fill", x, b.y, 1, b.size)

    if free then
        local pulse = 0.55 + 0.45 * effects.pulse(S.time, 0.9)
        drawStone(target, S.human, 0.86 + 0.05 * pulse, 0.36 + 0.16 * pulse)
        local colour = stoneColour(S.human)
        love.graphics.setColor(colour[1], colour[2], colour[3], 0.5 + 0.3 * pulse)
        love.graphics.setLineWidth(2)
        love.graphics.circle("line", x, y, b.cell * 0.48)
    else
        love.graphics.setColor(P.bad[1], P.bad[2], P.bad[3], 0.55)
        love.graphics.setLineWidth(2)
        local r = b.cell * 0.28
        love.graphics.line(x - r, y - r, x + r, y + r)
        love.graphics.line(x - r, y + r, x + r, y - r)
    end
    love.graphics.setColor(1, 1, 1, 1)
end

--- The shimmer that crosses the board while the model is thinking.
local function drawThinking()
    if not S.thinking then return end
    local b = S.frame.board
    local sweep = (S.time * 0.55) % 1.4 / 1.4
    local y = b.y + sweep * b.size
    love.graphics.setBlendMode("add", "alphamultiply")
    for i = 0, 10 do
        local alpha = 0.045 * (1 - i / 10)
        love.graphics.setColor(P.gold[1], P.gold[2], P.gold[3], alpha)
        love.graphics.rectangle("fill", b.x, y - i * 2, b.size, 2)
    end
    love.graphics.setBlendMode("alpha", "alphamultiply")
    love.graphics.setColor(1, 1, 1, 1)
end

local function drawTurnStone(x, y, radius, player, spin)
    local sprite = player == omok.BLACK and art.stone_black or art.stone_white
    local sw = sprite:getWidth()
    love.graphics.setColor(1, 1, 1, 1)
    love.graphics.draw(sprite, x, y, spin or 0, radius * 2 / sw, radius * 2 / sw,
                       sw / 2, sw / 2)
end

local function drawPanel()
    local p = S.frame.panel
    local portrait = S.frame.portrait

    love.graphics.setColor(P.panel)
    love.graphics.rectangle("fill", p.x, p.y, p.width, p.height)
    love.graphics.setColor(P.panelEdge)
    love.graphics.setLineWidth(1)
    love.graphics.rectangle("line", p.x + 0.5, p.y + 0.5, p.width - 1, p.height - 1)
    love.graphics.setColor(1, 1, 1, 1)

    -- Everything below is clipped to the panel, so a long move list or an
    -- unusually wide window can never spill text across the board.
    love.graphics.setScissor(p.x + 1, p.y + 1, p.width - 2, p.height - 2)

    local pad = 12
    local x = p.x + pad
    local y = p.y + pad
    local width = p.width - pad * 2
    local lineHeight = font:getHeight() + 3

    -- In portrait the panel is a wide strip, so it runs in two columns rather
    -- than as one long list that would not fit its height.
    local columnWidth = portrait and math.floor((width - pad * 2) / 3) or width
    local column2 = x + columnWidth + pad
    local column3 = x + (columnWidth + pad) * 2

    -- ---- title and engine
    text("CAUSEWAYBAY", x, y, 2, P.gold)
    text("OMOK", x + textWidth("CAUSEWAYBAY ", 2), y, 2, P.cream)
    y = y + lineHeight * 2 + 2
    text(S.engineLine, x, y, 1, P.dim)
    y = y + lineHeight + 4

    -- ---- whose turn
    local turnX, turnY = x, y
    if S.game:isOver() then
        local winner = S.game:winner()
        local label = winner == omok.EMPTY and "DRAW"
                      or ((winner == omok.BLACK and "INDIGO" or "AMBER") .. " WINS")
        text(label, turnX + 26, turnY + 3, 2,
             winner == omok.EMPTY and P.dim or stoneColour(winner))
        if winner ~= omok.EMPTY then
            drawTurnStone(turnX + 11, turnY + 11, 10, winner, S.time * 1.5)
        end
    else
        local player = S.game:toMove()
        local spin = S.thinking and S.time * 3.5 or 0
        local bob = S.thinking and math.sin(S.time * 6) * 2 or 0
        drawTurnStone(turnX + 11, turnY + 11 + bob, 10, player, spin)
        local label
        if S.thinking == "hint" then label = "ADVISING"
        elseif S.thinking then label = "THINKING"
        elseif humanTurn() then label = "YOUR MOVE"
        else label = "MODEL" end
        local colour = S.thinking and P.gold or P.cream
        text(label, turnX + 26, turnY + 3, 2, colour)
        if S.thinking then
            -- three dots, filling in turn
            local dots = math.floor(S.time * 3) % 4
            text(("."):rep(dots), turnX + 26 + textWidth(label, 2) + 4, turnY + 3, 2, P.gold)
        end
    end
    y = y + lineHeight * 2

    -- ---- evaluation bar
    local barX = portrait and column2 or x
    local barY = portrait and (p.y + pad + lineHeight * 2 + 2) or y
    local barWidth = portrait and columnWidth or width
    text("EVALUATION", barX, barY, 1, P.dim)
    barY = barY + lineHeight
    local barHeight = 12
    love.graphics.setColor(0.03, 0.03, 0.07, 0.9)
    love.graphics.rectangle("fill", barX, barY, barWidth, barHeight)
    local centre = barX + barWidth / 2
    local half = (barWidth / 2) * math.abs(S.evalShown)
    local colour = S.evalShown >= 0 and P.good or P.bad
    love.graphics.setColor(colour[1], colour[2], colour[3], 0.85)
    if S.evalShown >= 0 then
        love.graphics.rectangle("fill", centre, barY, half, barHeight)
    else
        love.graphics.rectangle("fill", centre - half, barY, half, barHeight)
    end
    love.graphics.setColor(P.cream[1], P.cream[2], P.cream[3], 0.6)
    love.graphics.rectangle("fill", centre, barY, 1, barHeight)
    love.graphics.setColor(P.panelEdge)
    love.graphics.rectangle("line", barX + 0.5, barY + 0.5, barWidth - 1, barHeight - 1)
    text(("%+.2f"):format(S.evalShown), barX, barY + barHeight + 3, 1, P.dim)
    text("for the side to move", barX, barY + barHeight + 3 + lineHeight, 1,
         {P.dim[1], P.dim[2], P.dim[3], 0.7})
    if not portrait then y = barY + barHeight + lineHeight * 2 + 8 end

    -- ---- difficulty
    local diffX = portrait and column2 or x
    local diffY = portrait and (barY + barHeight + lineHeight * 2 + 8) or y
    text("DIFFICULTY", diffX, diffY, 1, P.dim)
    diffY = diffY + lineHeight + 2
    local slotWidth = math.floor(((portrait and columnWidth or width) - 4 * 3) / 5)
    for i = 1, 5 do
        button("level" .. i, tostring(i), diffX + (i - 1) * (slotWidth + 3), diffY,
               slotWidth, 20, function() S.level = i say(LEVELS[i].name) end, S.level == i)
    end
    diffY = diffY + 24
    text(("%s  %d sims"):format(level().name, level().simulations), diffX, diffY, 1, P.gold)
    if not portrait then y = diffY + lineHeight + 8 end

    -- ---- buttons
    local actX = portrait and column3 or x
    local actY = portrait and (p.y + pad + lineHeight) or y
    local actWidth = portrait and columnWidth or width
    local half2 = math.floor((actWidth - 4) / 2)
    button("new", "NEW GAME", actX, actY, half2, 22, function() newGame() end)
    button("undo", "UNDO", actX + half2 + 4, actY, half2, 22, undo)
    actY = actY + 26
    button("hint", "HINT", actX, actY, half2, 22, function()
        if S.game:isOver() then say("the game is over") else startThinking("hint") end
    end)
    button("watch", "WATCH", actX + half2 + 4, actY, half2, 22, function()
        S.watch = not S.watch
        S.engine:cancel()
        S.thinking = nil
        say(S.watch and "the model plays itself" or "your move again")
    end, S.watch)
    actY = actY + 26
    button("swap", "SWAP SIDES", actX, actY, half2, 22, function()
        S.human = S.human == omok.BLACK and omok.WHITE or omok.BLACK
        S.engine:cancel()
        S.thinking = nil
        say("you are " .. (S.human == omok.BLACK and "indigo" or "amber"))
    end)
    button("full", love.window.getFullscreen() and "WINDOW" or "FULL",
           actX + half2 + 4, actY, half2, 22, function()
        love.window.setFullscreen(not love.window.getFullscreen(), "desktop")
        refreshLayout()
    end)
    actY = actY + 26
    if not portrait then y = actY + 8 end

    -- ---- move list, in whatever space the arrangement left over
    local listX = x
    local listY = portrait and (p.y + pad + lineHeight * 5 + 6) or y
    text("MOVES", listX, listY, 1, P.dim)
    listY = listY + lineHeight + 2
    local room = math.floor((p.y + p.height - pad - lineHeight - 4 - listY) / lineHeight)
    local first = math.max(1, #S.history - room + 1)
    for i = first, #S.history do
        text(S.history[i], listX, listY, 1, i == #S.history and P.cream or P.dim)
        listY = listY + lineHeight
    end

    -- ---- key hints along the bottom
    local hints = "SPACE PLACE  U UNDO  H HINT  N NEW  F11 FULL"
    if textWidth(hints, 1) > width then hints = "SPACE  U  H  N  F11" end
    text(hints, p.x + pad, p.y + p.height - pad - font:getHeight(), 1,
         {P.dim[1], P.dim[2], P.dim[3], 0.65})

    love.graphics.setScissor()
end

local function drawBanner()
    if not S.bannerTween then return end
    local w, h = love.graphics.getDimensions()
    local t = S.bannerTween:at()
    local winner = S.game:winner()
    local label = winner == omok.EMPTY and "DRAW"
                  or ((winner == S.human and not S.watch) and "YOU WIN"
                      or (winner == omok.BLACK and "INDIGO WINS" or "AMBER WINS"))
    local colour = winner == omok.EMPTY and P.dim or stoneColour(winner)

    local scale = 5
    local width = textWidth(label, scale)
    local boardCentreX = S.frame.board.x + S.frame.board.size / 2
    local y = effects.lerp(-80, S.frame.board.y + S.frame.board.size / 2 - 40, t)

    love.graphics.setColor(P.ink[1], P.ink[2], P.ink[3], 0.75 * math.min(1, t * 2))
    love.graphics.rectangle("fill", 0, y - 14, w, font:getHeight() * scale + 28)
    love.graphics.setColor(colour[1], colour[2], colour[3], 0.7)
    love.graphics.rectangle("fill", 0, y - 14, w, 2)
    love.graphics.rectangle("fill", 0, y + font:getHeight() * scale + 12, w, 2)

    text(label, boardCentreX - width / 2 + 2, y + 2, scale, {0, 0, 0, 0.6})
    text(label, boardCentreX - width / 2, y, scale, colour)
    text("press N for another", boardCentreX - textWidth("press N for another", 1) / 2,
         y + font:getHeight() * scale + 18, 1, P.cream)
end

local function drawMessage()
    if S.messageAge > 4 or S.message == "" then return end
    local alpha = math.min(1, (4 - S.messageAge) * 1.5)
    local b = S.frame.board
    local width = textWidth(S.message, 1)
    local x = b.x + (b.size - width) / 2
    local y = b.y + b.size + 8
    if y + font:getHeight() > love.graphics.getHeight() then y = b.y - font:getHeight() - 8 end
    love.graphics.setColor(P.ink[1], P.ink[2], P.ink[3], 0.7 * alpha)
    love.graphics.rectangle("fill", x - 8, y - 4, width + 16, font:getHeight() + 8)
    text(S.message, x, y, 1, {P.cream[1], P.cream[2], P.cream[3], alpha})
end

-- ------------------------------------------------------------------ love API

function love.load()
    love.graphics.setDefaultFilter("nearest", "nearest")
    love.graphics.setLineStyle("rough")
    math.randomseed(os.time())

    local glyphs = ""
    for c = 32, 126 do glyphs = glyphs .. string.char(c) end
    font = love.graphics.newImageFont("assets/font.png", glyphs)
    love.graphics.setFont(font)

    for _, name in ipairs({"backdrop", "board", "stone_black", "stone_white",
                           "spark", "halo"}) do
        art[name] = love.graphics.newImage("assets/" .. name .. ".png")
    end

    local libraryPath = omok.findLibrary()
    if not libraryPath then
        S.fatal = "The AI core is not built.\n\nRun:  make ai"
        S.size = 15
        return
    end
    omok.load(libraryPath)

    local modelPath = omok.findModel()
    if not modelPath then
        S.fatal = "No trained model found.\n\nRun:  make train\nor set OMOK_MODEL"
        S.size = 15
        return
    end

    local ok, engine = pcall(omok.Engine, modelPath)
    if not ok then
        S.fatal = "Could not start the engine:\n\n" .. tostring(engine)
        S.size = 15
        return
    end
    S.engine = engine
    S.size = engine:boardSize()
    -- The full description names the checkpoint path too, which is far wider
    -- than the panel; the panel gets the part that fits.
    local blocks, channels = engine:describe():match("(%d+) blocks x (%d+) channels")
    S.engineLine = blocks and ("MLX  %sX%s NET  %dX%d"):format(blocks, channels, S.size, S.size)
                          or "MLX"
    S.game = omok.Game(S.size)

    S.particles = effects.particles(art.spark)
    S.waves = effects.waves(art.halo)
    S.floaters = effects.floaters()
    S.shake = effects.shake()
    local w, h = love.graphics.getDimensions()
    S.motes = effects.motes(art.spark, 44, w, h)

    refreshLayout()
    newGame()

    S.shotPath = os.getenv("OMOK_SCREENSHOT")
    S.shotAfter = tonumber(os.getenv("OMOK_SHOT_AFTER") or "") or 6
    if S.shotPath then S.watch = true end
end

--- Dev aid: `OMOK_SCREENSHOT=path.png love love2d` plays a few moves, grabs a
-- frame and quits, so the README's pictures can be remade without a person at
-- the keyboard.  `OMOK_SHOT_AFTER` sets how long to play for first.
local function screenshotHook(dt)
    if not S.shotPath then return end
    S.shotClock = (S.shotClock or 0) + dt
    if S.shotClock < (S.shotAfter or 6) then return end
    local path = S.shotPath
    S.shotPath = nil
    love.graphics.captureScreenshot(function(imageData)
        -- Written straight out rather than through the save directory, which
        -- is on a different volume from the project on most machines.
        local file = io.open(path, "wb")
        if file then
            file:write(imageData:encode("png"):getString())
            file:close()
        end
        love.event.quit()
    end)
end

function love.update(dt)
    S.time = S.time + dt
    S.messageAge = S.messageAge + dt
    if S.fatal then return end

    if S.introTween then
        if S.introTween:update(dt) then S.introTween = nil end
    end
    if S.winTween then S.winTween:update(dt) end
    if S.bannerTween then S.bannerTween:update(dt) end
    for _, stone in pairs(S.stones) do stone.tween:update(dt) end

    S.particles:update(dt)
    S.waves:update(dt)
    S.floaters:update(dt)
    S.shake:update(dt)
    S.motes:update(dt, S.time)

    S.evalShown = effects.approach(S.evalShown, S.evalTarget, 6, dt)

    -- The backdrop drifts against the pointer, which gives the flat pixel art
    -- a little depth without moving anything the player is aiming at.
    local mx, my = love.mouse.getPosition()
    local w, h = love.graphics.getDimensions()
    S.parallaxX = effects.approach(S.parallaxX, (mx / w - 0.5) * -22, 4, dt)
    S.parallaxY = effects.approach(S.parallaxY, (my / h - 0.5) * -14, 4, dt)

    -- the engine
    if S.thinking then
        local result, err = S.engine:poll()
        if err then
            say(err)
            S.thinking = nil
        elseif result then
            applyResult(result)
        end
    elseif not S.game:isOver() and not humanTurn() then
        startThinking("move")
    end

    screenshotHook(dt)
end

function love.draw()
    if S.fatal then
        love.graphics.clear(P.ink)
        local w, h = love.graphics.getDimensions()
        text("CAUSEWAYBAY OMOK", 0, h / 2 - 60, 3, P.gold, "center", w)
        local y = h / 2 - 10
        for line in (S.fatal .. "\n"):gmatch("([^\n]*)\n") do
            text(line, 0, y, 2, P.cream, "center", w)
            y = y + font:getHeight() * 2 + 4
        end
        return
    end

    drawBackdrop()
    S.motes:draw(P.gold)

    love.graphics.push()
    S.shake:apply()
    drawBoard()
    drawThinking()
    drawOverlay()
    drawStones()
    S.waves:draw()
    S.particles:draw()
    S.floaters:draw(font)
    love.graphics.pop()

    S.buttons = {}
    drawPanel()
    drawButtons()
    drawMessage()
    drawBanner()
end

function love.resize()
    refreshLayout()
end

function love.mousemoved(x, y)
    if S.fatal then return end
    local row, col = layout.cellAt(S.frame.board, x, y)
    S.hover = row and (row * S.size + col) or nil
end

function love.mousepressed(x, y, whichButton)
    if S.fatal or whichButton ~= 1 then return end
    for _, b in ipairs(S.buttons) do
        if b.hot then b.action() return end
    end
    local row, col = layout.cellAt(S.frame.board, x, y)
    if row then
        S.cursor = row * S.size + col
        humanPlay(S.cursor)
    end
end

local function moveCursor(dr, dc)
    local row, col = omok.rc(S.cursor or 0, S.size)
    S.cursor = math.max(0, math.min(S.size - 1, row + dr)) * S.size
             + math.max(0, math.min(S.size - 1, col + dc))
    S.hover = nil
end

function love.keypressed(key)
    if key == "escape" then love.event.quit() return end
    if key == "f11" or (key == "f" and love.keyboard.isDown("lctrl", "lgui")) then
        love.window.setFullscreen(not love.window.getFullscreen(), "desktop")
        refreshLayout()
        return
    end
    if S.fatal then return end

    if key == "up" or key == "k" then moveCursor(-1, 0)
    elseif key == "down" or key == "j" then moveCursor(1, 0)
    elseif key == "left" then moveCursor(0, -1)
    elseif key == "right" or key == "l" then moveCursor(0, 1)
    elseif key == "space" or key == "return" then humanPlay(S.cursor)
    elseif key == "u" then undo()
    elseif key == "n" then newGame()
    elseif key == "h" then
        if S.game:isOver() then say("the game is over") else startThinking("hint") end
    elseif key == "w" then
        S.watch = not S.watch
        S.engine:cancel()
        S.thinking = nil
        say(S.watch and "the model plays itself" or "your move again")
    elseif key == "tab" then
        S.human = S.human == omok.BLACK and omok.WHITE or omok.BLACK
        S.engine:cancel()
        S.thinking = nil
        say("you are " .. (S.human == omok.BLACK and "indigo" or "amber"))
    elseif key:match("^[1-5]$") then
        S.level = tonumber(key)
        say(level().name .. ("  %d simulations"):format(level().simulations))
    end
end

function love.quit()
    if S.engine then S.engine:cancel() end
end

--- LÖVE draws crashes into its own window, which is no use when the game was
-- started from a terminal.  This also writes them to stderr, then falls back
-- to the built-in screen.
local defaultErrorHandler = love.errorhandler
function love.errorhandler(message)
    io.stderr:write("\ncausewaybay omok crashed:\n")
    io.stderr:write(tostring(message) .. "\n")
    io.stderr:write(debug.traceback("", 2) .. "\n")
    io.stderr:flush()
    return defaultErrorHandler(message)
end
