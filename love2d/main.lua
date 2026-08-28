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
-- N new game, 1-5 difficulty, TAB swap colours, W watch, V turn the board
-- upright or on its side, M mute, F11 fullscreen.

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
local sound = require("sound")
local store = require("store")
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
    -- The three the chrome is built from.  A console bevel is not a gradient:
    -- it is one colour lighter than the face and one darker, and nothing in
    -- between, which is why it stays hard at any scale.
    face       = {0.11, 0.12, 0.22},
    raised     = {0.17, 0.19, 0.32},
    bevelLight = {0.40, 0.45, 0.66},
    bevelDark  = {0.04, 0.04, 0.09},
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
    -- "title" or "game".  The game opens on its title screen, the way a
    -- console did, and the menu there is what starts a game; ESC comes back.
    screen = "title",
    menu = 1,              -- the selected row of the title menu
    menuRows = {},         -- where the rows were drawn, for the mouse
    resumable = false,     -- a game was left mid-way and CONTINUE is offered
    level = 2,
    -- How big the interface is drawn, and how big it was asked to be.  They
    -- differ when the window is too small to hold the size that was asked for:
    -- see `refreshLayout`.  `ui` is what everything measures itself against;
    -- `uiWanted` is the preference, and it is what gets remembered.
    --
    -- Three, not two: at two the buttons were the size of a fingernail on any
    -- display made this decade.  `loadSettings` goes one higher on a tall one.
    ui = 3,
    uiWanted = 3,
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
    -- Which button the pointer was over last frame, so crossing onto a new one
    -- ticks once instead of once a frame.
    lastHot = nil,
    fatal = nil,
    -- "portrait", "landscape", or nil for whatever the window's shape says.
    -- See `setOrientation`.
    orientation = nil,
    parallaxX = 0, parallaxY = 0,
    winTween = nil,
    introTween = nil,
    bannerTween = nil,
    -- The stone whose landing ends the game, while it is still in the air.
    -- See `checkFinished`.
    finish = nil,
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

--- How big the interface is drawn, as a whole-number multiple of the font.
--
-- The font is seven pixels by nine, which is a size for a console and not for
-- a 1440p monitor: at 1:1 it is the small grey text this interface used to be.
-- Everything in the panel is measured in multiples of it instead, so one
-- number moves the whole interface up and down in step -- text, buttons, bars,
-- padding and the panel that holds them -- and nothing has to be re-tuned.
--
-- Whole numbers only.  A fractional scale puts glyph edges between pixels,
-- which is exactly the softness a bitmap font is chosen to avoid.
local UI_MIN, UI_MAX = 1, 5

--- The character cell, the line, and the unit everything else is spaced by.
local function ch() return font:getHeight() * S.ui end
local function lh() return (font:getHeight() + 3) * S.ui end
local function unit() return S.ui end

--- Draw a string, with the options a 16-bit interface actually needs.
--
-- `opts.outline` puts a hard black ring around every letter, one *scaled* pixel
-- thick, which is how every console of this era made text survive being drawn
-- over artwork.  It is four extra draws and it is worth all four: without it
-- the panel's gold on the backdrop's amber lamps is a guess.
--
-- `opts.shadow` is the cheaper half of the same idea -- one offset copy, down
-- and right -- for text that sits on a solid panel and only needs weight.
local function text(str, x, y, scale, colour, align, width, opts)
    scale = scale or S.ui
    opts = opts or {}
    local w = font:getWidth(str) * scale
    if align == "center" then x = x + (width - w) / 2
    elseif align == "right" then x = x + width - w end
    x, y = math.floor(x), math.floor(y)

    if opts.outline then
        love.graphics.setColor(0, 0, 0, opts.outlineAlpha or 1)
        -- `thick` closes the corners too.  Four copies leave a notch at every
        -- corner that is one scaled pixel wide, which is invisible at the
        -- panel's size and a row of teeth along a logo drawn at ten.
        local ring = opts.thick
            and {{-scale, 0}, {scale, 0}, {0, -scale}, {0, scale},
                 {-scale, -scale}, {scale, -scale}, {-scale, scale}, {scale, scale}}
            or {{-scale, 0}, {scale, 0}, {0, -scale}, {0, scale}}
        for _, d in ipairs(ring) do
            love.graphics.print(str, x + d[1], y + d[2], 0, scale, scale)
        end
    elseif opts.shadow then
        love.graphics.setColor(0, 0, 0, 0.55)
        love.graphics.print(str, x + scale, y + scale, 0, scale, scale)
    end

    love.graphics.setColor(colour or P.cream)
    love.graphics.print(str, x, y, 0, scale, scale)
    love.graphics.setColor(1, 1, 1, 1)
    return w
end

local function textWidth(str, scale)
    return font:getWidth(str) * (scale or S.ui)
end

-- --------------------------------------------------------------- 16-bit chrome
--
-- Four shapes, and everything in the panel is made of them.  The look they are
-- after is a console menu of about 1991: no gradients, no soft shadows, no
-- rounded corners -- a flat fill, a hard black edge, and a one-pixel bevel that
-- is light along the top and left and dark along the bottom and right.  That
-- bevel is the entire 3D effect, and at this scale it is all one needs.

local function fill(colour, x, y, w, h, alpha)
    love.graphics.setColor(colour[1], colour[2], colour[3], alpha or colour[4] or 1)
    love.graphics.rectangle("fill", math.floor(x), math.floor(y),
                            math.floor(w), math.floor(h))
    love.graphics.setColor(1, 1, 1, 1)
end

--- A raised box: black edge, bevel, fill.  `sunken` turns the bevel over, which
-- is how a well for a meter differs from a button standing above the panel.
local function bevel(x, y, w, h, face, sunken)
    local u = unit()
    fill(P.ink, x, y, w, h)                                     -- the hard edge
    local light = sunken and P.bevelDark or P.bevelLight
    local dark = sunken and P.bevelLight or P.bevelDark
    fill(light, x + u, y + u, w - u * 2, h - u * 2)
    fill(dark, x + u * 2, y + u * 2, w - u * 3, h - u * 3)
    fill(face, x + u * 2, y + u * 2, w - u * 4, h - u * 4)
end

--- A heading: a filled bar with the name knocked out of it, which is how these
-- interfaces separated one block of information from the next without spending
-- a blank line on it.
local function labelBar(label, x, y, w, reading, ink)
    local u = unit()
    local h = ch() + u * 2
    fill(P.ink, x, y, w, h)
    fill(P.deepIndigo, x, y, w, u)
    text(label, x + u * 2, y + u, S.ui, P.gold)
    -- A heading can carry its own number at the other end of the bar.  It goes
    -- here rather than on a line of its own because a line of its own is a row
    -- of the panel spent on four characters.
    if reading then
        text(reading, x, y + u, S.ui, ink or P.cream, "right", w - u * 2)
    end
    return h + u * 2
end

--- Scanlines: one dark row every fourth, at a low alpha.
--
-- The one piece of pure decoration here, and it earns its place by killing the
-- flatness of a large area of one colour -- which is what a panel on a modern
-- display is, and what a panel on a CRT never was.
local function scanlines(x, y, w, h)
    love.graphics.setColor(0, 0, 0, 0.13)
    for row = 0, math.floor(h / 4) - 1 do
        love.graphics.rectangle("fill", x, math.floor(y + row * 4), w, 1)
    end
    love.graphics.setColor(1, 1, 1, 1)
end

-- ------------------------------------------------------------------ the game

-- Declared here and defined further down, because a stone landing is what sets
-- it off and the landing is written before the game knows how to end.
local celebrate

--- Recompute the frame, and with it the text size actually in force.
--
-- `layout.fit` is what decides that: the size is a preference, and a window too
-- small to hold it gets the largest one it can hold instead.  Grow the window
-- and the size that was asked for comes back on its own.
local function refreshLayout()
    local w, h = love.graphics.getDimensions()
    local frame = layout.fit(w, h, S.size, S.orientation, S.uiWanted)
    S.ui = frame.ui
    S.frame = frame
    if S.motes then S.motes:resize(w, h) end
    -- The board is a repeating texture; the quad only changes when the board
    -- does, so it is built here rather than in the draw loop.
    art.board:setWrap("repeat", "repeat")
    local tw, th = art.board:getDimensions()
    S.boardQuad = love.graphics.newQuad(0, 0, S.frame.board.size / 2,
                                        S.frame.board.size / 2, tw, th)
end

--- Which way the game is standing right now.
--
-- Falls back to the window's own shape, because the fatal screen never computes
-- a frame and V still has to work there: a window that opened fullscreen onto
-- "the AI core is not built" needs its view keys as much as a game does.
local function currentlyPortrait()
    if S.frame then return S.frame.portrait end
    local w, h = love.graphics.getDimensions()
    return layout.naturalOrientation(w, h) == "portrait"
end

--- Stand the game upright, or lay it on its side.
--
-- Two things happen, and they are separate on purpose.  The layout is *told*
-- which arrangement to use, which is the part that works anywhere -- on a
-- fullscreen 16:9 panel there is no window to reshape, and stacking the panel
-- under the board is still the way to give the board every pixel of the height.
-- And the window itself is turned to match, when there is one, so the shape on
-- screen agrees with the shape inside it.
--
-- `love.resize` clears the override again, so dragging the window into a
-- different shape hands the decision back to the window: the button states a
-- preference once, it is not a mode to get stuck in.
--- The three preferences that outlive the window: which way up, how big, and
-- whether it fills the screen.
--
-- Kept in `~/.causewaybayomok/settings.jsonl` with the sound setting -- see
-- `store.lua` for why there and why a log.  A person who stands the game on its
-- side, or who turns the text up because the one thing they could not read was
-- the text, has said something about how they want to play rather than about
-- this run of it, and being asked again every time is the interface forgetting.
local function saveSettings()
    store.write({
        orientation = S.orientation or "auto",
        -- `text_scale`, not the old `text_size`: the sizes were renumbered
        -- when the default went up, and a file that still says 2 from the days
        -- when 2 was the default is not a preference for small text.
        text_scale = S.uiWanted,
        window = love.window.getFullscreen() and "full" or "window",
    })
end

local function loadSettings()
    local saved = store.read()

    if saved.orientation == "portrait" or saved.orientation == "landscape" then
        S.orientation = saved.orientation
    end
    -- Nothing saved: size the text to the display it opened on, so the first
    -- frame is legible from a sofa on a big screen and still fits a laptop.
    -- Once a size has been chosen it is the choice that counts.
    local wanted = tonumber(saved.text_scale)
    if not wanted then
        local _, h = love.graphics.getDimensions()
        wanted = h >= 1300 and 4 or S.uiWanted
    end
    S.uiWanted = math.max(UI_MIN, math.min(UI_MAX, wanted))
    S.ui = S.uiWanted

    -- The window mode cannot be restored in `conf.lua` -- that runs before
    -- there is a filesystem to read -- so the game opens fullscreen and steps
    -- back out here if that is how it was left.  A size asked for on the
    -- command line outranks both: that is somebody wanting a particular window
    -- now, not the last thing they happened to choose.
    if saved.window == "window" and (os.getenv("OMOK_WINDOW") or "") == "" then
        love.window.setFullscreen(false, "desktop")
    end
end

--- Stand the game upright, or lay it on its side.  A plain toggle: it flips,
-- it stays flipped, and it is still flipped the next time the game opens.
--
-- Two things happen and they are separate on purpose.  The layout is *told*
-- which arrangement to use, which is the part that works anywhere -- on a
-- fullscreen 16:9 panel there is no window to reshape, and stacking the panel
-- under the board is still the way to give the board every pixel of the height.
-- And the window itself is turned to match, when there is one, so the shape on
-- screen agrees with the shape inside it.
local function setOrientation(portrait)
    S.orientation = portrait and "portrait" or "landscape"
    saveSettings()
    if not love.window.getFullscreen() then
        local w, h, flags = love.window.getMode()
        -- Only when the window disagrees: turning a portrait window portrait
        -- again would throw away a size somebody had dragged to.
        if layout.naturalOrientation(w, h) ~= S.orientation then
            love.window.setMode(h, w, flags)
        end
    end
    refreshLayout()
    sound.play("flip", {pitch = portrait and 1.0 or 0.85})
end

local function toggleOrientation()
    setOrientation(not currentlyPortrait())
end

local function toggleFullscreen()
    love.window.setFullscreen(not love.window.getFullscreen(), "desktop")
    saveSettings()
    refreshLayout()
    sound.play("flip", {pitch = love.window.getFullscreen() and 1.12 or 0.9})
end

--- Step the interface scale, wrapping round at the top.
--
-- One control rather than a plus and a minus: there are four sizes, the button
-- says which one it is on, and a second press is never more than three away
-- from any of them.  The panel is measured from this number, so the whole
-- interface -- and the board beside it -- rearranges to suit.
local function cycleUiScale(step)
    local wanted = S.uiWanted + (step or 1)
    if wanted > UI_MAX then wanted = UI_MIN elseif wanted < UI_MIN then wanted = UI_MAX end
    S.uiWanted = wanted
    saveSettings()
    refreshLayout()
    sound.play("level", {pitch = 0.8 + 0.14 * S.ui})
    if S.ui == S.uiWanted then
        say(("text size %d of %d"):format(S.ui, UI_MAX))
    else
        say(("text size %d -- %d needs a bigger window"):format(S.ui, S.uiWanted))
    end
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
    S.finish = nil
    S.evalTarget, S.evalShown = 0, 0
    S.cursor = math.floor(S.size / 2) * S.size + math.floor(S.size / 2)
    S.particles:clear()
    S.waves:clear()
    S.floaters:clear()
    -- The board fades in from the centre outwards, so a new game reads as a
    -- fresh start rather than as stones vanishing.
    S.introTween = effects.tween(0.75, ease.outCubic)
    -- A fanfare runs for a second; starting again during one has to cut it, or
    -- the new board opens under the end of the last game.
    sound.stop("win")
    sound.stop("lose")
    sound.stop("draw")
    sound.play("start")
    if not keepColours then
        say(S.watch and "the model plays itself" or
            ("you are " .. (S.human == omok.BLACK and "indigo" or "amber")))
    end
end

-- How a stone arrives.
--
-- It is thrown in from off screen rather than fading up on its point, and which
-- edge it comes from is the palette's own division: indigo falls out of the
-- night sky, amber rises from the lamplight below.  That is the same thing the
-- two stone sounds say a fourth apart -- whose move it was, told without the
-- eye having to leave the board and read the panel.
--
-- Held at a constant speed rather than a constant duration.  A fixed duration
-- would make a stone landing on the near edge crawl and one crossing the whole
-- board tear across it; a fixed speed with a floor and a ceiling means every
-- stone moves like the same object, whatever distance it happens to cover.
local FLIGHT_SPEED = 2700       -- pixels a second
local FLIGHT_MIN, FLIGHT_MAX = 0.17, 0.32
-- Not one of the harder 'out' curves: a fourth-power ease covers three
-- quarters of the distance in the first third of the time, which lands the
-- stone almost at once and then floats it, and a comet whose ghosts have
-- already bunched up is a smudge rather than a streak.  Cubic keeps it moving
-- across the board for most of the flight and still settles rather than stops.
local FLIGHT_CURVE = ease.outCubic

--- Where a stone is, some fraction of the way through its flight.
--
-- Takes the raw 0..1 rather than the eased value so the comet's ghosts can ask
-- for the position a few hundredths of a second *behind* the stone: they are
-- the same curve read at an earlier time, which is why they bunch up as it
-- slows into the board instead of trailing at a fixed distance.
local function flightPoint(f, raw)
    local at = FLIGHT_CURVE(math.max(0, math.min(1, raw)))
    return f.fromX + (f.toX - f.fromX) * at, f.fromY + (f.toY - f.fromY) * at
end

--- Is this point on the board's side of the window?  See `layout.field`.
local function inField(x, y)
    local f = S.frame.field
    return x >= f.x and x <= f.x + f.width and y >= f.y and y <= f.y + f.height
end

--- Throw a stone at its point.  The trimmings happen when it gets there.
local function placeStone(index, player, label)
    local x, y = boardCentre(index)
    local b = S.frame.board
    local height = love.graphics.getHeight()

    -- Off screen at both ends, so a stone is never seen to appear: it is
    -- already moving by the time it crosses into the window.
    local fromY = player == omok.BLACK and -b.cell * 2 or height + b.cell * 2
    -- A little sideways, and never the same amount twice.  Dead vertical drops
    -- read as a machine feeding stones down a chute.
    local fromX = x + (math.random() - 0.5) * b.cell * 3.5

    local dx, dy = x - fromX, y - fromY
    local distance = math.sqrt(dx * dx + dy * dy)
    local duration = math.max(FLIGHT_MIN,
                              math.min(FLIGHT_MAX, distance / FLIGHT_SPEED))

    S.stones[index] = {
        player = player,
        -- The settle is one delayed tween rather than a second piece of state
        -- to keep in step: it sits at zero for the whole flight and starts
        -- itself the instant the stone is down.
        tween = effects.tween(0.42, ease.outBack, duration),
        flight = {
            tween = effects.tween(duration, ease.linear),
            fromX = fromX, fromY = fromY, toX = x, toY = y,
            lastX = fromX, lastY = fromY,
            label = label,
            emit = 0,
        },
    }
end

--- The stone arrives: the impact, and everything that hangs off it.
local function landStone(index)
    local stone = S.stones[index]
    local f = stone.flight
    stone.flight = nil

    -- The point as it is now, not as it was when the stone was thrown: a
    -- window resized during the quarter second it was in the air moves every
    -- cell, and the impact belongs to the cell rather than to the throw.
    local x, y = boardCentre(index)
    local colour = stoneColour(stone.player)
    local cell = S.frame.board.cell

    S.waves:add(x, y, {
        from = cell * 0.25, to = cell * 2.1, life = 0.55,
        colour = colour, curve = ease.outQuart,
    })
    -- A second ring, thinner and quicker, that gets out ahead of the first.
    -- One ring expanding alone reads as a ripple; two at different speeds read
    -- as something having hit hard enough to make one.
    S.waves:add(x, y, {
        from = cell * 0.15, to = cell * 2.6, life = 0.26,
        colour = P.cream, curve = ease.outQuint,
    })

    -- The spray carries on down the line the stone came in on, the way anything
    -- thrown at a surface throws what it hits forwards rather than evenly.
    local angle = math.atan2(y - f.fromY, x - f.fromX)
    S.particles:burst(x, y, 16, {
        speed = cell * 6.5, life = 0.45, size = cell * 0.36, colour = colour,
        angle = angle, spread = math.pi * 1.15, gravity = cell * 3.0, drag = 3.2,
    })
    -- And a smaller even burst under it, which is the part that reads as dust.
    S.particles:burst(x, y, 12, {
        speed = cell * 3.2, life = 0.55, size = cell * 0.34, colour = colour,
        gravity = cell * 3.5, drag = 2.6,
    })

    S.shake:kick(3.0)
    -- Named for the side rather than for who played it, so watch mode needs no
    -- third sound and the two colours stay a fourth apart whoever is holding
    -- them.  See `tools/make_love2d_sfx.py`.
    sound.play(stone.player == omok.BLACK and "indigo" or "amber")
    if f.label then
        S.floaters:add(f.label, x, y - cell * 0.7,
                       {colour = colour, scale = S.ui, life = 1.0})
    end

    -- The game may have been won by this stone.  The fanfare waited for it.
    if S.finish == index then
        S.finish = nil
        celebrate()
    end
end

--- Is anything still on its way to the board?
local function stoneInFlight()
    for _, stone in pairs(S.stones) do
        if stone.flight then return true end
    end
    return false
end

--- Advance every stone still in the air, and lay down its trail.
local function updateFlights(dt)
    local cell = S.frame.board.cell
    for index, stone in pairs(S.stones) do
        local f = stone.flight
        if f then
            f.tween:update(dt)
            local x, y = flightPoint(f, f.tween:raw())

            -- Emitted per second rather than per frame, and spread along the
            -- ground covered since the last one, so the trail is the same
            -- density whether the machine is managing 30 frames or 144.
            f.emit = f.emit + dt
            local step = 1 / 220
            local count = math.floor(f.emit / step)
            -- Sparse emission is a dotted line, not a trail: at this speed
            -- the stone covers 40 pixels between two of them at 140 a second,
            -- and the gaps are what the eye picks out.
            if count > 0 then
                f.emit = f.emit - count * step
                -- The clock is spent either way; only the sparks are withheld,
                -- so a stone crossing into view does not arrive owing a burst
                -- of everything it should have shed while it was hidden.
                if inField(x, y) then
                    local c = stoneColour(stone.player)
                    S.particles:streak(f.lastX, f.lastY, x, y, count, {
                        -- Small, scattered and dim, and all three of those are
                        -- load-bearing.  Big ones overlap into a rope of beads;
                        -- ones that do not drift off the line stack into a
                        -- solid bar under additive blending; and at full colour
                        -- the pile-up blows out white and swallows the stone it
                        -- is supposed to be following.
                        speed = cell * 2.4, life = 0.24, size = cell * 0.20,
                        drag = 5.0, colour = {c[1] * 0.75, c[2] * 0.75, c[3] * 0.75},
                    })
                end
            end
            f.lastX, f.lastY = x, y

            if f.tween:done() then landStone(index) end
        end
    end
end

local function pushHistory(who, index, result)
    local line = ("%2d %s %s"):format(S.game:moveCount(), who, omok.name(index, S.size))
    if result then line = line .. (" %+.2f"):format(result.value) end
    S.history[#S.history + 1] = line
end

--- Everything that happens because the game just ended.
--
-- Separate from `checkFinished` because it does not happen when the game ends:
-- it happens when the stone that ended it lands.  The rules are settled the
-- moment the move is played -- the engine is told, the board is closed -- but
-- firing the fountain and the fanfare then would light up a winning line whose
-- fifth stone is still crossing the screen.
celebrate = function()
    S.winTween = effects.tween(1.4, ease.outCubic)
    S.bannerTween = effects.tween(0.9, ease.outElastic)
    S.shake:kick(14)

    local winner = S.game:winner()
    local line = S.game:winLine()
    if winner == omok.EMPTY then
        sound.play("draw")
        say("a draw -- the board is full")
        return
    end
    -- Watching the model play itself, nobody lost: the game ended, and the
    -- fanfare is for the board rather than for anyone at the keyboard.
    sound.play((S.watch or winner == S.human) and "win" or "lose")
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
        S.floaters:add(tostring(i), x, y, {colour = colour, scale = S.ui, life = 1.4})
    end
    say(winner == S.human and not S.watch and "you win!" or
        (winner == omok.BLACK and "indigo wins" or "amber wins"))
end

--- Did that move end it?  Records the stone to celebrate, and `landStone`
-- fires the celebration when that stone touches down.
local function checkFinished()
    if not S.game:isOver() then return end
    S.finish = S.game:lastMove()
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
    if S.game:isOver() then sound.play("deny") say("press N for a new game") return end
    if not humanTurn() then sound.play("deny") say("the model is thinking") return end
    if not S.game:play(index) then sound.play("deny") say("that point is taken") return end
    placeStone(index, S.human, omok.name(index, S.size))
    pushHistory("you", index)
    S.hint = nil
    checkFinished()
end

local function undo()
    if S.game:moveCount() == 0 then sound.play("deny") say("nothing to take back") return end
    S.engine:cancel()
    S.thinking = nil
    S.winTween, S.bannerTween, S.finish = nil, nil, nil

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
    -- One sound for the pair: the stone's own slide run backwards, played once
    -- however many stones came off, because it is one action.
    sound.play("undo")
    say("taken back")
end

-- The settings that have both a key and a button.  Written once and called from
-- both, so the two can never drift apart.
--
-- Each plays its own click, and the click the button press already made is
-- swallowed by the throttle in `sound.lua` -- two plays of one effect in the
-- same frame are what that gate is for.

local function toggleWatch()
    S.watch = not S.watch
    S.engine:cancel()
    S.thinking = nil
    sound.play("press")
    say(S.watch and "the model plays itself" or "your move again")
end

local function swapSides()
    S.human = S.human == omok.BLACK and omok.WHITE or omok.BLACK
    S.engine:cancel()
    S.thinking = nil
    sound.play("press")
    say("you are " .. (S.human == omok.BLACK and "indigo" or "amber"))
end

local function setLevel(n)
    S.level = n
    -- One note, played higher the harder the setting: the pitch is the reading,
    -- which is why the generator makes this a single tone and not a phrase.
    sound.play("level", {pitch = 0.82 + 0.12 * n})
    say(level().name .. ("  %d simulations"):format(level().simulations))
end

local function toggleSound()
    -- The one control that makes its own noise.  Left to the button, the click
    -- would play *before* the toggle, so turning the sound on would be the one
    -- press in the game that answers with silence.
    local on = sound.toggle()
    if on then sound.play("press") end
    say(on and "sound on" or "sound off")
end

local function applyResult(result)
    if S.thinking == "hint" then
        S.hint = result
        S.evalTarget = result.value
        S.cursor = result.move
        local x, y = boardCentre(result.move)
        S.waves:add(x, y, {from = 4, to = S.frame.board.cell * 1.8,
                           life = 0.7, colour = P.hint})
        sound.play("hint")
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
-- window through a resize, a change of text size or a flip to portrait without
-- any bookkeeping.
--
-- `silent` is for the buttons that answer for themselves -- a new game has its
-- own fanfare, and the sound toggle has to click *after* it has been turned
-- back on -- so they do not also get the generic press.
local function button(id, label, x, y, w, h, action, active, silent)
    local mx, my = love.mouse.getPosition()
    local hot = mx >= x and mx <= x + w and my >= y and my <= y + h
    S.buttons[#S.buttons + 1] = {
        id = id, label = label, x = x, y = y, w = w, h = h,
        action = action, hot = hot, active = active, silent = silent,
    }
    return hot
end

--- How tall a button is: the text, and five scaled pixels of air either side.
--
-- Five rather than three, because a button is a thing to press and three left
-- the label filling it edge to edge -- a caption with a border, not a key.  The
-- panel's height budget in `layout.fit` is the sum of these; change one, change
-- the other.
local function buttonHeight()
    return ch() + unit() * 10
end

local function drawButtons()
    -- Crossing onto a button ticks once.  Decided here rather than in `button`
    -- because only this loop knows about all of them, and two overlapping hot
    -- boxes would otherwise tick twice for one movement.
    local hot = nil
    for _, b in ipairs(S.buttons) do
        if b.hot then hot = b.id end
    end
    if hot and hot ~= S.lastHot then sound.play("hover") end
    S.lastHot = hot

    local u = unit()
    for _, b in ipairs(S.buttons) do
        -- A console button has two states and they are opposites: standing off
        -- the panel, or pressed into it and lit.  There is no third rendering
        -- for "hovered" beyond a brighter face -- where the pointer is resting
        -- is not a setting, and only settings get to look switched on.
        local face = b.active and P.gold or (b.hot and P.raised or P.face)
        bevel(b.x, b.y, b.w, b.h, face, b.active)

        local ink = b.active and P.ink or (b.hot and P.brightGold or P.cream)
        -- `layout.fit` already sizes the panel so the labels fit; this is the
        -- backstop for the one that does not, and it costs nothing to have.
        local scale = S.ui
        while scale > 1 and textWidth(b.label, scale) > b.w - u * 2 do
            scale = scale - 1
        end
        local y = b.y + (b.h - font:getHeight() * scale) / 2 + (b.active and u or 0)
        text(b.label, b.x, y, scale, ink, "center", b.w, {shadow = not b.active})
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

    -- Coordinates.  Held to double size however large the interface is set:
    -- they label a grid line rather than being read as text, and at triple a
    -- two-digit row number is wider than the margin it has to sit in.
    local cs = math.min(2, S.ui)
    local cwide, ctall = font:getWidth("M") * cs, font:getHeight() * cs
    local ink = {P.gold[1], P.gold[2], P.gold[3], 0.6 * intro}
    for i = 0, b.cells - 1 do
        local x, y = layout.cellCentre(b, i, i)
        text(string.char(string.byte("a") + i), x - cwide / 2, b.y - ctall - 4, cs, ink)
        local n = tostring(i)
        text(n, b.x - 6 - font:getWidth(n) * cs, y - ctall / 2, cs, ink)
    end
    love.graphics.setColor(1, 1, 1, 1)
end

local function drawStoneAt(x, y, player, scale, alpha, rotation)
    local sprite = player == omok.BLACK and art.stone_black or art.stone_white
    local sw, sh = sprite:getDimensions()
    local radius = S.frame.board.cell * 0.44
    local draw = radius * 2 / sw * scale

    -- The shadow grows with the stone, which is what sells the drop.
    love.graphics.setColor(0, 0, 0, 0.45 * alpha)
    love.graphics.draw(sprite, x + 2, y + 3, rotation or 0, draw, draw, sw / 2, sh / 2)
    love.graphics.setColor(1, 1, 1, alpha)
    love.graphics.draw(sprite, x, y, rotation or 0, draw, draw, sw / 2, sh / 2)
end

local function drawStone(index, player, scale, alpha)
    local x, y = boardCentre(index)
    drawStoneAt(x, y, player, scale, alpha)
end

--- A stone still on its way in, drawn as a comet.
--
-- Three copies of the sprite behind the real one, each read off the same curve
-- a few hundredths of a second earlier.  Reading the curve rather than trailing
-- at a fixed distance is the whole trick: they string out while the stone is
-- travelling and pile up into it as it slows, so the streak closes itself
-- rather than being switched off.
--
-- The halo goes underneath because the stones are dark against a dark board and
-- a moving one needs something to be seen by -- the same additive halo the last
-- move keeps, borrowed for the length of the flight.
local GHOSTS = 3

local function drawFlyingStone(stone)
    local f = stone.flight
    local raw = f.tween:raw()
    local colour = stoneColour(stone.player)
    local x, y = flightPoint(f, raw)
    local cell = S.frame.board.cell

    local sw = art.halo:getWidth()
    love.graphics.setBlendMode("add", "alphamultiply")
    love.graphics.setColor(colour[1], colour[2], colour[3], 0.20)
    local ring = cell * 1.7
    love.graphics.draw(art.halo, x, y, 0, ring / sw, ring / sw, sw / 2, sw / 2)
    love.graphics.setBlendMode("alpha", "alphamultiply")

    for i = GHOSTS, 1, -1 do
        local gx, gy = flightPoint(f, raw - i * 0.055)
        drawStoneAt(gx, gy, stone.player, 1.0 - i * 0.07, 0.30 - i * 0.07)
    end
    -- A shade over full size, so it reads as coming at you rather than lying
    -- flat, and settles to 1 through the landing tween.
    drawStoneAt(x, y, stone.player, 1.18, 1)
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
        -- The ones in the air are drawn by the pass below.
        if not stone.flight then
            local t = stone.tween:at()
            local scale = 1
            if not stone.tween:done() then
                -- Settling from the size it arrived at, overshooting once as it
                -- beds down.  It is no longer falling -- the flight did that -- so
                -- this starts near its own size rather than at two and a half.
                scale = effects.lerp(1.35, 1.0, stone.tween:raw(), ease.outBack)
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
    end

    -- The stones still in the air go last, over everything already on the
    -- board: a comet that passes behind the stones it is flying over reads as
    -- being under the table.  Clipped to the play area, so one coming up from
    -- below enters at the panel's edge instead of sliding up behind it.
    local field = S.frame.field
    love.graphics.setScissor(field.x, field.y, field.width, field.height)
    for _, stone in pairs(S.stones) do
        if stone.flight then drawFlyingStone(stone) end
    end
    love.graphics.setScissor()
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
            text(tostring(rank), x - font:getWidth("M") * S.ui / 2,
                 y - ch() / 2, S.ui, P.hint, nil, nil, {outline = true})
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

--- A cursor down one column of the panel.
--
-- The panel used to place every element against a running `y` and a pile of
-- `portrait and ... or ...` conditionals, which is workable for one column and
-- unreadable for three.  Each block below is handed one of these instead and
-- asks it for the room it needs, so the same code lays out a tall thin panel
-- beside the board and a short wide one under it.
local function column(x, y, w)
    return {
        x = x, y = y, w = w,
        room = function(self, h)
            local at = self.y
            self.y = self.y + h
            return at
        end,
        gap = function(self, n) self.y = self.y + unit() * (n or 2) end,
    }
end

--- The name, on one line.  The logo proper is on the title screen now; in
-- the panel it is a heading, and a heading that took four lines was what kept
-- the text a size smaller than it could be in any window under 900 pixels.
local function blockTitle(c)
    local u = unit()
    text("CAUSEWAYBAY OMOK", c.x, c:room(ch() + u), S.ui, P.gold, nil, nil, {outline = true})
    text(S.engineLine, c.x, c:room(lh()), S.ui, P.dim)
end

--- Whose turn it is: a sunken well with the stone in it, the way a portrait box
-- sat in the corner of every menu of this era.
local function blockTurn(c)
    local u = unit()
    local h = ch() + u * 8
    local y = c:room(h)
    bevel(c.x, y, c.w, h, P.ink, true)

    local radius = ch() / 2
    local cx, cy = c.x + u * 4 + radius, y + h / 2
    local label, colour

    if S.game:isOver() then
        local winner = S.game:winner()
        label = winner == omok.EMPTY and "DRAW"
                or ((winner == omok.BLACK and "INDIGO" or "AMBER") .. " WINS")
        colour = winner == omok.EMPTY and P.dim or stoneColour(winner)
        if winner ~= omok.EMPTY then
            drawTurnStone(cx, cy, radius, winner, S.time * 1.5)
        end
    else
        local player = S.game:toMove()
        -- Spinning and bobbing while it thinks, still while it waits: the one
        -- place in the panel that says the engine is alive.
        local bob = S.thinking and math.sin(S.time * 6) * u or 0
        drawTurnStone(cx, cy + bob, radius, player, S.thinking and S.time * 3.5 or 0)
        if S.thinking == "hint" then label = "ADVISING"
        elseif S.thinking then label = "THINKING" .. ("."):rep(math.floor(S.time * 3) % 4)
        elseif humanTurn() then label = "YOUR MOVE"
        else label = "MODEL" end
        colour = S.thinking and P.gold or P.cream
    end

    -- The well is narrow and the winner lines are long, so the text takes
    -- whichever whole size still fits beside the stone rather than running out
    -- through the side of its own box.
    local left = cx + radius + u * 3
    local room = c.x + c.w - u * 3 - left
    local scale = S.ui
    while scale > 1 and textWidth(label, scale) > room do scale = scale - 1 end
    text(label, left, y + (h - font:getHeight() * scale) / 2, scale, colour)
    c:gap(3)
end

--- The evaluation, as a row of lamps rather than as a bar.
--
-- A smooth bar is a thing this hardware could not draw and this interface has
-- no business having.  Sixteen segments filling out from the middle say the
-- same number, read at a glance from across the room, and put a floor under how
-- small a change has to be before it is worth showing at all.
local SEGMENTS = 16

local function blockEval(c)
    local u = unit()
    local colour = S.evalShown >= 0 and P.good or P.bad
    c:room(labelBar("EVAL", c.x, c.y, c.w, ("%+.2f"):format(S.evalShown), colour))

    local h = ch()
    local y = c:room(h + u * 4)
    bevel(c.x, y, c.w, h + u * 4, P.ink, true)

    local segW = math.floor((c.w - u * 4) / SEGMENTS)
    local half = SEGMENTS / 2
    local lit = math.floor(math.abs(S.evalShown) * half + 0.5)
    for i = 1, half do
        for _, side in ipairs({-1, 1}) do
            local on = (side > 0) == (S.evalShown >= 0) and i <= lit
            local index = side > 0 and (half + i - 1) or (half - i)
            fill(on and colour or P.raised,
                 c.x + u * 2 + index * segW, y + u * 2, segW - 1, h,
                 on and 1 or 0.9)
        end
    end
    -- The middle, so a reading of nothing still has something to be nothing at.
    fill(P.cream, c.x + u * 2 + half * segW - 1, y + u, 1, h + u * 2, 0.5)
    c:gap(2)
end

local function blockLevel(c)
    local u = unit()
    c:room(labelBar("DIFFICULTY", c.x, c.y, c.w))
    local h = buttonHeight()
    local y = c:room(h + u * 2)
    local slot = math.floor((c.w - u * 4) / 5)
    for i = 1, 5 do
        button("level" .. i, tostring(i), c.x + (i - 1) * (slot + u), y,
               slot, h, function() setLevel(i) end, S.level == i, true)
    end
    text(("%s  %d SIMS"):format(level().name, level().simulations),
         c.x, c:room(lh()), S.ui, P.gold)
    c:gap(2)
end

local function blockActions(c)
    local u = unit()
    local h = buttonHeight()
    local step = h + u * 2
    local half = math.floor((c.w - u * 2) / 2)
    local right = c.x + half + u * 2

    local y = c:room(step)
    -- NEW and UNDO are silent: both answer with a sound of their own, and a
    -- click in front of it would only blur the start of it.
    button("new", "NEW", c.x, y, half, h, function() newGame() end, false, true)
    button("undo", "UNDO", right, y, half, h, undo, false, true)

    y = c:room(step)
    button("hint", "HINT", c.x, y, half, h, function()
        if S.game:isOver() then sound.play("deny") say("the game is over")
        else startThinking("hint") end
    end)
    button("watch", "AUTO", right, y, half, h, toggleWatch, S.watch)

    y = c:room(step)
    button("swap", "SWAP", c.x, y, half, h, swapSides)
    button("sound", sound.enabled and "SFX ON" or "SFX OFF", right, y, half, h,
           toggleSound, sound.enabled, true)

    -- The view row: which way up, how much of the screen, and how big the
    -- writing.  Three settings that are about the window rather than about the
    -- game, kept together and away from the four that move stones.  All three
    -- are toggles and all three light up when they are on, so the row reads as
    -- the state of the window at a glance.
    y = c:room(step)
    local third = math.floor((c.w - u * 4) / 3)
    button("turn", S.frame.portrait and "WIDE" or "TALL", c.x, y, third, h,
           toggleOrientation, false, true)
    button("full", love.window.getFullscreen() and "WINDOW" or "FULL",
           c.x + third + u * 2, y, third, h, toggleFullscreen,
           love.window.getFullscreen(), true)
    button("size", "SIZE " .. S.ui, c.x + (third + u * 2) * 2, y, third, h,
           function() cycleUiScale(1) end, false, true)
    c:gap(2)
end

local function blockMoves(c, bottom)
    -- At a large text size the panel runs out before this does.  A heading with
    -- nothing under it is worse than no heading, so the whole block stands
    -- down rather than printing its own name over the key line.
    if bottom - c.y < lh() * 2 then return end
    c:room(labelBar("MOVES", c.x, c.y, c.w))
    local rows = math.max(0, math.floor((bottom - c.y) / lh()))
    local first = math.max(1, #S.history - rows + 1)
    for i = first, #S.history do
        -- Upper case at the point of drawing rather than at the point of
        -- writing, so the strings stay readable in the source.
        text(S.history[i]:upper(), c.x, c:room(lh()), S.ui,
             i == #S.history and P.cream or P.dim)
    end
end

--- The keys, along the bottom, in whichever length still fits.
--
-- Four tiers of the same line, longest first.  The keys stay in the same order
-- however narrow it gets, so a smaller panel drops words rather than
-- rearranging them, and even the shortest tier names every one of them --
-- clipping the last key off the end is worse than printing it tersely.
local function drawKeyHints(x, y, width)
    local hints = {
        "SPACE PLACE  U UNDO  H HINT  N NEW  V TURN  S SIZE  M MUTE  ESC TITLE",
        "SPACE  U UNDO  H HINT  N NEW  V TURN  S SIZE  M MUTE  ESC",
        "SPACE  U  H  N  V  S  M  ESC",
        "SPACE U H N V S M ESC",
        "U H N V S M ESC",
        "U H N V S M",
    }
    local line = hints[#hints]
    for _, candidate in ipairs(hints) do
        if textWidth(candidate, S.ui) <= width then line = candidate break end
    end
    local scale = S.ui
    while scale > 1 and textWidth(line, scale) > width do scale = scale - 1 end
    text(line, x, y, scale, {P.dim[1], P.dim[2], P.dim[3], 0.75})
end

local function drawPanel()
    local p = S.frame.panel
    local u = unit()
    local pad = u * 4

    -- The window: a hard black edge, a bevel, a flat face, and scanlines over
    -- the whole of it.
    bevel(p.x, p.y, p.width, p.height, P.face)
    scanlines(p.x + u * 2, p.y + u * 2, p.width - u * 4, p.height - u * 4)

    -- Everything below is clipped to the panel, so a long move list or an
    -- unusually large text size can never spill across the board.
    love.graphics.setScissor(p.x + u * 2, p.y + u * 2, p.width - u * 4, p.height - u * 4)

    local left = p.x + pad
    local inner = p.width - pad * 2
    local top = p.y + pad
    -- The keys own the last line of the panel in both arrangements.
    local floor = p.y + p.height - pad - ch()

    if S.frame.portrait then
        -- Two bands across: the game and its numbers on the left, the controls
        -- and the last few moves on the right.  Two rather than three because
        -- a third of a 900-pixel window is not wide enough for a row of three
        -- buttons at the size the text is now, and the size would have stepped
        -- down to fit -- which is the wrong thing to give up.
        local colW = math.floor((inner - pad) / 2)
        local a = column(left, top, colW)
        local b = column(left + colW + pad, top, colW)
        blockTitle(a)
        blockTurn(a)
        blockEval(a)
        blockLevel(a)
        blockActions(b)
        blockMoves(b, floor - u * 2)
    else
        local a = column(left, top, inner)
        blockTitle(a)
        blockTurn(a)
        blockEval(a)
        blockLevel(a)
        blockActions(a)
        blockMoves(a, floor - u * 2)
    end

    drawKeyHints(left, floor, inner)
    love.graphics.setScissor()
end

local function drawBanner()
    if not S.bannerTween then return end
    local b = S.frame.board
    local t = S.bannerTween:at()
    local winner = S.game:winner()
    local label = winner == omok.EMPTY and "DRAW"
                  or ((winner == S.human and not S.watch) and "YOU WIN"
                      or (winner == omok.BLACK and "INDIGO WINS" or "AMBER WINS"))
    local colour = winner == omok.EMPTY and P.dim or stoneColour(winner)

    -- As large as the board can carry.  Started high and stepped down rather
    -- than set to a number, because the board is any size at all and a title
    -- that runs off both ends of the thing it is about reads as a mistake
    -- rather than as a flourish.
    local scale = 8
    while scale > 1 and textWidth(label, scale) > b.size - 16 do scale = scale - 1 end
    local width = textWidth(label, scale)
    local boardCentreX = b.x + b.size / 2
    local y = effects.lerp(-80, b.y + b.size / 2 - 40, t)

    love.graphics.setColor(P.ink[1], P.ink[2], P.ink[3], 0.75 * math.min(1, t * 2))
    love.graphics.rectangle("fill", b.x, y - 14, b.size, font:getHeight() * scale + 28)
    love.graphics.setColor(colour[1], colour[2], colour[3], 0.7)
    love.graphics.rectangle("fill", b.x, y - 14, b.size, 2)
    love.graphics.rectangle("fill", b.x, y + font:getHeight() * scale + 12, b.size, 2)

    text(label, boardCentreX - width / 2, y, scale, colour, nil, nil, {outline = true})
    text("PRESS N FOR ANOTHER", boardCentreX - textWidth("PRESS N FOR ANOTHER", S.ui) / 2,
         y + font:getHeight() * scale + 18, S.ui, P.cream, nil, nil, {outline = true})
end

local function drawMessage()
    if S.messageAge > 4 or S.message == "" then return end
    local alpha = math.min(1, (4 - S.messageAge) * 1.5)
    local b = S.frame.board
    local field = S.frame.field
    local message = S.message:upper()

    -- Sized and placed against the play area rather than against the board.
    -- Centred under the board it is the right thing to look at, but at a large
    -- text size the line is wider than the board it is centred on, and it ran
    -- off the left of the screen and under the panel at both ends.
    local scale = S.ui
    local margin = unit() * 4
    while scale > 1 and textWidth(message, scale) > field.width - margin * 2 do
        scale = scale - 1
    end
    local width = textWidth(message, scale)
    local x = math.max(margin,
                       math.min(field.width - width - margin,
                                b.x + (b.size - width) / 2))
    local tall = font:getHeight() * scale
    local u = unit()
    local y = b.y + b.size + u * 4
    -- No room under the board: over its bottom edge, rather than above it,
    -- where the column letters are.  A line that lasts four seconds may sit on
    -- the last row of the board; it may not sit on the coordinates.
    if y + tall + u * 4 > field.height then y = b.y + b.size - tall - u * 6 end
    fill(P.ink, x - u * 3, y - u * 2, width + u * 6, tall + u * 4, 0.78 * alpha)
    fill(P.gold, x - u * 3, y - u * 2, width + u * 6, u, 0.5 * alpha)
    text(message, x, y, scale, {P.cream[1], P.cream[2], P.cream[3], alpha})
end

-- ------------------------------------------------------------------- title

-- The title screen: the Astronomical Clock, the name over it, and a menu.
--
-- A console game opened on a picture and a menu, and this one does the same,
-- for the same reason: the board is a working screen, and it is a poor place
-- to be asked which colour to play.  The choices that shape a game -- side,
-- difficulty, whether to watch -- live here, where they can be read at a
-- size that suits a title, and the board is reached with them already made.

--- The rows of the menu, built fresh each time because two of them change.
--
-- CONTINUE only exists while a game was left mid-way, and the two rows with
-- a `value` are adjusted in place with the left and right keys rather than
-- opening anything: a title menu of this era had no second screen to go to.
local function menuItems()
    local items = {}
    if S.resumable then
        items[#items + 1] = {id = "continue", label = "CONTINUE"}
    end
    items[#items + 1] = {id = "start", label = S.resumable and "NEW GAME" or "START"}
    items[#items + 1] = {id = "watch", label = "WATCH THE MODEL"}
    items[#items + 1] = {id = "colour", label = "YOU PLAY",
                         value = S.human == omok.BLACK and "INDIGO" or "AMBER"}
    items[#items + 1] = {id = "level", label = "LEVEL",
                         value = ("%d  %s"):format(S.level, level().name)}
    items[#items + 1] = {id = "quit", label = "QUIT"}
    return items
end

local function startGame(watch)
    S.watch = watch
    S.screen = "game"
    S.resumable = false
    newGame()
end

--- Back to the title.  The game is kept, so CONTINUE can pick it up; a search
-- in flight is not, because nothing is waiting for its answer.
local function goTitle()
    if S.engine then S.engine:cancel() end
    S.thinking = nil
    S.resumable = S.game:moveCount() > 0 and not S.game:isOver()
    S.screen = "title"
    S.menu = 1
    sound.play("flip", {pitch = 0.8})
end

local function menuMove(step)
    local n = #menuItems()
    local before = S.menu
    S.menu = ((S.menu - 1 + step) % n) + 1
    if S.menu ~= before then sound.play("blip") end
end

--- Left and right on a row with a value; the other rows ignore them.
local function menuAdjust(step)
    local item = menuItems()[S.menu]
    if item.id == "colour" then
        S.human = S.human == omok.BLACK and omok.WHITE or omok.BLACK
        sound.play("flip", {pitch = S.human == omok.BLACK and 0.9 or 1.1})
    elseif item.id == "level" then
        local n = S.level + step
        if n < 1 then n = #LEVELS elseif n > #LEVELS then n = 1 end
        setLevel(n)
    end
end

local function menuActivate()
    local item = menuItems()[S.menu]
    if item.id == "continue" then
        S.screen = "game"
        sound.play("press")
    elseif item.id == "start" then startGame(false)
    elseif item.id == "watch" then startGame(true)
    elseif item.id == "quit" then love.event.quit()
    else menuAdjust(1) end
end

--- The row of the title menu under a point, or nil.
local function menuRowAt(x, y)
    for _, row in ipairs(S.menuRows) do
        if x >= row.x and x <= row.x + row.w and y >= row.y and y <= row.y + row.h then
            return row.index
        end
    end
    return nil
end

local function drawTitle()
    local w, h = love.graphics.getDimensions()
    local u = unit()

    -- The dial, covering the window, drifting a little against the pointer and
    -- on its own, so a title left open is not a still.
    local picture = art.title
    local iw, ih = picture:getDimensions()
    local scale = math.max(w / iw, h / ih) * 1.08
    local sway = math.sin(S.time * 0.18) * w * 0.012
    love.graphics.setColor(1, 1, 1, 1)
    love.graphics.draw(picture, (w - iw * scale) / 2 + S.parallaxX * 0.7 + sway,
                       (h - ih * scale) / 2 + S.parallaxY * 0.7, 0, scale, scale)
    -- Darker towards the top and the bottom, where the writing is.
    fill(P.ink, 0, 0, w, h, 0.22)
    S.motes:draw(P.gold)

    -- The logo: the long word small, the short one as large as the window
    -- carries, with the two-flat-colour bevel every 16-bit logo was shaded
    -- with, a black ring, and a deep-indigo drop under it for weight.
    local big = 12
    while big > 3 and (textWidth("OMOK", big) > w * 0.55
                       or font:getHeight() * big > h * 0.20) do
        big = big - 1
    end
    local small = math.max(2, math.floor(big / 2))
    local lift = math.sin(S.time * 1.6) * u           -- it breathes
    local top = math.floor(h * 0.10) + lift
    text("CAUSEWAYBAY", 0, top, small, P.gold, "center", w, {outline = true, thick = true})
    local y = top + font:getHeight() * small + small * 4
    text("OMOK", 0, y + u * 3, big, P.deepIndigo, "center", w, {outline = true, thick = true})
    text("OMOK", 0, y, big, P.gold, "center", w, {outline = true, thick = true})
    text("OMOK", 0, y - u, big, P.brightGold, "center", w)
    y = y + font:getHeight() * big + small * 4
    text("OLD TOWN SQUARE, PRAGUE", 0, y, S.ui, P.cream, "center", w, {outline = true})

    -- The menu, in a window of its own.  Drawn at one size above the panel's,
    -- because it is the only thing on the screen to read; capped by what fits
    -- across the window with the cursor beside it.
    local items = menuItems()
    local ms = math.min(S.ui + 1, small)
    local widest = 0
    for _, item in ipairs(items) do
        local line = item.value and (item.label .. "  < " .. item.value .. " >") or item.label
        widest = math.max(widest, textWidth(line, ms))
    end
    while ms > 1 and widest + ms * 30 > w * 0.9 do
        ms = ms - 1
        widest = 0
        for _, item in ipairs(items) do
            local line = item.value and (item.label .. "  < " .. item.value .. " >") or item.label
            widest = math.max(widest, textWidth(line, ms))
        end
    end
    local rowH = (font:getHeight() + 6) * ms
    local cursorRoom = font:getHeight() * ms + ms * 4
    local pad = ms * 6
    local boxW = widest + cursorRoom + pad * 2
    local boxH = rowH * #items + pad * 2
    local boxX = math.floor((w - boxW) / 2)
    local boxY = math.floor(math.max(y + lh() * 2, h * 0.52))
    -- Never below the footer line; a small window drops the menu up instead.
    boxY = math.min(boxY, h - boxH - lh() * 2 - u * 4)

    -- Its own window: the same hard edge and bevel as the panel, over a face
    -- that lets the dial through, because the dial is the point of the screen.
    fill(P.ink, boxX, boxY, boxW, boxH, 0.80)
    fill(P.bevelLight, boxX + u, boxY + u, boxW - u * 2, u)
    fill(P.bevelLight, boxX + u, boxY + u, u, boxH - u * 2)
    fill(P.bevelDark, boxX + u, boxY + boxH - u * 2, boxW - u * 2, u)
    fill(P.bevelDark, boxX + boxW - u * 2, boxY + u, u, boxH - u * 2)
    fill(P.gold, boxX, boxY, boxW, u, 0.9)
    fill(P.gold, boxX, boxY + boxH - u, boxW, u, 0.9)

    S.menuRows = {}
    local textX = boxX + pad + cursorRoom
    for i, item in ipairs(items) do
        local rowY = boxY + pad + (i - 1) * rowH
        S.menuRows[i] = {x = boxX, y = rowY, w = boxW, h = rowH, index = i}
        local selected = i == S.menu
        local pulse = 0.5 + 0.5 * effects.pulse(S.time, 0.8)
        local colour = selected and {
            P.gold[1] + (P.brightGold[1] - P.gold[1]) * pulse,
            P.gold[2] + (P.brightGold[2] - P.gold[2]) * pulse,
            P.gold[3] + (P.brightGold[3] - P.gold[3]) * pulse,
        } or P.cream
        local textY = rowY + (rowH - font:getHeight() * ms) / 2
        if item.value then
            -- The arrows only on the selected row: they are an instruction,
            -- and an instruction on every row is noise.
            local line = selected and (item.label .. "  < " .. item.value .. " >")
                                   or (item.label .. "    " .. item.value)
            text(line, textX, textY, ms, colour, nil, nil, {shadow = true})
        else
            text(item.label, textX, textY, ms, colour, nil, nil, {shadow = true})
        end
        if selected then
            -- The cursor is a stone of the colour about to be played -- the
            -- mushroom beside the menu, in this game's own terms -- spinning.
            local radius = font:getHeight() * ms / 2
            drawTurnStone(boxX + pad + radius, rowY + rowH / 2, radius, S.human,
                          S.time * 2)
        end
    end

    -- The footer: the keys on one line, the engine and the year on the other,
    -- which is where every title screen kept its small print.
    local keysY = h - lh() * 2 - u * 2
    text("ARROWS CHOOSE   SPACE SELECT   ESC QUIT", 0, keysY, S.ui,
         {P.dim[1], P.dim[2], P.dim[3], 0.85}, "center", w, {outline = true})
    text(S.engineLine or "", u * 6, h - lh(), S.ui, P.dim, nil, nil, {outline = true})
    text("(C) 2026 CAUSEWAYBAY", 0, h - lh(), S.ui, P.dim, "right", w - u * 6,
         {outline = true})
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
    -- The title's picture is allowed to be missing -- a checkout from before
    -- it was drawn -- in which case the title opens on the square instead.
    art.title = love.filesystem.getInfo("assets/title.png")
                and love.graphics.newImage("assets/title.png") or art.backdrop

    -- The sound is allowed to be missing -- a checkout without `assets/sfx`, or
    -- a machine with no audio device, plays on in silence -- so this is loaded
    -- before the engine, where a real failure stops everything.
    sound.load()

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
    S.engineLine = blocks and ("MLX %sX%s  %dX%d"):format(blocks, channels, S.size, S.size)
                          or "MLX"
    S.game = omok.Game(S.size)

    S.particles = effects.particles(art.spark)
    S.waves = effects.waves(art.halo)
    S.floaters = effects.floaters()
    S.shake = effects.shake()
    local w, h = love.graphics.getDimensions()
    S.motes = effects.motes(art.spark, 44, w, h)

    loadSettings()
    refreshLayout()
    -- No game yet: the title's menu starts one.  `newGame` also plays the
    -- start fanfare, which belongs to pressing START rather than to opening.
    S.cursor = math.floor(S.size / 2) * S.size + math.floor(S.size / 2)

    S.shotPath = os.getenv("OMOK_SCREENSHOT")
    S.shotAfter = tonumber(os.getenv("OMOK_SHOT_AFTER") or "") or 6
    -- A screenshot of the game skips the title, as the smoke test and the
    -- README's pictures both want the board; `OMOK_SHOT_SCREEN=title` keeps it.
    if S.shotPath and os.getenv("OMOK_SHOT_SCREEN") ~= "title" then
        startGame(true)
    end
    if #sound.missing > 0 then
        say(("no sound: run  make love-sfx  (%d missing)"):format(#sound.missing))
    end
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
    -- Before the fatal-error return: the throttle's clock has to keep running
    -- even on the screen that only says the model is missing, or the first
    -- sound after any pause would arrive with a stale gate.
    sound.update(dt)
    if S.fatal then return end

    -- The pointer moves the picture on both screens.
    local mx, my = love.mouse.getPosition()
    local w, h = love.graphics.getDimensions()
    S.parallaxX = effects.approach(S.parallaxX, (mx / w - 0.5) * -22, 4, dt)
    S.parallaxY = effects.approach(S.parallaxY, (my / h - 0.5) * -14, 4, dt)

    if S.screen == "title" then
        S.motes:update(dt, S.time)
        screenshotHook(dt)
        return
    end

    if S.introTween then
        if S.introTween:update(dt) then S.introTween = nil end
    end
    if S.winTween then S.winTween:update(dt) end
    if S.bannerTween then S.bannerTween:update(dt) end
    for _, stone in pairs(S.stones) do stone.tween:update(dt) end
    updateFlights(dt)

    S.particles:update(dt)
    S.waves:update(dt)
    S.floaters:update(dt)
    S.shake:update(dt)
    S.motes:update(dt, S.time)

    S.evalShown = effects.approach(S.evalShown, S.evalTarget, 6, dt)

    -- the engine
    if S.thinking then
        local result, err = S.engine:poll()
        if err then
            say(err)
            S.thinking = nil
        elseif result then
            applyResult(result)
        end
    elseif not S.game:isOver() and not humanTurn() and not stoneInFlight() then
        -- The model waits for the board to settle before it starts on its
        -- reply.  Nothing in the rules needs this -- the move was legal the
        -- instant it was played -- but at 96 simulations the search comes back
        -- in a hundredth of a second, and without it a watched game answers a
        -- stone that is still crossing the screen: four or five of them in the
        -- air at once, none of them readable, and the board filling up faster
        -- than anybody can see it happen.  A quarter of a second of the game
        -- moving at the speed of its own animation is worth more than a quarter
        -- of a second saved.
        startThinking("move")
    end

    screenshotHook(dt)
end

function love.draw()
    if S.fatal then
        love.graphics.clear(P.ink)
        local w, h = love.graphics.getDimensions()
        local big, small = S.ui * 2, S.ui
        text("CAUSEWAYBAY OMOK", 0, h / 2 - font:getHeight() * big - 30, big,
             P.gold, "center", w, {outline = true})
        local y = h / 2 - 10
        for line in (S.fatal .. "\n"):gmatch("([^\n]*)\n") do
            text(line, 0, y, small, P.cream, "center", w)
            y = y + font:getHeight() * small + 6
        end
        return
    end

    if S.screen == "title" then
        drawTitle()
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

function love.resize(w, h)
    -- The orientation is deliberately left alone.  It is a toggle: once it has
    -- been pressed it holds, through a drag, a fullscreen and the next run of
    -- the game, and the only thing that changes it is pressing it again.  Until
    -- it is pressed for the first time it is unset, and the window's own shape
    -- decides -- which is what makes dragging one taller than it is wide still
    -- do the obvious thing.
    local _ = w, h
    refreshLayout()
end

function love.mousemoved(x, y)
    if S.fatal then return end
    if S.screen == "title" then
        -- Resting the pointer on a row selects it, and only a new row ticks.
        local index = menuRowAt(x, y)
        if index and index ~= S.menu then
            S.menu = index
            sound.play("blip")
        end
        return
    end
    local row, col = layout.cellAt(S.frame.board, x, y)
    local was = S.hover
    S.hover = row and (row * S.size + col) or nil
    -- Only crossing onto a *new* point, and only while there is a move to make:
    -- the pointer travels over the board constantly, and a tick for every pixel
    -- of that would be a buzz rather than feedback.
    if S.hover and S.hover ~= was and humanTurn() and not S.game:isOver() then
        sound.play("hover")
    end
end

function love.mousepressed(x, y, whichButton)
    if S.fatal or whichButton ~= 1 then return end
    if S.screen == "title" then
        local index = menuRowAt(x, y)
        if index then
            S.menu = index
            menuActivate()
        end
        return
    end
    for _, b in ipairs(S.buttons) do
        if b.hot then
            if not b.silent then sound.play("press") end
            b.action()
            return
        end
    end
    local row, col = layout.cellAt(S.frame.board, x, y)
    if row then
        S.cursor = row * S.size + col
        humanPlay(S.cursor)
    end
end

local function moveCursor(dr, dc)
    local before = S.cursor
    local row, col = omok.rc(S.cursor or 0, S.size)
    S.cursor = math.max(0, math.min(S.size - 1, row + dr)) * S.size
             + math.max(0, math.min(S.size - 1, col + dc))
    S.hover = nil
    -- Silent against the edge of the board.  The cursor is clamped there, so a
    -- held arrow key would otherwise keep ticking at a point that is not moving.
    if S.cursor ~= before then sound.play("blip") end
end

function love.keypressed(key)
    -- ESC steps back the way a console's B button did: from the board to the
    -- title, and from the title out of the game.
    if key == "escape" then
        if S.screen == "game" and not S.fatal then goTitle() else love.event.quit() end
        return
    end
    -- The view keys work on the fatal screen too: a window that opened
    -- fullscreen onto "the AI core is not built" still needs a way back out.
    if key == "f11" or (key == "f" and love.keyboard.isDown("lctrl", "lgui")) then
        toggleFullscreen()
        return
    end
    if key == "v" then toggleOrientation() return end
    if key == "m" then toggleSound() return end
    if key == "s" or key == "=" or key == "+" then cycleUiScale(1) return end
    if key == "-" then cycleUiScale(-1) return end
    if S.fatal then return end

    if S.screen == "title" then
        if key == "up" or key == "k" then menuMove(-1)
        elseif key == "down" or key == "j" then menuMove(1)
        elseif key == "left" or key == "h" then menuAdjust(-1)
        elseif key == "right" or key == "l" then menuAdjust(1)
        elseif key == "space" or key == "return" then menuActivate()
        elseif key:match("^[1-5]$") then setLevel(tonumber(key))
        end
        return
    end

    if key == "up" or key == "k" then moveCursor(-1, 0)
    elseif key == "down" or key == "j" then moveCursor(1, 0)
    elseif key == "left" then moveCursor(0, -1)
    elseif key == "right" or key == "l" then moveCursor(0, 1)
    elseif key == "space" or key == "return" then humanPlay(S.cursor)
    elseif key == "u" then undo()
    elseif key == "n" then newGame()
    elseif key == "h" then
        if S.game:isOver() then sound.play("deny") say("the game is over")
        else startThinking("hint") end
    elseif key == "w" then toggleWatch()
    elseif key == "tab" then swapSides()
    elseif key:match("^[1-5]$") then setLevel(tonumber(key))
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
