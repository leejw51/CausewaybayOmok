--- Where everything goes, for whatever window it is given.
--
-- The window is resizable and can go fullscreen on any display, so nothing is
-- positioned by constant.  There are two arrangements -- the panel beside the
-- board in landscape, below it in portrait -- and the board is always the
-- largest square that leaves the panel its room.
--
-- The panel is the part that flexes: a tall thin phone-shaped window gets a
-- short wide panel, a wide one gets a tall narrow panel, and the button strip
-- inside re-flows to match.

local layout = {}

local MIN_CELL = 14          -- below this the board is unreadable
local MARGIN = 18

--- Room to the left of and above the board for the coordinates.
--
-- They used to be printed into whatever slack the centring happened to leave,
-- which is nothing at all once the board is big enough to fill its space: at
-- the default window the row numbers were drawn off the left edge of the
-- screen and only their second digit survived.  A gutter is the honest fix --
-- the labels are part of the board, so the board's area has to include them.

--- How wide the panel has to be, for interface scale `ui`.
--
-- Measured in characters rather than in pixels, because that is what actually
-- constrains it: the panel holds a column of text about twenty characters
-- across, and the font is seven pixels wide, so the whole thing is a multiple
-- of the size the interface is drawn at.  Turn the text up and the panel grows
-- with it; turn it down and the board takes the room back.
--
-- The bounds either side of that are what stop it running away: a 4K window
-- does not want a quarter of itself given over to eight buttons, and no window
-- is allowed to spend more than 46% of its short side on the panel however
-- large the text is set.
local COORDS = 34

local function panelWidth(ui, along, across)
    local wanted = math.max(150 * ui, math.min(190 * ui, math.floor(along * 0.26)))
    return math.min(wanted, math.floor(along * 0.46), math.max(150, across - 120))
end

local function panelDepth(ui, height)
    -- Portrait's panel is a strip: it is measured by what has to stack inside
    -- it, which is a title, four rows of buttons and the air around them.
    local wanted = math.max(120 * ui, math.min(150 * ui, math.floor(height * 0.26)))
    return math.min(wanted, math.floor(height * 0.42))
end

--- Which arrangement a window of this shape asks for on its own.
--
-- The slack means a window that is square, or nearly so, stays in landscape
-- rather than flipping back and forth around an exact 1:1 while it is dragged.
function layout.naturalOrientation(width, height)
    return height > width * 1.05 and "portrait" or "landscape"
end

--- Compute the frame for a window of `width` x `height`.
--
-- `forced` is "portrait", "landscape" or nil.  Nil is the normal case and the
-- window's own shape decides; the other two are somebody having asked for an
-- arrangement, and they are what lets one be chosen on a display that cannot be
-- reshaped -- a fullscreen 16:9 panel can still stack the panel under the
-- board, which is the only way to give the board every pixel of the height.
--
-- Returns a table with `portrait`, `board` (x, y, size, cell, origin) and
-- `panel` (x, y, width, height), all in pixels.
function layout.compute(width, height, boardCells, forced, ui)
    local portrait = (forced or layout.naturalOrientation(width, height)) == "portrait"
    ui = math.max(1, math.floor(ui or 2))
    local frame = {portrait = portrait, width = width, height = height, ui = ui}

    local panelSize = portrait and panelDepth(ui, height)
                              or panelWidth(ui, width, height)

    -- The gutter comes off the top and the left, which is where the labels are.
    local availableWidth = (portrait and (width - MARGIN * 2)
                                     or (width - panelSize - MARGIN * 3)) - COORDS
    local availableHeight = (portrait and (height - panelSize - MARGIN * 3)
                                      or (height - MARGIN * 2)) - COORDS
    local boardSize = math.max(boardCells * MIN_CELL,
                               math.min(availableWidth, availableHeight))
    -- Whole-pixel cells keep the grid lines evenly spaced; without this the
    -- rounding wanders and some lines land a pixel thicker than others.
    local cell = math.max(MIN_CELL, math.floor(boardSize / boardCells))
    boardSize = cell * boardCells

    local boardX, boardY
    if portrait then
        boardX = math.max(MARGIN + COORDS, math.floor((width - boardSize) / 2))
        boardY = math.floor((availableHeight - boardSize) / 2) + MARGIN + COORDS
        frame.panel = {
            x = MARGIN,
            y = height - panelSize - MARGIN,
            width = width - MARGIN * 2,
            height = panelSize,
        }
    else
        boardX = math.floor((availableWidth - boardSize) / 2) + MARGIN + COORDS
        boardY = math.max(MARGIN + COORDS,
                          math.floor((height - boardSize) / 2))
        frame.panel = {
            x = width - panelSize - MARGIN,
            y = MARGIN,
            width = panelSize,
            height = height - MARGIN * 2,
        }
    end

    -- The part of the window the board lives in: everything up to the panel,
    -- which is one rectangle in both arrangements -- the panel is always either
    -- wholly to the right or wholly below.  A stone thrown in from off screen
    -- is clipped to this, so it enters past the edge of the play area rather
    -- than gliding up through the panel, which is only 90% opaque and shows it.
    frame.field = portrait
        and {x = 0, y = 0, width = width, height = frame.panel.y}
        or {x = 0, y = 0, width = frame.panel.x, height = height}

    frame.board = {
        x = boardX,
        y = boardY,
        size = boardSize,
        cell = cell,
        cells = boardCells,
        -- The centre of cell (0, 0); stones sit on cell centres, not corners.
        originX = boardX + cell / 2,
        originY = boardY + cell / 2,
    }
    return frame
end

--- What the panel needs, at interface scale `scale`, to hold its fixed blocks.
--
-- Everything in the panel is a multiple of the scale, so its requirements are
-- too, and these two numbers are those multiples.
--
-- The height is the sum of the blocks in `drawPanel`, in order: title 44, turn
-- 20, evaluation 40, difficulty 44, actions 70, keys 9, and 8 for the padding
-- above and below.  MOVES is left out deliberately -- it is the one block that
-- stands down when there is nothing left for it, which is what makes it the one
-- block that does not get a say in how large the text can be.
--
-- The width is the widest row in a column: three view buttons side by side,
-- whose longest label is six characters at seven pixels each, plus the gaps.
--
-- Both are estimates and cheap ones on purpose: they decide when to step the
-- text down a size, and being a few pixels out changes nothing.  A block added
-- to `drawPanel` belongs in the sum.
local NEEDS_H_LANDSCAPE, NEEDS_H_PORTRAIT = 235, 101
local NEEDS_W = 130

--- The frame for this window at the largest text size that actually fits.
--
-- `wanted` is a request rather than an instruction.  Somebody who turns the
-- text to its largest in a small window is asking for something that would push
-- the buttons off the bottom of the panel and print their labels through their
-- own borders -- which is not large text, it is a broken interface -- so they
-- get the largest size that does fit, and the preference itself is left alone
-- for when the window grows again.
function layout.fit(width, height, boardCells, forced, wanted)
    local scale = math.max(1, math.floor(wanted or 2))
    local frame = layout.compute(width, height, boardCells, forced, scale)
    while scale > 1 do
        local pad = scale * 4
        local inner = frame.panel.width - pad * 2
        local column = frame.portrait and math.floor((inner - pad * 2) / 3) or inner
        local needs = frame.portrait and NEEDS_H_PORTRAIT or NEEDS_H_LANDSCAPE
        if needs * scale <= frame.panel.height and NEEDS_W * scale <= column then
            break
        end
        scale = scale - 1
        frame = layout.compute(width, height, boardCells, forced, scale)
    end
    return frame
end

--- Screen position of the centre of a 0-based row and column.
function layout.cellCentre(board, row, col)
    return board.originX + col * board.cell, board.originY + row * board.cell
end

--- The 0-based row and column under a screen position, or nil if it is off the
-- board.  A small tolerance outside the edge cells makes the border rows as
-- easy to hit with a mouse as the middle ones.
function layout.cellAt(board, x, y)
    local col = math.floor((x - board.x) / board.cell)
    local row = math.floor((y - board.y) / board.cell)
    if row < 0 or col < 0 or row >= board.cells or col >= board.cells then
        return nil
    end
    return row, col
end

return layout
