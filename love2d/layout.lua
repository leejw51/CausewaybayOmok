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

--- Compute the frame for a window of `width` x `height`.
--
-- Returns a table with `portrait`, `board` (x, y, size, cell, origin) and
-- `panel` (x, y, width, height), all in pixels.
function layout.compute(width, height, boardCells)
    local portrait = height > width * 1.05
    local frame = {portrait = portrait, width = width, height = height}

    local panelSize
    if portrait then
        -- A quarter of the height, held between limits so the panel neither
        -- squeezes the board out nor floats in emptiness on a tall display.
        panelSize = math.max(150, math.min(260, math.floor(height * 0.26)))
    else
        panelSize = math.max(210, math.min(340, math.floor(width * 0.26)))
    end

    local availableWidth = portrait and (width - MARGIN * 2)
                                    or (width - panelSize - MARGIN * 3)
    local availableHeight = portrait and (height - panelSize - MARGIN * 3)
                                     or (height - MARGIN * 2)
    local boardSize = math.max(boardCells * MIN_CELL,
                               math.min(availableWidth, availableHeight))
    -- Whole-pixel cells keep the grid lines evenly spaced; without this the
    -- rounding wanders and some lines land a pixel thicker than others.
    local cell = math.max(MIN_CELL, math.floor(boardSize / boardCells))
    boardSize = cell * boardCells

    local boardX, boardY
    if portrait then
        boardX = math.floor((width - boardSize) / 2)
        boardY = math.floor((availableHeight - boardSize) / 2) + MARGIN
        frame.panel = {
            x = MARGIN,
            y = height - panelSize - MARGIN,
            width = width - MARGIN * 2,
            height = panelSize,
        }
    else
        boardX = math.floor((availableWidth - boardSize) / 2) + MARGIN
        boardY = math.floor((height - boardSize) / 2)
        frame.panel = {
            x = width - panelSize - MARGIN,
            y = MARGIN,
            width = panelSize,
            height = height - MARGIN * 2,
        }
    end

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
