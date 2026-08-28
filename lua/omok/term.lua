--- The small amount of terminal handling the text client needs.
--
-- No curses: an omok board is a fixed grid that changes a few cells per move,
-- so a full ANSI repaint into one buffered write is both simpler and flicker
-- free.  Raw mode comes from `stty`, with `VMIN 0 / VTIME 1` so reading a key
-- gives up after a tenth of a second -- that is what lets the draw loop keep
-- animating while the engine thinks.

local term = {}

local ESC = string.char(27)

-- ------------------------------------------------------------------ raw mode

local saved = nil

function term.raw()
    local pipe = io.popen("stty -g 2>/dev/null")
    if pipe then
        saved = pipe:read("*l")
        pipe:close()
    end
    os.execute("stty raw -echo min 0 time 1 2>/dev/null")
    io.write(ESC .. "[?25l")          -- hide the cursor
    io.write(ESC .. "[?1049h")        -- and draw on the alternate screen
end

function term.restore()
    io.write(ESC .. "[?1049l")
    io.write(ESC .. "[?25h")
    if saved and saved ~= "" then
        os.execute("stty " .. saved .. " 2>/dev/null")
    else
        os.execute("stty sane 2>/dev/null")
    end
    io.flush()
end

--- Read one key, or nil if none arrived within about a tenth of a second.
-- Arrow keys arrive as escape sequences and come back as "up"/"down"/... .
function term.key()
    local char = io.read(1)
    if not char then return nil end
    if char ~= ESC then return char end
    -- An escape sequence, or a bare Escape if nothing follows it.
    local bracket = io.read(1)
    if bracket ~= "[" and bracket ~= "O" then return "escape" end
    local final = io.read(1)
    local names = {A = "up", B = "down", C = "right", D = "left",
                   H = "home", F = "end"}
    if names[final] then return names[final] end
    -- Consume the tail of anything longer, e.g. page up, so it is not read as
    -- a series of stray key presses.
    while final and not final:match("[A-Za-z~]") do final = io.read(1) end
    return "escape"
end

-- --------------------------------------------------------------------- sizing

function term.size()
    local pipe = io.popen("stty size 2>/dev/null")
    if not pipe then return 24, 80 end
    local line = pipe:read("*l")
    pipe:close()
    local rows, cols = (line or ""):match("(%d+)%s+(%d+)")
    return tonumber(rows) or 24, tonumber(cols) or 80
end

-- --------------------------------------------------------------------- output

--- Collects the whole frame, then writes it in one go: two writes per frame
-- instead of a few hundred, which is the difference between a clean repaint
-- and visible tearing.
local Screen = {}
Screen.__index = Screen

function term.screen()
    return setmetatable({parts = {}}, Screen)
end

function Screen:put(text)
    self.parts[#self.parts + 1] = text
    return self
end

function Screen:at(row, col)
    return self:put(("%s[%d;%dH"):format(ESC, row, col))
end

--- 256-colour foreground; `nil` resets.
function Screen:fg(colour)
    if not colour then return self:put(ESC .. "[39m") end
    return self:put(("%s[38;5;%dm"):format(ESC, colour))
end

function Screen:bg(colour)
    if not colour then return self:put(ESC .. "[49m") end
    return self:put(("%s[48;5;%dm"):format(ESC, colour))
end

function Screen:bold()  return self:put(ESC .. "[1m") end
function Screen:dim()   return self:put(ESC .. "[2m") end
function Screen:reset() return self:put(ESC .. "[0m") end

function Screen:clear()
    return self:put(ESC .. "[2J")
end

function Screen:flush()
    io.write(table.concat(self.parts))
    io.flush()
    self.parts = {}
end

return term
