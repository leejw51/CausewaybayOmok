--- Window configuration.
--
-- The game opens fullscreen.  It is a board that wants every pixel it can have
-- and an interface drawn at console sizes, and neither is served by a window
-- somebody has to enlarge first -- so it fills the display and F11, or the
-- FULL/WINDOW button that is on screen from the first frame, comes back out.
-- Which of the two it was left in is remembered; see `loadSettings` in main.lua.
--
-- Windowed it is resizable, with a minimum that still fits a 15x15 board beside
-- its panel.  The default is landscape; dragging the window taller than it is
-- wide flips the layout to portrait, and V turns it either way without dragging
-- anything.

function love.conf(t)
    t.identity = "causewaybay-omok"
    t.version = "11.5"
    t.console = false

    t.window.title = "Causewaybay Omok"
    -- `OMOK_WINDOW=900x1200 love love2d` starts at a given size, which is how
    -- the portrait layout gets checked without dragging the window about.
    --
    -- Asking for a size is asking for a window, so it also turns fullscreen off
    -- -- otherwise the one flag that exists to photograph a particular shape
    -- would be ignored by the display it opened on.
    local size = os.getenv("OMOK_WINDOW") or ""
    local width, height = size:match("^(%d+)x(%d+)$")
    t.window.width = tonumber(width) or 1280
    t.window.height = tonumber(height) or 800
    t.window.fullscreen = size == ""
    -- Borderless at the display's own resolution rather than a mode change:
    -- nothing here needs an exclusive mode, and this way alt-tabbing works and
    -- the desktop does not resize itself around the game.
    t.window.fullscreentype = "desktop"
    -- Room for a readable board beside a panel drawn at the default text size.
    -- The panel is measured in characters now (see `layout.panelWidth`), so a
    -- 640-wide window spent nearly half of itself on it and left the board a
    -- nineteen-pixel cell.
    t.window.minwidth = 900
    t.window.minheight = 600
    t.window.resizable = true
    t.window.vsync = 1
    t.window.highdpi = true

    -- Nothing here uses physics, joysticks or the microphone.
    t.modules.physics = false
    t.modules.joystick = false
    t.modules.touch = true
end
