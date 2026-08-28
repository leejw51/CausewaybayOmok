--- Window configuration.
--
-- Resizable, with a minimum that still fits a 15x15 board beside its panel.
-- The default is landscape; dragging the window taller than it is wide flips
-- the layout to portrait, and F11 goes fullscreen on the current display.

function love.conf(t)
    t.identity = "causewaybay-omok"
    t.version = "11.5"
    t.console = false

    t.window.title = "Causewaybay Omok"
    -- `OMOK_WINDOW=900x1200 love love2d` starts at a given size, which is how
    -- the portrait layout gets checked without dragging the window about.
    local width, height = (os.getenv("OMOK_WINDOW") or ""):match("^(%d+)x(%d+)$")
    t.window.width = tonumber(width) or 1280
    t.window.height = tonumber(height) or 800
    t.window.minwidth = 640
    t.window.minheight = 480
    t.window.resizable = true
    t.window.vsync = 1
    t.window.highdpi = true

    -- Nothing here uses physics, joysticks or the microphone.
    t.modules.physics = false
    t.modules.joystick = false
    t.modules.touch = true
end
