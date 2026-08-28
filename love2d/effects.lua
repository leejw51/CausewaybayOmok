--- Easing, tweens, particles and screen shake.
--
-- Everything the game animates goes through here, so timing and feel live in
-- one file rather than being scattered through `love.draw`.  Nothing in this
-- module knows what omok is; it deals in numbers, points and sprites.

local effects = {}

-- ---------------------------------------------------------------- easing

--- `t` runs 0..1 and comes back shaped.  These are the Penner curves; the two
-- that carry most of the game's character are `outBack`, which overshoots and
-- settles (a stone landing), and `outElastic`, which rings (a banner arriving).
local ease = {}
effects.ease = ease

function ease.linear(t) return t end

function ease.inQuad(t) return t * t end
function ease.outQuad(t) return 1 - (1 - t) * (1 - t) end
function ease.inOutQuad(t)
    return t < 0.5 and 2 * t * t or 1 - (-2 * t + 2) ^ 2 / 2
end

function ease.inCubic(t) return t * t * t end
function ease.outCubic(t) return 1 - (1 - t) ^ 3 end
function ease.inOutCubic(t)
    return t < 0.5 and 4 * t * t * t or 1 - (-2 * t + 2) ^ 3 / 2
end

function ease.outQuart(t) return 1 - (1 - t) ^ 4 end
function ease.outQuint(t) return 1 - (1 - t) ^ 5 end

function ease.outBack(t)
    local c1, c3 = 1.70158, 2.70158
    return 1 + c3 * (t - 1) ^ 3 + c1 * (t - 1) ^ 2
end

function ease.inBack(t)
    local c1, c3 = 1.70158, 2.70158
    return c3 * t * t * t - c1 * t * t
end

function ease.outElastic(t)
    if t <= 0 then return 0 end
    if t >= 1 then return 1 end
    local c4 = (2 * math.pi) / 3
    return 2 ^ (-10 * t) * math.sin((t * 10 - 0.75) * c4) + 1
end

function ease.outBounce(t)
    local n1, d1 = 7.5625, 2.75
    if t < 1 / d1 then return n1 * t * t end
    if t < 2 / d1 then t = t - 1.5 / d1 return n1 * t * t + 0.75 end
    if t < 2.5 / d1 then t = t - 2.25 / d1 return n1 * t * t + 0.9375 end
    t = t - 2.625 / d1
    return n1 * t * t + 0.984375
end

--- Interpolate `a`..`b` at `t` (0..1) through `curve`.
function effects.lerp(a, b, t, curve)
    t = math.max(0, math.min(1, t))
    if curve then t = curve(t) end
    return a + (b - a) * t
end

--- Frame-rate independent approach: pulls `value` towards `target`, covering
-- `rate` of the remaining distance every second.  Used for the eval bar and
-- the parallax, where a fixed-duration tween would restart on every change.
function effects.approach(value, target, rate, dt)
    return value + (target - value) * (1 - math.exp(-rate * dt))
end

--- A 0..1 triangle wave, for anything that pulses.
function effects.pulse(time, period)
    local phase = (time % period) / period
    return phase < 0.5 and phase * 2 or 2 - phase * 2
end

-- ---------------------------------------------------------------- tweens

local Tween = {}
Tween.__index = Tween

--- A named clock that runs 0..1 once.  The game keeps one per animated thing
-- and reads `:at()` while drawing, rather than storing a position per frame.
function effects.tween(duration, curve, delay)
    return setmetatable({
        elapsed = -(delay or 0),
        duration = math.max(1e-6, duration),
        curve = curve or ease.outCubic,
    }, Tween)
end

function Tween:update(dt)
    self.elapsed = self.elapsed + dt
    return self:done()
end

function Tween:raw()
    return math.max(0, math.min(1, self.elapsed / self.duration))
end

--- The eased 0..1 value.
function Tween:at()
    return self.curve(self:raw())
end

--- The eased value mapped onto `a`..`b`.
function Tween:between(a, b)
    return a + (b - a) * self:at()
end

function Tween:done()
    return self.elapsed >= self.duration
end

function Tween:started()
    return self.elapsed >= 0
end

-- -------------------------------------------------------------- particles

local Particles = {}
Particles.__index = Particles

--- A plain array of particles drawn as additive sprites.  LÖVE's own particle
-- system would do most of this, but bursts here need per-burst colour, gravity
-- and drag chosen at emit time, and a single flat list draws in one pass.
function effects.particles(sprite)
    return setmetatable({sprite = sprite, list = {}}, Particles)
end

--- Emit `count` particles from (x, y).
--
-- `spec` fields: speed, spread (radians), angle, life, size, colour,
-- gravity, drag, spin.
function Particles:burst(x, y, count, spec)
    spec = spec or {}
    local angle = spec.angle or 0
    local spread = spec.spread or math.pi * 2
    for i = 1, count do
        local theta = angle + (math.random() - 0.5) * spread
        local speed = (spec.speed or 200) * (0.35 + math.random() * 0.85)
        local life = (spec.life or 0.8) * (0.6 + math.random() * 0.7)
        self.list[#self.list + 1] = {
            x = x, y = y,
            vx = math.cos(theta) * speed,
            vy = math.sin(theta) * speed,
            life = life, age = 0,
            size = (spec.size or 12) * (0.5 + math.random()),
            rotation = math.random() * math.pi * 2,
            spin = (spec.spin or 2) * (math.random() - 0.5),
            gravity = spec.gravity or 0,
            drag = spec.drag or 1.6,
            colour = spec.colour or {1, 1, 1},
        }
    end
end

--- Emit particles that fall inward towards (x, y) -- the undo effect.
function Particles:implode(x, y, count, spec)
    spec = spec or {}
    local radius = spec.radius or 60
    for i = 1, count do
        local theta = math.random() * math.pi * 2
        local speed = (spec.speed or 140) * (0.6 + math.random() * 0.6)
        self.list[#self.list + 1] = {
            x = x + math.cos(theta) * radius,
            y = y + math.sin(theta) * radius,
            vx = -math.cos(theta) * speed,
            vy = -math.sin(theta) * speed,
            life = (spec.life or 0.45) * (0.7 + math.random() * 0.6),
            age = 0,
            size = (spec.size or 10) * (0.5 + math.random()),
            rotation = math.random() * math.pi * 2,
            spin = 0,
            gravity = 0,
            drag = 0.4,
            colour = spec.colour or {1, 1, 1},
        }
    end
end

function Particles:update(dt)
    local list = self.list
    local write = 1
    for read = 1, #list do
        local p = list[read]
        p.age = p.age + dt
        if p.age < p.life then
            local damping = math.exp(-p.drag * dt)
            p.vx = p.vx * damping
            p.vy = p.vy * damping + p.gravity * dt
            p.x = p.x + p.vx * dt
            p.y = p.y + p.vy * dt
            p.rotation = p.rotation + p.spin * dt
            list[write] = p
            write = write + 1
        end
    end
    for i = #list, write, -1 do list[i] = nil end
end

function Particles:draw()
    if #self.list == 0 then return end
    local mode, alphaMode = love.graphics.getBlendMode()
    love.graphics.setBlendMode("add", "alphamultiply")
    local sw, sh = self.sprite:getDimensions()
    local ox, oy = sw / 2, sh / 2
    for _, p in ipairs(self.list) do
        local t = p.age / p.life
        local fade = 1 - t * t          -- bright for most of the life, then gone
        local scale = p.size / sw * (1.1 - 0.5 * t)
        local c = p.colour
        love.graphics.setColor(c[1], c[2], c[3], fade)
        love.graphics.draw(self.sprite, p.x, p.y, p.rotation, scale, scale, ox, oy)
    end
    love.graphics.setBlendMode(mode, alphaMode)
    love.graphics.setColor(1, 1, 1, 1)
end

function Particles:count()
    return #self.list
end

function Particles:clear()
    self.list = {}
end

-- ------------------------------------------------------------- shockwaves

local Waves = {}
Waves.__index = Waves

--- Expanding rings, drawn from the halo sprite.
function effects.waves(sprite)
    return setmetatable({sprite = sprite, list = {}}, Waves)
end

function Waves:add(x, y, spec)
    spec = spec or {}
    self.list[#self.list + 1] = {
        x = x, y = y, age = 0,
        life = spec.life or 0.6,
        from = spec.from or 8,
        to = spec.to or 90,
        colour = spec.colour or {1, 1, 1},
        curve = spec.curve or ease.outQuart,
        width = spec.width or 1,
    }
end

function Waves:update(dt)
    local list = self.list
    local write = 1
    for read = 1, #list do
        local w = list[read]
        w.age = w.age + dt
        if w.age < w.life then
            list[write] = w
            write = write + 1
        end
    end
    for i = #list, write, -1 do list[i] = nil end
end

function Waves:draw()
    if #self.list == 0 then return end
    local mode, alphaMode = love.graphics.getBlendMode()
    love.graphics.setBlendMode("add", "alphamultiply")
    local sw, sh = self.sprite:getDimensions()
    for _, w in ipairs(self.list) do
        local t = w.age / w.life
        local radius = effects.lerp(w.from, w.to, t, w.curve)
        local alpha = (1 - t) ^ 2
        local c = w.colour
        love.graphics.setColor(c[1], c[2], c[3], alpha)
        love.graphics.draw(self.sprite, w.x, w.y, 0,
                           radius * 2 / sw, radius * 2 / sh, sw / 2, sh / 2)
    end
    love.graphics.setBlendMode(mode, alphaMode)
    love.graphics.setColor(1, 1, 1, 1)
end

function Waves:clear()
    self.list = {}
end

-- ------------------------------------------------------------ floating text

local Floaters = {}
Floaters.__index = Floaters

--- Short-lived text that rises and fades -- move names, "nice one", and so on.
function effects.floaters()
    return setmetatable({list = {}}, Floaters)
end

function Floaters:add(text, x, y, spec)
    spec = spec or {}
    self.list[#self.list + 1] = {
        text = text, x = x, y = y, age = 0,
        life = spec.life or 1.1,
        rise = spec.rise or 44,
        colour = spec.colour or {1, 1, 1},
        scale = spec.scale or 1,
    }
end

function Floaters:update(dt)
    local list = self.list
    local write = 1
    for read = 1, #list do
        local f = list[read]
        f.age = f.age + dt
        if f.age < f.life then
            list[write] = f
            write = write + 1
        end
    end
    for i = #list, write, -1 do list[i] = nil end
end

function Floaters:draw(font)
    for _, f in ipairs(self.list) do
        local t = f.age / f.life
        local y = f.y - ease.outCubic(t) * f.rise
        -- A quick pop on arrival, then a steady fade out.
        local scale = f.scale * effects.lerp(0.6, 1.0, math.min(1, t * 6), ease.outBack)
        local c = f.colour
        local width = font:getWidth(f.text) * scale
        love.graphics.setColor(0, 0, 0, (1 - t) * 0.5)
        love.graphics.print(f.text, f.x - width / 2 + 1, y + 1, 0, scale, scale)
        love.graphics.setColor(c[1], c[2], c[3], 1 - t * t)
        love.graphics.print(f.text, f.x - width / 2, y, 0, scale, scale)
    end
    love.graphics.setColor(1, 1, 1, 1)
end

function Floaters:clear()
    self.list = {}
end

-- ------------------------------------------------------------- screen shake

local Shake = {}
Shake.__index = Shake

--- Decaying positional shake.  `:apply()` translates the current transform, so
-- it wraps whatever is drawn between it and `love.graphics.pop()`.
function effects.shake()
    return setmetatable({amount = 0, time = 0}, Shake)
end

function Shake:kick(amount)
    self.amount = math.max(self.amount, amount)
end

function Shake:update(dt)
    self.time = self.time + dt
    self.amount = math.max(0, self.amount - self.amount * 6 * dt - 8 * dt)
end

function Shake:apply()
    if self.amount <= 0.01 then return end
    -- Two out-of-phase sines read as a shudder rather than as noise.
    local x = math.sin(self.time * 47) * self.amount
    local y = math.cos(self.time * 61) * self.amount
    love.graphics.translate(x, y)
end

-- ------------------------------------------------------------ ambient motes

local Motes = {}
Motes.__index = Motes

--- Slow specks drifting across the square, so the scene is never quite still.
function effects.motes(sprite, count, width, height)
    local self = setmetatable({sprite = sprite, list = {}}, Motes)
    self:resize(width, height)
    for _ = 1, count do
        self.list[#self.list + 1] = self:spawn(true)
    end
    return self
end

function Motes:resize(width, height)
    self.width, self.height = width, height
end

function Motes:spawn(anywhere)
    return {
        x = math.random() * self.width,
        y = anywhere and math.random() * self.height or self.height + 20,
        vx = (math.random() - 0.5) * 14,
        vy = -8 - math.random() * 18,
        size = 2 + math.random() * 5,
        alpha = 0.12 + math.random() * 0.3,
        phase = math.random() * math.pi * 2,
        rate = 0.6 + math.random() * 1.4,
    }
end

function Motes:update(dt, time)
    for i, m in ipairs(self.list) do
        m.x = m.x + (m.vx + math.sin(time * m.rate + m.phase) * 10) * dt
        m.y = m.y + m.vy * dt
        if m.y < -20 or m.x < -20 or m.x > self.width + 20 then
            self.list[i] = self:spawn(false)
        end
    end
end

function Motes:draw(colour)
    local mode, alphaMode = love.graphics.getBlendMode()
    love.graphics.setBlendMode("add", "alphamultiply")
    local sw = self.sprite:getWidth()
    local sh = self.sprite:getHeight()
    for _, m in ipairs(self.list) do
        love.graphics.setColor(colour[1], colour[2], colour[3], m.alpha)
        love.graphics.draw(self.sprite, m.x, m.y, 0,
                           m.size / sw, m.size / sh, sw / 2, sh / 2)
    end
    love.graphics.setBlendMode(mode, alphaMode)
    love.graphics.setColor(1, 1, 1, 1)
end

return effects
