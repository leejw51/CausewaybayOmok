--- The sound chip: playing the effects in `assets/sfx`.
--
-- Those files are square waves, a stepped triangle and a shift register,
-- synthesised by `tools/make_love2d_sfx.py` and committed.  There is no sample
-- pack and no licence to honour -- a stone's clack is edited by changing a
-- number in that script and running it again.
--
-- Playing a sound is one line of LÖVE.  Everything here exists because of the
-- three ways doing it naively goes wrong:
--
-- **A Source is one voice.**  Calling `play` on a Source that is already
-- playing restarts it, so the model's stone landing on top of a fading spark
-- would cut it off.  Each effect keeps a small pool of clones and takes them in
-- turn.
--
-- **The game fires far more often than an ear wants.**  Hover is evaluated on
-- every pointer move, and a held arrow key steps the cursor at the key-repeat
-- rate.  Without a throttle the result is not a sound effect, it is a buzz, so
-- every effect has a minimum gap and the ones that fire most have the longest.
--
-- **The same sample twice in a row sounds like a stuck machine.**  A few
-- percent of random detune on the effects that repeat is enough for the ear to
-- hear two separate events rather than one glitch.
--
-- And it has to survive not being there.  A checkout without `assets/sfx`, a
-- machine with no audio device and a headless run all have to work: missing
-- sounds are recorded, and every `play` after that is a no-op.  Nothing in here
-- is allowed to be load-bearing.

local store = require("store")

local sound = {}

--- Every effect, and how loud it is *relative to the others*.
--
-- The absolute levels are baked into the files by the generator's `MIX` table.
-- These are the second, smaller adjustment: the place to make one effect sit
-- back a little without regenerating anything.
local WANTED = {
    -- the pointer and the keyboard
    hover = 0.7,
    blip  = 0.9,
    press = 1.0,
    level = 0.9,
    flip  = 0.9,
    -- the board.  One sound per side, not per player: see the generator.
    indigo = 1.0,
    amber  = 1.0,
    deny   = 0.9,
    undo   = 1.0,
    hint   = 1.0,
    start  = 1.0,
    -- the end of it
    win   = 1.0,
    lose  = 1.0,
    draw  = 1.0,
}

--- How many of each sound can overlap.  Three is enough for a stone landing
-- into the tail of the one before it, and small enough that fourteen effects
-- cost nothing.
local VOICES = 3

--- The shortest gap between two plays of the same effect, in seconds.
--
-- Tuned by what triggers it rather than by what it sounds like: `hover` fires
-- on every pointer movement and `blip` follows the key-repeat rate, while the
-- rest are things somebody did deliberately and want no more gate than enough
-- to swallow a double-fire in one frame.
local THROTTLE = {
    hover = 0.09,
    blip  = 0.045,
    level = 0.04,
}
local THROTTLE_DEFAULT = 0.02

--- How much random detune each effect gets, as a fraction of its pitch.
--
-- Only the ones that repeat.  A stone gets a little because a hundred of them
-- land in a game and identical clacks read as one looping sample; the three
-- outcome fanfares get none, because they are musical phrases played once and
-- detuning those only makes them sound out of tune.
local JITTER = {
    hover  = 0.08,
    blip   = 0.06,
    press  = 0.03,
    indigo = 0.05,
    amber  = 0.05,
    undo   = 0.05,
}

sound.pools = {}
sound.missing = {}
sound.last = {}
sound.enabled = true
sound.volume = 0.8

--- A clock advanced by `update`, not read from `love.timer`.
--
-- The throttle is the one piece of policy in this file worth testing, and a
-- test cannot advance a real clock.  Driving it from `dt` means it can be
-- stepped by any amount, and the game's behaviour is identical.
sound.clock = 0

local function audioAvailable()
    return love ~= nil and love.audio ~= nil and love.sound ~= nil
end

function sound.load()
    if not audioAvailable() then return end

    for name in pairs(WANTED) do
        local path = "assets/sfx/" .. name .. ".wav"
        if love.filesystem.getInfo(path) then
            -- Decoded once into SoundData and shared by every clone, so three
            -- voices cost three cursors rather than three copies of the audio.
            local data = love.sound.newSoundData(path)
            local pool = {index = 0, voices = {}}
            for i = 1, VOICES do
                pool.voices[i] = love.audio.newSource(data, "static")
            end
            sound.pools[name] = pool
        else
            sound.missing[#sound.missing + 1] = name
        end
    end
    table.sort(sound.missing)

    local saved = store.read().sound
    if saved ~= nil then sound.enabled = saved ~= "off" and saved ~= false end
end

function sound.update(dt)
    sound.clock = sound.clock + dt
end

--- Whether policy lets this effect sound right now, and records that it did.
--
-- Split out from `play` on purpose: this is the part with a decision in it and
-- it touches no LÖVE call, so the throttle can be checked by stepping
-- `sound.clock` with no audio device anywhere.
function sound.allowed(name)
    if not sound.enabled then return false end
    local gate = THROTTLE[name] or THROTTLE_DEFAULT
    local last = sound.last[name]
    if last and sound.clock - last < gate then return false end
    sound.last[name] = sound.clock
    return true
end

--- Play an effect.  Returns true if it actually sounded.
--
-- `options.pitch` multiplies the pitch -- the caller's way of saying "the same
-- sound, but higher because this is the fifth difficulty".  `options.volume`
-- likewise.  Both are on top of the tables above rather than instead of them.
function sound.play(name, options)
    options = options or {}
    if not sound.allowed(name) then return false end

    local pool = sound.pools[name]
    if not pool then return false end

    -- Round-robin rather than "find one that is not playing": if all three are
    -- busy the oldest is the right one to steal, and taking them in order does
    -- that for free.
    pool.index = pool.index % #pool.voices + 1
    local voice = pool.voices[pool.index]

    local spread = JITTER[name] or 0
    local pitch = (options.pitch or 1) * (1 + (math.random() * 2 - 1) * spread)

    voice:stop()
    voice:setPitch(math.max(0.05, pitch))
    voice:setVolume(sound.volume * (WANTED[name] or 1) * (options.volume or 1))
    voice:play()
    return true
end

--- Stop every voice of one effect.  For the fanfares, which a new game cuts.
function sound.stop(name)
    local pool = sound.pools[name]
    if not pool then return end
    for _, voice in ipairs(pool.voices) do voice:stop() end
end

function sound.stopAll()
    for name in pairs(sound.pools) do sound.stop(name) end
end

--- Turn the sound on or off, and remember which.
--
-- Kept with the rest of the preferences in `~/.causewaybayomok/settings.jsonl`;
-- see `store.lua`.  If it cannot be written the sound is simply on again next
-- time, which is the right way for a preference to fail.
function sound.toggle()
    sound.enabled = not sound.enabled
    if not sound.enabled then sound.stopAll() end
    store.write({sound = sound.enabled and "on" or "off"})
    return sound.enabled
end

return sound
