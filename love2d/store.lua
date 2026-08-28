--- Where the game's preferences live: `~/.causewaybayomok/settings.jsonl`.
--
-- Not LÖVE's save directory.  That is a path nobody can guess -- on a Mac it is
-- buried under `~/Library/Application Support/LOVE/` -- and the things kept in
-- it here are four values a person might reasonably want to look at, edit or
-- delete: which way up the board is, how big the text is, whether the window
-- fills the screen, and whether the sound is on.  A dotfolder in the home
-- directory is where a program of this size has always put that.
--
-- ## Why a log rather than a file that gets overwritten
--
-- JSON Lines: one object per line, and a line is only ever *appended*.  The
-- file is read by walking it from the top and letting each line overwrite the
-- keys it names, so the last word on any setting wins.
--
--     {"orientation":"landscape"}
--     {"text_size":3}
--     {"window":"window"}
--     {"text_size":2}
--
-- Which is worth the small strangeness of storing four values in a growing
-- file, for two reasons.  Appending never reads first, so two things changing a
-- setting cannot lose each other's write the way a read-modify-write pair can.
-- And a write that is cut off half way through -- the machine went down, the
-- disk filled -- damages one line at the end rather than the whole file: the
-- reader skips what it cannot parse and every setting before it survives.
--
-- It is compacted back to a single line once it gets long, so it does not grow
-- without end.
--
-- ## It has to survive not being there
--
-- Same rule as the art and the sound: a home directory that cannot be written
-- to, a first run with no file at all, and the headless test suite all have to
-- work.  Every function here fails to "no preferences", which is the right way
-- for a preference to fail.

local store = {}

local HOME = os.getenv("HOME") or os.getenv("USERPROFILE") or "."
store.dir = HOME .. "/.causewaybayomok"
store.file = store.dir .. "/settings.jsonl"

--- How many lines it is allowed to reach before being rewritten as one.
--
-- Every setting change is a line, so this is a few dozen changes -- far more
-- than a session makes, and small enough that the file never leaves a page.
local COMPACT_AT = 64

-- ------------------------------------------------------------------ the json
--
-- Flat objects of strings, numbers and booleans, which is all a settings file
-- has ever needed.  Written out by hand rather than pulled in: a JSON library
-- is a dependency, and this is thirty lines that cannot surprise anybody.

local function escape(text)
    return (tostring(text):gsub('[\\"]', "\\%0"):gsub("%c", " "))
end

local function encode(fields)
    -- Keys sorted, so two files written from the same settings are the same
    -- file -- which matters the moment anybody looks at one in a diff.
    local keys = {}
    for key in pairs(fields) do keys[#keys + 1] = key end
    table.sort(keys)

    local parts = {}
    for _, key in ipairs(keys) do
        local value = fields[key]
        local written
        if type(value) == "number" then written = ("%.14g"):format(value)
        elseif type(value) == "boolean" then written = tostring(value)
        else written = '"' .. escape(value) .. '"' end
        parts[#parts + 1] = ('"%s":%s'):format(escape(key), written)
    end
    return "{" .. table.concat(parts, ",") .. "}"
end

--- One line to a table, or nil if it is not one.
--
-- Deliberately forgiving: anything it cannot make sense of is skipped rather
-- than raised.  The only line likely to be malformed is a half-written last
-- one, and the whole point of the format is that such a line costs nothing.
local function decode(line)
    -- It has to be a *whole* object: an opening brace is not enough, because a
    -- line cut off half way through still has one, and the pairs before the cut
    -- would parse and be applied.  The closing brace is what says the writer
    -- finished, and refusing a line without one is the whole of what makes a
    -- torn write cost only the setting it was carrying.
    if not line:match("^%s*{.*}%s*$") then return nil end
    local fields = {}
    for key, value in line:gmatch('"([^"]+)"%s*:%s*([^,}]+)') do
        value = value:match("^%s*(.-)%s*$")
        local quoted = value:match('^"(.*)"$')
        if quoted then fields[key] = quoted
        elseif value == "true" then fields[key] = true
        elseif value == "false" then fields[key] = false
        elseif tonumber(value) then fields[key] = tonumber(value)
        end
    end
    return next(fields) and fields or nil
end

-- ------------------------------------------------------------------ the file

--- Make sure the file can be opened for appending.  Returns false if it cannot,
-- which is the signal to stop trying rather than an error to report: a game
-- that will not start because it could not write a preference is worse than one
-- that forgets which way up it was.
local function ready()
    local probe = io.open(store.file, "a")
    if probe then probe:close() return true end
    -- `mkdir -p` rather than a filesystem call, because LÖVE has none for a
    -- path outside its own save directory.
    os.execute(('mkdir -p "%s" 2>/dev/null'):format(store.dir))
    probe = io.open(store.file, "a")
    if not probe then return false end
    probe:close()
    return true
end

--- Every setting, merged in the order it was written.
function store.read()
    local fields, lines = {}, 0
    local handle = io.open(store.file, "r")
    if not handle then return fields, lines end
    for line in handle:lines() do
        lines = lines + 1
        local record = decode(line)
        if record then
            for key, value in pairs(record) do fields[key] = value end
        end
    end
    handle:close()
    return fields, lines
end

--- Append one record.  `fields` is whatever changed, not the whole of it.
function store.write(fields)
    if not next(fields) then return false end
    if not ready() then return false end

    local _, lines = store.read()
    if lines >= COMPACT_AT then
        -- Long enough to be worth flattening.  Read everything, write it back
        -- as one line, and carry on appending.  Done to a temporary file and
        -- moved into place, so a compaction that fails half way leaves the
        -- original rather than a stump.
        local merged = store.read()
        for key, value in pairs(fields) do merged[key] = value end
        local temporary = store.file .. ".new"
        local out = io.open(temporary, "w")
        if out then
            out:write(encode(merged), "\n")
            out:close()
            os.remove(store.file)
            os.rename(temporary, store.file)
            return true
        end
    end

    local handle = io.open(store.file, "a")
    if not handle then return false end
    handle:write(encode(fields), "\n")
    handle:close()
    return true
end

return store
