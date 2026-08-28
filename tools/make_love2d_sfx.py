#!/usr/bin/env python3
"""Bake the LÖVE game's sound effects, the way a 1983 sound chip would have.

    python3 tools/make_love2d_sfx.py            # all of them
    python3 tools/make_love2d_sfx.py indigo win  # just these

Writes ``love2d/assets/sfx/*.wav`` and prints what it made.  Both this script
and its output are committed, so nobody needs to run it to play; it only needs
running to *change* a sound.

Synthesised rather than downloaded, for the same reason the art is generated
rather than pasted in: a sample pack is somebody's licence to honour, somebody's
server to stay up and a megabyte to carry, and none of it is editable.  A PSG is
two square waves, a stepped triangle, a noise register and a four-bit volume --
about a hundred lines of arithmetic -- so here a stone's clack is a number in
this file, not a waveform in a binary nobody can open.

Three constraints make it sound like a chip rather than a soft synth, and all
three are deliberate:

* **The volume is four bits.**  An AY-3-8910 or a 2A03 had sixteen levels and no
  more, so every envelope here is quantised to sixteen steps.  That stair-step
  on a decay tail is a large part of the sound.
* **The waveforms are the ones the hardware had.**  Square at a few duty cycles,
  a stepped triangle, and noise from a linear-feedback shift register.  No sine
  anywhere: a PSG could not make one.
* **The noise is a real LFSR**, the same 15-bit one the NES used, clocked by a
  divider.  White noise from a random number generator has a different, hissier
  character; this is why the stones sound like wood and not like static.

Output is 8-bit unsigned mono at 22050 Hz, which is roughly what these chips
were sampled at and keeps the whole set under a hundred kilobytes.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(ROOT, "love2d", "assets", "sfx")

RATE = 22050

# Sixteen volume levels, signed: the four-bit envelope the hardware had.  The
# stair-step this puts on a decay is audible, and wanted.
LEVELS = 15


# ---------------------------------------------------------------- oscillators

def square(phase, duty=0.5):
    """A pulse wave.  The duty cycle is the whole character of it.

    0.5 is hollow and flute-like, 0.25 is the classic lead, 0.125 is thin and
    nasal -- the three an NES could actually select, and the reason a chiptune
    lead and a chiptune bass sound different when they are the same waveform.
    """
    return 1.0 if (phase % 1.0) < duty else -1.0


def triangle(phase):
    """A stepped triangle, sixteen steps up and sixteen down.

    Not a smooth ramp: the 2A03's triangle channel walked a 4-bit counter, so
    the steps are in the hardware.  They are what stops it sounding like a sine.
    """
    p = phase % 1.0
    value = 4.0 * p - 1.0 if p < 0.5 else 3.0 - 4.0 * p
    return math.floor(value * 8.0 + 0.5) / 8.0


class Noise:
    """The NES's 15-bit linear-feedback shift register.

    Clocked by a divider rather than run at the sample rate, which is what gives
    noise a *pitch*: a high divider is a hiss, a low one is a rumble, and
    sweeping it downwards is every impact sound ever made on this hardware.

    The short-mode tap (bit 1 rather than bit 6) shortens the period enough that
    it turns tonal and woody -- that is the stone hitting the board.
    """

    def __init__(self, short=False):
        self.register = 1
        self.tap = 1 if short else 6
        self.phase = 0.0
        self.value = 1.0

    def sample(self, freq):
        self.phase += freq / RATE
        while self.phase >= 1.0:
            self.phase -= 1.0
            bit = (self.register & 1) ^ ((self.register >> self.tap) & 1)
            self.register = (self.register >> 1) | (bit << 14)
            self.value = -1.0 if (self.register & 1) else 1.0
        return self.value


# ------------------------------------------------------------------ envelopes

def decay(power=3.0):
    """Struck and left to ring.  The workhorse: every blip and every stone."""
    return lambda u: max(0.0, 1.0 - u) ** power


def hold(release=0.15):
    """Flat, then off.  A sustained note that does not fade while it plays."""
    def shape(u):
        if u >= 1.0:
            return 0.0
        if u > 1.0 - release:
            return (1.0 - u) / release
        return 1.0
    return shape


def swell(attack=0.25, power=2.0):
    """Rises, then falls.  Something arriving rather than something struck."""
    def shape(u):
        if u < attack:
            return (u / attack) ** 0.6
        return max(0.0, 1.0 - (u - attack) / (1.0 - attack)) ** power
    return shape


# ---------------------------------------------------------------------- pitch

NAMES = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6,
         "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}


def note(name):
    """``"A4"`` -> 440.0.  Written as notes because the intervals carry."""
    pitch, octave = name[:-1], int(name[-1])
    semitones = NAMES[pitch] + (octave - 4) * 12 - 9
    return 440.0 * (2.0 ** (semitones / 12.0))


def sweep(start, end, curve=1.0):
    """A pitch that slides.  ``curve`` above 1 holds low, then rushes."""
    return lambda u: start * ((end / start) ** (min(1.0, max(0.0, u)) ** curve))


# --------------------------------------------------------------------- voices

def tone(duration, freq, envelope, wave_shape="square", duty=0.5, gain=1.0,
         vibrato=0.0, vibrato_rate=6.0, delay=0.0):
    """One voice, rendered to a list of floats.

    ``freq`` is a number or a function of progress, so a slide costs nothing
    extra.  Phase is accumulated rather than recomputed from the time, which is
    the only way a slide stays continuous instead of clicking every sample.
    """
    total = int((delay + duration) * RATE)
    lead = int(delay * RATE)
    out = [0.0] * total
    phase = 0.0
    noise = Noise(short=(wave_shape == "wood"))

    for i in range(lead, total):
        u = (i - lead) / max(1, duration * RATE)
        f = freq(u) if callable(freq) else freq
        if vibrato:
            f *= 1.0 + vibrato * math.sin(2.0 * math.pi * vibrato_rate * u * duration)

        amplitude = envelope(u) * gain
        # Four bits, and the quantising happens *before* the waveform is
        # scaled: an envelope step is a step in level, not a smooth fade.
        amplitude = math.floor(amplitude * LEVELS + 0.5) / LEVELS

        if wave_shape in ("noise", "wood"):
            value = noise.sample(f)
        elif wave_shape == "triangle":
            phase += f / RATE
            value = triangle(phase)
        else:
            phase += f / RATE
            value = square(phase, duty)

        out[i] = value * amplitude

    return out


def mix(*voices):
    """Sum the voices.  Deliberately does not clip.

    Clipping here would be irreversible and invisible -- a sound would arrive at
    the levelling stage already squared off.  Levelling happens once, at the
    end, where it can be seen: see ``MIX``.
    """
    length = max((len(v) for v in voices), default=0)
    out = [0.0] * length
    for voice in voices:
        for i, value in enumerate(voice):
            out[i] += value
    return out


def arpeggio(names, step, envelope=None, wave_shape="square", duty=0.5,
             gain=0.8, last=None):
    """A run of notes, one after another.

    The chip's way of playing a chord: it had no polyphony to spare, so it
    played the notes fast enough in sequence that the ear hears one sound.
    """
    envelope = envelope or decay(2.0)
    voices = []
    for index, name in enumerate(names):
        length = last if (last and index == len(names) - 1) else step
        voices.append(tone(length, note(name), envelope, wave_shape=wave_shape,
                           duty=duty, gain=gain, delay=index * step))
    return mix(*voices)


# ----------------------------------------------------------------- the sounds

def build():
    """Every effect, named for the moment it belongs to rather than its shape."""
    sounds = {}

    # The cursor stepping between points.  Deliberately tiny -- this plays more
    # than anything else, and a tail would turn a held arrow key into a drone.
    sounds["blip"] = tone(0.04, note("E6"), decay(4.0), duty=0.25, gain=0.5)

    # The pointer crossing onto a new point.  Quieter and higher than the blip,
    # because it is feedback for something nobody has committed to yet.
    sounds["hover"] = tone(0.028, note("B6"), decay(5.0), duty=0.125, gain=0.22)

    # A panel button going in: two notes, up.  The interval does the work -- a
    # fifth reads as "yes, that happened" where a single note reads as a tick.
    sounds["press"] = arpeggio(["A5", "E6"], 0.035, decay(3.0), duty=0.25, gain=0.55)

    # ------------------------------------------------------------ the stones
    #
    # A stone landing is the sound this game is *for*, so it gets the most care.
    # It is wood on wood, which is a click and a hollow ring at once: the LFSR
    # in short mode gives the click its grain, a body tone sweeping down gives
    # the ring, and a triangle underneath is the board itself moving.
    #
    # Indigo and amber get the same sound a fourth apart.  Not two sounds --
    # they are the same stone on the same board -- but far enough apart that a
    # move is heard as one side or the other without looking up from the board,
    # which is what makes the model's replies legible while you are still
    # reading the shape you just made.  They are named for the two sides rather
    # than for who played them, so watch mode needs no third sound.
    def stone(body, click, gain=1.0):
        return mix(
            tone(0.045, click, decay(3.2), wave_shape="wood", gain=0.55 * gain),
            tone(0.14, sweep(body, body * 0.55, 0.7), decay(3.4), duty=0.5,
                 gain=0.45 * gain),
            tone(0.20, body * 0.5, decay(2.6), wave_shape="triangle",
                 gain=0.32 * gain),
        )

    sounds["indigo"] = stone(note("D4"), 2600.0)
    sounds["amber"] = stone(note("A3"), 1900.0)

    # A point that is already taken.  A tritone, held then dropped: the interval
    # every game has used for "no" since there were games to use it.
    sounds["deny"] = mix(
        tone(0.16, sweep(note("A3"), note("D#3")), hold(0.4), duty=0.5, gain=0.7),
        tone(0.20, 1300.0, decay(2.0), wave_shape="noise", gain=0.22),
    )

    # Taking a move back: the stone's own sweep, run the other way.  A stone
    # being lifted off is the one event that undoes another, and reversing the
    # direction of the slide is the whole of how the ear hears that.
    sounds["undo"] = mix(
        tone(0.18, sweep(note("D3"), note("A4"), 1.4), decay(2.4), duty=0.125,
             gain=0.42),
        tone(0.13, sweep(1200.0, 5200.0), decay(2.8), wave_shape="noise", gain=0.22),
    )

    # The model's advice arriving: a bright rising figure, and a shimmer over
    # it.  Deliberately unlike either stone -- nothing was played, something was
    # suggested, and the two must never be confused by ear.
    sounds["hint"] = mix(
        arpeggio(["E5", "B5", "F#6"], 0.055, decay(2.6), duty=0.125, gain=0.42,
                 last=0.34),
        tone(0.34, 6400.0, decay(3.0), wave_shape="wood", gain=0.16, delay=0.08),
    )

    # A fresh board.  Short, up, and over before the intro tween has finished
    # sweeping the grid in.
    sounds["start"] = mix(
        arpeggio(["D5", "A5", "D6"], 0.05, decay(2.8), duty=0.25, gain=0.45,
                 last=0.30),
        tone(0.36, note("D3"), decay(2.4), wave_shape="triangle", gain=0.38),
    )

    # The difficulty changing.  One note, and the game plays it higher for a
    # higher level -- see `sound.play`'s pitch option in love2d/sound.lua, which
    # is why this is a single tone and not a phrase.
    sounds["level"] = tone(0.07, note("A5"), decay(3.5), duty=0.25, gain=0.45)

    # Turning the board on its side, or going fullscreen.  A swoosh: noise
    # sweeping up under a rising blip, which is a thing physically moving rather
    # than another button.
    sounds["flip"] = mix(
        tone(0.22, sweep(900.0, 6000.0, 0.8), swell(0.35, 1.6),
             wave_shape="noise", gain=0.5),
        tone(0.14, sweep(note("A4"), note("E6"), 1.6), decay(2.4), duty=0.125,
             gain=0.3, delay=0.05),
    )

    # ----------------------------------------------------------- the outcomes
    #
    # These are the only three sounds allowed to take over the room, and they
    # are the longest by a wide margin: the winning line is lit one stone at a
    # time under a fountain of particles, and the sound has to still be there
    # when the last of them lands.

    # Winning.  A major fanfare over a bass that moves under it, so it lands
    # rather than merely stopping.
    sounds["win"] = mix(
        arpeggio(["G5", "C6", "E6", "G6"], 0.085, decay(1.9), duty=0.25,
                 gain=0.5, last=0.62),
        tone(1.0, note("C3"), decay(1.8), wave_shape="triangle", gain=0.5),
        tone(0.6, 6000.0, decay(3.0), wave_shape="wood", gain=0.16, delay=0.26),
    )

    # Losing.  The same shape walking downwards, minor, and slower -- the ear
    # reads a descending phrase as a defeat with no other cue needed.
    sounds["lose"] = mix(
        arpeggio(["E5", "C5", "A4", "F4"], 0.11, decay(1.8), duty=0.5,
                 gain=0.45, last=0.70),
        tone(1.1, sweep(note("F3"), note("D3"), 1.6), decay(1.6),
             wave_shape="triangle", gain=0.5),
    )

    # A draw.  Neither of the above: two notes a whole tone apart, held and let
    # go, which resolves nowhere on purpose.
    sounds["draw"] = mix(
        arpeggio(["G4", "F4"], 0.16, hold(0.35), duty=0.5, gain=0.42, last=0.52),
        tone(0.7, note("F3"), decay(2.0), wave_shape="triangle", gain=0.38),
    )

    return sounds


# --------------------------------------------------------------------- output

# How loud each sound ends up, as a fraction of full scale.
#
# Set here rather than left to whatever the voice gains happened to sum to.
# Per-voice levelling does not work: `deny` stacks a square on a noise burst and
# came out three times the level of `blip`, which would have made a mistake the
# loudest thing in the game.  This is a mixing desk, and the ordering in it is
# the actual design --
#
#   whispers   things that fire constantly and must not be noticed
#   taps       things somebody did, heard once and gone
#   events     a stone landed, or a move was refused
#   moments    the three sounds allowed to take over: the end of the game
#
# Levelling to peak also means nothing clips, whatever the voices sum to.
MIX = {
    "hover": 0.20,                                                  # whispers
    "blip": 0.40, "press": 0.50, "level": 0.46, "flip": 0.50,           # taps
    "indigo": 0.72, "amber": 0.66, "undo": 0.52,                        # events
    "deny": 0.60, "hint": 0.62, "start": 0.66,
    "win": 0.90, "lose": 0.82, "draw": 0.72,                         # moments
}


def level_to(samples, target):
    """Scale so the loudest sample lands exactly on ``target``."""
    peak = max((abs(v) for v in samples), default=0.0)
    if peak <= 0.0:
        return samples
    return [v * (target / peak) for v in samples]


def write(path, samples):
    """8-bit unsigned mono, the format these chips were sampled to.

    With a short fade over the last few milliseconds: a waveform cut mid-cycle
    ends on a step, and a step is a click.  Two hundred samples is inaudible as
    a fade and removes it completely.
    """
    tail = min(200, len(samples))
    for i in range(tail):
        samples[len(samples) - tail + i] *= 1.0 - (i / tail)

    frames = bytearray()
    for value in samples:
        clipped = max(-1.0, min(1.0, value))
        frames.append(int(round(clipped * 127.0)) + 128)

    with wave.open(path, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(1)
        out.setframerate(RATE)
        out.writeframes(bytes(frames))

    return len(frames)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("only", nargs="*", metavar="SOUND",
                        help="limit to these sounds (default: all of them)")
    args = parser.parse_args(argv)

    os.makedirs(OUT_DIR, exist_ok=True)
    sounds = build()

    # Every sound must have a level and every level must have a sound: a typo in
    # either table would otherwise ship one effect at whatever its voices summed
    # to, which is exactly the mistake `MIX` exists to prevent.
    if set(sounds) != set(MIX):
        parser.error("MIX and build() disagree: "
                     + ", ".join(sorted(set(sounds) ^ set(MIX))))
    sounds = {name: level_to(data, MIX[name]) for name, data in sounds.items()}

    if args.only:
        unknown = sorted(set(args.only) - set(sounds))
        if unknown:
            parser.error("no such sound: " + ", ".join(unknown))
        sounds = {n: d for n, d in sounds.items() if n in args.only}

    total = 0
    for name, samples in sorted(sounds.items()):
        written = write(os.path.join(OUT_DIR, f"{name}.wav"), samples)
        total += written
        print(f"  {name:<7} {written / RATE:5.2f}s  {written / 1024:6.1f} KB")

    print(f"  {len(sounds)} sounds, {total / 1024:.1f} KB total -> "
          f"{os.path.relpath(OUT_DIR, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
