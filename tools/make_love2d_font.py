#!/usr/bin/env python3
"""Draw the game's bitmap font, glyph by glyph, on a 7x9 grid.

    python3 tools/make_love2d_font.py

Writes ``love2d/assets/font.png`` -- a LÖVE image font, committed like the rest
of the art, so this only needs running to change a letter.

## Why the font is drawn here rather than taken from somewhere

It used to be Pillow's built-in default, which is a fine terminal face and the
wrong thing entirely: light, narrow, and drawn for a nine-pixel line of code
rather than for a menu on a console.  Set against chunky 16-bit sprites at the
size a modern window shows it, it read as small grey text on a pixel-art game.

A UI font from this era is not a typeface that happens to be small.  It is
designed on the grid, and every decision in it is about legibility at a hundred
times the size of a printed letter:

* **One stroke weight, everywhere.**  Every stem in here is a single pixel, so
  scaled two or three times it is a solid two- or three-pixel bar.  Nothing
  tapers, nothing is half a pixel, and there is no grey anywhere -- the alpha is
  0 or 255 and nothing between, which is what keeps it hard-edged when the game
  scales it up with a nearest-neighbour filter.
* **Caps fill the box.**  Seven rows of the nine, with the two under them left
  for the descenders, so a capital reads as a solid block of letter.
* **Square terminals and 45-degree diagonals.**  A curve is a staircase, drawn
  as one; there is no attempt to imply a smooth arc, because at three times the
  size the attempt is what looks wrong.

The result is deliberately closer to an MSX or a Mega Drive menu than to a
typeface: it is meant to be read across a room at a glance, alongside sprites
made of the same size of pixel.

## The format

``love.graphics.newImageFont`` wants the glyphs in a row, separated by a column
of the colour it finds at pixel (0, 0).  Each glyph here is six columns of
letter and a seventh of built-in spacing, so a run of text needs no letter
spacing added on top and the advance is a round seven pixels.

Glyph pixels are white, so the game tints them to whatever the panel needs.
Written with nothing but the standard library -- a PNG is a zlib stream with a
length and a checksum around it, and a font that needs a pillow installed to
edit one letter is a font nobody edits.
"""

from __future__ import annotations

import binascii
import os
import struct
import sys
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "love2d", "assets", "font.png")

# The grid.  Six columns of letter, one of spacing; seven rows of capital, two
# below them for the tails of g, j, p, q and y.
CELL_W, CELL_H = 7, 9
INK_W = 6

# Every printable ASCII character, in order.  The game hands the same string to
# `newImageFont`, and the two have to agree exactly.
GLYPHS = "".join(chr(c) for c in range(32, 127))

# Rows are written top to bottom, separated by slashes, and any left off the end
# are blank -- so a capital is seven rows and only the letters with tails need
# all nine.  `#` is ink and `.` is nothing.
#
# Lowercase starts two rows down, which is what gives it its x-height: written
# out, the two leading `....../` on every one of them is the whole reason `o` is
# smaller than `O` rather than merely narrower.
FONT = {
    " ": "",
    "!": "..##../..##../..##../..##../..##../....../..##..",
    '"': ".#..#./.#..#.",
    "#": ".#..#./######/.#..#./.#..#./######/.#..#.",
    "$": "..##../.#####/#.#.../.####./...#.#/#####./..##..",
    "%": "##...#/##..#./...#../..#.../.#..##/#...##",
    "&": ".###../#...#./#...#./.##.../#..#.#/#...##/.####.",
    "'": "..##../..##..",
    "(": "...##./..##../.##.../.##.../.##.../..##../...##.",
    ")": ".##.../..##../...##./...##./...##./..##../.##...",
    "*": "..#.../#.#.#./.###../#.#.#./..#...",
    "+": "....../..#.../..#.../######/..#.../..#...",
    ",": "....../....../....../....../..##../..##../.##...",
    "-": "....../....../....../######",
    ".": "....../....../....../....../....../..##../..##..",
    "/": ".....#/....#./...#../..#.../.#..../#.....",
    "0": ".####./#....#/#...##/#.##.#/##...#/#....#/.####.",
    "1": "..##../.###../..##../..##../..##../..##../.####.",
    "2": ".####./#....#/.....#/...##./.##.../#...../######",
    "3": ".####./#....#/.....#/..###./.....#/#....#/.####.",
    "4": "...##./..###./.#..#./#...#./######/....#./....#.",
    "5": "######/#...../#####./.....#/.....#/#....#/.####.",
    "6": "..###./.#..../#...../#####./#....#/#....#/.####.",
    "7": "######/.....#/....#./...#../..#.../..#.../..#...",
    "8": ".####./#....#/#....#/.####./#....#/#....#/.####.",
    "9": ".####./#....#/#....#/.#####/.....#/....#./.###..",
    ":": "....../..##../..##../....../..##../..##..",
    ";": "....../..##../..##../....../..##../..##../.##...",
    "<": "....#./...#../..#.../.#..../..#.../...#../....#.",
    "=": "....../....../######/....../######",
    ">": ".#..../..#.../...#../....#./...#../..#.../.#....",
    "?": ".####./#....#/.....#/...##./..##../....../..##..",
    "@": ".####./#....#/#.####/#.#..#/#.####/#...../.####.",
    "A": ".####./#....#/#....#/######/#....#/#....#/#....#",
    "B": "#####./#....#/#....#/#####./#....#/#....#/#####.",
    "C": ".####./#....#/#...../#...../#...../#....#/.####.",
    "D": "#####./#....#/#....#/#....#/#....#/#....#/#####.",
    "E": "######/#...../#...../#####./#...../#...../######",
    "F": "######/#...../#...../#####./#...../#...../#.....",
    "G": ".####./#....#/#...../#..###/#....#/#....#/.####.",
    "H": "#....#/#....#/#....#/######/#....#/#....#/#....#",
    "I": ".####./..##../..##../..##../..##../..##../.####.",
    "J": "..####/....#./....#./....#./....#./#...#./.###..",
    "K": "#....#/#...#./#..#../###.../#..#../#...#./#....#",
    "L": "#...../#...../#...../#...../#...../#...../######",
    "M": "#....#/##..##/#.##.#/#....#/#....#/#....#/#....#",
    "N": "#....#/##...#/#.#..#/#..#.#/#...##/#....#/#....#",
    "O": ".####./#....#/#....#/#....#/#....#/#....#/.####.",
    "P": "#####./#....#/#....#/#####./#...../#...../#.....",
    "Q": ".####./#....#/#....#/#....#/#..#.#/#...#./.###.#",
    "R": "#####./#....#/#....#/#####./#..#../#...#./#....#",
    "S": ".####./#....#/#...../.####./.....#/#....#/.####.",
    "T": "######/..##../..##../..##../..##../..##../..##..",
    "U": "#....#/#....#/#....#/#....#/#....#/#....#/.####.",
    "V": "#....#/#....#/#....#/#....#/#....#/.#..#./..##..",
    "W": "#....#/#....#/#....#/#.##.#/#.##.#/##..##/#....#",
    "X": "#....#/#....#/.#..#./..##../.#..#./#....#/#....#",
    "Y": "#....#/#....#/.#..#./..##../..##../..##../..##..",
    "Z": "######/.....#/....#./..##../.#..../#...../######",
    "[": "..###./..##../..##../..##../..##../..##../..###.",
    "\\": "#...../.#..../..#.../...#../....#./.....#",
    "]": ".###../..##../..##../..##../..##../..##../.###..",
    "^": "..##../.#..#./#....#",
    "_": "....../....../....../....../....../....../######",
    "`": "..##../...##.",
    "a": "....../....../.####./.....#/.#####/#....#/.#####",
    "b": "#...../#...../#####./#....#/#....#/#....#/#####.",
    "c": "....../....../.####./#...../#...../#....#/.####.",
    "d": ".....#/.....#/.#####/#....#/#....#/#....#/.#####",
    "e": "....../....../.####./#....#/######/#...../.####.",
    "f": "...##./..#.../.####./..#.../..#.../..#.../..#...",
    "g": "....../....../.####./#....#/#....#/.#####/.....#/#....#/.####.",
    "h": "#...../#...../#####./#....#/#....#/#....#/#....#",
    "i": "..##../....../.###../..##../..##../..##../.####.",
    "j": "...##./....../..###./...##./...##./...##./...##./#..##./.###..",
    "k": "#...../#...../#...#./#..#../###.../#..#../#...#.",
    "l": ".###../..##../..##../..##../..##../..##../..###.",
    "m": "....../....../##.##./#.#..#/#.#..#/#.#..#/#.#..#",
    "n": "....../....../#####./#....#/#....#/#....#/#....#",
    "o": "....../....../.####./#....#/#....#/#....#/.####.",
    "p": "....../....../#####./#....#/#....#/#####./#...../#...../#.....",
    "q": "....../....../.#####/#....#/#....#/.#####/.....#/.....#/.....#",
    "r": "....../....../#.###./##..../#...../#...../#.....",
    "s": "....../....../.#####/#...../.####./.....#/#####.",
    "t": "..#.../..#.../.####./..#.../..#.../..#.../...##.",
    "u": "....../....../#....#/#....#/#....#/#....#/.#####",
    "v": "....../....../#....#/#....#/#....#/.#..#./..##..",
    "w": "....../....../#....#/#....#/#.##.#/##..##/#....#",
    "x": "....../....../#....#/.#..#./..##../.#..#./#....#",
    "y": "....../....../#....#/#....#/#....#/.#####/.....#/#....#/.####.",
    "z": "....../....../######/....#./..##../.#..../######",
    "{": "...##./..##../..##../.##.../..##../..##../...##.",
    "|": "..##../..##../..##../..##../..##../..##../..##..",
    "}": ".##.../..##../..##../...##./..##../..##../.##...",
    "~": "....../....../.##..#/#..##.",
}

# The colour that marks the gap between two glyphs.  Magenta because it has to
# be a colour no letter is drawn in, and this is the one every tool that has
# ever done this has used.
SEPARATOR = (255, 0, 255, 255)


def bitmap(char):
    """One glyph as CELL_H rows of CELL_W booleans, spacing column included."""
    rows = [row for row in FONT[char].split("/") if row != ""] if FONT[char] else []
    out = []
    for y in range(CELL_H):
        row = rows[y] if y < len(rows) else ""
        # Every row is padded out to the full cell, so a short line in the table
        # above is blank rather than ragged, and the spacing column is simply
        # the one nothing is ever drawn in.
        out.append([x < len(row) and row[x] == "#" for x in range(CELL_W)])
    return out


def chunk(tag, data):
    body = tag + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I",
                                                             binascii.crc32(body) & 0xFFFFFFFF)


def write_png(path, width, height, pixels):
    """8-bit RGBA, one filter byte a row, all of it filter type 0.

    Filtering is what a PNG encoder spends its effort on, and there is nothing
    here for it to win: the image is a hard two-colour bitmap a few kilobytes
    wide, and zlib flattens the long runs of transparent black on its own.
    """
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        for x in range(width):
            raw.extend(pixels[y][x])

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(raw), 9))
    png += chunk(b"IEND", b"")
    with open(path, "wb") as out:
        out.write(png)
    return len(png)


def main(argv=None):
    missing = [c for c in GLYPHS if c not in FONT]
    if missing:
        raise SystemExit("no glyph drawn for: " + " ".join(missing))

    width = (CELL_W + 1) * len(GLYPHS) + 1
    clear = (0, 0, 0, 0)
    ink = (255, 255, 255, 255)
    pixels = [[clear] * width for _ in range(CELL_H)]

    for i, char in enumerate(GLYPHS):
        left = (CELL_W + 1) * i
        for y in range(CELL_H):
            pixels[y][left] = SEPARATOR
        glyph = bitmap(char)
        for y in range(CELL_H):
            for x in range(CELL_W):
                if glyph[y][x]:
                    pixels[y][left + 1 + x] = ink
    for y in range(CELL_H):
        pixels[y][width - 1] = SEPARATOR

    size = write_png(OUT, width, CELL_H, pixels)
    print(f"  {len(GLYPHS)} glyphs, {CELL_W}x{CELL_H} each -> "
          f"{os.path.relpath(OUT, ROOT)}  ({width}x{CELL_H}, {size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
