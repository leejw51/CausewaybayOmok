#!/usr/bin/env python3
"""Draw the macOS app icon for the LÖVE game from the game's own art.

    make love-icon            # writes love2d/icon.png (1024x1024)

The icon is the board: the wood tile from `love2d/assets/board.png` laid out
at eight times its size, the grid the game draws over it, and five stones in
the shape every omok player recognises -- an open four for black, two white
stones that came too late.  `make app` turns this one PNG into the `.icns`
with `sips` and `iconutil`, so this is the only picture that has to exist.

Nearest-neighbour throughout: the art is pixel art and stays pixel art.  Read
and written with nothing but the standard library, the same way the font is,
so remaking the icon needs no pillow and no API key.  The result is committed;
this only has to run when the art changes.
"""

from __future__ import annotations

import binascii
import math
import os
import struct
import sys
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ASSETS = os.path.join(ROOT, "love2d", "assets")
OUT = os.path.join(ROOT, "love2d", "icon.png")

SIZE = 1024
# Big Sur icons are supplied already rounded; a square one looks pasted on.
CORNER = 180
# The game draws its grid lines in a dark wood brown, two pixels wide at
# scale 2.  At scale 8 that is eight, which reads as a line at any size.
GRID = 8
GRID_COLOUR = (60, 34, 12, 255)
CELL = 128
# Stones are 32px sprites; the game draws them at scale 2, one per cell.  Here
# the cell is 128 so they are drawn at 4, which fills the cell the way the
# board does at its own scale.
STONE_SCALE = 4


# -------------------------------------------------------------------- png

def read_png(path):
    """Non-interlaced PNG at any of the usual depths; returns rows of RGBA.

    The board is a 4-bit palette image and the stones are 8-bit RGBA, so both
    the packed and the byte-per-sample paths get exercised on every run.
    """
    with open(path, "rb") as handle:
        data = handle.read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise SystemExit(f"{path}: not a PNG")
    pos = 8
    width = height = None
    colour_type = depth = interlace = None
    palette, trns = [], b""
    idat = bytearray()
    while pos < len(data):
        length, = struct.unpack(">I", data[pos:pos + 4])
        tag = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if tag == b"IHDR":
            width, height, depth, colour_type, _, _, interlace = struct.unpack(">IIBBBBB", body)
        elif tag == b"PLTE":
            palette = [tuple(body[i:i + 3]) for i in range(0, len(body), 3)]
        elif tag == b"tRNS":
            trns = body
        elif tag == b"IDAT":
            idat.extend(body)
        elif tag == b"IEND":
            break
    if interlace != 0:
        raise SystemExit(f"{path}: interlaced PNGs are not read here")
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(colour_type)
    if channels is None or depth not in (1, 2, 4, 8):
        raise SystemExit(f"{path}: unsupported PNG (depth {depth}, colour type {colour_type})")
    bpp = max(1, channels * depth // 8)
    stride = (width * channels * depth + 7) // 8
    raw = zlib.decompress(bytes(idat))

    rows = []
    previous = bytearray(stride)
    offset = 0
    for _ in range(height):
        filter_type = raw[offset]
        line = bytearray(raw[offset + 1:offset + 1 + stride])
        offset += 1 + stride
        for i in range(stride):
            a = line[i - bpp] if i >= bpp else 0
            b = previous[i]
            c = previous[i - bpp] if i >= bpp else 0
            if filter_type == 0:
                add = 0
            elif filter_type == 1:
                add = a
            elif filter_type == 2:
                add = b
            elif filter_type == 3:
                add = (a + b) // 2
            elif filter_type == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                add = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
            else:
                raise SystemExit(f"{path}: unknown filter type {filter_type}")
            line[i] = (line[i] + add) & 0xFF
        previous = line

        # Unpack the samples, then turn each pixel into RGBA.
        if depth == 8:
            samples = list(line)
        else:
            per_byte = 8 // depth
            mask = (1 << depth) - 1
            samples = []
            for byte in line:
                for k in range(per_byte):
                    samples.append((byte >> (8 - depth * (k + 1))) & mask)
            samples = samples[:width * channels]
        scale = 255 // ((1 << depth) - 1) if colour_type in (0, 4) else 1
        row = []
        for x in range(width):
            s = samples[x * channels:(x + 1) * channels]
            if colour_type == 3:
                r, g, b = palette[s[0]]
                a = trns[s[0]] if s[0] < len(trns) else 255
                row.append((r, g, b, a))
            elif colour_type == 6:
                row.append(tuple(s))
            elif colour_type == 2:
                row.append(tuple(s) + (255,))
            elif colour_type == 4:
                row.append((s[0] * scale,) * 3 + (s[1] * scale,))
            else:
                row.append((s[0] * scale,) * 3 + (255,))
        rows.append(row)
    return width, height, rows


def chunk(tag, data):
    body = tag + data
    return struct.pack(">I", len(data)) + body + struct.pack(
        ">I", binascii.crc32(body) & 0xFFFFFFFF)


def write_png(path, width, height, pixels):
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


# ---------------------------------------------------------------- drawing

def blend(under, over):
    """Source-over, the way the game composites a sprite onto the board."""
    a = over[3] / 255.0
    if a >= 1.0:
        return over
    if a <= 0.0:
        return under
    ua = under[3] / 255.0
    out_a = a + ua * (1 - a)
    if out_a == 0:
        return (0, 0, 0, 0)
    return tuple(
        int(round((over[i] * a + under[i] * ua * (1 - a)) / out_a)) for i in range(3)
    ) + (int(round(out_a * 255)),)


def stamp(canvas, sprite, x0, y0, scale):
    w, h, rows = sprite
    for sy in range(h):
        for sx in range(w):
            pixel = rows[sy][sx]
            if pixel[3] == 0:
                continue
            for dy in range(scale):
                for dx in range(scale):
                    x, y = x0 + sx * scale + dx, y0 + sy * scale + dy
                    if 0 <= x < SIZE and 0 <= y < SIZE:
                        canvas[y][x] = blend(canvas[y][x], pixel)


def rounded(x, y):
    """Whether (x, y) is inside the rounded square."""
    r = CORNER
    cx = min(max(x, r), SIZE - 1 - r)
    cy = min(max(y, r), SIZE - 1 - r)
    return math.hypot(x - cx, y - cy) <= r


def main(argv=None):
    board = read_png(os.path.join(ASSETS, "board.png"))
    black = read_png(os.path.join(ASSETS, "stone_black.png"))
    white = read_png(os.path.join(ASSETS, "stone_white.png"))

    # The wood, tiled at eight times its size.  128px tile x 8 = the icon.
    tw, th, tile = board
    tile_scale = SIZE // tw
    canvas = [[(0, 0, 0, 0)] * SIZE for _ in range(SIZE)]
    for y in range(SIZE):
        row = canvas[y]
        src = tile[(y // tile_scale) % th]
        for x in range(SIZE):
            row[x] = src[(x // tile_scale) % tw]

    # The grid: lines through the centre of every cell, like the game's board,
    # so the stones sit on intersections and not in squares.
    for k in range(SIZE // CELL):
        centre = k * CELL + CELL // 2
        for offset in range(-GRID // 2, GRID // 2):
            line = centre + offset
            for i in range(SIZE):
                canvas[line][i] = GRID_COLOUR
                canvas[i][line] = GRID_COLOUR

    # The position: black has four in a row on the diagonal with both ends
    # open, which is the shape that ends a game of omok; white tried the
    # middle too late.  Cells are counted from the top-left intersection.
    def place(sprite, col, row):
        w, h, _ = sprite
        cx = col * CELL + CELL // 2
        cy = row * CELL + CELL // 2
        stamp(canvas, sprite, cx - (w * STONE_SCALE) // 2, cy - (h * STONE_SCALE) // 2,
              STONE_SCALE)

    for i in range(4):
        place(black, 2 + i, 2 + i)
    place(white, 4, 3)
    place(white, 3, 5)

    # Rounded corners, and one dark pixel-row of edge so the wood does not
    # bleed into a light dock.
    for y in range(SIZE):
        for x in range(SIZE):
            if not rounded(x, y):
                canvas[y][x] = (0, 0, 0, 0)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    size = write_png(OUT, SIZE, SIZE, canvas)
    print(f"  {os.path.relpath(OUT, ROOT)}  {SIZE}x{SIZE}  {size // 1024} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
