#!/usr/bin/env python3
"""Generate the LÖVE game's 16-bit pixel art with Grok, then make it game-ready.

    export XAI_API_KEY=...
    python3 tools/make_love2d_assets.py            # reuse raw images already there
    python3 tools/make_love2d_assets.py --force    # ask Grok for fresh ones

Raw generations land in ``assets_raw/love2d/`` (git-ignored); the finished
sprites land in ``love2d/assets/`` and are committed, so nobody needs an API
key to play.  The prompts live here rather than in a chat log -- that is the
only way the art stays reproducible.

The setting is Prague's Old Town Square at dusk: the Astronomical Clock, the
twin Tyn spires, amber lamps on wet cobbles, drawn the way a 1993 SNES or Amiga
background artist would have drawn it.

Two things make the difference between "an image of pixel art" and pixel art.
An image model draws at 1024px with soft anti-aliased edges, so every asset is
**downsampled to its true pixel grid with a box filter and then quantised to a
small palette** -- that is what turns smooth gradients into flat colour blocks
and gives the edges their hard staircase.  And every asset is **PNG**: these
are composited over lit backgrounds with additive glows, where JPEG's ringing
shows up as haloes around the lamps and dirty fringes along the stones' cut
edges, and a 16-colour image is smaller as PNG anyway.

The game draws these with a nearest-neighbour filter and scales them by whole
numbers, so the pixel grid stays square on screen.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_assets import generate  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW_DIR = os.path.join(ROOT, "assets_raw", "love2d")
OUT_DIR = os.path.join(ROOT, "love2d", "assets")

# Repeated in every prompt: without it the model drifts back to smooth digital
# painting, which no amount of downsampling turns into convincing pixel art.
RETRO = ("16-bit pixel art in the style of a 1993 SNES or Amiga game, chunky visible square "
         "pixels, hard aliased edges with no blur and no anti-aliasing, flat blocks of colour, "
         "limited palette, dithered gradients, crisp pixel-perfect shapes. ")

PROMPTS = {
    # The game's backdrop.  The Astronomical Clock is the subject rather than
    # a thing on the left: it is the one landmark on the square everybody
    # knows, and the panel and board cover most of the width anyway, so what
    # survives in the gaps has to be the clock.
    "backdrop":
        RETRO +
        "Wide scenic background of Old Town Square in Prague at dusk, centred on the famous "
        "Prague Astronomical Clock (the Orloj) on the Old Town Hall tower. The clock is large "
        "and clearly legible in the middle of the frame: the big blue and gold astrolabe dial "
        "with its golden zodiac ring and ornate hands, the painted calendar dial below it, the "
        "gilded stone gothic tower rising above with a pointed roof. The twin gothic spires of "
        "the Tyn Church stand further back on the right against a deep indigo night sky with a "
        "few pixel stars. Pastel baroque facades either side, glowing amber windows, warm "
        "street lamps casting pools of light on wet cobblestones in the foreground. Rich "
        "saturated colours, strong dithering in the sky gradient, dark enough overall that "
        "bright sprites read clearly on top. No people, no text, no watermark, no user "
        "interface.",
    # The title screen's picture: the dial itself, close, as the thing the
    # logo is set against.  A landmark drawn at a distance is scenery; drawn
    # this close it is the game's face.
    "title":
        RETRO +
        "Close-up of the Prague Astronomical Clock at night filling the frame: the great blue "
        "and gold astrolabe dial in the centre with its golden zodiac ring, roman numerals, "
        "ornate gilded hands and the small golden sun and moon, the gothic stone tower wall "
        "and carved statues around it, the painted calendar dial just visible below. Lit by "
        "warm amber lamplight from below against a deep indigo night sky with a few pixel "
        "stars at the top. Rich saturated blue, gold and amber, heavy ordered dithering, "
        "symmetrical composition with the dial centred and the upper third of the frame "
        "quieter and darker so a title can be printed over it. No people, no text, no "
        "watermark, no user interface.",
    "board":
        RETRO +
        "Seamless tileable texture of a warm honey-golden wooden table top seen from directly "
        "above, lit by amber lamplight. Chunky pixel wood grain in three or four shades of "
        "golden brown, subtle dithering, semi-gloss. Perfectly even lighting, no shadows, "
        "no objects, no drawn lines, no grid, no text. Flat texture filling the entire frame.",
    "stone_black":
        RETRO +
        "A single round game stone sprite seen from directly above, a PERFECT CIRCLE, not an "
        "oval, not tilted. Glossy dark obsidian, near-black with deep indigo shading, a bright "
        "highlight of two or three pixels in the upper left and a warm amber lamp reflection "
        "along the lower right rim. The circle is centered and fills the middle of the frame. "
        "Plain flat pure white background, no shadow, no text.",
    "stone_white":
        RETRO +
        "A single round game stone sprite seen from directly above, a PERFECT CIRCLE, not an "
        "oval, not tilted. Creamy ivory white with warm beige shading, a bright white highlight "
        "of two or three pixels in the upper left and a cool indigo reflection along the lower "
        "right rim. The circle is centered and fills the middle of the frame. Plain flat pure "
        "black background, no shadow, no text.",
    "spark":
        RETRO +
        "A perfectly circular radial burst of light centered on a flat pure black background. "
        "White-hot core stepping outward through concentric rings of pale yellow, warm gold and "
        "deep amber to pure black, with heavy ordered dithering between the rings. Perfectly "
        "symmetrical circle, equal width and height. No oval, no streaks, no other objects, "
        "no text.",
    "halo":
        RETRO +
        "A perfectly circular ring of warm gold light on a flat pure black background, like the "
        "gilded outer ring of an astronomical clock dial seen head on. Bright gold ring, dark "
        "centre, dithered falloff on both sides of the ring. Perfectly symmetrical, equal width "
        "and height. No oval, no streaks, no other objects, no text.",
}


# ------------------------------------------------------------------ pixel art

def pixelate(image: Image.Image, width: int, height: int, colours: int) -> Image.Image:
    """Downsample onto the real pixel grid, then flatten to a small palette.

    BOX averaging (not LANCZOS) is what keeps the result free of the ringing
    that would put stray off-palette pixels along every edge; quantising
    afterwards is what turns the model's smooth shading into flat blocks.
    """
    small = image.convert("RGB").resize((width, height), Image.BOX)
    return small.quantize(colors=colours, method=Image.MEDIANCUT, dither=Image.FLOYDSTEINBERG)


def scenic(image: Image.Image, width: int, height: int, colours: int,
           dim: float = 0.62) -> Image.Image:
    """Crop to the window's aspect, pixelate, and knock the brightness back."""
    target = width / height
    if image.width / image.height > target:
        side = int(image.height * target)
        left = (image.width - side) // 2
        box = (left, 0, left + side, image.height)
    else:
        side = int(image.width / target)
        top = (image.height - side) // 2
        box = (0, top, image.width, top + side)
    cropped = image.convert("RGB").crop(box)
    pixels = np.asarray(cropped, dtype=np.float32) / 255.0
    # The backdrop sits behind everything, so it is dimmed here rather than by
    # drawing a scrim over it every frame; the blue lift keeps it reading as
    # dusk instead of as a flat grey wash.
    pixels *= dim
    pixels[..., 2] = np.clip(pixels[..., 2] * 1.10, 0.0, 1.0)
    dimmed = Image.fromarray((pixels * 255).astype(np.uint8))
    return pixelate(dimmed, width, height, colours)


def round_sprite(image: Image.Image, size: int, dark_subject: bool,
                 trim: float = 0.94, colours: int = 16) -> Image.Image:
    """Find the stone against its flat backdrop and give it a round alpha.

    The mask is built on the small grid rather than a big one and then shrunk:
    a supersampled mask would give the sprite a smooth anti-aliased edge, which
    is exactly the thing that would stop it looking 16-bit.
    """
    grey = np.asarray(image.convert("L"), dtype=np.float32)
    subject = grey < 140 if dark_subject else grey > 90
    rows, cols = np.nonzero(subject)
    if len(rows) == 0:
        raise ValueError("no subject found -- is the backdrop the expected colour?")
    top, bottom, left, right = rows.min(), rows.max(), cols.min(), cols.max()
    side = max(bottom - top, right - left) + 1
    cy, cx = (top + bottom) // 2, (left + right) // 2
    box = (cx - side // 2, cy - side // 2, cx - side // 2 + side, cy - side // 2 + side)

    stone = pixelate(image.crop(box), size, size, colours).convert("RGB")
    yy, xx = np.mgrid[0:size, 0:size]
    centre = (size - 1) / 2.0
    radius = np.hypot(yy - centre, xx - centre)
    # A hard cut: a pixel is either in the stone or it is not.
    alpha = (radius <= centre * trim).astype(np.uint8) * 255
    out = Image.new("RGBA", (size, size))
    out.paste(stone, (0, 0))
    out.putalpha(Image.fromarray(alpha))
    return out


def glow_sprite(image: Image.Image, size: int, colours: int = 12) -> Image.Image:
    """Luminance becomes alpha, so the black backdrop becomes transparency."""
    grey = np.asarray(image.convert("L"), dtype=np.float32)
    rows, cols = np.nonzero(grey > 24)
    if len(rows) == 0:
        raise ValueError("the glow is not brighter than its background")
    top, bottom, left, right = rows.min(), rows.max(), cols.min(), cols.max()
    side = max(bottom - top, right - left) + 1
    cy, cx = (top + bottom) // 2, (left + right) // 2
    box = (cx - side // 2, cy - side // 2, cx - side // 2 + side, cy - side // 2 + side)

    crop = pixelate(image.crop(box), size, size, colours).convert("RGB")
    pixels = np.asarray(crop, dtype=np.float32)
    alpha = pixels.max(axis=2)
    alpha *= 255.0 / max(alpha.max(), 1.0)
    # Push the colour to full brightness: the alpha now carries the falloff, and
    # the particle system tints the sprite anyway.
    colour = pixels * (255.0 / np.maximum(pixels.max(axis=2, keepdims=True), 1.0))
    rgba = np.dstack([colour, alpha]).clip(0, 255).astype(np.uint8)
    return Image.fromarray(rgba)


# Sizes are the true pixel-art resolution; the game scales them up by whole
# numbers with a nearest-neighbour filter, so these stay crisp rather than soft.
PROCESS = {
    "backdrop": lambda im: scenic(im, 480, 300, 64),
    # Brighter than the backdrop: nothing sits on it but the logo and a menu,
    # and the dial is the point of the screen.
    "title": lambda im: scenic(im, 480, 300, 64, dim=0.80),
    "board": lambda im: pixelate(im, 128, 128, 16),
    "stone_black": lambda im: round_sprite(im, 32, True, trim=0.92, colours=8),
    "stone_white": lambda im: round_sprite(im, 32, False, trim=0.96, colours=8),
    "spark": lambda im: glow_sprite(im, 32, colours=8),
    "halo": lambda im: glow_sprite(im, 64, colours=8),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true",
                        help="regenerate even when a raw image is already present")
    parser.add_argument("--only", action="append", default=[], choices=sorted(PROCESS),
                        help="limit to these assets (repeatable)")
    args = parser.parse_args(argv)

    names = args.only or sorted(PROCESS)
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    # The font is not here: it is hand-drawn on a pixel grid by
    # `tools/make_love2d_font.py`, which needs neither an API key nor Pillow.
    missing = [n for n in names if n in PROMPTS
               and (args.force or not os.path.exists(os.path.join(RAW_DIR, f"{n}.png")))]
    if missing:
        print(f"generating {len(missing)} image(s) with Grok...")
        with ThreadPoolExecutor(max_workers=3) as pool:
            for path in pool.map(lambda n: generate(n, PROMPTS[n], RAW_DIR), missing):
                print(f"  raw  {os.path.relpath(path, ROOT)}")

    for name in names:
        raw = None
        if name in PROMPTS:
            raw = Image.open(os.path.join(RAW_DIR, f"{name}.png"))
        out = PROCESS[name](raw)
        path = os.path.join(OUT_DIR, f"{name}.png")
        out.save(path, optimize=True)
        print(f"  {out.size[0]}x{out.size[1]}  {os.path.relpath(path, ROOT)}  "
              f"({os.path.getsize(path) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
