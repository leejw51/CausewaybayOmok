#!/usr/bin/env python3
"""Generate the GUI's 2D art with Grok, then cut it into game-ready sprites.

    export XAI_API_KEY=...
    python3 tools/make_assets.py            # reuse any raw images already there
    python3 tools/make_assets.py --force    # ask Grok for fresh ones

Raw generations land in ``assets_raw/`` (git-ignored); the processed sprites
land in ``omok/assets/`` and are committed, so nobody needs an API key to run
the game.  The prompts live here rather than in a chat log: that is the only
way the art stays reproducible.

The processing matters as much as the prompt.  An image model returns an opaque
rectangle, and a game needs a sprite with an alpha channel, so:

* the stones are found by contrast against their flat backdrop, cropped square
  around that bounding box, and given an anti-aliased circular alpha;
* the spark becomes alpha-from-luminance, which is exactly what a glow wants --
  its black background turns into transparency instead of a dark square.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

MODEL = "grok-imagine-image"
ENDPOINT = "https://api.x.ai/v1/images/generations"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW_DIR = os.path.join(ROOT, "assets_raw")
OUT_DIR = os.path.join(ROOT, "omok", "assets")

PROMPTS = {
    "board": "Seamless top-down texture of a fine kaya wood go board surface. Warm honey-golden "
             "wood, straight vertical grain lines, matte finish, perfectly even soft studio "
             "lighting, no shadows, no objects, no drawn lines, no text. Flat photographic "
             "texture filling the entire frame.",
    "panel": "Seamless top-down texture of dark charcoal slate stone, near-black with very subtle "
             "cool grey mineral veining, matte finish, perfectly even soft lighting, no shadows, "
             "no objects, no text, flat photographic texture filling the entire frame.",
    "stone_black": "Flat lay photograph of one polished black slate Go stone, camera exactly 90 "
                   "degrees straight above the stone so its outline is a PERFECT CIRCLE, not an "
                   "oval, not tilted. The round glossy jet black disc is centered and fills the "
                   "middle of the frame, one soft crescent specular highlight in the upper left, "
                   "plain pure white background, no shadow, no text.",
    "stone_white": "Flat lay photograph of one white clamshell Go stone, camera exactly 90 degrees "
                   "straight above the stone so its outline is a PERFECT CIRCLE, not an oval, not "
                   "tilted. The round creamy off-white disc is centered and fills the middle of "
                   "the frame, faint shell grain, soft specular highlight in the upper left, "
                   "plain pure black background, no shadow, no text.",
    "spark": "A perfectly circular radial glow of light centered on a pure black background. "
             "Blazing white-hot core fading evenly outward in all directions through warm golden "
             "amber to pure black. Perfectly symmetrical circle, equal width and height, like a "
             "round firefly glow or lens bloom. No oval, no streaks, no other objects, no text.",
}


# ------------------------------------------------------------------ generate
def generate(name: str, prompt: str, raw_dir: str = RAW_DIR) -> str:
    key = os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")
    if not key:
        raise SystemExit("set XAI_API_KEY (or GROK_API_KEY) to generate art")
    body = json.dumps({"model": MODEL, "prompt": prompt, "n": 1,
                       "response_format": "b64_json"}).encode()
    request = urllib.request.Request(
        ENDPOINT, data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        payload = json.load(response)
    path = os.path.join(raw_dir, f"{name}.png")
    with open(path, "wb") as fh:
        fh.write(base64.b64decode(payload["data"][0]["b64_json"]))
    return path


# ------------------------------------------------------------------- process
def square_texture(image: Image.Image, size: int) -> Image.Image:
    """Centre-crop to a square and resample -- for the tiling backgrounds."""
    side = min(image.size)
    left = (image.width - side) // 2
    top = (image.height - side) // 2
    return image.crop((left, top, left + side, top + side)).resize(
        (size, size), Image.LANCZOS)


def circular_sprite(image: Image.Image, size: int, dark_subject: bool,
                    trim: float = 0.94) -> Image.Image:
    """Find the stone against its flat backdrop and give it a round alpha.

    ``trim`` shrinks the circle relative to the detected bounding box.  The
    model renders a soft edge, so a mask that reaches the full bounding box
    keeps a halo of the backdrop -- very visible as a white ring around the
    black stone.
    """
    grey = np.asarray(image.convert("L"), dtype=np.float32)
    subject = grey < 140 if dark_subject else grey > 90
    rows, cols = np.nonzero(subject)
    if len(rows) == 0:
        raise ValueError("no subject found -- is the backdrop the expected colour?")
    top, bottom, left, right = rows.min(), rows.max(), cols.min(), cols.max()
    # A square centred on the stone, so the circular mask lines up with it.
    side = max(bottom - top, right - left) + 1
    cy, cx = (top + bottom) // 2, (left + right) // 2
    box = (cx - side // 2, cy - side // 2, cx - side // 2 + side, cy - side // 2 + side)
    # Supersample so the cut edge is smooth rather than stair-stepped.
    scale = 4
    stone = image.convert("RGB").crop(box).resize((size * scale, size * scale), Image.LANCZOS)
    yy, xx = np.mgrid[0:size * scale, 0:size * scale]
    centre = (size * scale - 1) / 2.0
    radius = np.hypot(yy - centre, xx - centre)
    mask = np.clip((centre * trim - radius) * 1.5, 0.0, 1.0)
    out = Image.new("RGBA", (size * scale, size * scale))
    out.paste(stone, (0, 0))
    out.putalpha(Image.fromarray((mask * 255).astype(np.uint8)))
    return out.resize((size, size), Image.LANCZOS)


def glow_sprite(image: Image.Image, size: int) -> Image.Image:
    """Luminance becomes alpha, so the black backdrop becomes transparency."""
    grey = np.asarray(image.convert("L"), dtype=np.float32)
    rows, cols = np.nonzero(grey > 24)
    if len(rows) == 0:
        raise ValueError("the glow is not brighter than its background")
    top, bottom, left, right = rows.min(), rows.max(), cols.min(), cols.max()
    side = max(bottom - top, right - left) + 1
    cy, cx = (top + bottom) // 2, (left + right) // 2
    box = (cx - side // 2, cy - side // 2, cx - side // 2 + side, cy - side // 2 + side)
    crop = image.convert("RGB").crop(box).resize((size, size), Image.LANCZOS)
    pixels = np.asarray(crop, dtype=np.float32)
    alpha = pixels.max(axis=2)
    alpha *= 255.0 / max(alpha.max(), 1.0)
    # Push the colour to full brightness; the alpha now carries the falloff, and
    # a particle system tints the sprite anyway.
    colour = pixels * (255.0 / np.maximum(pixels.max(axis=2, keepdims=True), 1.0))
    rgba = np.dstack([colour, alpha]).clip(0, 255).astype(np.uint8)
    return Image.fromarray(rgba)


# The two backgrounds carry no transparency, so they ship as JPEG -- as PNG the
# wood grain alone costs more than every other asset put together.
PROCESS = {
    "board": (lambda im: square_texture(im, 512), "jpg"),
    "panel": (lambda im: square_texture(im, 512), "jpg"),
    "stone_black": (lambda im: circular_sprite(im, 256, True, trim=0.92), "png"),
    "stone_white": (lambda im: circular_sprite(im, 256, False, trim=0.96), "png"),
    "spark": (lambda im: glow_sprite(im, 128), "png"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true",
                        help="regenerate even when a raw image is already present")
    parser.add_argument("--only", action="append", default=[], choices=sorted(PROMPTS),
                        help="limit to these assets (repeatable)")
    args = parser.parse_args(argv)

    names = args.only or sorted(PROMPTS)
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    missing = [n for n in names
               if args.force or not os.path.exists(os.path.join(RAW_DIR, f"{n}.png"))]
    if missing:
        print(f"generating {len(missing)} image(s) with {MODEL}...")
        with ThreadPoolExecutor(max_workers=4) as pool:
            for path in pool.map(lambda n: generate(n, PROMPTS[n]), missing):
                print(f"  raw  {os.path.relpath(path, ROOT)}")

    for name in names:
        raw = Image.open(os.path.join(RAW_DIR, f"{name}.png"))
        process, extension = PROCESS[name]
        out = process(raw)
        path = os.path.join(OUT_DIR, f"{name}.{extension}")
        if extension == "jpg":
            out.convert("RGB").save(path, quality=88, optimize=True)
        else:
            out.save(path, optimize=True)
        print(f"  {out.size[0]:>4}px  {os.path.relpath(path, ROOT)}  "
              f"({os.path.getsize(path) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
