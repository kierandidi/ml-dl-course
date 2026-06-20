#!/usr/bin/env python3
"""Build per-day lesson hero images in assets/img/lessons/dayNN.png.

Each hero is a single, concept-illustrating visual hand-picked from the
extracted figures, cropped tightly to the diagram itself (title bands, captions,
margins, and logos removed). The lecture post front matter already points at
``/assets/img/lessons/dayNN.png`` (set by generate_lectures.py), so this script
only needs to produce good-looking cropped PNGs.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "assets" / "figures"
HEROES = ROOT / "assets" / "img" / "lessons"

# day -> (source figure relative to assets/figures, (left, top, right, bottom) crop
# as fractions of width/height applied BEFORE a final whitespace trim).
HERO_SOURCES: dict[int, tuple[str, tuple[float, float, float, float]]] = {
    1:  ("day01/ode_vectorfield.png",      (0.03, 0.06, 0.99, 0.62)),  # ODE velocity field + trajectories
    2:  ("day02/mml_svm_margin.png",       (0.29, 0.00, 1.00, 1.00)),  # SVM margin / hyperplane
    3:  ("day03/dnn_compgraph.png",        (0.00, 0.13, 0.92, 0.86)),  # neural net as a computational graph
    4:  ("day04/cnn_convop.png",           (0.00, 0.14, 0.83, 0.66)),  # the convolution operation
    5:  ("day05/attn_alignment.png",       (0.13, 0.11, 0.88, 0.80)),  # attention alignment matrix
    6:  ("day06/pdm_ddpm_overview.png",    (0.08, 0.05, 0.94, 0.46)),  # DDPM forward/reverse chain
    7:  ("day07/pdm_score_field.png",      (0.02, 0.00, 0.99, 0.86)),  # score vector field
    8:  ("day08/pdm_cfg.png",              (0.00, 0.02, 1.00, 0.72)),  # classifier-free guidance
    9:  ("day09/llmks_attention_dict.png", (0.05, 0.24, 0.97, 0.76)),  # attention as a soft dictionary lookup
    10: ("day10/llmks_kvcache.png",        (0.02, 0.24, 0.98, 0.93)),  # KV cache
}


def trim_whitespace(img: Image.Image, pad: int = 12, tol: int = 8) -> Image.Image:
    """Crop near-white borders, leaving a small uniform padding."""
    rgb = img.convert("RGB")
    bg = Image.new("RGB", rgb.size, (255, 255, 255))
    diff = ImageChops.difference(rgb, bg)
    # Amplify so faint anti-aliased pixels still register, then threshold via tol.
    diff = ImageChops.add(diff, diff, 2.0, -tol)
    bbox = diff.getbbox()
    if not bbox:
        return img
    l, t, r, b = bbox
    l = max(0, l - pad)
    t = max(0, t - pad)
    r = min(img.width, r + pad)
    b = min(img.height, b + pad)
    return img.crop((l, t, r, b))


def frac_crop(img: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    w, h = img.size
    l, t, r, b = box
    return img.crop((int(l * w), int(t * h), int(r * w), int(b * h)))


def pad_to_aspect(img: Image.Image, ratio: float = 16 / 9,
                  margin: float = 0.06, bg=(255, 255, 255)) -> Image.Image:
    """Letterbox to a target aspect ratio so 16:9 cards show the whole visual.

    A small uniform margin is added first so the diagram never touches the card edge.
    """
    w, h = img.size
    m = int(round(min(w, h) * margin))
    w, h = w + 2 * m, h + 2 * m
    if w / h < ratio:           # too tall -> add width
        new_w, new_h = int(round(h * ratio)), h
    else:                       # too wide -> add height
        new_w, new_h = w, int(round(w / ratio))
    canvas = Image.new("RGB", (new_w, new_h), bg)
    canvas.paste(img, ((new_w - img.width) // 2, (new_h - img.height) // 2))
    return canvas


def main() -> None:
    HEROES.mkdir(parents=True, exist_ok=True)
    for day in range(1, 11):
        rel, box = HERO_SOURCES[day]
        src = FIGURES / rel
        if not src.exists():
            print(f"day{day:02d}: MISSING source {src}")
            continue
        img = Image.open(src)
        if img.mode in ("RGBA", "P", "LA"):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img.convert("RGBA"), mask=img.convert("RGBA").split()[-1])
            img = bg
        else:
            img = img.convert("RGB")
        img = frac_crop(img, box)
        img = trim_whitespace(img)
        img = pad_to_aspect(img, 16 / 9)
        dest = HEROES / f"day{day:02d}.png"
        img.save(dest)
        print(f"day{day:02d}: {rel} {img.size} -> {dest.name}")


if __name__ == "__main__":
    main()
