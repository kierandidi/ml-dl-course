#!/usr/bin/env python3
"""Create per-day lesson hero images in assets/img/lessons/ from extracted figures."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "assets" / "figures"
HEROES = ROOT / "assets" / "img" / "lessons"
POSTS = ROOT / "lectures" / "_posts"

# Prefer these filename substrings when picking a representative hero image
PREFER = ["page005", "page010", "page008", "page003", "pptx", "page000"]


def pick_hero(day: int) -> Path | None:
    d = FIGURES / f"day{day:02d}"
    if not d.is_dir():
        return None
    pngs = sorted(d.glob("*.png"))
    if not pngs:
        return None
    for key in PREFER:
        for p in pngs:
            if key in p.name.lower():
                return p
    return pngs[len(pngs) // 3]  # middle slide often has a good diagram


def main() -> None:
    HEROES.mkdir(parents=True, exist_ok=True)
    for day in range(1, 11):
        src = pick_hero(day)
        dest = HEROES / f"day{day:02d}.png"
        if src:
            shutil.copy2(src, dest)
            print(f"Hero day{day:02d}: {src.name} -> {dest}")
        else:
            fallback = ROOT / "assets/img/sampling_space.png"
            if fallback.exists():
                shutil.copy2(fallback, dest)
                print(f"Hero day{day:02d}: fallback sampling_space.png")

    # Update lecture post front matter image: field
    for day in range(1, 11):
        hero_url = f"/assets/img/lessons/day{day:02d}.png"
        for post in POSTS.glob("*.md"):
            if f"day{day:02d}" in post.name or (
                day == 1 and "day01" in post.name
            ):
                pass
        # match by date order in filenames
    date_to_day = {
        "2026-08-17": 1, "2026-08-18": 2, "2026-08-19": 3, "2026-08-20": 4,
        "2026-08-21": 5, "2026-08-24": 6, "2026-08-25": 7, "2026-08-26": 8,
        "2026-08-27": 9, "2026-08-28": 10,
    }
    for post in POSTS.glob("*.md"):
        m = re.match(r"(\d{4}-\d{2}-\d{2})", post.name)
        if not m:
            continue
        day = date_to_day.get(m.group(1))
        if not day:
            continue
        text = post.read_text(encoding="utf-8")
        hero = f"/assets/img/lessons/day{day:02d}.png"
        text = re.sub(
            r"^image: .*$",
            f"image: {hero}",
            text,
            count=1,
            flags=re.MULTILINE,
        )
        post.write_text(text, encoding="utf-8")
        print(f"Updated {post.name} -> image: {hero}")


if __name__ == "__main__":
    main()
