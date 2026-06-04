#!/usr/bin/env python3
"""Extract illustrations from PDFs and PPTX into assets/figures/dayNN/."""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

try:
    import fitz  # pymupdf
except ImportError:
    fitz = None

ROOT = Path(__file__).resolve().parents[1]
MATERIALS = ROOT.parent
OUT = ROOT / "assets" / "figures"

# day -> list of (source_path_relative_to_materials, page_indices_to_render)
def page_range(start: int, stop: int, step: int = 3) -> list[int]:
    return list(range(start, stop, step))


PDF_SOURCES = {
    1: [("Re_ ML and DL Materials (3)/L1_introduction.pptx", None)],
    2: [("Re_ ML and DL Materials (3)/L3_regressions.pptx", None)],
    3: [("L2 - UCLxDeepMind DL2020.pdf", page_range(0, 30, 2))],
    4: [
        ("L3 - UUCLxDeepMind DL2020.pdf", page_range(0, 40, 3)),
        ("L4 - UCLxDeepMind DL2020.pdf", page_range(0, 35, 3)),
    ],
    5: [
        ("L6 - UCLxDeepMind DL2020.pdf", page_range(0, 35, 3)),
        ("L7 - UCLxDeepMind DL2020.pdf", page_range(0, 40, 4)),
        ("L8 - UCLxDeepMind DL2020.pdf", page_range(0, 30, 3)),
    ],
    6: [
        ("20260120_Lecture_01.pdf", page_range(0, 25, 2)),
        ("principles_of_diffusion_models.pdf", page_range(0, 50, 4)),
    ],
    7: [
        ("20260122_Lecture_02.pdf", page_range(0, 20, 2)),
        ("20260123_Lecture_03.pdf", page_range(0, 30, 3)),
    ],
    8: [
        ("20260123_Lecture_03.pdf", page_range(15, 45, 3)),
        ("mit_course_notes.pdf", page_range(0, 80, 5)),
    ],
    9: [
        ("L7 - UCLxDeepMind DL2020.pdf", page_range(0, 45, 4)),
        ("L8 - UCLxDeepMind DL2020.pdf", page_range(0, 40, 4)),
    ],
    10: [
        ("L11 - UCLxDeepMind DL2020.pdf", page_range(0, 20, 2)),
        ("20260128_Lecture_04_edited.pdf", page_range(0, 25, 2)),
    ],
}


def extract_pptx_images(pptx_path: Path, out_dir: Path, prefix: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(pptx_path, "r") as zf:
        for name in zf.namelist():
            if name.startswith("ppt/media/") and not name.endswith("/"):
                ext = Path(name).suffix.lower()
                if ext in {".png", ".jpg", ".jpeg", ".gif", ".emf", ".wmf"}:
                    data = zf.read(name)
                    dest = out_dir / f"{prefix}_{count:02d}{ext if ext != '.emf' else '.png'}"
                    if ext in {".emf", ".wmf"}:
                        continue
                    dest.write_bytes(data)
                    count += 1
                    if count >= 12:
                        break
    return count


def render_pdf_pages(pdf_path: Path, pages: list[int], out_dir: Path, prefix: str) -> int:
    if fitz is None:
        print("pymupdf not installed, skipping PDF render")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    count = 0
    for p in pages:
        if p >= len(doc):
            continue
        pix = doc[p].get_pixmap(matrix=fitz.Matrix(2, 2))
        dest = out_dir / f"{prefix}_page{p:03d}.png"
        pix.save(dest)
        count += 1
    doc.close()
    return count


def main():
    for day, sources in PDF_SOURCES.items():
        day_dir = OUT / f"day{day:02d}"
        day_dir.mkdir(parents=True, exist_ok=True)
        for i, item in enumerate(sources):
            src_name, pages = item
            src = MATERIALS / src_name
            if not src.exists():
                print(f"skip missing: {src}")
                continue
            if src.suffix.lower() == ".pptx":
                n = extract_pptx_images(src, day_dir, f"pptx{i}")
                print(f"day{day:02d}: {n} images from {src.name}")
            elif src.suffix.lower() == ".pdf" and pages:
                n = render_pdf_pages(src, pages, day_dir, f"pdf{i}")
                print(f"day{day:02d}: {n} pages from {src.name}")

    # Copy a few PPTX folders wholesale for days 1-2
    for folder, day in [
        ("Re_ ML and DL Materials (3)", 1),
        ("Re_ ML and DL Materials (4)", 2),
        ("Re_ ML and DL Materials (5)", 3),
        ("Re_ ML and DL Materials (6)", 5),
    ]:
        p = MATERIALS / folder
        if not p.exists():
            continue
        for pptx in list(p.glob("*.pptx"))[:2]:
            extract_pptx_images(pptx, OUT / f"day{day:02d}", pptx.stem[:12])

    print("Done. Figures in", OUT)


if __name__ == "__main__":
    main()
