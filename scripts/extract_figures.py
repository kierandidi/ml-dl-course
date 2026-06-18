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

try:
    from PIL import Image, ImageChops
except ImportError:
    Image = None
    ImageChops = None

ROOT = Path(__file__).resolve().parents[1]
MATERIALS = ROOT.parent
COURSE_MATERIAL = ROOT / "material"
OUT = ROOT / "assets" / "figures"

# Day 1 figures are *cropped* from the source PDFs (not full-page renders).
# MML book uses margin captions (caption-band crop); the ODE appendix has a
# centred caption (region crop); the integration tutorial has below-figure
# captions (largest-cluster crop). Page indices are 0-based.

# (figure_id, pdf_page, output_name) — cropped via caption band
DAY01_MML_FIGURES = [
    ("2.1", 22, "mml_vectors_types"),
    ("2.3", 26, "mml_linear_system"),
    ("2.8", 55, "mml_linear_mapping"),
    ("2.6", 44, "mml_subspace"),
    ("2.12", 64, "mml_kernel_nullspace"),
    ("3.1", 75, "mml_geometry_mindmap"),
    ("3.2", 76, "mml_triangle_ineq"),
    ("3.6", 82, "mml_angle"),
    ("3.11", 90, "mml_projection"),
    ("5.3", 146, "mml_gradient"),
    ("5.4", 149, "mml_taylor"),
    ("4.6", 118, "mml_eigen"),
    ("4.9", 127, "mml_svd"),
    ("5.8", 165, "mml_forward_pass"),
    ("6.4", 194, "mml_distributions"),
    ("6.7", 202, "mml_gaussian"),
    ("6.11", 212, "mml_conjugate"),
]

# Explicit plot-region overrides (x0, y0, x1, y1 in PDF points) for MML figures
# whose automatic cluster detection is unreliable (background fills, multi-panel
# layouts, or diagrams made of many tiny strokes). Verified visually.
MML_PLOT_BOX = {
    "2.1": (115, 575, 372, 688),   # types of vectors (bottom panels near caption)
    "4.6": (110, 124, 380, 200),   # eigenvalue area interpretation (two small panels)
    "5.3": (160, 124, 300, 262),   # average incline / secant slope
    "5.8": (140, 122, 418, 185),   # forward-pass NN diagram (thin wide row)
    "6.4": (115, 255, 348, 508),   # mean/mode/median over a contour density
}
# (figure_no, pdf_page, output_name) — cropped via largest cluster
DAY01_INTEG_FIGURES = [
    ("8", 18, "integ_unscented"),
    ("6", 13, "integ_samples"),
]

# ---------------------------------------------------------------------------
# Week 2 — Principles of Diffusion Models (caption-anchored crops).
# The book is single-column with centred captions *below* each figure, so we
# grow the figure region upward from the detected caption. Page indices 0-based.
# (figure_id, pdf_page, output_name)
PDM = "principles_of_diffusion_models.pdf"

DAY06_PDM_FIGURES = [
    ("1.1", 22, "pdm_dgm_target"),        # target of deep generative modeling
    ("1.2", 31, "pdm_dgm_zoo"),           # computation graphs of DGMs
    ("2.1", 35, "pdm_vae"),               # VAE
    ("2.3", 45, "pdm_ddpm_overview"),     # DDPM overview
    ("2.4", 46, "pdm_ddpm_forward"),      # DDPM forward process
    ("2.5", 47, "pdm_ddpm_reverse"),      # DDPM reverse process
    ("2.6", 48, "pdm_ddpm_conditioning"), # conditioning trick
    ("2.7", 56, "pdm_denoise_renoise"),   # denoise-then-renoise view
]

DAY07_PDM_FIGURES = [
    ("3.1", 59, "pdm_ebm_training"),      # EBM training
    ("3.2", 61, "pdm_score_field"),       # score vector fields
    ("3.3", 62, "pdm_langevin"),          # Langevin sampling
    ("3.4", 65, "pdm_score_matching"),    # score matching
    ("3.7", 81, "pdm_ncsn"),              # NCSN multi-noise
    ("4.1", 88, "pdm_score_landscape"),   # time-dependent score landscape
    ("4.3", 91, "pdm_forward_1d"),        # 1D forward process
    ("4.5", 99, "pdm_three_dynamics"),    # forward SDE / reverse SDE / PF-ODE
    ("4.6", 106, "pdm_dsm_trick"),        # denoising score matching trick
    ("5.2", 130, "pdm_nf"),               # normalizing flow
    ("5.5", 137, "pdm_cond_transition"),  # conditional transition dist
    ("5.6", 138, "pdm_cond_vs_marginal"), # conditional vs marginal velocity
    ("5.9", 159, "pdm_curved_paths"),     # marginal ODE trajectories curved
    ("5.10", 164, "pdm_reflow"),          # reflow straightening
    ("6.1", 180, "pdm_param_equiv"),      # four parameterizations equivalence
    ("6.2", 190, "pdm_unified"),          # unified variational/SDE/ODE
]

DAY08_PDM_FIGURES = [
    ("4.4", 93, "pdm_reverse_sde"),       # reverse-time stochastic process
    ("4.7", 108, "pdm_score_sde_2d"),     # 2D sampling from Score SDE
    ("8.1", 232, "pdm_guidance"),         # steered diffusion sampling
    ("8.2", 239, "pdm_cfg"),              # classifier-free guidance
    ("9.1", 271, "pdm_ddim_euler"),       # DDIM = Euler discretization
    ("9.2", 283, "pdm_deis"),             # DEIS multistep
    ("9.3", 296, "pdm_heun_logsnr"),      # Heun in log-SNR time
    ("11.1", None, "pdm_flowmap_timeline"),  # filled at runtime by caption search
    ("11.2", None, "pdm_flowmap"),        # flow map illustration
    ("11.3", None, "pdm_flowmap_semigroup"),  # semigroup property
]

# UCL x DeepMind lecture decks → Week 1 days (diagram-rich pages auto-selected)
DAY_UCL_DECKS = {
    3: ["L2 - UCLxDeepMind DL2020.pdf", "L5 - UCLxDeepMind DL2020.pdf"],
    4: ["L3 - UUCLxDeepMind DL2020.pdf", "L4 - UCLxDeepMind DL2020.pdf"],
    5: [
        "L6 - UCLxDeepMind DL2020.pdf",
        "L7 - UCLxDeepMind DL2020.pdf",
        "L8 - UCLxDeepMind DL2020.pdf",
    ],
}


def _graphic_rects(page):
    pr = page.rect
    area = pr.width * pr.height
    out = []
    for d in page.get_drawings():
        r = fitz.Rect(d["rect"]) & pr
        if r.width < 3 or r.height < 3:
            continue
        if r.width > pr.width * 0.85 and r.height < 4:  # horizontal rules
            continue
        if r.width > pr.width * 0.55 and r.height > pr.height * 0.18 and d.get("fill"):
            continue  # large background highlight boxes
        if r.width * r.height > area * 0.5:
            continue
        out.append(r)
    for i in page.get_image_info():
        r = fitz.Rect(i["bbox"]) & pr
        if r.width > 3 and r.height > 3:
            out.append(r)
    return out


def _caption_rect(page, regex):
    import re

    best = None
    for b in page.get_text("dict")["blocks"]:
        if "lines" not in b:
            continue
        s = " ".join(sp["text"] for l in b["lines"] for sp in l["spans"]).strip()
        if re.match(regex, s):
            r = fitz.Rect(b["bbox"])
            if best is None or r.height > best.height:
                best = r
    return best


def _cluster(rects, gap=16):
    boxes = [fitz.Rect(r) for r in rects]
    changed = True
    while changed:
        changed = False
        merged_out = []
        while boxes:
            b = boxes.pop()
            grew = True
            while grew:
                grew = False
                rest = []
                for o in boxes:
                    be = fitz.Rect(b.x0 - gap, b.y0 - gap, b.x1 + gap, b.y1 + gap)
                    if be.intersects(o):
                        b |= o
                        grew = changed = True
                    else:
                        rest.append(o)
                boxes = rest
            merged_out.append(b)
        boxes = merged_out
    return boxes


def _save_crop(page, box, dest, zoom=3):
    box = box & page.rect
    if box.width < 20 or box.height < 20:
        return False
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=box)
    pix.save(dest)
    return True


def _text_line_rects(page, min_width=0.0):
    """All text-line bounding boxes, optionally filtered to 'wide' body lines."""
    out = []
    for b in page.get_text("dict")["blocks"]:
        if "lines" not in b:
            continue
        for l in b["lines"]:
            r = fitz.Rect(l["bbox"])
            if r.width >= min_width:
                out.append(r)
    return out


def _figure_graphic_rects(page):
    """Graphic rects with body-text artefacts removed.

    Drops (a) drawings that sit *inside* a text line (fraction bars, inline
    math glyphs rendered as paths) and (b) large fills/clip rects that span two
    or more *wide* body-text lines (page backgrounds), which otherwise merge the
    whole text column into the figure cluster.
    """
    pr = page.rect
    all_lines = _text_line_rects(page)
    wide_lines = _text_line_rects(page, min_width=140)
    cand = []
    for d in page.get_drawings():
        r = fitz.Rect(d["rect"]) & pr
        if r.width > 1 and r.height > 1:
            cand.append(r)
    for i in page.get_image_info():
        r = fitz.Rect(i["bbox"]) & pr
        if r.width > 1 and r.height > 1:
            cand.append(r)
    out = []
    for r in cand:
        ra = r.get_area()
        if ra <= 0:
            continue
        if any((r & t).is_valid and (r & t).get_area() > 0.55 * ra for t in all_lines):
            continue  # inline glyph / fraction bar inside a text line
        wide_hits = sum(
            1
            for t in wide_lines
            if (r & t).is_valid and (r & t).get_area() > 0.5 * t.get_area()
        )
        if wide_hits >= 2:
            continue  # background / clip rect spanning body text
        out.append(r)
    return out


def _cluster_amax(rects, gap=16):
    """Cluster rects (proximity merge); track the max single-element area."""
    cs = []
    for r in sorted(rects, key=lambda r: (r.y0, r.x0)):
        hit = None
        for c in cs:
            bb = c["bb"]
            if fitz.Rect(bb.x0 - gap, bb.y0 - gap, bb.x1 + gap, bb.y1 + gap).intersects(r):
                hit = c
                break
        if hit:
            hit["bb"] |= r
            hit["amax"] = max(hit["amax"], r.get_area())
        else:
            cs.append({"bb": fitz.Rect(r), "amax": r.get_area()})
    changed = True
    while changed:
        changed = False
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                a, b = cs[i]["bb"], cs[j]["bb"]
                if fitz.Rect(a.x0 - gap, a.y0 - gap, a.x1 + gap, a.y1 + gap).intersects(b):
                    cs[i]["bb"] = a | b
                    cs[i]["amax"] = max(cs[i]["amax"], cs[j]["amax"])
                    cs.pop(j)
                    changed = True
                    break
            if changed:
                break
    return cs


def _render_pil(page, box, scale=3.5):
    box = fitz.Rect(box) & page.rect
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=box)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def _pil_trim(img, pad=10, thresh=8):
    """Crop surrounding whitespace from a rendered PIL image."""
    bg = Image.new("RGB", img.size, (255, 255, 255))
    diff = ImageChops.difference(img, bg).convert("L").point(
        lambda p: 255 if p > thresh else 0
    )
    bb = diff.getbbox()
    if not bb:
        return img
    l, t, r, b = bb
    return img.crop(
        (max(0, l - pad), max(0, t - pad), min(img.width, r + pad), min(img.height, b + pad))
    )


def crop_mml_figure_image(page, fig_id):
    """Return a clean PIL image of an MML figure (plot + margin caption).

    The MML book uses margin captions placed beside the figure. We render the
    figure (plot) region and the caption block *separately* — auto-trimming
    whitespace on each — then compose them side-by-side. This guarantees the
    full figure is visible and avoids sweeping in adjacent body text.
    """
    import re

    if Image is None:
        return None
    cap = _caption_rect(page, rf"Figure {re.escape(fig_id)}\b")
    if cap is None:
        return None

    box = MML_PLOT_BOX.get(fig_id)
    if box is not None:
        plot_box = fitz.Rect(box)
    else:
        clusters = _cluster_amax(_figure_graphic_rects(page), gap=16)
        if not clusters:
            return None

        def score(c):
            bb = c["bb"]
            v_overlap = max(0, min(bb.y1, cap.y1) - max(bb.y0, cap.y0))
            return (v_overlap > 0, c["amax"], bb.get_area())

        plot_box = max(clusters, key=score)["bb"]

    plot_img = _pil_trim(_render_pil(page, plot_box))
    cap_img = _pil_trim(_render_pil(page, cap))
    cap_left = cap.x0 < plot_box.x0
    gap = 44
    h = max(plot_img.height, cap_img.height)
    out = Image.new("RGB", (plot_img.width + gap + cap_img.width, h), (255, 255, 255))
    if cap_left:
        out.paste(cap_img, (0, 0))
        out.paste(plot_img, (cap_img.width + gap, 0))
    else:
        out.paste(plot_img, (0, 0))
        out.paste(cap_img, (plot_img.width + gap, 0))
    return out


def crop_cluster_figure(page, regex):
    """Crop the largest graphic cluster, attaching an adjacent caption."""
    grs = _graphic_rects(page)
    clusters = [c for c in _cluster(grs) if c.width > 55 and c.height > 38]
    if not clusters:
        return None
    clusters.sort(key=lambda c: -c.width * c.height)
    fig = clusters[0]
    cap = _caption_rect(page, regex)
    if cap is not None:
        gx = max(0, max(fig.x0, cap.x0) - min(fig.x1, cap.x1))
        gy = max(0, max(fig.y0, cap.y0) - min(fig.y1, cap.y1))
        if gx < 70 and gy < 70:
            fig |= cap
    box = fitz.Rect(fig)
    box.x0 -= 8
    box.y0 -= 8
    box.x1 += 8
    box.y1 += 8
    return box


def _norm_ws(s):
    import re

    return re.sub(r"\s+", " ", s).strip()


def _book_caption_rect(page, fig_id):
    """Caption block 'Figure X.Y:' (tolerant to the book's irregular spacing)."""
    import re

    for b in page.get_text("dict")["blocks"]:
        if "lines" not in b:
            continue
        s = _norm_ws(" ".join(sp["text"] for l in b["lines"] for sp in l["spans"]))
        if re.search(rf"Figure {re.escape(fig_id)}[:.]", s):
            return fitz.Rect(b["bbox"])
    return None


def crop_book_figure(page, fig_id, top_min=58):
    """Crop a single-column textbook figure whose caption sits *below* it.

    Strategy: locate the caption, then grow the figure bounding box upward
    through contiguous graphic elements (drawings + images) until a vertical
    gap separates the figure from the body text above it.
    """
    pr = page.rect
    cap = _book_caption_rect(page, fig_id)
    if cap is None:
        return None
    grs = [r for r in _graphic_rects(page) if r.y1 <= cap.y0 + 4 and r.y0 >= top_min - 2]
    if not grs:
        box = fitz.Rect(cap)
    else:
        grs.sort(key=lambda r: -r.y1)  # bottom-up
        box = fitz.Rect(grs[0])
        cur_top = box.y0
        for r in grs[1:]:
            if cur_top - r.y1 < 45 or r.y1 > cur_top:  # contiguous or overlapping
                box |= r
                cur_top = min(cur_top, r.y0)
        box |= cap
    box.x0 = min(box.x0, cap.x0)
    box.x1 = max(box.x1, cap.x1)
    box.y0 = max(box.y0, top_min)
    box.y1 = min(box.y1, cap.y1 + 5)  # trim the faint "Source: ..." line
    box.x0 -= 8
    box.y0 -= 8
    box.x1 += 8
    box.y1 += 8
    return box & pr


def _page_diagram_score(page):
    """Heuristic 'how diagram-rich is this slide' score.

    Rewards a large *single* figure (image or drawing cluster) and many
    medium-sized vector elements; penalizes text-heavy / title / author slides
    and tiny corner logos.
    """
    pr = page.rect
    area = pr.width * pr.height
    # Largest single embedded image (ignore small corner logos).
    big_img = 0.0
    for i in page.get_image_info():
        r = fitz.Rect(i["bbox"]) & pr
        frac = (r.width * r.height) / area
        if frac > 0.06:  # ignore logos / icons
            big_img = max(big_img, frac)
    # Medium-sized vector drawings (diagram strokes/boxes), and their footprint.
    draw_area = 0.0
    n_draw = 0
    for d in page.get_drawings():
        r = fitz.Rect(d["rect"]) & pr
        if r.width > 15 and r.height > 15 and r.width * r.height < area * 0.7:
            draw_area += r.width * r.height
            n_draw += 1
    words = len(page.get_text().split())
    score = big_img + min(draw_area / area, 0.5) + min(n_draw, 30) / 120.0
    # Penalize text/title slides (lots of words, little graphic content).
    if words > 55 and big_img < 0.25:
        score *= 0.35
    if words < 4 and big_img < 0.1:  # near-empty / section divider
        score *= 0.3
    return score


def select_diagram_pages(doc, max_n=7, skip_first=10):
    """Pick the most diagram-rich slide pages, in reading order."""
    scored = []
    for p in range(skip_first, len(doc)):
        sc = _page_diagram_score(doc[p])
        if sc > 0.28:
            scored.append((sc, p))
    scored.sort(key=lambda t: -t[0])
    chosen = sorted(p for _, p in scored[:max_n])
    return chosen


def crop_ode_vectorfield(page):
    """Crop the ODE vector-field figure (A.1): graphics in the lower half + caption."""
    rects = [fitz.Rect(d["rect"]) for d in page.get_drawings()]
    rects = [
        r
        for r in rects
        if r.width > 2 and r.height > 2 and not (r.width > 400 and r.height > 400)
    ]
    low = [r for r in rects if r.y0 > 300]
    if not low:
        return None
    box = fitz.Rect(low[0])
    for r in low[1:]:
        box |= r
    box &= page.rect
    box.x0 -= 8
    box.y0 -= 8
    box.x1 += 8
    box.y1 += 8
    return box


def extract_day01_materials() -> None:
    """Crop curated Day 1 figures from the course material PDFs."""
    if fitz is None:
        print("pymupdf not installed, skipping Day 1 figure crops")
        return
    day_dir = OUT / "day01"
    day_dir.mkdir(parents=True, exist_ok=True)

    mml = COURSE_MATERIAL / "mml-book.pdf"
    if mml.exists():
        doc = fitz.open(mml)
        ok = 0
        for fig_id, pno, name in DAY01_MML_FIGURES:
            img = crop_mml_figure_image(doc[pno], fig_id)
            if img is not None and img.width > 20 and img.height > 20:
                img.save(str(day_dir / f"{name}.png"))
                ok += 1
            else:
                print(f"  ! MML Fig {fig_id} (p{pno}) crop failed")
        doc.close()
        print(f"day01: {ok}/{len(DAY01_MML_FIGURES)} MML figures cropped")

    integ = COURSE_MATERIAL / "integration-methods.pdf"
    if integ.exists():
        doc = fitz.open(integ)
        ok = 0
        for fig_no, pno, name in DAY01_INTEG_FIGURES:
            box = crop_cluster_figure(doc[pno], rf"Figure {fig_no}\b")
            if box and _save_crop(doc[pno], box, day_dir / f"{name}.png"):
                ok += 1
            else:
                print(f"  ! Integration Fig {fig_no} (p{pno}) crop failed")
        doc.close()
        print(f"day01: {ok}/{len(DAY01_INTEG_FIGURES)} integration figures cropped")

    ode = COURSE_MATERIAL / "2510.21890v2.pdf"
    if ode.exists():
        doc = fitz.open(ode)
        box = crop_ode_vectorfield(doc[404])
        if box and _save_crop(doc[404], box, day_dir / "ode_vectorfield.png"):
            print("day01: ODE vector-field figure (A.1) cropped")
        else:
            print("  ! ODE Fig A.1 crop failed")
        doc.close()


def _find_figure_page(doc, fig_id, hint=None):
    """Locate the page index whose caption is 'Figure X.Y:' (search if needed)."""
    import re

    if hint is not None and hint < len(doc):
        if _book_caption_rect(doc[hint], fig_id) is not None:
            return hint
    pat = re.compile(rf"Figure {re.escape(fig_id)}[:.]")
    for p in range(len(doc)):
        if pat.search(_norm_ws(doc[p].get_text())):
            if _book_caption_rect(doc[p], fig_id) is not None:
                return p
    return None


def extract_principles_figures():
    """Crop curated Principles-of-Diffusion figures for Week-2 days 6–8."""
    src = COURSE_MATERIAL / PDM
    if not src.exists():
        print(f"skip missing: {src}")
        return
    doc = fitz.open(src)
    for day, figs in [(6, DAY06_PDM_FIGURES), (7, DAY07_PDM_FIGURES), (8, DAY08_PDM_FIGURES)]:
        day_dir = OUT / f"day{day:02d}"
        day_dir.mkdir(parents=True, exist_ok=True)
        ok = 0
        for fig_id, pno, name in figs:
            pidx = _find_figure_page(doc, fig_id, pno)
            if pidx is None:
                print(f"  ! PDM Fig {fig_id}: caption not found")
                continue
            box = crop_book_figure(doc[pidx], fig_id)
            if box and _save_crop(doc[pidx], box, day_dir / f"{name}.png"):
                ok += 1
            else:
                print(f"  ! PDM Fig {fig_id} (p{pidx}) crop failed")
        print(f"day{day:02d}: {ok}/{len(figs)} Principles figures cropped")
    doc.close()


def extract_ucl_decks():
    """Render the most diagram-rich slide pages from UCL x DeepMind decks (days 3–5)."""
    for day, decks in DAY_UCL_DECKS.items():
        day_dir = OUT / f"day{day:02d}"
        day_dir.mkdir(parents=True, exist_ok=True)
        total = 0
        for i, deck in enumerate(decks):
            src = COURSE_MATERIAL / deck
            if not src.exists():
                print(f"skip missing: {src}")
                continue
            doc = fitz.open(src)
            pages = select_diagram_pages(doc, max_n=7, skip_first=8)
            for p in pages:
                pix = doc[p].get_pixmap(matrix=fitz.Matrix(2, 2))
                pix.save(day_dir / f"ucl{i}_p{p:03d}.png")
                total += 1
            doc.close()
        print(f"day{day:02d}: {total} UCL slide pages rendered")


# day -> list of (source_path_relative_to_materials, page_indices_to_render)
def page_range(start: int, stop: int, step: int = 3) -> list[int]:
    return list(range(start, stop, step))


# Days 1 (curated crops), 3–5 (UCL decks) and 6–8 (Principles book) are handled
# by dedicated extractors. Days 9–10 (LLMs) are authored later.
PDF_SOURCES = {
    2: [("Re_ ML and DL Materials (3)/L3_regressions.pptx", None)],
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
    extract_day01_materials()
    if fitz is not None:
        extract_ucl_decks()           # Week 1: days 3–5 (UCL x DeepMind)
        extract_principles_figures()  # Week 2: days 6–8 (Principles of Diffusion)
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

    # Copy a few PPTX folders wholesale for day 2
    for folder, day in [
        ("Re_ ML and DL Materials (4)", 2),
    ]:
        p = MATERIALS / folder
        if not p.exists():
            continue
        for pptx in list(p.glob("*.pptx"))[:2]:
            extract_pptx_images(pptx, OUT / f"day{day:02d}", pptx.stem[:12])

    print("Done. Figures in", OUT)


if __name__ == "__main__":
    main()
