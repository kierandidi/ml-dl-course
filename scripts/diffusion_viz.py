"""Registry of the 14 interactive visualizations from *The Principles of
Diffusion Models* (https://the-principles-of-diffusion-models.github.io/).

Each widget is a standalone HTML page hosted under ``/assets/`` on the book
site, so we can embed it live with an ``<iframe>`` inside the Jekyll lecture
notes and link to it (with a short URL) from the PDF slide decks.

Usage:
    from diffusion_viz import VIZ, viz_iframe, viz_link
    html = viz_iframe("noise_schedule_explorer")     # for lecture notes
    md   = viz_link("noise_schedule_explorer")       # for slides / markdown
"""
from __future__ import annotations

BASE = "https://the-principles-of-diffusion-models.github.io/assets"

# key -> (title, html_file, iframe_height_px, day)
VIZ: dict[str, tuple[str, str, int, int]] = {
    # Day 6 — forward process, DDPM, change of variables
    "noise_schedule_explorer": ("Noise Schedule Explorer", "noise_schedule_explorer.html", 820, 6),
    "ddpm_conditional_trick": ("DDPM Conditional Trick", "ddpm_conditional_trick.html", 580, 6),
    "cov_2d_map": ("Change-of-Variable 2D Map", "cov_2d_map.html", 640, 6),
    # Day 7 — score / SDE / flow matching
    "score_landscape": ("Score Landscape", "score_landscape.html", 580, 7),
    "score_global_vs_local": ("Global vs Local View of the Score", "score_global_vs_local.html", 580, 7),
    "denoising_score_matching": ("Denoising Score Matching", "denoising_score_matching.html", 820, 7),
    "score_sde_three_dynamics": ("Three Dynamics: Forward SDE, Reverse SDE, PF-ODE", "score_sde_three_dynamics.html", 580, 7),
    "ddpm_prediction_equiv": ("DDPM Prediction Equivalences", "ddpm_prediction_equiv.html", 580, 7),
    "four_predictions": ("Four Prediction Parameterizations", "four_predictions.html", 580, 7),
    "conditional_vs_marginal_velocity": (
        "Conditional vs Marginal Velocity Fields",
        "conditional_vs_marginal_flow_matching_velocity.html",
        820,
        7,
    ),
    "conditional_vs_marginal_paths": ("Conditional vs Marginal Paths", "conditional_vs_marginal_paths.html", 620, 7),
    # Day 8 — solvers, Fokker–Planck, flow maps
    "euler_vs_heun_solver": ("Euler vs Heun Solver", "euler_vs_heun_solver.html", 560, 8),
    "fokker_planck": ("Fokker–Planck Equation", "fokker_planck.html", 640, 8),
    "flow_map_models": ("Flow Map Models Comparison", "flow_map_models.html", 640, 8),
}


def viz_url(key: str) -> str:
    return f"{BASE}/{VIZ[key][1]}"


def viz_iframe(key: str) -> str:
    """Return a Jekyll-friendly HTML block embedding the live widget."""
    title, _file, height, _day = VIZ[key]
    url = viz_url(key)
    return (
        f'<div class="interactive-viz" markdown="0">\n'
        f'  <iframe src="{url}" width="100%" height="{height}" loading="lazy" '
        f'style="border:1px solid #ddd;border-radius:8px;" title="{title}"></iframe>\n'
        f'  <p style="font-size:0.85em;color:#888;margin-top:4px;">'
        f'Interactive: <strong>{title}</strong> — '
        f'<a href="{url}" target="_blank" rel="noopener">open in new tab</a>. '
        f'Source: <em>The Principles of Diffusion Models</em>.</p>\n'
        f"</div>"
    )


def viz_link(key: str) -> str:
    """Return a short markdown link (for slides / inline references)."""
    title = VIZ[key][0]
    return f"[Interactive — {title}]({viz_url(key)})"


def viz_for_day(day: int) -> list[str]:
    return [k for k, v in VIZ.items() if v[3] == day]
