#import "../lib.typ": *

#show: course-theme.with(title: [Diffusion Inference], subtitle: [Day 8 | Aug 2026])

= Day 8: Diffusion Inference

== Welcome

- *Diffusion Inference* — SDEs, probability-flow ODEs, and samplers
- 3 hours lecture + practical
- Slides, notes, and code on the course site

== Outline

- Continuous-Time View
- Samplers
- Practical Inference
- Connections & Outlook

= Continuous-Time View

== Forward SDE

- $d x = f(x,t) d t + g(t) d w$
- Variance-preserving (VP) and VE variants
- Marginal $p_t(x)$ approaches noise

== pdf0 page015

#align(center)[#image("/assets/figures/day08/pdf0_page015.png", width: 92%)]

#text(size: 14pt, fill: gray)[Continuous-Time View — Forward SDE (source: course materials)]

== Reverse SDE

- $d x = [f(x,t) - g(t)^2 nabla_x log p_t(x)] d t + g(t) dif overline(w)$
- Score replaces unknown drift correction
- Anderson's reverse-time SDE theorem

== pdf0 page018

#align(center)[#image("/assets/figures/day08/pdf0_page018.png", width: 92%)]

#text(size: 14pt, fill: gray)[Continuous-Time View — Reverse SDE (source: course materials)]

== Probability Flow ODE

- Same marginals without stochastic term
- $d x = [f(x,t) - (1/2) g(t)^2 nabla_x log p_t(x)] d t$
- Deterministic sampling path

== pdf0 page021

#align(center)[#image("/assets/figures/day08/pdf0_page021.png", width: 92%)]

#text(size: 14pt, fill: gray)[Continuous-Time View — Probability Flow ODE (source: course materials)]

== Discretization

- Euler–Maruyama for SDEs
- DDIM as non-Markovian deterministic integrator
- Step count vs quality tradeoff

== pdf0 page024

#align(center)[#image("/assets/figures/day08/pdf0_page024.png", width: 92%)]

#text(size: 14pt, fill: gray)[Continuous-Time View — Discretization (source: course materials)]

= Samplers

== DDPM Sampling

- Markovian ancestral sampling
- $T$ steps — slow at high resolution
- Stochasticity helps diversity

== pdf0 page025

#align(center)[#image("/assets/figures/day08/pdf0_page025.png", width: 92%)]

#text(size: 14pt, fill: gray)[Samplers — DDPM Sampling (source: course materials)]

== DDIM

- Skip timesteps with adjusted updates
- eta=0 fully deterministic
- 10–50 steps often sufficient

== pdf0 page027

#align(center)[#image("/assets/figures/day08/pdf0_page027.png", width: 92%)]

#text(size: 14pt, fill: gray)[Samplers — DDIM (source: course materials)]

== Higher-Order Solvers

- DPM-Solver, Heun, Runge–Kutta on ODE
- Fewer function evaluations (NFE)
- Active research area

== pdf0 page030

#align(center)[#image("/assets/figures/day08/pdf0_page030.png", width: 92%)]

#text(size: 14pt, fill: gray)[Samplers — Higher-Order Solvers (source: course materials)]

== Guidance at Inference

- Classifier guidance (separate classifier grad)
- CFG combines cond and uncond score
- Large guidance → artifacts

== pdf0 page033

#align(center)[#image("/assets/figures/day08/pdf0_page033.png", width: 92%)]

#text(size: 14pt, fill: gray)[Samplers — Guidance at Inference (source: course materials)]

= Practical Inference

== Scheduler Choice

- Timestep spacing: uniform vs SNR-based
- Respacing pretrained models
- Distillation for 1–4 step models

== pdf0 page035

#align(center)[#image("/assets/figures/day08/pdf0_page035.png", width: 92%)]

#text(size: 14pt, fill: gray)[Practical Inference — Scheduler Choice (source: course materials)]

== Memory & Speed

- Attention at 512² vs 1024²
- VAE decode bottleneck
- Batch size 1 for interactive apps

== pdf0 page036

#align(center)[#image("/assets/figures/day08/pdf0_page036.png", width: 92%)]

#text(size: 14pt, fill: gray)[Practical Inference — Memory & Speed (source: course materials)]

== Editing & Control

- Img2img: partial noise then denoise
- Inpainting with masked regions
- ControlNet auxiliary conditioning

== pdf0 page039

#align(center)[#image("/assets/figures/day08/pdf0_page039.png", width: 92%)]

#text(size: 14pt, fill: gray)[Practical Inference — Editing & Control (source: course materials)]

== Failure Modes

- Mode averaging at low steps
- Text neglect with weak guidance
- Watermark and safety filters

== pdf0 page042

#align(center)[#image("/assets/figures/day08/pdf0_page042.png", width: 92%)]

#text(size: 14pt, fill: gray)[Practical Inference — Failure Modes (source: course materials)]

= Connections & Outlook

== Flow Matching

- Learn vector field transporting noise → data
- Rectified flows and straight paths
- Unified view with diffusion/flows

== Consistency Models

- Single-step or few-step generation
- Distill iterative sampler

== Bridge to Autoregressive

- Different inductive bias: parallel vs sequential
- Multimodal systems combine both
- Days 9–10: language side

== Summary

- Day 8: *Diffusion Inference*
- SDEs, probability-flow ODEs, and samplers
- Questions welcome — practical follows

== Questions?

#align(center + horizon)[
  #text(size: 44pt, weight: "bold", fill: course-primary)[Questions?]

  #v(1em)
  Practical session follows — see course site.
]
