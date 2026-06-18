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

== Forward SDE — illustration

#align(center)[#image("/assets/figures/day08/pdm_cfg.png", width: 80%)]

#text(size: 14pt, fill: gray)[Continuous-Time View — Forward SDE (source: course materials)]

== Reverse SDE

- $d x = [f(x,t) - g(t)^2 nabla_x log p_t(x)] d t + g(t) dif overline(w)$
- Score replaces unknown drift correction
- Anderson's reverse-time SDE theorem

== Reverse SDE — illustration

#align(center)[#image("/assets/figures/day08/pdm_ddim_euler.png", width: 80%)]

#text(size: 14pt, fill: gray)[Continuous-Time View — Reverse SDE (source: course materials)]

== Probability Flow ODE

- Same marginals without stochastic term
- $d x = [f(x,t) - (1/2) g(t)^2 nabla_x log p_t(x)] d t$
- Deterministic sampling path

== Probability Flow ODE — illustration

#align(center)[#image("/assets/figures/day08/pdm_deis.png", width: 80%)]

#text(size: 14pt, fill: gray)[Continuous-Time View — Probability Flow ODE (source: course materials)]

== Discretization

- Euler–Maruyama for SDEs
- DDIM as non-Markovian deterministic integrator
- Step count vs quality tradeoff

== Discretization — illustration

#align(center)[#image("/assets/figures/day08/pdm_flowmap.png", width: 80%)]

#text(size: 14pt, fill: gray)[Continuous-Time View — Discretization (source: course materials)]

= Samplers

== DDPM Sampling

- Markovian ancestral sampling
- $T$ steps — slow at high resolution
- Stochasticity helps diversity

== DDPM Sampling — illustration

#align(center)[#image("/assets/figures/day08/pdm_flowmap_semigroup.png", width: 80%)]

#text(size: 14pt, fill: gray)[Samplers — DDPM Sampling (source: course materials)]

== DDIM

- Skip timesteps with adjusted updates
- eta=0 fully deterministic
- 10–50 steps often sufficient

== DDIM — illustration

#align(center)[#image("/assets/figures/day08/pdm_flowmap_timeline.png", width: 80%)]

#text(size: 14pt, fill: gray)[Samplers — DDIM (source: course materials)]

== Higher-Order Solvers

- DPM-Solver, Heun, Runge–Kutta on ODE
- Fewer function evaluations (NFE)
- Active research area

== Higher-Order Solvers — illustration

#align(center)[#image("/assets/figures/day08/pdm_guidance.png", width: 80%)]

#text(size: 14pt, fill: gray)[Samplers — Higher-Order Solvers (source: course materials)]

== Guidance at Inference

- Classifier guidance (separate classifier grad)
- CFG combines cond and uncond score
- Large guidance → artifacts

== Guidance at Inference — illustration

#align(center)[#image("/assets/figures/day08/pdm_heun_logsnr.png", width: 80%)]

#text(size: 14pt, fill: gray)[Samplers — Guidance at Inference (source: course materials)]

= Practical Inference

== Scheduler Choice

- Timestep spacing: uniform vs SNR-based
- Respacing pretrained models
- Distillation for 1–4 step models

== Scheduler Choice — illustration

#align(center)[#image("/assets/figures/day08/pdm_reverse_sde.png", width: 80%)]

#text(size: 14pt, fill: gray)[Practical Inference — Scheduler Choice (source: course materials)]

== Memory & Speed

- Attention at 512² vs 1024²
- VAE decode bottleneck
- Batch size 1 for interactive apps

== Memory & Speed — illustration

#align(center)[#image("/assets/figures/day08/pdm_score_sde_2d.png", width: 80%)]

#text(size: 14pt, fill: gray)[Practical Inference — Memory & Speed (source: course materials)]

== Editing & Control

- Img2img: partial noise then denoise
- Inpainting with masked regions
- ControlNet auxiliary conditioning

== Failure Modes

- Mode averaging at low steps
- Text neglect with weak guidance
- Watermark and safety filters

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
