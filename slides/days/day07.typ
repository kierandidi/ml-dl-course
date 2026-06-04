#import "../lib.typ": *

#show: course-theme.with(title: [Training Flow & Diffusion Models], subtitle: [Day 7 | Aug 2026])

= Day 7: Training Flow & Diffusion Models

== Welcome

- *Training Flow & Diffusion Models* — Continuous transforms and denoising objectives
- 3 hours lecture + practical
- Slides, notes, and code on the course site

== Outline

- Normalizing Flows
- Diffusion Intuition
- Score Matching View
- Training Practice

= Normalizing Flows

== Change of Variables

- $p_X(x) = p_Z(f^(-1)(x)) |det J_(f^(-1))(x)|$
- Bijective $f$ with tractable inverse
- Composition of simple coupling layers

== pdf0 page000

#align(center)[#image("/assets/figures/day07/pdf0_page000.png", width: 92%)]

#text(size: 14pt, fill: gray)[Normalizing Flows — Change of Variables (source: course materials)]

== Coupling Layers

- Split dimensions; transform one part conditioned on other
- Affine coupling: scale and shift
- RealNVP, Glow architectures

== pdf0 page002

#align(center)[#image("/assets/figures/day07/pdf0_page002.png", width: 92%)]

#text(size: 14pt, fill: gray)[Normalizing Flows — Coupling Layers (source: course materials)]

== Training Flows

- Maximize log-likelihood directly
- Jacobian log-determinant cost
- Exact sampling and density

== pdf0 page004

#align(center)[#image("/assets/figures/day07/pdf0_page004.png", width: 92%)]

#text(size: 14pt, fill: gray)[Normalizing Flows — Training Flows (source: course materials)]

== Limitations

- Architectural constraints for invertibility
- Scaling to high-res images is hard
- Diffusion trades exact likelihood for flexibility

== pdf0 page006

#align(center)[#image("/assets/figures/day07/pdf0_page006.png", width: 92%)]

#text(size: 14pt, fill: gray)[Normalizing Flows — Limitations (source: course materials)]

= Diffusion Intuition

== Forward Process

- Gradually add Gaussian noise: $q(x_t|x_(t-1))$
- Closed form $q(x_t|x_0) = cal(N)(sqrt(overline(alpha)_t) x_0, (1-overline(alpha)_t) I)$
- Ends at pure noise

== pdf0 page008

#align(center)[#image("/assets/figures/day07/pdf0_page008.png", width: 92%)]

#text(size: 14pt, fill: gray)[Diffusion Intuition — Forward Process (source: course materials)]

== Reverse Process

- Learn to denoise step by step
- $p_theta(x_(t-1)|x_t)$ parameterized by neural net
- Sampling walks from noise to data

== pdf0 page010

#align(center)[#image("/assets/figures/day07/pdf0_page010.png", width: 92%)]

#text(size: 14pt, fill: gray)[Diffusion Intuition — Reverse Process (source: course materials)]

== DDPM Objective

- Predict noise $epsilon$ added at step $t$
- Simple MSE on $epsilon$ with random $t$
- Equivalent variants: predict $x_0$ or score

== pdf0 page012

#align(center)[#image("/assets/figures/day07/pdf0_page012.png", width: 92%)]

#text(size: 14pt, fill: gray)[Diffusion Intuition — DDPM Objective (source: course materials)]

== Noise Schedules

- Linear, cosine $overline(alpha)_t$ schedules
- Affects training stability and sample quality
- Signal-to-noise ratio view

== pdf0 page014

#align(center)[#image("/assets/figures/day07/pdf0_page014.png", width: 92%)]

#text(size: 14pt, fill: gray)[Diffusion Intuition — Noise Schedules (source: course materials)]

= Score Matching View

== Denoising Score Matching

- Learn $s_theta(x_t, t) approx nabla_(x_t) log p(x_t)$
- Tweedie's formula links $epsilon$ and score
- Unifies diffusion training objectives

== pdf0 page016

#align(center)[#image("/assets/figures/day07/pdf0_page016.png", width: 92%)]

#text(size: 14pt, fill: gray)[Score Matching View — Denoising Score Matching (source: course materials)]

== SDE Formulation Preview

- Forward SDE adds noise continuously
- Reverse SDE uses score function
- Day 8 inference details

== pdf0 page018

#align(center)[#image("/assets/figures/day07/pdf0_page018.png", width: 92%)]

#text(size: 14pt, fill: gray)[Score Matching View — SDE Formulation Preview (source: course materials)]

== Classifier-Free Guidance

- Train conditional model with dropped labels
- Guidance scale trades diversity vs fidelity
- Standard in text-to-image systems

== pdf1 page000

#align(center)[#image("/assets/figures/day07/pdf1_page000.png", width: 92%)]

#text(size: 14pt, fill: gray)[Score Matching View — Classifier-Free Guidance (source: course materials)]

== Latent Diffusion

- Diffuse in VAE latent space (Stable Diffusion)
- Lower dimension → cheaper training
- Text encoder provides conditioning

== pdf1 page003

#align(center)[#image("/assets/figures/day07/pdf1_page003.png", width: 92%)]

#text(size: 14pt, fill: gray)[Score Matching View — Latent Diffusion (source: course materials)]

= Training Practice

== Network Architecture

- U-Net with time embedding $t$
- Attention at lower resolutions
- GroupNorm + SiLU activations

== Compute & Data

- Large-scale image-text pairs
- Mixed precision and EMA weights
- Checkpointing long runs

== Monitoring

- Loss per noise level
- Periodic sample grids
- FID on small val set

== Summary

- Day 7: *Training Flow & Diffusion Models*
- Continuous transforms and denoising objectives
- Questions welcome — practical follows

== Questions?

#align(center + horizon)[
  #text(size: 44pt, weight: "bold", fill: course-primary)[Questions?]

  #v(1em)
  Practical session follows — see course site.
]
