#import "../lib.typ": *

#show: course-theme.with(title: [Generative Modeling], subtitle: [Day 6 | Aug 2026])

= Day 6: Generative Modeling

== Welcome

- *Generative Modeling* — Likelihoods, KL, ELBO, model families
- 3 hours lecture + practical
- Slides, notes, and code on the course site

== Outline

- What Is Generative Modeling?
- Likelihood & Divergences
- VAEs & GANs (Context)
- Building Blocks for Week 2

= What Is Generative Modeling?

== Discriminative vs Generative

- Discriminative: model $p(y|x)$
- Generative: model $p(x)$ or $p(x|z)$
- Sample generation, density estimation, editing

== pdf0 page000

#align(center)[#image("/assets/figures/day06/pdf0_page000.png", width: 92%)]

#text(size: 14pt, fill: gray)[What Is Generative Modeling? — Discriminative vs Generative (source: course materials)]

== Explicit vs Implicit

- Explicit likelihood: VAEs, flows, autoregressive
- Implicit: GANs (no tractable $p(x)$)
- Tradeoffs: training stability, evaluation

== pdf0 page002

#align(center)[#image("/assets/figures/day06/pdf0_page002.png", width: 92%)]

#text(size: 14pt, fill: gray)[What Is Generative Modeling? — Explicit vs Implicit (source: course materials)]

== Latent Variable Models

- $p(x) = integral p(x|z) p(z) d z$
- Latent $z$ captures factors of variation
- Posterior $p(z|x)$ usually intractable

== pdf0 page004

#align(center)[#image("/assets/figures/day06/pdf0_page004.png", width: 92%)]

#text(size: 14pt, fill: gray)[What Is Generative Modeling? — Latent Variable Models (source: course materials)]

== Model Families Roadmap

- Week 2: flows, diffusion, autoregressive LLMs
- Each defines different tractability / quality

== pdf0 page005

#align(center)[#image("/assets/figures/day06/pdf0_page005.png", width: 92%)]

#text(size: 14pt, fill: gray)[What Is Generative Modeling? — Model Families Roadmap (source: course materials)]

= Likelihood & Divergences

== Maximum Likelihood

- $theta^* = "arg max"_theta sum_i log p_theta(x_i)$
- Equivalent to minimizing cross-entropy
- MLE is consistent under mild conditions

== pdf0 page006

#align(center)[#image("/assets/figures/day06/pdf0_page006.png", width: 92%)]

#text(size: 14pt, fill: gray)[Likelihood & Divergences — Maximum Likelihood (source: course materials)]

== KL Divergence

- $D_"KL"(q || p) = EE_q[log q - log p]$
- Non-negative; zero iff $q = p$
- Not symmetric — direction matters

== pdf0 page008

#align(center)[#image("/assets/figures/day06/pdf0_page008.png", width: 92%)]

#text(size: 14pt, fill: gray)[Likelihood & Divergences — KL Divergence (source: course materials)]

== Forward vs Reverse KL

- Forward KL: mode-covering (mean-field)
- Reverse KL: mode-seeking
- VAE uses reverse KL to approximate posterior

== pdf0 page010

#align(center)[#image("/assets/figures/day06/pdf0_page010.png", width: 92%)]

#text(size: 14pt, fill: gray)[Likelihood & Divergences — Forward vs Reverse KL (source: course materials)]

== Evidence Lower Bound

- $log p(x) >= EE_(q(z|x))[log p(x|z)] - D_"KL"(q(z|x) || p(z))$
- ELBO tight when $q = p(z|x)$
- Reparameterization trick for gradients

== pdf0 page012

#align(center)[#image("/assets/figures/day06/pdf0_page012.png", width: 92%)]

#text(size: 14pt, fill: gray)[Likelihood & Divergences — Evidence Lower Bound (source: course materials)]

= VAEs & GANs (Context)

== Variational Autoencoder

- Encoder $q_phi(z|x)$, decoder $p_theta(x|z)$
- Train by maximizing ELBO
- Blurry samples — Gaussian assumption

== pdf0 page014

#align(center)[#image("/assets/figures/day06/pdf0_page014.png", width: 92%)]

#text(size: 14pt, fill: gray)[VAEs & GANs (Context) — Variational Autoencoder (source: course materials)]

== GAN Objective

- Min-max game: generator vs discriminator
- No explicit likelihood
- Mode collapse and training instability

== pdf0 page015

#align(center)[#image("/assets/figures/day06/pdf0_page015.png", width: 92%)]

#text(size: 14pt, fill: gray)[VAEs & GANs (Context) — GAN Objective (source: course materials)]

== Evaluation Challenges

- FID, IS for images; human eval for text
- Likelihood can misalign with sample quality
- Coverage vs fidelity

== pdf0 page016

#align(center)[#image("/assets/figures/day06/pdf0_page016.png", width: 92%)]

#text(size: 14pt, fill: gray)[VAEs & GANs (Context) — Evaluation Challenges (source: course materials)]

== Modern Landscape

- Diffusion dominates image generation
- Autoregressive dominates language
- Hybrid and unified models emerging

== pdf0 page018

#align(center)[#image("/assets/figures/day06/pdf0_page018.png", width: 92%)]

#text(size: 14pt, fill: gray)[VAEs & GANs (Context) — Modern Landscape (source: course materials)]

= Building Blocks for Week 2

== Score Functions

- Score: $nabla_x log p(x)$
- Score matching and denoising connections
- Foundation for diffusion (Days 7–8)

== Normalizing Flows Preview

- Invertible maps with tractable Jacobian
- $log p(x) = log p(z) + log |det partial f / partial z|$
- Day 7 training details

== Autoregressive Factorization

- $log p(x) = sum_t log p(x_t | x_(<t))$
- Causal masks enforce ordering
- GPT family (Days 9–10)

== Summary

- Day 6: *Generative Modeling*
- Likelihoods, KL, ELBO, model families
- Questions welcome — practical follows

== Questions?

#align(center + horizon)[
  #text(size: 44pt, weight: "bold", fill: course-primary)[Questions?]

  #v(1em)
  Practical session follows — see course site.
]
