---
layout: post
title: Day 8 - Inference for Diffusion and Flow Models
image: /assets/img/lessons/day08.png
accent_image: 
  background: url('/assets/img/sampling_space.png') center/cover
  overlay: false
accent_color: '#ccc'
theme_color: '#ccc'
description: >
  Reverse SDEs, probability-flow ODEs, Euler–Maruyama, and classifier-free guidance.
invert_sidebar: true
---

# Day 8 - Inference for Diffusion and Flow Models

### [Slides](/assets/slides/day08.pdf)

### [Practical](/projects/day08-practical/)

### Optional reading for this lesson
- [Song et al. — Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456)
- [Ho et al. — Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Ho & Salimans — Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)
- [Complete reading list for Day 8](/publications/#day-8) (all resources for this lecture)


* toc
{:toc}

Training gives a score or velocity field; **inference** integrates dynamics backward in time to produce samples.
We compare **stochastic** (SDE) and **deterministic** (ODE) samplers and how guidance steers generation.

## 1. Reverse SDE

> **Time reversal (informal).** If forward dynamics add noise according to an SDE, the reverse-time SDE
> removes noise using the **score** of the marginal $$p_t(\mathbf{x})$$.
{:.lead}

For a forward Itô SDE

$$
d\mathbf{x}_t = \mathbf{f}(\mathbf{x}_t, t)\,dt + g(t)\,d\mathbf{w}_t,
$$

the reverse SDE (Anderson, 1982) involves $$\nabla_{\mathbf{x}} \log p_t(\mathbf{x}_t)$$:

$$
d\mathbf{x}_t = \left[\mathbf{f}(\mathbf{x}_t, t) - g(t)^2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x}_t)\right] dt + g(t)\,d\bar{\mathbf{w}}_t,
$$

run from $$t=T$$ down to $$0$$. In practice $$s_\theta(\mathbf{x}, t) \approx \nabla_{\mathbf{x}} \log p_t(\mathbf{x})$$.

![Reverse-time dynamics](/assets/figures/day08/pdf0_page025.png)
*Figure: stochastic denoising trajectory.*

### 1.1 Discrete DDPM updates

DDPM defines a Markov reverse kernel $$p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$$ parameterized via
predicted noise. Each step is a Gaussian whose mean depends on $$s_\theta$$ or $$\boldsymbol{\epsilon}_\theta$$.

## 2. Probability flow ODE

> **Probability flow ODE.** There exists an ODE with the **same marginals** $$p_t$$ as the forward SDE but
> without injected noise; it uses the score to cancel the diffusion term.
{:.lead}

A common form (VP parameterization):

$$
\frac{d\mathbf{x}_t}{dt} = \mathbf{f}(\mathbf{x}_t, t) - \tfrac{1}{2} g(t)^2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x}_t).
$$

**Deterministic sampling** integrates this ODE from noise to data—often with fewer steps than the SDE when using
high-order solvers (Heun, DPM-Solver, etc.).

![ODE vs SDE marginals](/assets/figures/day08/pdf1_page020.png)
*Figure: same $$p_t$$, different sample paths.*

### 2.1 When to prefer ODE vs SDE

- **ODE:** reproducible samples, fast solvers, good FID with few steps.
- **SDE:** extra stochasticity can improve diversity; closer to training noise process.

## 3. Euler–Maruyama discretization

To simulate

$$
d\mathbf{x}_t = \mathbf{a}(\mathbf{x}_t, t)\,dt + g(t)\,d\mathbf{w}_t
$$

on a grid $$0 = t_0 < t_1 < \cdots < t_K = T$$, **Euler–Maruyama** uses

$$
\mathbf{x}_{n+1} = \mathbf{x}_n + \mathbf{a}(\mathbf{x}_n, t_n)\,\Delta t_n + g(t_n)\sqrt{\Delta t_n}\,\boldsymbol{\zeta}_n,
\qquad \boldsymbol{\zeta}_n \sim \mathcal{N}(\mathbf{0}, \mathbf{I}).
$$

For the **reverse** SDE, replace $$\nabla_{\mathbf{x}} \log p_t$$ with $$s_\theta(\mathbf{x}, t)$$ and step **backward**
($$\Delta t < 0$$). Step size controls quality–speed trade-off.

![Numerical integration](/assets/figures/day08/pdf1_page040.png)
*Figure: discretization error vs step count.*

### 3.1 Flow-model inference

Given learned $$\mathbf{v}_\theta(\mathbf{x}, t)$$, integrate

$$
\mathbf{x}_{n+1} = \mathbf{x}_n + \mathbf{v}_\theta(\mathbf{x}_n, t_n)\,\Delta t_n
$$

from $$t=0$$ (prior) to $$t=1$$ (data). Same integrators apply; no Brownian term unless you add stochasticity.

## 4. Classifier-free guidance

Conditional models use $$s_\theta(\mathbf{x}, t, y)$$ or $$\boldsymbol{\epsilon}_\theta(\mathbf{x}, t, y)$$.
**Classifier guidance** perturbs scores with $$\nabla_{\mathbf{x}} \log p(y\mid\mathbf{x}_t)$$—requires a separate classifier.

**Classifier-free guidance (CFG)** trains with random dropout of $$y$$, then at sample time mixes conditional and unconditional predictions:

$$
\tilde{\boldsymbol{\epsilon}}_\theta(\mathbf{x}, t, y) =
\boldsymbol{\epsilon}_\theta(\mathbf{x}, t, \varnothing)
+ w\left(\boldsymbol{\epsilon}_\theta(\mathbf{x}, t, y) - \boldsymbol{\epsilon}_\theta(\mathbf{x}, t, \varnothing)\right),
$$

where $$w \ge 1$$ is the **guidance scale**. Larger $$w$$ increases adherence to $$y$$ but can reduce diversity or stability.

![Guidance effect](/assets/figures/day08/pdf1_page060.png)
*Figure: text/image conditioning strength.*

### 4.1 Practical guidance tips

- Train with 10–20% null labels for $$\varnothing$$.
- Tune $$w$$ per task; too high → oversaturated or mode-collapsed outputs.
- CFG applies equally to flow velocities by mixing $$\mathbf{v}_\theta(\mathbf{x}, t, y)$$.

### 4.2 Step schedulers and distillation

**DDIM** and related samplers skip Markovian noise injection by integrating a non-Markovian generative process
consistent with the same training objective—often 10–50 steps instead of hundreds.

**Consistency models** and **progressive distillation** train a student that maps $$\mathbf{x}_t \to \mathbf{x}_{t-\Delta}$$
in one shot, amortizing solver cost at deployment.

### 4.3 Error analysis for Euler–Maruyama

For step size $$\Delta t$$, local weak error is $$\mathcal{O}(\Delta t)$$ under smooth coefficients; global error accumulates
over $$K = T/\Delta t$$ steps. Adaptive solvers monitor drift norms and shrink $$\Delta t$$ when score magnitudes spike—common
near $$t \approx 0$$ where data detail is recovered.

![Sampler comparison](/assets/figures/day08/pdf0_page035.png)
*Figure: SDE, ODE, and distilled few-step samplers.*

## Checkpoint summary

- **Reverse SDE** = forward drift correction minus score term + noise.
- **Probability flow ODE** shares marginals with deterministic integration.
- **Euler–Maruyama** discretizes SDEs; flow models use the ODE limit.
- **CFG** trades off fidelity to conditioning vs sample diversity via scale $$w$$.
