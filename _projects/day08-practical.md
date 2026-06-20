---
layout: project
title: Day 8 Exercise — SDE vs ODE Sampling
caption: Simulate generative dynamics
description: >
  Euler–Maruyama vs probability-flow ODE on 2D moons (MIT Lab 1 inspired).
date: '26-08-2026'
sitemap: false
links:
  - title: Exercise notebook
    url: /notebooks/practicals/day08.ipynb
  - title: Lecture slides (PDF)
    url: /assets/slides/day08.pdf
  - title: Lecture notes
    url: /blog/lectures/2026/08/26/day08-diffusion-flow-inference/
---

# Day 8 Exercise — SDE vs ODE Sampling


## Figures from lecture materials

![Source illustration — Day 8 (extracted from course PDFs/PPTX)](/assets/figures/day08/pdm_cfg.png)
*Source illustration — Day 8 (extracted from course PDFs/PPTX)*

![Source illustration — Day 8 (extracted from course PDFs/PPTX)](/assets/figures/day08/pdm_ddim_euler.png)
*Source illustration — Day 8 (extracted from course PDFs/PPTX)*

![Source illustration — Day 8 (extracted from course PDFs/PPTX)](/assets/figures/day08/pdm_deis.png)
*Source illustration — Day 8 (extracted from course PDFs/PPTX)*

![Source illustration — Day 8 (extracted from course PDFs/PPTX)](/assets/figures/day08/pdm_flowmap.png)
*Source illustration — Day 8 (extracted from course PDFs/PPTX)*


## Learning objectives

- Simulate ODEs and SDEs with Euler schemes
- Compare stochastic vs deterministic sampling paths
- Implement a few-step reverse diffusion sampler on 2D data

## Key derivations

**Euler–Maruyama.**

$$X_{t+\Delta t} = X_t + u_t(X_t)\Delta t + \sigma_t \sqrt{\Delta t}\, \xi, \quad \xi \sim \mathcal{N}(0,I).$$

**Probability-flow ODE** (schematic): replace noise term with a drift correction using score $$\nabla \log p_t(x)$$.

**Reverse diffusion (DDPM-style, few steps):**

$$x_{t-\Delta} \approx x_t + \tfrac{1}{2}\beta_t \nabla \log p_t(x_t) + \text{noise term}.$$

## Exercises

1. Integrate 2D ODE — plot trajectories
2. SDE with additive noise — ensemble of paths
3. 10-step sampler from noise to data cloud
4. **Reflection:** when is stochastic sampling preferred?

## Notebook

[`notebooks/practicals/day08.ipynb`](/notebooks/practicals/day08.ipynb)
