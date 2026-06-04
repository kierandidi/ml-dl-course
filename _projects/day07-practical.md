---
layout: project
title: Day 7 Practical — Flow & Diffusion Training
caption: Flow matching and denoising
description: >
  1D flow matching and denoising score targets (MIT Lab 2 inspired).
date: '25-08-2026'
sitemap: false
links:
  - title: Jupyter notebook
    url: /notebooks/practicals/day07.ipynb
  - title: Lecture slides (PDF)
    url: /assets/slides/day07.pdf
  - title: Lecture notes
    url: /blog/lectures/2026/08/25/day07-training-diffusion-flow/
---

# Day 7 Practical — Flow & Diffusion Training


## Figures from lecture materials

![Source illustration — Day 7 (extracted from course PDFs/PPTX)](/assets/figures/day07/pdf0_page000.png)
*Source illustration — Day 7 (extracted from course PDFs/PPTX)*

![Source illustration — Day 7 (extracted from course PDFs/PPTX)](/assets/figures/day07/pdf0_page002.png)
*Source illustration — Day 7 (extracted from course PDFs/PPTX)*

![Source illustration — Day 7 (extracted from course PDFs/PPTX)](/assets/figures/day07/pdf0_page004.png)
*Source illustration — Day 7 (extracted from course PDFs/PPTX)*

![Source illustration — Day 7 (extracted from course PDFs/PPTX)](/assets/figures/day07/pdf0_page006.png)
*Source illustration — Day 7 (extracted from course PDFs/PPTX)*


## Learning objectives

- Define a probability path $$p_t(x)$$ from noise to data
- Train a velocity field / score network with simple regression
- Relate flow matching to diffusion denoising objectives

## Key derivations

**Flow matching objective.**

$$\mathcal{L}_{\mathrm{FM}} = \mathbb{E}_{t,x_0,x_1}\big[ \| v_\theta(x_t, t) - (x_1 - x_0) \|^2 \big]$$

for appropriate interpolant $$x_t$$.

**Denoising score matching.**

$$\mathcal{L} = \mathbb{E}_{t,x_0,\epsilon}\big[ \| s_\theta(x_t, t) + \sigma_t^{-1}\epsilon \|^2 \big], \quad x_t = x_0 + \sigma_t \epsilon.$$

## Exercises

1. Sample 1D interpolants $$x_t = (1-t)x_0 + t x_1$$
2. Train MLP $$v_\theta(x,t)$$ on synthetic 1D data
3. Plot learned field vs ground truth
4. **Reflection:** three views of diffusion (variational, score, flow)

## Notebook

[`notebooks/practicals/day07.ipynb`](/notebooks/practicals/day07.ipynb)
