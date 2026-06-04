---
layout: project
title: Day 6 Practical — Generative Modeling
caption: KL, ELBO, and VAE
description: >
  Derive ELBO, compute KL for Gaussians, train a 1D VAE.
date: '24-08-2026'
sitemap: false
links:
  - title: Jupyter notebook
    url: /notebooks/practicals/day06.ipynb
  - title: Lecture slides (PDF)
    url: /assets/slides/day06.pdf
  - title: Lecture notes
    url: /blog/lectures/2026/08/24/day06-generative-modeling/
---

# Day 6 Practical — Generative Modeling


## Figures from lecture materials

![Source illustration — Day 6 (extracted from course PDFs/PPTX)](/assets/figures/day06/pdf0_page000.png)
*Source illustration — Day 6 (extracted from course PDFs/PPTX)*

![Source illustration — Day 6 (extracted from course PDFs/PPTX)](/assets/figures/day06/pdf0_page002.png)
*Source illustration — Day 6 (extracted from course PDFs/PPTX)*

![Source illustration — Day 6 (extracted from course PDFs/PPTX)](/assets/figures/day06/pdf0_page004.png)
*Source illustration — Day 6 (extracted from course PDFs/PPTX)*

![Source illustration — Day 6 (extracted from course PDFs/PPTX)](/assets/figures/day06/pdf0_page005.png)
*Source illustration — Day 6 (extracted from course PDFs/PPTX)*


## Learning objectives

- Compute $$D_{\mathrm{KL}}(q\|p)$$ for univariate Gaussians in closed form
- Derive the ELBO and implement a 1D Gaussian VAE
- Classify model families (explicit vs implicit)

## Key derivations

**KL between Gaussians** $$\mathcal{N}(\mu_0,\sigma_0^2)$$ and $$\mathcal{N}(\mu_1,\sigma_1^2)$$:

$$D_{\mathrm{KL}} = \log\frac{\sigma_1}{\sigma_0} + \frac{\sigma_0^2 + (\mu_0-\mu_1)^2}{2\sigma_1^2} - \frac{1}{2}.$$

**ELBO.**

$$\log p(x) \geq \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{\mathrm{KL}}(q(z|x)\|p(z)).$$

**VAE loss** = reconstruction + KL to prior $$\mathcal{N}(0,I)$$.

## Exercises

1. Pen-and-paper KL (then verify numerically)
2. Implement ELBO for factorized Gaussian $$q(z|x)$$
3. Train 1D VAE on synthetic mixture — plot samples
4. **Reflection:** explicit vs implicit — one example each

## Notebook

[`notebooks/practicals/day06.ipynb`](/notebooks/practicals/day06.ipynb)
