---
layout: project
title: Day 1 Practical — Math Foundations
caption: Gradients, probability, and MLE
description: >
  Numerical gradients, Gaussian MLE, and probability exercises with worked derivations.
date: '17-08-2026'
sitemap: false
links:
  - title: Jupyter notebook
    url: /notebooks/practicals/day01.ipynb
  - title: Lecture slides (PDF)
    url: /assets/slides/day01.pdf
  - title: Lecture notes
    url: /blog/lectures/2026/08/17/day01-math-foundations/
---

# Day 1 Practical — Math Foundations


## Figures from lecture materials

![Source illustration — Day 1 (extracted from course PDFs/PPTX)](/assets/figures/day01/mml_gradient.png)
*Source illustration — Day 1 (extracted from course PDFs/PPTX)*

![Source illustration — Day 1 (extracted from course PDFs/PPTX)](/assets/figures/day01/mml_projection.png)
*Source illustration — Day 1 (extracted from course PDFs/PPTX)*

![Source illustration — Day 1 (extracted from course PDFs/PPTX)](/assets/figures/day01/mml_gaussian.png)
*Source illustration — Day 1 (extracted from course PDFs/PPTX)*

![Source illustration — Day 1 (extracted from course PDFs/PPTX)](/assets/figures/day01/ode_vectorfield.png)
*Source illustration — Day 1 (extracted from course PDFs/PPTX)*


## Learning objectives

- Implement finite-difference gradients and compare to symbolic answers
- Derive and code MLE for Gaussian mean/variance
- Connect calculus to optimization used in ML

## Key derivations

**Gradient of a quadratic.** For $$f(w) = \frac{1}{2} w^T A w - b^T w$$ with symmetric $$A$$,

$$\nabla_w f(w) = A w - b.$$

**Gaussian MLE.** For $$x_1,\ldots,x_n \sim \mathcal{N}(\mu, \sigma^2)$$,

$$\hat{\mu}_{\mathrm{MLE}} = \frac{1}{n}\sum_i x_i, \qquad
\hat{\sigma}^2_{\mathrm{MLE}} = \frac{1}{n}\sum_i (x_i - \hat{\mu})^2.$$

**Multivariate MLE.** For $$\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$$,

$$\hat{\boldsymbol{\mu}} = \frac{1}{n}\sum_i \mathbf{x}^{(i)}, \qquad
\hat{\Sigma} = \frac{1}{n}\sum_i (\mathbf{x}^{(i)} - \hat{\boldsymbol{\mu}})(\mathbf{x}^{(i)} - \hat{\boldsymbol{\mu}})^T.$$

## Exercises (notebook)

1. Numerical vs analytic gradient on $$f(x,y) = x^2 + xy + y^2$$
2. MLE for 1D Gaussian — plot likelihood surface
3. Bivariate Gaussian: estimate $$\boldsymbol{\mu}, \Sigma$$ and verify via `scipy.stats`
4. **Reflection:** where do non-convex objectives appear later in the course?

## Notebook

Open [`notebooks/practicals/day01.ipynb`](/notebooks/practicals/day01.ipynb) (also in the GitHub repo).
