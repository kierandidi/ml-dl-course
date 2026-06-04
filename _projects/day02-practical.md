---
layout: project
title: Day 2 Practical — Statistical Learning
caption: Regression and classification
description: >
  Ridge regression, logistic classification, cross-validation, and bias–variance experiments.
date: '18-08-2026'
sitemap: false
links:
  - title: Jupyter notebook
    url: /notebooks/practicals/day02.ipynb
  - title: Lecture slides (PDF)
    url: /assets/slides/day02.pdf
  - title: Lecture notes
    url: /blog/lectures/2026/08/18/day02-statistical-learning/
---

# Day 2 Practical — Statistical Learning


## Figures from lecture materials

![Source illustration — Day 2 (extracted from course PDFs/PPTX)](/assets/figures/day02/L5_probabili_01.png)
*Source illustration — Day 2 (extracted from course PDFs/PPTX)*

![Source illustration — Day 2 (extracted from course PDFs/PPTX)](/assets/figures/day02/L5_probabili_02.png)
*Source illustration — Day 2 (extracted from course PDFs/PPTX)*

![Source illustration — Day 2 (extracted from course PDFs/PPTX)](/assets/figures/day02/L6_decision__00.png)
*Source illustration — Day 2 (extracted from course PDFs/PPTX)*

![Source illustration — Day 2 (extracted from course PDFs/PPTX)](/assets/figures/day02/L6_decision__01.png)
*Source illustration — Day 2 (extracted from course PDFs/PPTX)*


## Learning objectives

- Fit linear and ridge regression; interpret regularization path
- Train logistic regression; report cross-entropy and accuracy
- Observe bias–variance tradeoff via train/validation curves

## Key derivations

**Ridge regression (closed form).**

$$\hat{\mathbf{w}} = (\mathbf{X}^T\mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}.$$

**Logistic loss.** For label $$y \in \{0,1\}$$ and probability $$\hat{p} = \sigma(\mathbf{w}^T\mathbf{x})$$,

$$\mathcal{L} = -\big[ y \log \hat{p} + (1-y)\log(1-\hat{p}) \big].$$

**Bias–variance decomposition (squared loss).**

$$\mathbb{E}[(y - \hat{f})^2] = \mathrm{Bias}^2 + \mathrm{Var}(\hat{f}) + \sigma^2.$$

## Exercises

1. OLS vs ridge on polynomial features — plot validation MSE vs $$\lambda$$
2. Logistic regression on synthetic 2D data — decision boundary plot
3. k-fold cross-validation for model selection
4. **Reflection:** when does high variance dominate in deep learning?

## Notebook

[`notebooks/practicals/day02.ipynb`](/notebooks/practicals/day02.ipynb)
