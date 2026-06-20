---
layout: project
title: Day 2 Exercise — Statistical Learning
caption: Regression, PCA, GMM, and SVM
description: >
  MML Ch. 8–12 exercises: OLS/ridge, PCA, GMM with EM, and SVM on synthetic data.
date: '18-08-2026'
sitemap: false
links:
  - title: Exercise notebook
    url: /notebooks/practicals/day02.ipynb
  - title: Lecture slides (PDF)
    url: /assets/slides/day02.pdf
  - title: Lecture notes
    url: /blog/lectures/2026/08/18/day02-statistical-learning/
---

# Day 2 Exercise — Statistical Learning


## Figures from lecture materials

![Source illustration — Day 2 (extracted from course PDFs/PPTX)](/assets/figures/day02/ml_taxonomy.png)
*Source illustration — Day 2 (extracted from course PDFs/PPTX)*

![Source illustration — Day 2 (extracted from course PDFs/PPTX)](/assets/figures/day02/mml_linear_regression.png)
*Source illustration — Day 2 (extracted from course PDFs/PPTX)*

![Source illustration — Day 2 (extracted from course PDFs/PPTX)](/assets/figures/day02/mml_pca_illustration.png)
*Source illustration — Day 2 (extracted from course PDFs/PPTX)*

![Source illustration — Day 2 (extracted from course PDFs/PPTX)](/assets/figures/day02/mml_svm_margin.png)
*Source illustration — Day 2 (extracted from course PDFs/PPTX)*


## Learning objectives

- Fit linear and ridge regression; relate OLS to Gaussian MLE
- Run PCA for visualization and reconstruction; report explained variance
- Fit a 2-component GMM with EM; interpret responsibilities
- Train a linear SVM; compare margin to logistic regression

## Key derivations

**OLS / MLE.**

$$\hat{\boldsymbol{\theta}} = (\Phi^{\top}\Phi)^{-1}\Phi^{\top}\mathbf{y}.$$

**Ridge regression.**

$$\hat{\boldsymbol{\theta}} = (\Phi^{\top}\Phi + \lambda \mathbf{I})^{-1}\Phi^{\top}\mathbf{y}.$$

**PCA (top-$M$ eigenvectors of sample covariance).**

$$\mathbf{S} = \frac{1}{N}\sum_n \tilde{\mathbf{x}}_n \tilde{\mathbf{x}}_n^{\top}, \quad \mathbf{z}_n = \mathbf{B}^{\top}\tilde{\mathbf{x}}_n.$$

**GMM responsibility (E-step).**

$$r_{nk} = \frac{\pi_k\,\mathcal{N}(\mathbf{x}_n\mid\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_j \pi_j\,\mathcal{N}(\mathbf{x}_n\mid\boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}.$$

## Exercises

1. **Regression** — polynomial features on 1D data; plot train/validation MSE vs degree and vs ridge $$\lambda$$
2. **PCA** — project 2D/3D data to 2 components; plot reconstruction error vs $$M$$
3. **GMM** — fit $$K=2$$ on a bimodal 1D dataset; plot fitted density and responsibilities
4. **SVM** — linearly separable 2D points; plot decision boundary and support vectors; sweep soft-margin $$C$$
5. **Reflection** — where does each method sit in the ML taxonomy (supervised vs unsupervised)?

## Notebook

[`notebooks/practicals/day02.ipynb`](/notebooks/practicals/day02.ipynb)
