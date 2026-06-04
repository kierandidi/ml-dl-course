---
layout: project
title: Day 4 Practical — Convolutional Networks
caption: Fashion-MNIST CNN
description: >
  Implement a CNN for image classification; visualize filters and receptive fields.
date: '20-08-2026'
sitemap: false
links:
  - title: Jupyter notebook
    url: /notebooks/practicals/day04.ipynb
  - title: Lecture slides (PDF)
    url: /assets/slides/day04.pdf
  - title: Lecture notes
    url: /blog/lectures/2026/08/20/day04-convolutional-networks/
---

# Day 4 Practical — Convolutional Networks


## Figures from lecture materials

![Source illustration — Day 4 (extracted from course PDFs/PPTX)](/assets/figures/day04/pdf0_page000.png)
*Source illustration — Day 4 (extracted from course PDFs/PPTX)*

![Source illustration — Day 4 (extracted from course PDFs/PPTX)](/assets/figures/day04/pdf0_page003.png)
*Source illustration — Day 4 (extracted from course PDFs/PPTX)*

![Source illustration — Day 4 (extracted from course PDFs/PPTX)](/assets/figures/day04/pdf0_page006.png)
*Source illustration — Day 4 (extracted from course PDFs/PPTX)*

![Source illustration — Day 4 (extracted from course PDFs/PPTX)](/assets/figures/day04/pdf0_page008.png)
*Source illustration — Day 4 (extracted from course PDFs/PPTX)*


## Learning objectives

- Implement Conv2d → ReLU → Pool → Linear head
- Understand parameter sharing and translation equivariance
- Visualize first-layer convolutional filters

## Key ideas

**2D convolution.**

$$(f * k)[i,j] = \sum_{u,v} f[i-u,j-v]\, k[u,v].$$

**Receptive field** grows with depth: RF $$\approx 1 + \sum_\ell (k_\ell - 1) \prod_{m<\ell} s_m$$.

**Cross-entropy for $$K$$ classes.**

$$\mathcal{L} = -\sum_{k=1}^K y_k \log \hat{p}_k.$$

## Exercises

1. CNN on Fashion-MNIST — target $$>88\%$$ test accuracy
2. Plot training curves (loss/accuracy)
3. Visualize 8 first-layer filters
4. **Reflection:** what inductive bias does convolution encode?

## Notebook

[`notebooks/practicals/day04.ipynb`](/notebooks/practicals/day04.ipynb)
