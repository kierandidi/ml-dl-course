---
layout: project
title: Day 3 Practical — Deep Neural Networks
caption: MLP and autograd
description: >
  Train an MLP on MNIST; verify backprop with PyTorch autograd.
date: '19-08-2026'
sitemap: false
links:
  - title: Jupyter notebook
    url: /notebooks/practicals/day03.ipynb
  - title: Lecture slides (PDF)
    url: /assets/slides/day03.pdf
  - title: Lecture notes
    url: /blog/lectures/2026/08/19/day03-deep-neural-networks/
---

# Day 3 Practical — Deep Neural Networks


## Figures from lecture materials

![Source illustration — Day 3 (extracted from course PDFs/PPTX)](/assets/figures/day03/L1_introduct_00.png)
*Source illustration — Day 3 (extracted from course PDFs/PPTX)*

![Source illustration — Day 3 (extracted from course PDFs/PPTX)](/assets/figures/day03/L1_introduct_06.png)
*Source illustration — Day 3 (extracted from course PDFs/PPTX)*

![Source illustration — Day 3 (extracted from course PDFs/PPTX)](/assets/figures/day03/L1_introduct_08.png)
*Source illustration — Day 3 (extracted from course PDFs/PPTX)*

![Source illustration — Day 3 (extracted from course PDFs/PPTX)](/assets/figures/day03/L5- CNN_00.png)
*Source illustration — Day 3 (extracted from course PDFs/PPTX)*


## Learning objectives

- Build a multi-layer perceptron in PyTorch
- Verify gradients with `torch.autograd.gradcheck`
- Compare SGD and Adam on convergence speed

## Key derivations

**Backprop for a layer.** With $$\mathbf{z} = \mathbf{W}\mathbf{h} + \mathbf{b}$$, $$\mathbf{h}' = g(\mathbf{z})$$,

$$\frac{\partial \mathcal{L}}{\partial \mathbf{z}} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}'} \odot g'(\mathbf{z}), \quad
\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}} \mathbf{h}^T.$$

**Adam update.**

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \quad
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2, \quad
\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}.$$

## Exercises

1. Train 2-layer MLP on MNIST — report test accuracy
2. `gradcheck` on a single linear+ReLU block
3. Plot train loss for SGD vs Adam (same architecture)
4. **Reflection:** why ReLU helps optimization vs sigmoid

## Notebook

[`notebooks/practicals/day03.ipynb`](/notebooks/practicals/day03.ipynb)
