---
layout: project
title: Day 9 Practical — Train a Tiny GPT
caption: Decoder-only language model
description: >
  Character-level decoder-only transformer; cross-entropy training loop.
date: '27-08-2026'
sitemap: false
links:
  - title: Jupyter notebook
    url: /notebooks/practicals/day09.ipynb
  - title: Lecture slides (PDF)
    url: /assets/slides/day09.pdf
  - title: Lecture notes
    url: /blog/lectures/2026/08/27/day09-autoregressive-llms/
---

# Day 9 Practical — Train a Tiny GPT


## Figures from lecture materials

![Source illustration — Day 9 (extracted from course PDFs/PPTX)](/assets/figures/day09/pdf0_page000.png)
*Source illustration — Day 9 (extracted from course PDFs/PPTX)*

![Source illustration — Day 9 (extracted from course PDFs/PPTX)](/assets/figures/day09/pdf0_page004.png)
*Source illustration — Day 9 (extracted from course PDFs/PPTX)*

![Source illustration — Day 9 (extracted from course PDFs/PPTX)](/assets/figures/day09/pdf0_page008.png)
*Source illustration — Day 9 (extracted from course PDFs/PPTX)*

![Source illustration — Day 9 (extracted from course PDFs/PPTX)](/assets/figures/day09/pdf0_page012.png)
*Source illustration — Day 9 (extracted from course PDFs/PPTX)*


## Learning objectives

- Implement causal self-attention and a transformer block
- Train on a tiny corpus (~2M parameters budget)
- Track loss curves and generate samples

## Key derivations

**Causal LM loss.**

$$\mathcal{L} = -\sum_{i=1}^{L} \log p_\theta(x_i \mid x_{<i}).$$

**Pre-norm block.**

$$h' = h + \mathrm{MHA}(\mathrm{LN}(h)), \quad h'' = h' + \mathrm{MLP}(\mathrm{LN}(h')).$$

**RoPE** encodes relative position via rotations in Q/K pairs (see lecture notes).

## Exercises

1. Tokenize character-level corpus
2. Train small GPT — plot loss
3. Sample 200 characters from trained model
4. **Reflection:** estimate FLOPs per token (order of magnitude)

## Notebook

[`notebooks/practicals/day09.ipynb`](/notebooks/practicals/day09.ipynb)
