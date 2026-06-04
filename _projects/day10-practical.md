---
layout: project
title: Day 10 Practical — KV Cache & Inference
caption: Efficient autoregressive decoding
description: >
  KV cache memory formula, timed generation with/without cache.
date: '28-08-2026'
sitemap: false
links:
  - title: Jupyter notebook
    url: /notebooks/practicals/day10.ipynb
  - title: Lecture slides (PDF)
    url: /assets/slides/day10.pdf
  - title: Lecture notes
    url: /blog/lectures/2026/08/28/day10-ar-inference/
---

# Day 10 Practical — KV Cache & Inference


## Figures from lecture materials

![Source illustration — Day 10 (extracted from course PDFs/PPTX)](/assets/figures/day10/pdf0_page000.png)
*Source illustration — Day 10 (extracted from course PDFs/PPTX)*

![Source illustration — Day 10 (extracted from course PDFs/PPTX)](/assets/figures/day10/pdf0_page002.png)
*Source illustration — Day 10 (extracted from course PDFs/PPTX)*

![Source illustration — Day 10 (extracted from course PDFs/PPTX)](/assets/figures/day10/pdf0_page004.png)
*Source illustration — Day 10 (extracted from course PDFs/PPTX)*

![Source illustration — Day 10 (extracted from course PDFs/PPTX)](/assets/figures/day10/pdf0_page005.png)
*Source illustration — Day 10 (extracted from course PDFs/PPTX)*


## Learning objectives

- Derive KV cache memory scaling
- Implement cached incremental decoding
- Compare temperature and top-$$p$$ sampling

## Key derivations

**KV cache size (per layer, batch $$B$$, seq $$T$$, head dim $$d_h$$, $$n_{\mathrm{kv}}$$ heads):**

$$\mathrm{Memory}_{\mathrm{KV}} \approx 2 \cdot L \cdot B \cdot T \cdot n_{\mathrm{kv}} \cdot d_h \cdot \texttt{bytes}.$$

**Autoregressive factorization.**

$$p(x_{1:T}) = \prod_{t=1}^T p(x_t \mid x_{<t}).$$

**Top-$$p$$ (nucleus):** keep smallest set of tokens with cumulative prob $$\geq p$$.

## Exercises

1. Compute KV bytes for given $$(L, B, T, d_{\mathrm{model}})$$
2. Time 500 tokens with vs without cache
3. Compare greedy vs temperature vs top-$$p$$ outputs
4. **Reflection:** throughput vs memory tradeoff at deployment

## Notebook

[`notebooks/practicals/day10.ipynb`](/notebooks/practicals/day10.ipynb)
