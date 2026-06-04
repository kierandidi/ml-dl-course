---
layout: project
title: Day 5 Practical — Sequences & Attention
caption: RNN and attention
description: >
  Sentiment RNN and manual scaled dot-product attention on short sequences.
date: '21-08-2026'
sitemap: false
links:
  - title: Jupyter notebook
    url: /notebooks/practicals/day05.ipynb
  - title: Lecture slides (PDF)
    url: /assets/slides/day05.pdf
  - title: Lecture notes
    url: /blog/lectures/2026/08/21/day05-rnns-and-transformers/
---

# Day 5 Practical — Sequences & Attention


## Figures from lecture materials

![Source illustration — Day 5 (extracted from course PDFs/PPTX)](/assets/figures/day05/L6_LSTM_00.png)
*Source illustration — Day 5 (extracted from course PDFs/PPTX)*

![Source illustration — Day 5 (extracted from course PDFs/PPTX)](/assets/figures/day05/L6_LSTM_02.png)
*Source illustration — Day 5 (extracted from course PDFs/PPTX)*

![Source illustration — Day 5 (extracted from course PDFs/PPTX)](/assets/figures/day05/L6_LSTM_04.png)
*Source illustration — Day 5 (extracted from course PDFs/PPTX)*

![Source illustration — Day 5 (extracted from course PDFs/PPTX)](/assets/figures/day05/L6_LSTM_06.png)
*Source illustration — Day 5 (extracted from course PDFs/PPTX)*


## Learning objectives

- Train an LSTM/GRU for sequence classification
- Implement scaled dot-product attention by hand
- Compare sequential vs parallel computation

## Key derivations

**Scaled dot-product attention.**

$$\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V.$$

**LSTM cell (schematic).** Gates $$f,i,o$$ control forget, input, and output; state $$c_t$$ carries long-range memory.

**Causal mask:** $$A_{ij} = -\infty$$ for $$j > i$$ so position $$i$$ cannot attend to the future.

## Exercises

1. IMDB/binary sentiment with embedding + LSTM
2. Manual attention on length-$$T$$ toy sequences — heatmap
3. Count parameters: RNN vs 1-layer transformer block
4. **Reflection:** why transformers replaced RNNs at scale

## Notebook

[`notebooks/practicals/day05.ipynb`](/notebooks/practicals/day05.ipynb)
