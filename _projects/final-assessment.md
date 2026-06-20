---
layout: project
title: Final Assessment
caption: Math + coding (~3 hours)
description: >
  Closed-book math questions plus PyTorch coding: MLP classifier and diffusion sampler or KV-cache forward pass.
date: '28-08-2026'
sitemap: false
links:
  - title: Assessment notebook
    url: /notebooks/assessment/final_assessment.ipynb
  - title: Dataset
    url: /notebooks/data/assessment_synthetic.csv
---

# Final Assessment

**Duration:** ~3 hours · **Total:** 100 points (50 math + 50 coding) · **Submission:** PDF export of notebook + honor code statement

Work through [`notebooks/assessment/final_assessment.ipynb`](/notebooks/assessment/final_assessment.ipynb); write math answers in its markdown cells and code in its code cells. Before submitting, run **Kernel → Restart & Run All**.

## Part A — Mathematics (50 points)

1. **A1 — Chain rule and gradients (10 pts).** Differentiate $$f(x,y) = \sin(xy) + x^2$$, evaluate at a point, then apply the chain rule with $$x = t^2,\ y = 3t$$.
2. **A2 — Backpropagation sketch (10 pts).** For a one-hidden-unit network with squared loss, draw the computation graph and derive $$\partial \mathcal{L}/\partial w_1$$.
3. **A3 — KL divergence and ELBO (10 pts).** Write the closed-form $$\mathrm{KL}(q \,\|\, p)$$ for two univariate Gaussians, then explain why maximizing the ELBO $$\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)] - \mathrm{KL}(q_\phi(z\mid x) \,\|\, p(z))$$ trades off reconstruction against a latent regularizer.
4. **A4 — Forward diffusion and score matching (15 pts).** Show $$x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1 - \bar{\alpha}_t}\,\epsilon$$, relate the optimal noise predictor to the score $$\nabla_{x_t}\log q(x_t)$$, and explain why the signal-to-noise ratio falls with $$t$$.
5. **A5 — Scaled dot-product attention (15 pts).** Give the shapes in $$\mathrm{softmax}(QK^\top/\sqrt{d_k})V$$, justify the $$1/\sqrt{d_k}$$ scaling, and describe the causal mask.

## Part B — Coding (50 points)

Use [`notebooks/assessment/final_assessment.ipynb`](/notebooks/assessment/final_assessment.ipynb) and `notebooks/data/assessment_synthetic.csv`.

| Task | Points |
|------|--------|
| **B1 —** Train an MLP classifier (≥2 hidden layers) on the tabular data; report validation accuracy and plot the training loss | 25 |
| **B2a —** 1-D reverse diffusion sampler for a known Gaussian-mixture score | 25 |
| **B2b —** KV-cache greedy decoding for a tiny causal Transformer; report the tokens/sec speedup | 25 |

For **B2**, complete **one** track (B2a *or* B2b), not both.

## Grading rubric

- Correct math reasoning and notation
- Code runs end-to-end; plots where requested
- Brief written interpretation of results (2–3 sentences per part)

## Figure reference

![The deep generative modeling landscape: transform a simple reference distribution into the data distribution by different model families.](/assets/figures/day06/pdm_dgm_zoo.png)
*The generative-modeling map from Day 6 — useful background for Part A.*
