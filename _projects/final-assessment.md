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

**Duration:** ~3 hours · **Submission:** PDF export of notebook + honor code statement

## Part A — Written (40 points)

1. **(10 pts)** Derive the ELBO for a latent variable model with joint $$p_\theta(x,z) = p_\theta(x|z)p(z)$$ and variational $$q_\phi(z|x)$$. Identify each term.
2. **(10 pts)** Compute $$D_{\mathrm{KL}}(\mathcal{N}(0,1) \| \mathcal{N}(1, 4))$$ in closed form.
3. **(10 pts)** Explain **explicit** vs **implicit** generative models with one example each (VAE/GAN acceptable).
4. **(10 pts)** Write the **KV cache** memory formula for $$L$$ layers, batch $$B$$, sequence length $$T$$, hidden size $$d$$, and GQA with $$n_{\mathrm{kv}}$$ KV heads.

## Part B — Coding (60 points)

Use [`notebooks/assessment/final_assessment.ipynb`](/notebooks/assessment/final_assessment.ipynb) and `notebooks/data/assessment_synthetic.csv`.

| Task | Points |
|------|--------|
| Train 3-layer MLP classifier on tabular data | 25 |
| **Track A:** 10-step reverse diffusion sampler on 2D moons | 35 |
| **Track B:** KV-cache forward pass for provided tiny GPT | 35 |

Choose **one** generative track (A or B), not both.

## Grading rubric

- Correct math reasoning and notation
- Code runs end-to-end; plots where requested
- Brief written interpretation of results (2–3 sentences per part)

## Figure reference

![Assessment data](/assets/figures/day06/pdf0_page010.png)
*Example generative modeling figure (course materials)*
