---
layout: post
title: Day 6 - Generative Modeling
image: /assets/img/lessons/day06.png
accent_image: 
  background: url('/assets/img/sampling_space.png') center/cover
  overlay: false
accent_color: '#ccc'
theme_color: '#ccc'
description: >
  KL divergence, ELBO, MLE/MAP, and model families from VAEs to energy-based models.
invert_sidebar: true
---

# Day 6 - Generative Modeling

### [Slides](/assets/slides/day06.pdf)

### [Practical](/projects/day06-practical/)

### Optional reading for this lesson
- [The Principles of Diffusion Models](https://arxiv.org/abs/2510.21890) — Ch. 1–2 (variational view)
- [Kingma & Welling — Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
- [Rezende & Mohamed — Normalizing Flows](https://arxiv.org/abs/1505.05770)
- [Complete reading list for Day 6](/publications/#day-6) (all resources for this lecture)


* toc
{:toc}

Week 2 opens with the probabilistic language of generative modeling: how we compare distributions,
how we optimize latent-variable models, and how explicit density models differ from implicit samplers.

## 1. Likelihoods, MLE, and MAP

> **Definition (generative model).** A generative model specifies a distribution $$p_\theta(\mathbf{x})$$
> over data $$\mathbf{x} \in \mathcal{X}$$, either directly or via latent variables
> $$\mathbf{z}$$ and a joint $$p_\theta(\mathbf{x}, \mathbf{z})$$.
{:.lead}

Given i.i.d. samples $$\{\mathbf{x}^{(i)}\}_{i=1}^N$$ from an unknown data distribution $$p_{\mathrm{data}}$$,
**maximum likelihood estimation (MLE)** chooses parameters that maximize the empirical log-likelihood:

$$
\hat{\theta}_{\mathrm{MLE}} = \arg\max_\theta \frac{1}{N}\sum_{i=1}^N \log p_\theta(\mathbf{x}^{(i)}).
$$

When we have a prior $$p(\theta)$$ over parameters, **maximum a posteriori (MAP)** adds a regularizer:

$$
\hat{\theta}_{\mathrm{MAP}} = \arg\max_\theta \left[ \sum_{i=1}^N \log p_\theta(\mathbf{x}^{(i)}) + \log p(\theta) \right].
$$

For deep generative models the likelihood is often intractable; we then optimize surrogates (ELBO, score
matching, flow-matching losses) that remain consistent with likelihood principles.

![Generative modeling overview](/assets/figures/day06/pdf0_page000.png)
*Figure: generative modeling landscape (slides).*

### 1.1 KL divergence

> **Definition.** For distributions $$p$$ and $$q$$ on the same space, the Kullback–Leibler divergence is
> $$D_{\mathrm{KL}}(p \| q) = \mathbb{E}_{\mathbf{x}\sim p}\left[\log \frac{p(\mathbf{x})}{q(\mathbf{x})}\right]$$.
{:.lead}

Properties used throughout the course:

- $$D_{\mathrm{KL}}(p \| q) \ge 0$$ with equality iff $$p = q$$ (a.e.).
- Not symmetric: $$D_{\mathrm{KL}}(p \| q) \neq D_{\mathrm{KL}}(q \| p)$$ in general.
- **Information inequality:** minimizing $$D_{\mathrm{KL}}(p_{\mathrm{data}} \| p_\theta)$$ is equivalent to MLE when $$p_\theta$$ is flexible enough.

For latent-variable models with joint $$p_\theta(\mathbf{x}, \mathbf{z}) = p_\theta(\mathbf{z})\,p_\theta(\mathbf{x}\mid\mathbf{z})$$,

$$
\log p_\theta(\mathbf{x}) = \log \int p_\theta(\mathbf{x}, \mathbf{z})\,d\mathbf{z},
$$

which is typically intractable for neural encoders/decoders.

## 2. ELBO and variational inference

Introduce an approximate posterior $$q_\phi(\mathbf{z}\mid\mathbf{x})$$. For any fixed $$\mathbf{x}$$,

$$
\log p_\theta(\mathbf{x}) = \mathcal{L}(\theta, \phi; \mathbf{x}) + D_{\mathrm{KL}}\!\left(q_\phi(\mathbf{z}\mid\mathbf{x}) \,\|\, p_\theta(\mathbf{z}\mid\mathbf{x})\right),
$$

where the **evidence lower bound (ELBO)** is

$$
\mathcal{L}(\theta, \phi; \mathbf{x}) =
\mathbb{E}_{q_\phi(\mathbf{z}\mid\mathbf{x})}\!\left[\log p_\theta(\mathbf{x}\mid\mathbf{z})\right]
- D_{\mathrm{KL}}\!\left(q_\phi(\mathbf{z}\mid\mathbf{x}) \,\|\, p_\theta(\mathbf{z})\right).
$$

Since $$D_{\mathrm{KL}} \ge 0$$, we have $$\mathcal{L}(\theta,\phi;\mathbf{x}) \le \log p_\theta(\mathbf{x})$$.
**Maximizing the ELBO** tightens the bound and improves the marginal likelihood proxy.

![ELBO decomposition](/assets/figures/day06/pdf0_page005.png)
*Figure: variational bound on log-likelihood.*

### 2.1 Amortized inference

In a **VAE**, $$q_\phi(\mathbf{z}\mid\mathbf{x}) = \mathcal{N}(\mathbf{z}; \boldsymbol{\mu}_\phi(\mathbf{x}), \mathrm{diag}(\boldsymbol{\sigma}_\phi^2(\mathbf{x})))$$
and $$p_\theta(\mathbf{x}\mid\mathbf{z})$$ is a decoder. Training maximizes

$$
\mathbb{E}_{\mathbf{x}\sim p_{\mathrm{data}}}\left[\mathcal{L}(\theta,\phi;\mathbf{x})\right],
$$

using the reparameterization trick $$\mathbf{z} = \boldsymbol{\mu}_\phi(\mathbf{x}) + \boldsymbol{\sigma}_\phi(\mathbf{x}) \odot \boldsymbol{\epsilon}$$,
$$\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0}, \mathbf{I})$$.

## 3. Explicit vs implicit generative models

| Family | Tractable density? | Sample | Examples |
|--------|-------------------|--------|----------|
| **Explicit** | Yes (up to constant) | MCMC or exact | Flows, autoregressive, some EBMs |
| **Implicit** | No closed form | Forward pass | GANs, diffusion (learn score/velocity) |
| **Latent explicit** | Marginal hard | Encode–decode | VAE, hierarchical VAE |
| **Energy-based** | $$p(\mathbf{x}) \propto e^{-E(\mathbf{x})}$$ | MCMC / Langevin | Hopfield, modern EBMs |

> **Explicit model:** you can evaluate $$p_\theta(\mathbf{x})$$ (or $$\log p_\theta(\mathbf{x})$$) up to normalization for a single $$\mathbf{x}$$.
> **Implicit model:** you can sample $$\mathbf{x}\sim p_\theta$$ without a normalized density.
{:.lead}

![Model taxonomy](/assets/figures/day06/pdf0_page010.png)
*Figure: where common architectures sit in the taxonomy.*

### 3.1 Normalizing flows

A flow is an invertible map $$\mathbf{f}_\theta: \mathbb{R}^d \to \mathbb{R}^d$$ with tractable Jacobian determinant:

$$
p_\theta(\mathbf{x}) = p_0(\mathbf{f}_\theta^{-1}(\mathbf{x}))
\left|\det \frac{\partial \mathbf{f}_\theta^{-1}(\mathbf{x})}{\partial \mathbf{x}}\right|.
$$

Composition of coupling layers yields expressive densities and exact MLE training.

### 3.2 Energy-based models

An **EBM** defines $$p_\theta(\mathbf{x}) = \frac{1}{Z(\theta)} e^{-E_\theta(\mathbf{x})}$$ with partition function
$$Z(\theta) = \int e^{-E_\theta(\mathbf{x})}\,d\mathbf{x}$$. Training often uses contrastive objectives or score matching
because $$Z(\theta)$$ is intractable in high dimensions.

## 4. VAEs, flows, and EBMs in practice

**VAEs** trade exact likelihood for fast amortized inference; blurry samples can result from Gaussian decoders and ELBO gap.

**Flows** give exact likelihoods but architectural constraints (invertibility, dimension preservation).

**EBMs** model unnormalized densities and connect naturally to **score functions**
$$\nabla_{\mathbf{x}} \log p(\mathbf{x}) = -\nabla_{\mathbf{x}} E(\mathbf{x})$$, foreshadowing diffusion training.

![VAE and flow sketches](/assets/figures/day06/pdf1_page000.png)
*Figure: encoder–decoder vs invertible layers.*

### 4.1 Design checklist

1. Do you need exact $$\log p(\mathbf{x})$$? → flows or autoregressive models.
2. Do you need fast latent codes? → VAE or VAE + diffusion in latent space (LDM).
3. Do you only need high-quality samples? → diffusion, flows, or GAN-style objectives.

### 4.2 GANs as implicit competitors

Generative adversarial networks optimize a minimax game between generator $$G_\theta$$ and discriminator $$D_\phi$$:

$$
\min_\theta \max_\phi \;
\mathbb{E}_{\mathbf{x}\sim p_{\mathrm{data}}}[\log D_\phi(\mathbf{x})]
+ \mathbb{E}_{\mathbf{z}\sim p_0}[\log(1 - D_\phi(G_\theta(\mathbf{z})))].
$$

There is no tractable global density unless combined with additional constraints; mode collapse and training
instability motivated diffusion and flow alternatives covered in days 7–8.

![Energy landscape](/assets/figures/day06/pdf1_page010.png)
*Figure: implicit vs explicit sampling geometry.*

## Checkpoint summary

- **MLE/MAP** fit parameters to data (with optional priors).
- **KL** measures how one distribution diverges from another; ELBO = log-likelihood minus posterior KL gap.
- **Explicit** models expose densities; **implicit** models expose samplers or scores.
- **VAE / flows / EBMs** are the three classical families we extend in days 7–10.
