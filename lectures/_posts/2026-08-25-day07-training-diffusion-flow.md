---
layout: post
title: Day 7 - Training Diffusion and Flow Models
image: /assets/img/lessons/day07.png
accent_image: 
  background: url('/assets/img/sampling_space.png') center/cover
  overlay: false
accent_color: '#ccc'
theme_color: '#ccc'
description: >
  Forward noising, score matching, and the flow-matching training objective.
invert_sidebar: true
---

# Day 7 - Training Diffusion and Flow Models

### [Slides](/assets/slides/day07.pdf)

### [Practical](/projects/day07-practical/)

### Optional reading for this lesson
- [MIT 6.S184 — Flow Matching and Diffusion (2026)](https://diffusion.csail.mit.edu/2026/index.html)
- [Song et al. — Score-Based Generative Modeling](https://arxiv.org/abs/2011.13456)
- [Lipman et al. — Flow Matching](https://arxiv.org/abs/2210.02747)
- [Complete reading list for Day 7](/publications/#day-7) (all resources for this lecture)


* toc
{:toc}

Diffusion and flow models learn **time-dependent vector fields** or **scores** that transport a simple prior
to the data distribution. This lecture focuses on **training objectives**, not sampling.

## 1. Forward (noising) process

> **Definition (forward SDE, informal).** A forward process gradually adds noise:
> $$d\mathbf{x}_t = \mathbf{f}(\mathbf{x}_t, t)\,dt + g(t)\,d\mathbf{w}_t$$, $$t \in [0, T]$$,
> with $$\mathbf{x}_0 \sim p_{\mathrm{data}}$$ and $$\mathbf{x}_T \approx \mathcal{N}(\mathbf{0}, \mathbf{I})$$.
{:.lead}

A common **variance-preserving (VP)** discrete schedule uses

$$
q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}\!\left(\mathbf{x}_t;\, \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0,\, (1-\bar{\alpha}_t)\mathbf{I}\right),
$$

with noise schedule $$\beta_t$$ and $$\bar{\alpha}_t = \prod_{s=1}^{t}(1-\beta_s)$$.

The forward kernel tells us how corrupted data looks at time $$t$$; learning reverses this corruption.

![Forward noising](/assets/figures/day07/pdf0_page000.png)
*Figure: data → noise continuum.*

### 1.1 Conditional means and scores

From the Gaussian kernel,

$$
\mathbb{E}[\mathbf{x}_t \mid \mathbf{x}_0] = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0,
\qquad
\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t \mid \mathbf{x}_0) = -\frac{\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0}{1-\bar{\alpha}_t}.
$$

Denoising networks often predict noise $$\boldsymbol{\epsilon}$$, score, or velocity depending on parameterization.

## 2. Score matching

> **Score function.** $$s(\mathbf{x}) = \nabla_{\mathbf{x}} \log p(\mathbf{x})$$ points toward high-density regions.
{:.lead}

**Denoising score matching (DSM):** train $$s_\theta(\mathbf{x}_t, t)$$ to match the score of the noisy distribution
$$q(\mathbf{x}_t)$$. With sampled pairs $$(\mathbf{x}_0, \mathbf{x}_t \sim q(\mathbf{x}_t\mid\mathbf{x}_0))$$,

$$
\mathcal{L}_{\mathrm{DSM}}(\theta) =
\mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_t}\left[
\lambda(t)\,\left\| s_\theta(\mathbf{x}_t, t) - \nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t \mid \mathbf{x}_0) \right\|_2^2
\right].
$$

Weight $$\lambda(t)$$ balances signal across noise levels. Predicting $$\boldsymbol{\epsilon}$$ is an equivalent
reparameterization widely used in DDPM implementations.

![Score matching](/assets/figures/day07/pdf0_page008.png)
*Figure: denoising target at each noise level.*

### 2.1 Connection to EBMs

If $$p(\mathbf{x}) \propto e^{-E(\mathbf{x})}$$, then $$s(\mathbf{x}) = -\nabla_{\mathbf{x}} E(\mathbf{x})$$.
Score-based diffusion avoids estimating the partition function $$Z$$ directly.

## 3. Flow matching objective

**Flow matching** learns a velocity field $$\mathbf{v}_\theta(\mathbf{x}, t)$$ such that the ODE

$$
\frac{d\mathbf{x}_t}{dt} = \mathbf{v}_\theta(\mathbf{x}_t, t)
$$

transports a prior $$p_0$$ to $$p_{\mathrm{data}}$$. Construct a **probability path**
$$p_t(\mathbf{x})$$ with marginals $$p_0$$ and $$p_1 = p_{\mathrm{data}}$$.

> **Conditional flow matching.** Sample $$\mathbf{x}_1 \sim p_{\mathrm{data}}$$, $$\mathbf{x}_0 \sim p_0$$,
> build a path $$\mathbf{x}_t$$ (e.g. linear interpolation $$\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$$),
> and regress $$\mathbf{v}_\theta(\mathbf{x}_t, t)$$ onto the path velocity $$\dot{\mathbf{x}}_t$$.
{:.lead}

A typical loss:

$$
\mathcal{L}_{\mathrm{FM}}(\theta) =
\mathbb{E}_{t \sim \mathcal{U}(0,1),\, \mathbf{x}_0,\, \mathbf{x}_1,\, \mathbf{x}_t}
\left\| \mathbf{v}_\theta(\mathbf{x}_t, t) - \dot{\mathbf{x}}_t \right\|_2^2.
$$

**Rectified flows** and **optimal transport** paths reduce curvature and improve sample efficiency at inference.

![Flow matching paths](/assets/figures/day07/pdf1_page010.png)
*Figure: coupling noise to data along a path.*

### 3.1 Diffusion as a special flow

Diffusion SDEs induce a family of marginals $$p_t$$; the **probability flow ODE** uses a velocity field derived from
the score so that marginals match the SDE. Training can therefore be viewed in a unified **flow / score** picture.

## 4. Training loop in practice

1. Sample minibatch $$\mathbf{x}_0 \sim p_{\mathrm{data}}$$.
2. Sample time $$t$$ (continuous or discrete index).
3. Form $$\mathbf{x}_t$$ via forward kernel or path sampler.
4. Compute loss (DSM, $$\boldsymbol{\epsilon}$$-prediction, or flow matching).
5. Backprop through U-Net / DiT backbone.

![Training pipeline](/assets/figures/day07/pdf1_page020.png)
*Figure: denoiser architecture and time conditioning.*

### 4.1 Engineering notes

- **Time embedding:** sinusoidal or learned embeddings injected into every block.
- **EMA weights:** exponential moving average of parameters stabilizes sampling metrics.
- **Class conditioning:** auxiliary label $$y$$ enters via embedding or cross-attention (day 8: guidance).

### 4.2 Loss weighting and continuous time

In continuous-time VP-SDEs, the DSM weight $$\lambda(t)$$ compensates for shrinking signal-to-noise ratio as
$$t \to T$$. A practical recipe:

$$
\lambda(t) \propto \mathbb{E}\left[\left\|\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t \mid \mathbf{x}_0)\right\|_2^2\right]^{-1}
$$

so each noise level contributes equally to gradients. Discrete DDPM uses uniform $$t \sim \mathcal{U}\{1,\ldots,T\}$$ with
optional importance sampling on hard timesteps.

### 4.3 Bridging diffusion and flow training

Given score $$s_\theta(\mathbf{x}, t) \approx \nabla_{\mathbf{x}} \log p_t(\mathbf{x})$$, the probability-flow velocity can be written

$$
\mathbf{v}^*(\mathbf{x}, t) = \mathbf{f}(\mathbf{x}, t) - \tfrac{1}{2} g(t)^2 s_\theta(\mathbf{x}, t),
$$

allowing a single network to support both DSM pretraining and flow-matching fine-tuning on rectified paths.

![Discrete vs continuous time](/assets/figures/day07/pdf0_page016.png)
*Figure: timestep schedules and signal-to-noise ratio.*

## Checkpoint summary

- **Forward process** defines $$q(\mathbf{x}_t\mid\mathbf{x}_0)$$ and noisy scores.
- **Score matching** trains $$\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$$ without normalized densities.
- **Flow matching** trains velocity fields along explicit paths between prior and data.
- Diffusion training is the discrete-time score-matching special case of this broader picture.
