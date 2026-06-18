---
layout: post
title: Day 7 - Score-Based Models, SDEs, and Flow Matching
image: /assets/img/sampling_space.png
accent_image: 
  background: url('/assets/img/sampling_space.png') center/cover
  overlay: false
accent_color: '#ccc'
theme_color: '#ccc'
description: >
  The continuous-time view of diffusion: score functions, the score SDE and probability-flow ODE, and flow matching.
invert_sidebar: true
---

# Day 7 - Score-Based Models, SDEs, and Flow Matching

### Optional reading for this lesson
- [The Principles of Diffusion Models](https://arxiv.org/abs/2510.21890), Ch. 3–6
- [Song et al. — Score-Based Generative Modeling through SDEs (2021)](https://arxiv.org/abs/2011.13456)
- [Lipman et al. — Flow Matching for Generative Modeling (2023)](https://arxiv.org/abs/2210.02747)
- [Interactive companion — The Principles of Diffusion Models](https://the-principles-of-diffusion-models.github.io/)

### [Slides](/assets/slides/day07.pdf)

### [Practical](/projects/day07-practical/)

Day 6 built diffusion from the variational, discrete-time DDPM viewpoint. Today we take the continuous-time view that unifies modern generative modeling. The central object is the **score** — the gradient of the log-density — which we can learn by denoising and use to sample. Letting the number of noising steps go to infinity turns the forward chain into a stochastic differential equation (SDE), whose time-reversal and deterministic *probability-flow ODE* give two ways to generate data. Flow matching arrives at the same place from a different door, regressing a velocity field directly. We finish by showing that DDPM, score-SDE, and flow matching are three views of one model.

* toc
{:toc}

## 1. The Score Function

### 1.1 The score as a vector field

> The **(Stein) score** of a density is the gradient of its log: $$\boldsymbol{s}(\boldsymbol{x}) = \nabla_{\boldsymbol{x}}\log p(\boldsymbol{x}).$$ It is a vector field that, at every point, points in the direction of steepest increase of probability.
{:.lead}

**Why this matters.** On Day 6 we trained a network to *denoise*. Today we reveal what that network is really learning — the score — and why the score is exactly the quantity you need to turn noise into data. Note this is **not** the "score" of maximum-likelihood statistics (the gradient w.r.t. *parameters*); here the gradient is w.r.t. the *input* $$\boldsymbol{x}$$.

![The score field of a distribution: arrows point toward regions of higher probability density (Principles Fig 3.2).](/assets/figures/day07/pdm_score_field.png)

Geometrically, the score is a map $$\boldsymbol{x}\mapsto\nabla\log p(\boldsymbol{x})$$ that, like a gravitational field, pulls points toward the modes of the distribution and is zero at stationary points. If we know this field, we can "flow" or "diffuse" random points into samples from $$p$$ — no normalized density required. Explore how the field relates to the density:


<div class="interactive-viz" markdown="0">
  <iframe src="https://the-principles-of-diffusion-models.github.io/assets/score_landscape.html" width="100%" height="760" loading="lazy" style="border:1px solid #ddd;border-radius:8px;" title="Score Landscape"></iframe>
  <p style="font-size:0.85em;color:#888;margin-top:4px;">Interactive: <strong>Score Landscape</strong> — <a href="https://the-principles-of-diffusion-models.github.io/assets/score_landscape.html" target="_blank" rel="noopener">open in new tab</a>. Source: <em>The Principles of Diffusion Models</em>.</p>
</div>


The local arrows tell you which way to nudge a sample to make it more probable; the global structure tells you where the mass is.


<div class="interactive-viz" markdown="0">
  <iframe src="https://the-principles-of-diffusion-models.github.io/assets/score_global_vs_local.html" width="100%" height="760" loading="lazy" style="border:1px solid #ddd;border-radius:8px;" title="Global vs Local View of the Score"></iframe>
  <p style="font-size:0.85em;color:#888;margin-top:4px;">Interactive: <strong>Global vs Local View of the Score</strong> — <a href="https://the-principles-of-diffusion-models.github.io/assets/score_global_vs_local.html" target="_blank" rel="noopener">open in new tab</a>. Source: <em>The Principles of Diffusion Models</em>.</p>
</div>


### 1.2 Energy-based models and the normalizer problem

> An **energy-based model (EBM)** writes $$p_\theta(\boldsymbol{x}) = \frac{e^{-E_\theta(\boldsymbol{x})}}{Z_\theta},\qquad Z_\theta = \int e^{-E_\theta(\boldsymbol{x})}\,\mathrm{d}\boldsymbol{x},$$ where the **partition function** $$Z_\theta$$ is generally intractable.
{:.lead}

![Training an energy-based model shapes the energy landscape so that data sits in low-energy basins (Principles Fig 3.1).](/assets/figures/day07/pdm_ebm_training.png)

EBMs are maximally flexible: any nonnegative function defines a distribution after normalization. The catch is precisely that normalization. Maximum-likelihood training needs $$\nabla_\theta\log p_\theta = -\nabla_\theta E_\theta - \nabla_\theta\log Z_\theta$$, and the second term involves an expectation under the model that requires expensive MCMC to estimate.

The escape route motivates the whole day: instead of modeling the normalized density, model its **score**. As we show next, the score is blind to $$Z_\theta$$.

### 1.3 Why the score sidesteps the normalizer

> Because the partition function is constant in $$\boldsymbol{x}$$, it disappears under the gradient: $$\nabla_{\boldsymbol{x}}\log p_\theta(\boldsymbol{x}) = -\nabla_{\boldsymbol{x}} E_\theta(\boldsymbol{x}).$$
{:.lead}

Take logs of the EBM and differentiate with respect to the input:

$$\log p_\theta(\boldsymbol{x}) = -E_\theta(\boldsymbol{x}) - \log Z_\theta \;\;\Longrightarrow\;\; \nabla_{\boldsymbol{x}}\log p_\theta(\boldsymbol{x}) = -\nabla_{\boldsymbol{x}} E_\theta(\boldsymbol{x}) - \underbrace{\nabla_{\boldsymbol{x}}\log Z_\theta}_{=\,\mathbf{0}}.$$

The term $$\log Z_\theta$$ is a number — it does not depend on $$\boldsymbol{x}$$ — so its gradient is exactly zero. The intractable obstacle simply **vanishes**. This is the key insight of score-based modeling: by working with $$\nabla_{\boldsymbol{x}}\log p$$ we learn the *shape* of the distribution (where mass concentrates and how it falls off) without ever computing how much total mass there is. The remaining question is how to fit a model to a score we cannot observe directly.

## 2. Learning the Score

### 2.1 Score matching

> **Score matching** fits $$\boldsymbol{s}_\theta(\boldsymbol{x})\approx\nabla\log p_{\text{data}}(\boldsymbol{x})$$ by minimizing the expected squared difference between the model and the true score.
{:.lead}

![Score matching learns a vector field that agrees with the data score across space (Principles Fig 3.4).](/assets/figures/day07/pdm_score_matching.png)

The obvious objective,

$$J_{\text{ESM}}(\theta) = \tfrac12\,\mathbb{E}_{p_{\text{data}}}\big\Vert \boldsymbol{s}_\theta(\boldsymbol{x}) - \nabla\log p_{\text{data}}(\boldsymbol{x})\big\Vert ^2,$$

is useless as written because it contains the unknown true score $$\nabla\log p_{\text{data}}$$. Hyvärinen's remarkable result is that this objective can be rewritten — via integration by parts — into an equivalent one that depends only on $$\boldsymbol{s}_\theta$$ and its derivatives. We derive it next.

### 2.2 Derivation: implicit score matching

> Integration by parts converts the intractable objective into $$J(\theta) = \mathbb{E}_{p_{\text{data}}}\!\left[\tfrac12\Vert \boldsymbol{s}_\theta(\boldsymbol{x})\Vert ^2 + \operatorname{tr}\!\big(\nabla_{\boldsymbol{x}}\boldsymbol{s}_\theta(\boldsymbol{x})\big)\right] + \text{const}.$$
{:.lead}

Expand the square in the explicit objective:

$$J_{\text{ESM}} = \mathbb{E}_{p}\Big[\tfrac12\Vert \boldsymbol{s}_\theta\Vert ^2 - \textcolor{teal}{\boldsymbol{s}_\theta^{\top}\nabla\log p} + \tfrac12\Vert \nabla\log p\Vert ^2\Big].$$

The last term does not involve $$\theta$$ (a constant). The middle (cross) term is the problem; rewrite it using $$p\,\nabla\log p = \nabla p$$:

$$\textcolor{teal}{\mathbb{E}_{p}\big[\boldsymbol{s}_\theta^{\top}\nabla\log p\big]} = \int p(\boldsymbol{x})\,\boldsymbol{s}_\theta(\boldsymbol{x})^{\top}\frac{\nabla p(\boldsymbol{x})}{p(\boldsymbol{x})}\,\mathrm{d}\boldsymbol{x} = \int \boldsymbol{s}_\theta(\boldsymbol{x})^{\top}\nabla p(\boldsymbol{x})\,\mathrm{d}\boldsymbol{x}.$$

Now integrate by parts (component-wise), assuming $$p(\boldsymbol{x})\,\boldsymbol{s}_\theta(\boldsymbol{x})\to\mathbf{0}$$ as $$\Vert \boldsymbol{x}\Vert \to\infty$$ so the boundary term drops:

$$\int \boldsymbol{s}_\theta^{\top}\nabla p\,\mathrm{d}\boldsymbol{x} = -\int p\,(\nabla\cdot\boldsymbol{s}_\theta)\,\mathrm{d}\boldsymbol{x} = -\,\mathbb{E}_{p}\big[\operatorname{tr}(\nabla_{\boldsymbol{x}}\boldsymbol{s}_\theta)\big].$$

Substituting back gives the **implicit score matching** objective

$$J(\theta) = \mathbb{E}_{p_{\text{data}}}\!\left[\tfrac12\Vert \boldsymbol{s}_\theta(\boldsymbol{x})\Vert ^2 + \textcolor{purple}{\operatorname{tr}\!\big(\nabla_{\boldsymbol{x}}\boldsymbol{s}_\theta(\boldsymbol{x})\big)}\right] + \text{const},$$

which no longer references the true score. The price is the **trace of the Jacobian** $$\operatorname{tr}(\nabla\boldsymbol{s}_\theta)$$ — a sum of $$d$$ second derivatives that costs $$d$$ backward passes in $$d$$ dimensions. For images ($$d\sim10^5$$) this is prohibitive, which is why we turn to denoising.

### 2.3 Denoising score matching

> **Denoising score matching (DSM)** perturbs the data with Gaussian noise and fits the score of the *noisy* density. The target is then available in closed form: for $$\boldsymbol{x}_t = \boldsymbol{x}_0 + \sigma\boldsymbol{\epsilon}$$, $$\nabla_{\boldsymbol{x}_t}\log p(\boldsymbol{x}_t\mid\boldsymbol{x}_0) = -\frac{\boldsymbol{x}_t-\boldsymbol{x}_0}{\sigma^2} = -\frac{\boldsymbol{\epsilon}}{\sigma}.$$
{:.lead}

![Denoising score matching: the score of the noised distribution points from the noisy sample back toward the clean data (Principles Fig 4.6).](/assets/figures/day07/pdm_dsm_trick.png)

The perturbation kernel is Gaussian, $$p(\boldsymbol{x}_t\mid\boldsymbol{x}_0)=\mathcal{N}(\boldsymbol{x}_0,\sigma^2 I)$$, whose log-gradient we can write down exactly (differentiate $$-\Vert \boldsymbol{x}_t-\boldsymbol{x}_0\Vert ^2/2\sigma^2$$). Vincent's identity shows that matching the *conditional* score also matches the *marginal* noisy score, so the trainable objective is just a regression:

$$J_{\text{DSM}}(\theta) = \mathbb{E}_{\boldsymbol{x}_0,\boldsymbol{\epsilon}}\left\Vert \boldsymbol{s}_\theta(\boldsymbol{x}_t) + \frac{\boldsymbol{x}_t-\boldsymbol{x}_0}{\sigma^2}\right\Vert ^2 = \mathbb{E}\left\Vert \boldsymbol{s}_\theta(\boldsymbol{x}_t) + \frac{\boldsymbol{\epsilon}}{\sigma}\right\Vert ^2.$$

No Jacobian trace, no MCMC — just predict the (scaled) noise that was added. This is **exactly** the DDPM noise-prediction loss of Day 6 in disguise: learning the score *is* learning to denoise, with $$\boldsymbol{s}_\theta = -\boldsymbol{\epsilon}_\theta/\sigma$$. Build intuition for the equivalence:


<div class="interactive-viz" markdown="0">
  <iframe src="https://the-principles-of-diffusion-models.github.io/assets/denoising_score_matching.html" width="100%" height="860" loading="lazy" style="border:1px solid #ddd;border-radius:8px;" title="Denoising Score Matching"></iframe>
  <p style="font-size:0.85em;color:#888;margin-top:4px;">Interactive: <strong>Denoising Score Matching</strong> — <a href="https://the-principles-of-diffusion-models.github.io/assets/denoising_score_matching.html" target="_blank" rel="noopener">open in new tab</a>. Source: <em>The Principles of Diffusion Models</em>.</p>
</div>


### 2.4 Multiple noise scales (NCSN)

> A single noise level cannot teach the score everywhere. **Noise-Conditional Score Networks (NCSN)** train one network $$\boldsymbol{s}_\theta(\boldsymbol{x},\sigma)$$ across a range of noise scales $$\sigma$$.
{:.lead}

![Training across many noise scales covers both the data manifold and the empty regions far from it (Principles Fig 3.7).](/assets/figures/day07/pdm_ncsn.png)

The difficulty with DSM at a single $$\sigma$$ is a coverage problem:

- **Small $$\sigma$$**: the noisy samples hug the data manifold, so the score is accurate *near* the data but never trained in the vast empty regions where sampling actually starts.
- **Large $$\sigma$$**: noise fills the whole space (good coverage) but smears out the fine structure of $$p_{\text{data}}$$.

The fix is to use **many** noise scales and a single network conditioned on $$\sigma$$. At sampling time we **anneal** from large $$\sigma$$ (which moves us into the right region from anywhere) down to small $$\sigma$$ (which refines detail). Taking the number of scales to infinity is precisely the continuous-time SDE we build below.

## 3. Sampling with the Score

### 3.1 Langevin dynamics

> **Langevin dynamics** turns a score into samples by repeatedly stepping uphill in log-density and injecting noise: $$\boldsymbol{x}_{k+1} = \boldsymbol{x}_k + \tau\,\boldsymbol{s}_\theta(\boldsymbol{x}_k) + \sqrt{2\tau}\,\boldsymbol{z}_k,\quad \boldsymbol{z}_k\sim\mathcal{N}(\mathbf{0},I).$$
{:.lead}

![Langevin sampling: noisy gradient ascent on the log-density converges to samples from $$p$$ (Principles Fig 3.3).](/assets/figures/day07/pdm_langevin.png)

The update has two parts: a **drift** $$\tau\,\boldsymbol{s}_\theta$$ that climbs toward high-probability regions, and a **diffusion** $$\sqrt{2\tau}\,\boldsymbol{z}_k$$ that keeps the chain exploring rather than collapsing onto a single mode. Under mild conditions, as $$\tau\to0$$ and the number of steps grows, the iterates converge in distribution to $$p$$ — the noise is exactly calibrated so that $$p$$ is the **stationary distribution**.

Pure Langevin mixes slowly between far-apart modes. Combining it with the NCSN annealing schedule — run Langevin at large $$\sigma$$, then progressively smaller — gives **annealed Langevin dynamics**, the original score-based sampler. The continuous-time limit of this annealed process is the reverse SDE we meet next.

## 4. The Continuous-Time View

### 4.1 The forward SDE

> Taking the noising chain to infinitely many infinitesimal steps yields a **stochastic differential equation** $$\mathrm{d}\boldsymbol{x} = \boldsymbol{f}(\boldsymbol{x},t)\,\mathrm{d}t + g(t)\,\mathrm{d}\boldsymbol{w},$$ with **drift** $$\boldsymbol{f}$$ and **diffusion** $$g$$, driven by a Wiener process $$\boldsymbol{w}$$.
{:.lead}

![The forward process as a continuous-time SDE: a 1-D density smoothly transported from data to noise (Principles Fig 4.3).](/assets/figures/day07/pdm_forward_1d.png)

The discrete DDPM update $$\boldsymbol{x}_t=\sqrt{1-\beta_t}\,\boldsymbol{x}_{t-1}+\sqrt{\beta_t}\,\boldsymbol{\epsilon}$$ is the Euler–Maruyama discretization (Day 1) of an SDE. Two standard families:

- **Variance-Preserving (VP) SDE** — the continuous limit of DDPM, with $$\boldsymbol{f}=-\tfrac12\beta(t)\boldsymbol{x}$$ and $$g=\sqrt{\beta(t)}$$; keeps total variance bounded.
- **Variance-Exploding (VE) SDE** — the continuous limit of NCSN, with $$\boldsymbol{f}=\mathbf{0}$$ and growing $$g$$; variance blows up as noise is added.

Crucially the marginal at each time stays Gaussian in the conditioning variable, $$p_t(\boldsymbol{x}_t\mid\boldsymbol{x}_0)=\mathcal{N}(\alpha_t\boldsymbol{x}_0,\sigma_t^2 I)$$, so the unified $$\boldsymbol{x}_t=\alpha_t\boldsymbol{x}_0+\sigma_t\boldsymbol{\epsilon}$$ notation from Day 6 carries over unchanged — the SDE just describes how $$\alpha_t,\sigma_t$$ evolve.

### 4.2 The time-dependent score

> Each forward time defines a distribution $$p_t$$, so the score becomes **time-dependent**: $$\boldsymbol{s}(\boldsymbol{x},t) = \nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x}).$$ A single network $$\boldsymbol{s}_\theta(\boldsymbol{x},t)$$ learns it for all $$t$$.
{:.lead}

![The score landscape changes with time: smooth and easy at high noise, sharp and detailed at low noise (Principles Fig 4.1).](/assets/figures/day07/pdm_score_landscape.png)

This is the NCSN idea recast in continuous time: instead of a discrete set of noise levels, there is a continuum $$t\in[0,T]$$, and the network is conditioned on $$t$$. At large $$t$$ the density $$p_t$$ is a broad, smooth Gaussian whose score is easy to learn; at small $$t$$ it concentrates on the data manifold and the score becomes sharp and hard. Training is denoising score matching at every time:

$$J(\theta) = \mathbb{E}_{t}\,\lambda(t)\,\mathbb{E}_{\boldsymbol{x}_0,\boldsymbol{\epsilon}}\left\Vert \boldsymbol{s}_\theta(\boldsymbol{x}_t,t) + \frac{\boldsymbol{\epsilon}}{\sigma_t}\right\Vert ^2,$$

with a time-weighting $$\lambda(t)$$. One network, trained once, gives the score at every noise level — everything we need to run the dynamics below.

### 4.3 The reverse SDE and the probability-flow ODE

> Anderson's theorem gives a **reverse-time SDE** that undoes the forward process, and there is a deterministic **probability-flow ODE** with the *same marginals*: $$\mathrm{d}\boldsymbol{x} = \big[\boldsymbol{f}-g^2\nabla\log p_t\big]\mathrm{d}t + g\,\mathrm{d}\bar{\boldsymbol{w}}\quad\text{vs}\quad \mathrm{d}\boldsymbol{x} = \big[\boldsymbol{f}-\tfrac12 g^2\nabla\log p_t\big]\mathrm{d}t.$$
{:.lead}

![Three dynamics with identical time marginals: the forward SDE, the reverse SDE, and the probability-flow ODE (Principles Fig 4.5).](/assets/figures/day07/pdm_three_dynamics.png)

This is the payoff of the whole machinery. Running the **reverse SDE** from $$\boldsymbol{x}_T\sim\mathcal{N}(\mathbf{0},I)$$ back to $$t=0$$ produces a sample from $$p_{\text{data}}$$ — a stochastic sampler that generalizes annealed Langevin and DDPM ancestral sampling. The only unknown in the reverse drift is the score, which we have learned.

The **probability-flow ODE** is a deterministic process that, remarkably, has the *same marginal distribution* $$p_t$$ at every time. It gives:

- **Deterministic sampling** — fix the initial noise and you fix the output (useful for editing, interpolation, reproducibility).
- **Exact likelihoods** — being an ODE (a continuous normalizing flow), you can compute $$\log p_0(\boldsymbol{x})$$ via the instantaneous change-of-variables.
- **Fast solvers** — ODEs admit high-order integrators, the focus of Day 8.

Compare all three dynamics side by side:


<div class="interactive-viz" markdown="0">
  <iframe src="https://the-principles-of-diffusion-models.github.io/assets/score_sde_three_dynamics.html" width="100%" height="780" loading="lazy" style="border:1px solid #ddd;border-radius:8px;" title="Three Dynamics: Forward SDE, Reverse SDE, PF-ODE"></iframe>
  <p style="font-size:0.85em;color:#888;margin-top:4px;">Interactive: <strong>Three Dynamics: Forward SDE, Reverse SDE, PF-ODE</strong> — <a href="https://the-principles-of-diffusion-models.github.io/assets/score_sde_three_dynamics.html" target="_blank" rel="noopener">open in new tab</a>. Source: <em>The Principles of Diffusion Models</em>.</p>
</div>


### 4.4 Derivation: the denoising score matching objective

> The marginal score equals the conditional expectation of the (closed-form) conditional score: $$\nabla\log p_t(\boldsymbol{x}_t) = \mathbb{E}\big[\nabla\log p_t(\boldsymbol{x}_t\mid\boldsymbol{x}_0)\,\big\vert \,\boldsymbol{x}_t\big].$$ Hence regressing on the conditional score trains the marginal score.
{:.lead}

Write the marginal as $$p_t(\boldsymbol{x}_t)=\int p_t(\boldsymbol{x}_t\mid\boldsymbol{x}_0)\,p_{\text{data}}(\boldsymbol{x}_0)\,\mathrm{d}\boldsymbol{x}_0$$ and differentiate its log:

$$\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t) = \frac{\nabla_{\boldsymbol{x}_t} p_t(\boldsymbol{x}_t)}{p_t(\boldsymbol{x}_t)} = \frac{\int \textcolor{teal}{\nabla_{\boldsymbol{x}_t} p_t(\boldsymbol{x}_t\mid\boldsymbol{x}_0)}\,p_{\text{data}}(\boldsymbol{x}_0)\,\mathrm{d}\boldsymbol{x}_0}{p_t(\boldsymbol{x}_t)}.$$

Use $$\nabla p_t(\cdot\mid\boldsymbol{x}_0)=p_t(\cdot\mid\boldsymbol{x}_0)\,\nabla\log p_t(\cdot\mid\boldsymbol{x}_0)$$ in the numerator and recognize the posterior $$p_t(\boldsymbol{x}_0\mid\boldsymbol{x}_t)=p_t(\boldsymbol{x}_t\mid\boldsymbol{x}_0)p_{\text{data}}(\boldsymbol{x}_0)/p_t(\boldsymbol{x}_t)$$:

$$\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t) = \int p_t(\boldsymbol{x}_0\mid\boldsymbol{x}_t)\,\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t\mid\boldsymbol{x}_0)\,\mathrm{d}\boldsymbol{x}_0 = \textcolor{purple}{\mathbb{E}_{\boldsymbol{x}_0\mid\boldsymbol{x}_t}\big[\nabla\log p_t(\boldsymbol{x}_t\mid\boldsymbol{x}_0)\big]}.$$

So the marginal score is the conditional-mean of a quantity we know in closed form. Since the minimizer of a squared-error regression is exactly the conditional mean, regressing $$\boldsymbol{s}_\theta(\boldsymbol{x}_t,t)$$ onto the **conditional** score trains it to equal the **marginal** score. With the Gaussian kernel, $$\nabla\log p_t(\boldsymbol{x}_t\mid\boldsymbol{x}_0) = -(\boldsymbol{x}_t-\alpha_t\boldsymbol{x}_0)/\sigma_t^2 = -\boldsymbol{\epsilon}/\sigma_t$$, recovering the DSM/noise-prediction loss — and proving rigorously that **predicting the score is predicting the noise**. This connects directly to **Tweedie's formula**, $$\mathbb{E}[\boldsymbol{x}_0\mid\boldsymbol{x}_t] = (\boldsymbol{x}_t+\sigma_t^2\,\nabla\log p_t(\boldsymbol{x}_t))/\alpha_t$$, the optimal denoiser.

## 5. Flow Matching

### 5.1 Continuous normalizing flows and velocity fields

> A **continuous normalizing flow (CNF)** transports samples by an ODE $$\mathrm{d}\boldsymbol{x} = \boldsymbol{v}_\theta(\boldsymbol{x},t)\,\mathrm{d}t,$$ whose **velocity field** $$\boldsymbol{v}$$ moves a base distribution into the data distribution as $$t$$ goes from $$0$$ to $$1$$.
{:.lead}

![A normalizing flow transports a simple base density to the data density along a learned path (Principles Fig 5.2).](/assets/figures/day07/pdm_nf.png)

If samples move with velocity $$\boldsymbol{v}$$, their density evolves by the **continuity equation** $$\partial_t p_t + \nabla\cdot(p_t\,\boldsymbol{v}_t)=0$$ (conservation of probability mass). Classical CNFs trained this velocity by maximizing likelihood, which requires integrating the ODE — and its divergence — during training. That is slow and unstable.

**Flow matching** asks a simpler question: can we *regress* the velocity field directly, without simulating the ODE? The answer is yes, provided we choose the paths cleverly.

### 5.2 Conditional flow matching

> **Conditional flow matching (CFM)** specifies a simple per-example path from noise to a data point and regresses the network onto its known velocity. For the linear path $$\boldsymbol{x}_t=(1-t)\boldsymbol{x}_0+t\,\boldsymbol{x}_1$$, the velocity is the constant $$\boldsymbol{u}_t=\boldsymbol{x}_1-\boldsymbol{x}_0.$$
{:.lead}

![A conditional transition path interpolates a single noise sample to a single data sample; its velocity is known in closed form (Principles Fig 5.5).](/assets/figures/day07/pdm_cond_transition.png)

Pick a base sample $$\boldsymbol{x}_0\sim\mathcal{N}(\mathbf{0},I)$$ and a data sample $$\boldsymbol{x}_1\sim p_{\text{data}}$$, and connect them with a simple path — for the straight line, $$\boldsymbol{x}_t=(1-t)\boldsymbol{x}_0+t\boldsymbol{x}_1$$, the velocity is simply $$\dot{\boldsymbol{x}}_t=\boldsymbol{x}_1-\boldsymbol{x}_0$$. The objective is a plain regression:

$$J_{\text{CFM}}(\theta) = \mathbb{E}_{t,\boldsymbol{x}_0,\boldsymbol{x}_1}\big\Vert \boldsymbol{v}_\theta(\boldsymbol{x}_t,t) - (\boldsymbol{x}_1-\boldsymbol{x}_0)\big\Vert ^2.$$

No ODE simulation during training, no divergence term — just sample a pair, sample a time, and regress. This is the appeal of flow matching: the stability of a supervised regression with the flexibility of a CNF.

### 5.3 Conditional versus marginal velocity

> Although we regress on per-sample (conditional) velocities, the network learns the **marginal** velocity, which is their conditional average through each point: $$\boldsymbol{v}(\boldsymbol{x},t) = \mathbb{E}\big[\boldsymbol{u}_t(\boldsymbol{x}\mid\boldsymbol{x}_1)\,\big\vert \,\boldsymbol{x}_t=\boldsymbol{x}\big].$$
{:.lead}

![Many conditional paths cross any given point; the marginal velocity is their average, and that is what the network learns (Principles Fig 5.6).](/assets/figures/day07/pdm_cond_vs_marginal.png)

A subtlety: many different $$(\boldsymbol{x}_0,\boldsymbol{x}_1)$$ pairs route their straight-line paths through the *same* point $$\boldsymbol{x}_t=\boldsymbol{x}$$, each with its own velocity. The single-valued field the network must learn is their **average**. Because the minimizer of the squared-error CFM loss at each point is the conditional mean of the target, the network automatically converges to this marginal velocity:

$$\nabla_\theta J_{\text{CFM}} = \nabla_\theta J_{\text{FM}} \quad\Longrightarrow\quad \boldsymbol{v}_\theta^\star(\boldsymbol{x},t) = \mathbb{E}\big[\boldsymbol{u}_t(\boldsymbol{x}\mid\boldsymbol{x}_1)\mid\boldsymbol{x}_t=\boldsymbol{x}\big].$$

The two objectives differ only by a constant, so optimizing the tractable conditional one is equivalent to optimizing the intractable marginal one. This is the exact analogue of the score-matching argument above. See how conditional and marginal fields relate:


<div class="interactive-viz" markdown="0">
  <iframe src="https://the-principles-of-diffusion-models.github.io/assets/conditional_vs_marginal_flow_matching_velocity.html" width="100%" height="860" loading="lazy" style="border:1px solid #ddd;border-radius:8px;" title="Conditional vs Marginal Velocity Fields"></iframe>
  <p style="font-size:0.85em;color:#888;margin-top:4px;">Interactive: <strong>Conditional vs Marginal Velocity Fields</strong> — <a href="https://the-principles-of-diffusion-models.github.io/assets/conditional_vs_marginal_flow_matching_velocity.html" target="_blank" rel="noopener">open in new tab</a>. Source: <em>The Principles of Diffusion Models</em>.</p>
</div>



<div class="interactive-viz" markdown="0">
  <iframe src="https://the-principles-of-diffusion-models.github.io/assets/conditional_vs_marginal_paths.html" width="100%" height="780" loading="lazy" style="border:1px solid #ddd;border-radius:8px;" title="Conditional vs Marginal Paths"></iframe>
  <p style="font-size:0.85em;color:#888;margin-top:4px;">Interactive: <strong>Conditional vs Marginal Paths</strong> — <a href="https://the-principles-of-diffusion-models.github.io/assets/conditional_vs_marginal_paths.html" target="_blank" rel="noopener">open in new tab</a>. Source: <em>The Principles of Diffusion Models</em>.</p>
</div>


### 5.4 Rectified flow and reflow

> **Rectified flow** uses straight-line conditional paths; even so, the learned **marginal** trajectories are generally curved. **Reflow** retrains on the model's own (noise, sample) pairs to straighten them, enabling few-step sampling.
{:.lead}

![Marginal ODE trajectories are curved even when conditional paths are straight, which forces many integration steps (Principles Fig 5.9).](/assets/figures/day07/pdm_curved_paths.png)

The number of function evaluations a deterministic sampler needs is governed by how *curved* the marginal trajectories are: a straight path can be integrated exactly in one Euler step, while a curved one needs many. Reflow is an elegant fix: generate pairs $$(\boldsymbol{x}_0,\boldsymbol{x}_1=\text{ODE}(\boldsymbol{x}_0))$$ from the trained model, then *retrain* flow matching using those coupled pairs as the new conditional endpoints. The resulting flow is **straighter**, and iterating drives it toward straight marginal paths that can be sampled in very few — even one — steps. This is one of the routes to the fast samplers of Day 8.

## 6. One Model, Many Views

### 6.1 Four equivalent parameterizations

> Given the forward rule $$\boldsymbol{x}_t=\alpha_t\boldsymbol{x}_0+\sigma_t\boldsymbol{\epsilon}$$, predicting the **noise** $$\boldsymbol{\epsilon}$$, the **data** $$\boldsymbol{x}_0$$, the **score** $$\boldsymbol{s}$$, or the **velocity** $$\boldsymbol{v}$$ are all linearly interchangeable.
{:.lead}

![The four common prediction targets are equivalent reparameterizations of the same network output (Principles Fig 6.1).](/assets/figures/day07/pdm_param_equiv.png)

From the single relation $$\boldsymbol{x}_t=\alpha_t\boldsymbol{x}_0+\sigma_t\boldsymbol{\epsilon}$$ we can solve for any target in terms of any other:

$$\boldsymbol{s}(\boldsymbol{x}_t,t) = -\frac{\boldsymbol{\epsilon}}{\sigma_t},\qquad \boldsymbol{x}_0 = \frac{\boldsymbol{x}_t-\sigma_t\boldsymbol{\epsilon}}{\alpha_t},\qquad \boldsymbol{v} = \dot\alpha_t\,\boldsymbol{x}_0 + \dot\sigma_t\,\boldsymbol{\epsilon}.$$

So a single network can be trained to output any one of them, and the others follow by algebra. The choice is purely practical: **$$\boldsymbol{\epsilon}$$-prediction** is stable at high noise, **$$\boldsymbol{x}_0$$-prediction** at low noise, and **$$\boldsymbol{v}$$-prediction** balances both and is popular for distillation. Explore the equivalence:


<div class="interactive-viz" markdown="0">
  <iframe src="https://the-principles-of-diffusion-models.github.io/assets/four_predictions.html" width="100%" height="800" loading="lazy" style="border:1px solid #ddd;border-radius:8px;" title="Four Prediction Parameterizations"></iframe>
  <p style="font-size:0.85em;color:#888;margin-top:4px;">Interactive: <strong>Four Prediction Parameterizations</strong> — <a href="https://the-principles-of-diffusion-models.github.io/assets/four_predictions.html" target="_blank" rel="noopener">open in new tab</a>. Source: <em>The Principles of Diffusion Models</em>.</p>
</div>



<div class="interactive-viz" markdown="0">
  <iframe src="https://the-principles-of-diffusion-models.github.io/assets/ddpm_prediction_equiv.html" width="100%" height="760" loading="lazy" style="border:1px solid #ddd;border-radius:8px;" title="DDPM Prediction Equivalences"></iframe>
  <p style="font-size:0.85em;color:#888;margin-top:4px;">Interactive: <strong>DDPM Prediction Equivalences</strong> — <a href="https://the-principles-of-diffusion-models.github.io/assets/ddpm_prediction_equiv.html" target="_blank" rel="noopener">open in new tab</a>. Source: <em>The Principles of Diffusion Models</em>.</p>
</div>


### 6.2 The unified picture

> The variational (DDPM), score-based SDE, and flow-matching views all learn the **same** time-indexed field over the same family of marginals $$p_t$$, and differ only in parameterization and sampler.
{:.lead}

![The variational, score-SDE, and flow-matching formulations are three views of one underlying model (Principles Fig 6.2).](/assets/figures/day07/pdm_unified.png)

Stepping back, every approach this week shares the same skeleton:

1. a **forward process** $$\boldsymbol{x}_t=\alpha_t\boldsymbol{x}_0+\sigma_t\boldsymbol{\epsilon}$$ that interpolates data and noise;
2. a **network** trained by a denoising/regression objective to predict noise / data / score / velocity (all equivalent);
3. a **sampler** that runs a dynamics backward — ancestral (DDPM), annealed Langevin, the reverse SDE, or the probability-flow ODE.

DDPM, score-SDE, and flow matching are not competitors but different coordinates on the same object. This unification is what makes the field so powerful: an advance in one view (a better objective, a better sampler) transfers immediately to the others. **Day 8** exploits exactly this, using the deterministic ODE view to build guidance, high-order solvers, and few-step samplers.

## Checkpoint summary

Before moving to the practical, confirm you can:

- Define the score and explain why it sidesteps the intractable normalizer of an energy-based model.
- Derive implicit score matching by integration by parts and say why the Jacobian-trace term is costly.
- State the denoising-score-matching identity and explain why learning the score equals learning to denoise.
- Write the Langevin update and explain the roles of the drift and noise terms.
- Explain the forward SDE, the reverse SDE, and the probability-flow ODE, and what each is good for.
- Derive that the marginal score is the conditional mean of the (Gaussian) conditional score.
- Set up the conditional flow-matching loss and argue it learns the marginal velocity field.
- Convert between the noise, data, score, and velocity parameterizations and justify the unified view.
