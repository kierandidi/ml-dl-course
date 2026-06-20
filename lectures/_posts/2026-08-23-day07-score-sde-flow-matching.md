---
layout: post
title: Day 7 - Score-Based Models, SDEs, and Flow Matching
image: /assets/img/lessons/day07.png
accent_image: 
  background: url('/assets/img/lessons/day07.png') center/cover
  overlay: false
accent_color: '#ccc'
theme_color: '#ccc'
description: >
  The continuous-time view of diffusion: score functions, the score SDE and probability-flow ODE, and flow matching.
invert_sidebar: true
---

# Day 7 - Score-Based Models, SDEs, and Flow Matching

### Optional reading for this lesson
- [The Principles of Diffusion Models](https://arxiv.org/abs/2510.21890), Ch. 3–6; Appendices B–D (optional deep dives in notes)
- [Song et al. — Score-Based Generative Modeling through SDEs (2021)](https://arxiv.org/abs/2011.13456)
- [Lipman et al. — Flow Matching for Generative Modeling (2023)](https://arxiv.org/abs/2210.02747)
- [Interactive companion — The Principles of Diffusion Models](https://the-principles-of-diffusion-models.github.io/)
- [SDE course — Lesson 2: FPE, time reversal (Nelson/Anderson), DSM proof, PF-ODE](/material/sde-course/) (local notes; see also [Generative Modelling with SDEs](https://kierandidi.github.io/))
- [Ludwig Winkler — Simple sketch of the reverse SDE](https://ludwigwinkler.github.io/blog/SimpleReverseSDE/)

### [Slides](/assets/slides/day07.pdf)

### [Exercise](/projects/day07-practical/)

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

### 4.3 Heuristic: reversing an SDE

> To reverse the forward SDE, factor the joint density with the chain rule and approximate forward transitions by Euler–Maruyama; a Taylor expansion of $$\log p_t$$ reveals the **score** $$\boldsymbol{s}=\nabla\log p_t$$ in the reverse drift $$\boldsymbol{f}-g^2\boldsymbol{s}$$.
{:.lead}

The forward SDE (Principles notation)

$$\mathrm{d}\boldsymbol{x} = \boldsymbol{f}(\boldsymbol{x},t)\,\mathrm{d}t + g(t)\,\mathrm{d}\boldsymbol{w}$$

has Euler–Maruyama transitions $$p_{t+\delta\mid t}(\boldsymbol{y}\mid\boldsymbol{x})=\mathcal{N}(\boldsymbol{y}\mid\boldsymbol{x}+\boldsymbol{f}\delta,\,\delta\,g^2 I)$$. The chain rule gives $$p_{t\mid t+\delta}(\boldsymbol{x}\mid\boldsymbol{y})\propto p_{t+\delta\mid t}(\boldsymbol{y}\mid\boldsymbol{x})\,p_t(\boldsymbol{x})/p_{t+\delta}(\boldsymbol{y})$$; Taylor-expanding $$\log p_t$$ around $$\boldsymbol{y}$$ and **completing the square** in the reverse kernel yields drift $$\boldsymbol{f}(\boldsymbol{x},t)-g(t)^2\,\boldsymbol{s}(\boldsymbol{x},t)$$ — the same score we train with DSM.

At sampling time replace $$\boldsymbol{s}$$ with $$\boldsymbol{s}_\theta(\boldsymbol{x},t)$$. The expanded step-by-step derivation (Winkler; SDE course §2.1) is in the optional block below.

<details class="optional-derivation" markdown="1">
<summary><strong>Heuristic reverse SDE (chain rule + Euler–Maruyama + Taylor)</strong> (optional — click to expand)</summary>

**Setup (Principles / Day 6–7 notation).** Forward SDE

$$\mathrm{d}\boldsymbol{x} = \boldsymbol{f}(\boldsymbol{x},t)\,\mathrm{d}t + g(t)\,\mathrm{d}\boldsymbol{w}, \qquad \boldsymbol{s}(\boldsymbol{x},t)=\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x}).$$

Discretise with Euler–Maruyama over $$\delta>0$$: if $$\boldsymbol{x}$$ is the state at time $$t$$ and $$\boldsymbol{y}$$ at $$t+\delta$$,

$$\boldsymbol{y} \approx \boldsymbol{x} + \boldsymbol{f}(\boldsymbol{x},t)\,\delta + g(t)\sqrt{\delta}\,\boldsymbol{\epsilon}, \qquad \boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},I).$$

Hence the **forward transition kernel** is Gaussian:

$$p_{t+\delta\mid t}(\boldsymbol{y}\mid\boldsymbol{x}) = \mathcal{N}\!\Big(\boldsymbol{y}\;\Big\vert \;\boldsymbol{x}+\boldsymbol{f}(\boldsymbol{x},t)\,\delta,\;\delta\,g(t)^2 I\Big).$$

**Step 1 — chain rule.** For exact transition densities,

$$p_{t,t+\delta}(\boldsymbol{x},\boldsymbol{y}) = p_{t+\delta\mid t}(\boldsymbol{y}\mid\boldsymbol{x})\,p_t(\boldsymbol{x}) = p_{t\mid t+\delta}(\boldsymbol{x}\mid\boldsymbol{y})\,p_{t+\delta}(\boldsymbol{y}).$$

Rearranging,

$$p_{t\mid t+\delta}(\boldsymbol{x}\mid\boldsymbol{y}) = p_{t+\delta\mid t}(\boldsymbol{y}\mid\boldsymbol{x})\,\frac{p_t(\boldsymbol{x})}{p_{t+\delta}(\boldsymbol{y})}.$$

**Step 2 — Taylor expansion of the log-density.** Expand $$\log p_t$$ around $$\boldsymbol{y}$$ (Winkler / SDE course §2.1):

$$\log p_t(\boldsymbol{x}) - \log p_{t+\delta}(\boldsymbol{y}) = (\boldsymbol{x}-\boldsymbol{y})^{\top}\nabla_{\boldsymbol{y}}\log p_t(\boldsymbol{y}) + \mathcal{O}(\Vert \boldsymbol{x}-\boldsymbol{y}\Vert ^2).$$

Under a standard Lipschitz bound on $$\log p_t - \log p_{t+\delta}$$, the $$p_{t+\delta}(\boldsymbol{y})$$ factor in the denominator can be absorbed into the remainder as $$\mathcal{O}(\delta^2)$$ when $$\delta\to 0$$. Thus

$$\frac{p_t(\boldsymbol{x})}{p_{t+\delta}(\boldsymbol{y})} \approx \exp\big((\boldsymbol{x}-\boldsymbol{y})^{\top}\boldsymbol{s}(\boldsymbol{y},t) + \mathcal{O}(\delta^2)\big).$$

**Step 3 — complete the square.** Multiply the forward Gaussian kernel by this exponential. Writing $$\boldsymbol{\mu}_+ = \boldsymbol{x}+\boldsymbol{f}(\boldsymbol{x},t)\delta$$ and $$\sigma^2 = \delta\,g(t)^2$$,

$$p_{t\mid t+\delta}(\boldsymbol{x}\mid\boldsymbol{y}) \propto \exp\!\Big(-\frac{\Vert \boldsymbol{x}-\boldsymbol{\mu}_+\Vert ^2}{2\sigma^2} + (\boldsymbol{x}-\boldsymbol{y})^{\top}\boldsymbol{s}(\boldsymbol{y},t)\Big).$$

Complete the square in $$\boldsymbol{x}$$ (expand, collect linear and quadratic terms). The linear term picks up a contribution from $$\boldsymbol{s}$$; the resulting Gaussian has mean shifted by $$+\,g(t)^2\,\boldsymbol{s}(\boldsymbol{y},t)\,\delta$$ relative to the naive time-reversal of the forward drift.

**Step 4 — continuous-time limit.** Reversing time ($$t\mapsto T-t$$) and taking $$\delta\to 0$$ yields the **reverse-time SDE**

$$\mathrm{d}\boldsymbol{x} = \big[\boldsymbol{f}(\boldsymbol{x},t) - g(t)^2\,\boldsymbol{s}(\boldsymbol{x},t)\big]\,\mathrm{d}t + g(t)\,\mathrm{d}\bar{\boldsymbol{w}}.$$

The score appears because reversing a diffusion requires knowing how density changes in space — the quantity we train with denoising score matching. See [Winkler's sketch](https://ludwigwinkler.github.io/blog/SimpleReverseSDE/) and Anderson (1982) for the rigorous treatment below.

</details>

### 4.4 Anderson's reverse-time SDE

> Anderson (1982): the time-reversal of $$\mathrm{d}\boldsymbol{x} = \boldsymbol{f}\,\mathrm{d}t + g\,\mathrm{d}\boldsymbol{w}$$ is $$\mathrm{d}\boldsymbol{x} = [\boldsymbol{f}-g^2\boldsymbol{s}]\,\mathrm{d}t + g\,\mathrm{d}\bar{\boldsymbol{w}}$$ with $$\boldsymbol{s}=\nabla\log p_t$$ — the rigorous form of the heuristic above.
{:.lead}

Running this SDE from $$\boldsymbol{x}_T\sim\mathcal{N}(\mathbf{0},I)$$ to $$t=0$$ generates samples from $$p_{\text{data}}$$ once $$\boldsymbol{s}$$ is replaced by the learned $$\boldsymbol{s}_\theta(\boldsymbol{x},t)$$. Two sign conventions appear in the literature (reverse Wiener vs. $$t\mapsto T-t$$); they agree on the **score term** $$g^2\boldsymbol{s}$$.

The reverse SDE generalizes annealed Langevin dynamics and DDPM ancestral sampling. Anderson's proof via forward/backward Kolmogorov equations is optional below.

<details class="optional-derivation" markdown="1">
<summary><strong>Anderson's reverse-time SDE (sketch)</strong> (optional — click to expand)</summary>

**Goal.** Show that the time-reversal of

$$\mathrm{d}\boldsymbol{X}_t = \boldsymbol{f}(\boldsymbol{X}_t,t)\,\mathrm{d}t + g(t)\,\mathrm{d}\boldsymbol{w}$$

is

$$\mathrm{d}\boldsymbol{X}_t = \big[\boldsymbol{f}(\boldsymbol{X}_t,t) - g(t)^2\,\boldsymbol{s}(\boldsymbol{X}_t,t)\big]\,\mathrm{d}t + g(t)\,\mathrm{d}\bar{\boldsymbol{w}},$$

with the **same marginals** $$\{p_t\}$$ when run backward from $$t=T$$ to $$0$$.

**Step 1 — joint density.** Write the joint $$p_{s,t}(\boldsymbol{x}_s,\boldsymbol{x}_t)=p_{s\mid t}(\boldsymbol{x}_s\mid\boldsymbol{x}_t)\,p_t(\boldsymbol{x}_t)$$ for $$s<t$$.

**Step 2 — forward Kolmogorov equation.** The transition density $$p_{s\mid t}(\boldsymbol{x}_s\mid\boldsymbol{x}_t)$$ (as a function of $$\boldsymbol{x}_t,t$$) satisfies the **backward Kolmogorov** equation, which encodes the forward SDE drift $$\boldsymbol{f}$$ and diffusion $$g$$.

**Step 3 — differentiate the joint in $$t$$.** Using $$\partial_t p_t = \partial_t[p_{s\mid t}\,p_t]$$ and substituting the backward equation for $$p_{s\mid t}$$ plus the **Fokker–Planck** equation for $$p_t$$,

$$\partial_t p_{s,t} = \int \Big[\text{(backward)}\,p_{s\mid t}\,p_t + p_{s\mid t}\,\text{(FPE)}\,p_t\Big]\,\mathrm{d}\boldsymbol{x}_t.$$

**Step 4 — complete the square.** After integrating by parts in $$\boldsymbol{x}_t$$, the terms involving $$\nabla p_t$$ combine into a single **score** term $$g(t)^2\nabla\log p_t$$ multiplying $$p_{s,t}$$. The remaining terms match the FPE of an SDE with drift $$\boldsymbol{f}-g^2\boldsymbol{s}$$ and diffusion $$g$$.

**Step 5 — read off the reverse SDE.** The reverse-time process (with $$\mathrm{d}\bar{\boldsymbol{w}}$$) has the stated drift. At sampling time replace $$\boldsymbol{s}$$ with $$\boldsymbol{s}_\theta(\boldsymbol{x},t)$$.

*Reference:* Anderson (1982); SDE course [Lesson 2 §2.2](https://kierandidi.github.io/); Song et al. (2021) Eq. (4.1.6).

</details>

### 4.5 The probability-flow ODE (same marginals, no noise)

> Every SDE admits a deterministic **probability-flow ODE** with the *same* marginals $$\{p_t\}$$: $$\mathrm{d}\boldsymbol{x} = \big[\boldsymbol{f}-\tfrac12 g^2\boldsymbol{s}\big]\mathrm{d}t.$$ The $$\tfrac12$$ factor and absence of noise follow from the FPE (Day 8).
{:.lead}

![Three dynamics with identical time marginals: the forward SDE, the reverse SDE, and the probability-flow ODE (Principles Fig 4.5).](/assets/figures/day07/pdm_three_dynamics.png)

The reverse SDE is a **stochastic** sampler; the **probability-flow ODE** is deterministic ($$\tilde{\boldsymbol{\mu}}=\boldsymbol{f}-\tfrac12 g^2\boldsymbol{s}$$, no $$\mathrm{d}\bar{\boldsymbol{w}}$$). Practical uses: reproducible sampling, exact likelihoods (CNF), fast ODE solvers (Day 8).

Compare all three dynamics:


<div class="interactive-viz" markdown="0">
  <iframe src="https://the-principles-of-diffusion-models.github.io/assets/score_sde_three_dynamics.html" width="100%" height="780" loading="lazy" style="border:1px solid #ddd;border-radius:8px;" title="Three Dynamics: Forward SDE, Reverse SDE, PF-ODE"></iframe>
  <p style="font-size:0.85em;color:#888;margin-top:4px;">Interactive: <strong>Three Dynamics: Forward SDE, Reverse SDE, PF-ODE</strong> — <a href="https://the-principles-of-diffusion-models.github.io/assets/score_sde_three_dynamics.html" target="_blank" rel="noopener">open in new tab</a>. Source: <em>The Principles of Diffusion Models</em>.</p>
</div>


### 4.6 The denoising score matching objective

> The marginal score equals the conditional expectation of the closed-form conditional score: $$\nabla\log p_t(\boldsymbol{x}_t) = \mathbb{E}[\nabla\log p_t(\boldsymbol{x}_t\mid\boldsymbol{x}_0)\,\big\vert \,\boldsymbol{x}_t]$$. Hence regressing on $$-\boldsymbol{\epsilon}/\sigma_t$$ trains the marginal score.
{:.lead}

With $$\boldsymbol{x}_t=\alpha_t\boldsymbol{x}_0+\sigma_t\boldsymbol{\epsilon}$$,

$$\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t\mid\boldsymbol{x}_0) = -\frac{\boldsymbol{\epsilon}}{\sigma_t}.$$

Minimizing $$\mathbb{E}\Vert \boldsymbol{s}_\theta(\boldsymbol{x}_t,t)-\nabla\log p_t(\boldsymbol{x}_t\mid\boldsymbol{x}_0)\Vert ^2$$ is squared-error regression whose optimum is $$\mathbb{E}[\nabla\log p_t(\cdot\mid\boldsymbol{x}_0)\mid\boldsymbol{x}_t]=\nabla\log p_t(\boldsymbol{x}_t)$$ — so **predicting the score is predicting the noise** (Day 6). Full proof and Tweedie's formula: optional block below.

<details class="optional-derivation" markdown="1">
<summary><strong>DSM objective: conditional → marginal score (full proof)</strong> (optional — click to expand)</summary>

**Claim (Proposition 4.3.1, Principles).** The minimizer of

$$J(\theta)=\mathbb{E}_{t,\boldsymbol{x}_0,\boldsymbol{\epsilon}}\Big\Vert \boldsymbol{s}_\theta(\boldsymbol{x}_t,t) - \nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t\mid\boldsymbol{x}_0)\Big\Vert ^2$$

satisfies $$\boldsymbol{s}_\theta^\star(\boldsymbol{x}_t,t)=\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t)$$ for a.e. $$\boldsymbol{x}_t\sim p_t$$.

**Step 1 — regression identity.** For any target $$\boldsymbol{g}(\boldsymbol{x}_t,\boldsymbol{x}_0)$$,

$$\arg\min_{\boldsymbol{s}}\,\mathbb{E}\Vert \boldsymbol{s}(\boldsymbol{x}_t,t)-\boldsymbol{g}\Vert ^2 = \mathbb{E}[\boldsymbol{g}\mid\boldsymbol{x}_t].$$

Set $$\boldsymbol{g}=\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t\mid\boldsymbol{x}_0)$$.

**Step 2 — posterior average.** Write $$p_t(\boldsymbol{x}_t)=\int p_t(\boldsymbol{x}_t\mid\boldsymbol{x}_0)p_{\text{data}}(\boldsymbol{x}_0)\,\mathrm{d}\boldsymbol{x}_0$$ and differentiate:

$$\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t) = \frac{\int \nabla_{\boldsymbol{x}_t}p_t(\boldsymbol{x}_t\mid\boldsymbol{x}_0)\,p_{\text{data}}(\boldsymbol{x}_0)\,\mathrm{d}\boldsymbol{x}_0}{p_t(\boldsymbol{x}_t)}.$$

Use $$\nabla p_t(\cdot\mid\boldsymbol{x}_0)=p_t(\cdot\mid\boldsymbol{x}_0)\,\nabla\log p_t(\cdot\mid\boldsymbol{x}_0)$$ and recognise $$p_t(\boldsymbol{x}_0\mid\boldsymbol{x}_t)$$:

$$\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t) = \mathbb{E}_{\boldsymbol{x}_0\mid\boldsymbol{x}_t}\big[\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t\mid\boldsymbol{x}_0)\big].$$

**Step 3 — Gaussian conditional score.** With $$\boldsymbol{x}_t=\alpha_t\boldsymbol{x}_0+\sigma_t\boldsymbol{\epsilon}$$,

$$\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t\mid\boldsymbol{x}_0) = -\frac{\boldsymbol{x}_t-\alpha_t\boldsymbol{x}_0}{\sigma_t^2} = -\frac{\boldsymbol{\epsilon}}{\sigma_t}.$$

**Conclusion.** Regressing on the known conditional score trains the **marginal** score; predicting $$\boldsymbol{s}_\theta\approx -\boldsymbol{\epsilon}/\sigma_t$$ is predicting the noise (Day 6). Related: **Tweedie's formula** $$\mathbb{E}[\boldsymbol{x}_0\mid\boldsymbol{x}_t]=(\boldsymbol{x}_t+\sigma_t^2\boldsymbol{s})/\alpha_t$$.

</details>

<details class="optional-derivation" markdown="1">
<summary><strong>Appendix B — density evolution and the Fokker–Planck equation</strong> (optional — click to expand)</summary>

*Condensed from [Principles of Diffusion Models](https://arxiv.org/abs/2510.21890), Appendix B.*

**B.1 — Change of variables → continuity → FPE.**

1. **Single bijection** $$\boldsymbol{x}_1=\Psi(\boldsymbol{x}_0)$$: $$p_1(\boldsymbol{x}_1)=p_0(\Psi^{-1}(\boldsymbol{x}_1))\,\vert \det\partial\Psi^{-1}/\partial\boldsymbol{x}_1\vert $$.
2. **Composition** of maps: log-density accumulates $$-\sum_k \log\vert \det\partial\Psi_k/\partial\boldsymbol{x}_{k-1}\vert $$ (normalizing flows, Eq. B.1.2).
3. **Continuous limit** $$\boldsymbol{x}_{t+\delta}=\boldsymbol{x}_t+\delta\,\boldsymbol{f}(\boldsymbol{x}_t,t)$$: Jacobian $$\det(I+\delta\nabla\boldsymbol{f})=1+\delta\,\nabla\cdot\boldsymbol{f}+\mathcal{O}(\delta^2)$$ gives the **continuity equation**

$$\partial_t p_t + \nabla\cdot(p_t\,\boldsymbol{f}) = 0.$$

4. **Add noise** $$\mathrm{d}\boldsymbol{x}=\boldsymbol{f}\,\mathrm{d}t+g(t)\,\mathrm{d}\boldsymbol{w}$$: spreading term $$\tfrac12 g(t)^2\Delta p_t$$ yields the **Fokker–Planck equation**

$$\partial_t p_t = -\nabla\cdot(\boldsymbol{f}\,p_t) + \tfrac12 g(t)^2 \Delta p_t = -\nabla\cdot\Big(\big(\boldsymbol{f}-\tfrac12 g(t)^2\,\boldsymbol{s}\big)\,p_t\Big).$$

**B.2 — Intuition.** In a small box, mass changes only through net flux $$\boldsymbol{j}=p_t\boldsymbol{v}$$; conservation $$\partial_t p_t + \nabla\cdot\boldsymbol{j}=0$$ is the continuity equation. The divergence theorem upgrades the box argument to arbitrary control volumes.

</details>

<details class="optional-derivation" markdown="1">
<summary><strong>Appendix C — Itô's formula and Girsanov's theorem</strong> (optional — click to expand)</summary>

*Condensed from Principles Appendix C.*

**C.1 — Itô's formula (chain rule for SDEs).** For smooth $$h(\boldsymbol{x},t)$$ and $$\mathrm{d}\boldsymbol{x}=\boldsymbol{f}\,\mathrm{d}t+g\,\mathrm{d}\boldsymbol{w}$$,

$$\mathrm{d}h = \Big(\partial_t h + \nabla h^{\top}\boldsymbol{f} + \tfrac12 g^2 \Delta h\Big)\mathrm{d}t + g\,\nabla h^{\top}\mathrm{d}\boldsymbol{w}.$$

Key rule: $$(\mathrm{d}\boldsymbol{w})^2 = \mathrm{d}t$$, so second-order terms survive (unlike ordinary calculus). **C.1.4** uses Itô's formula on $$h=\log p_t$$ to derive the Fokker–Planck equation.

**C.2 — Girsanov's theorem.** Two SDEs with the same diffusion $$g$$ but drifts $$\boldsymbol{f}$$ and $$\boldsymbol{f}+g\,\boldsymbol{u}$$ assign different path probabilities. The **likelihood ratio** (Radon–Nikodym derivative) on a path $$\boldsymbol{x}_{[0,T]}$$ is

$$\frac{\mathrm{d}\mathbb{Q}}{\mathrm{d}\mathbb{P}} = \exp\Big(\int_0^T \boldsymbol{u}^{\top}\mathrm{d}\boldsymbol{w} - \tfrac12\int_0^T \Vert \boldsymbol{u}\Vert ^2\,\mathrm{d}t\Big).$$

For the reverse SDE, $$\boldsymbol{u}=-g\,\boldsymbol{s}$$ — the score reweights forward paths to reverse paths. This explains why score matching implicitly performs **likelihood-based** training (Song et al., 2021).

</details>

<details class="optional-derivation" markdown="1">
<summary><strong>Appendix D — score matching and PF-ODE proofs</strong> (optional — click to expand)</summary>

*Selected proofs from Principles Appendix D.*

**D.2.1 — Score matching via integration by parts (Prop. 3.2.1).** Expand

$$\tfrac12\mathbb{E}\Vert \boldsymbol{s}_\theta-\boldsymbol{s}\Vert ^2 = \tfrac12\mathbb{E}\Vert \boldsymbol{s}_\theta\Vert ^2 - \mathbb{E}[\boldsymbol{s}_\theta^{\top}\boldsymbol{s}] + \text{const}.$$

Use $$\boldsymbol{s}=\nabla\log p_{\text{data}}$$ and $$\mathbb{E}[\boldsymbol{s}_\theta^{\top}\nabla p/p]=-\mathbb{E}[\nabla\cdot\boldsymbol{s}_\theta]$$ (integration by parts, vanishing boundary) to obtain the tractable objective with Jacobian trace.

**D.2.2 — Denoising score matching (Prop. 3.3.1 / 4.3.1).** Add noise $$\tilde{\boldsymbol{x}}=\boldsymbol{x}+\sigma\boldsymbol{\epsilon}$$; the cross term becomes an expectation under $$p_\sigma(\tilde{\boldsymbol{x}}\mid\boldsymbol{x})$$, which integrates by parts to remove $$\nabla\log p_{\text{data}}$$. The minimizer is the **marginal** noisy score — the DSM proof in the block above.

**D.2.6 — PF-ODE shares marginals (Prop. 4.1.1).** Part 1: verify that $$\tilde{\boldsymbol{\mu}}=\boldsymbol{f}-\tfrac12 g^2\boldsymbol{s}$$ reproduces the FPE. Part 2: show the reverse SDE with drift $$\boldsymbol{f}-g^2\boldsymbol{s}$$ has the same $$p_t$$ when time is reversed — connecting Anderson, PF-ODE, and FPE in one proof chain.

</details>

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
