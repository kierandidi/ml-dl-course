---
layout: post
title: Day 8 - Guidance, Solvers, and Few-Step Sampling
image: /assets/img/lessons/day08.png
description: >
  Controlling diffusion with guidance, sampling as ODE/SDE solving, fast high-order solvers, and one-step flow maps.
invert_sidebar: true
---

# Day 8 - Guidance, Solvers, and Few-Step Sampling

### Optional reading for this lesson
- [The Principles of Diffusion Models](https://arxiv.org/abs/2510.21890), Ch. 8–11; Appendix B (FPE), D.2.6 (PF-ODE proof)
- [Ho & Salimans — Classifier-Free Diffusion Guidance (2022)](https://arxiv.org/abs/2207.12598)
- [Song et al. — Denoising Diffusion Implicit Models (DDIM, 2021)](https://arxiv.org/abs/2010.02502)
- [Karras et al. — Elucidating the Design Space of Diffusion Models (EDM, 2022)](https://arxiv.org/abs/2206.00364)
- [Song et al. — Consistency Models (2023)](https://arxiv.org/abs/2303.01469)
- [SDE course — Lesson 2 §4: PF-ODE derivation from the FPE](https://kierandidi.github.io/) (Generative Modelling with SDEs)

### [Slides](/assets/slides/day08.pdf)

### Exercise

[Download the notebook](/notebooks/practicals/day08.ipynb) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kierandidi/ml-dl-course/blob/main/notebooks/practicals/day08.ipynb)

Days 6–7 built diffusion models and showed that sampling means running a learned dynamics backward in time. Today we make them **useful and fast**. First, **guidance** lets us steer generation toward a condition — a class label or a text prompt — and trade diversity for fidelity. Second, we recognize sampling as **numerical integration** of the probability-flow ODE (or reverse SDE), which lets us import centuries of numerical-analysis wisdom: DDIM is just Euler's method, and high-order and exponential integrators reach high quality in a handful of steps. Finally, **flow maps** and **consistency models** collapse the trajectory into a single learned jump, pushing toward real-time, few-step generation.

* toc
{:toc}

## 1. Conditional Generation and Guidance

### 1.1 Conditioning a diffusion model

> A **conditional** diffusion model samples $$p(\boldsymbol{x}\mid c)$$ for a condition $$c$$ (class, text, image) by training a condition-aware denoiser $$\boldsymbol{\epsilon}_\theta(\boldsymbol{x}_t,t,c)$$, equivalently a conditional score $$\nabla\log p_t(\boldsymbol{x}\mid c)$$.
{:.lead}

**Why this matters.** Unconditional generation is a curiosity; the applications people care about — text-to-image, class-conditional synthesis, inpainting, super-resolution — are all *conditional*. The simplest recipe is to feed the condition $$c$$ to the network alongside $$(\boldsymbol{x}_t,t)$$ and train exactly as before. Then the learned score is the conditional score, and any sampler from Day 7 produces samples from $$p(\boldsymbol{x}\mid c)$$.

In practice this "plain" conditioning often **under-uses** the condition: samples are diverse but only loosely match the prompt. We want a knob that trades some diversity for much stronger adherence to $$c$$. That knob is **guidance**, and it falls straight out of Bayes' rule on scores.

### 1.2 Classifier guidance

> **Classifier guidance** adds the gradient of a noise-aware classifier to the score: $$\nabla\log p_t(\boldsymbol{x}\mid c) = \nabla\log p_t(\boldsymbol{x}) + \nabla\log p_t(c\mid\boldsymbol{x}),$$ scaled by a guidance weight $$w$$.
{:.lead}

![Guidance steers the sampling trajectory toward regions consistent with the condition.](/assets/figures/day08/pdm_guidance.png)

By Bayes' rule, the conditional score decomposes into the unconditional score plus the gradient of a classifier's log-likelihood. So if we train a classifier $$p_\phi(c\mid\boldsymbol{x}_t)$$ **on noisy inputs** (it must operate at every noise level), we can steer an unconditional model:

$$\tilde{\boldsymbol{s}}(\boldsymbol{x}_t,t) = \nabla\log p_t(\boldsymbol{x}_t) + w\,\nabla_{\boldsymbol{x}_t}\log p_\phi(c\mid\boldsymbol{x}_t).$$

The weight $$w$$ controls how hard we push toward the class. The drawbacks are practical: you need a separate classifier trained on noised data, and its gradients can be brittle. Classifier-free guidance removes the external classifier entirely.

### 1.3 Classifier-free guidance

> **Classifier-free guidance (CFG)** trains a single network to be both conditional and unconditional (by randomly dropping $$c$$), then extrapolates at sampling time: $$\tilde{\boldsymbol{\epsilon}} = (1+w)\,\boldsymbol{\epsilon}_\theta(\boldsymbol{x}_t,t,c) - w\,\boldsymbol{\epsilon}_\theta(\boldsymbol{x}_t,t,\varnothing).$$
{:.lead}

![Classifier-free guidance interpolates/extrapolates between conditional and unconditional predictions to control fidelity.](/assets/figures/day08/pdm_cfg.png)

The trick is to avoid a separate classifier by noting that its gradient equals the *difference* of conditional and unconditional scores (next subsection). During training we replace $$c$$ with a null token $$\varnothing$$ a fraction of the time, so one network learns both predictions. At sampling we extrapolate **away** from the unconditional prediction and **toward** the conditional one:

$$\tilde{\boldsymbol{\epsilon}}_\theta = \boldsymbol{\epsilon}_\theta(\boldsymbol{x}_t,t,\varnothing) + (1+w)\big[\boldsymbol{\epsilon}_\theta(\boldsymbol{x}_t,t,c) - \boldsymbol{\epsilon}_\theta(\boldsymbol{x}_t,t,\varnothing)\big].$$

The guidance scale $$w$$ is the single most important sampling knob in modern text-to-image models: $$w=0$$ is plain conditioning (diverse, loosely matched), while larger $$w$$ sharpens prompt adherence and image fidelity at the cost of diversity (and, if pushed too far, saturation artifacts).

### 1.4 Derivation: guidance from Bayes' rule

> Guidance is Bayes' rule differentiated. Sharpening the implicit classifier by an exponent $$w$$ yields the guided score, and the score↔noise identity converts it to the CFG noise formula.
{:.lead}

Start from Bayes' rule at noise level $$t$$, $$p_t(\boldsymbol{x}\mid c)\propto p_t(\boldsymbol{x})\,p_t(c\mid\boldsymbol{x})$$, and take $$\nabla_{\boldsymbol{x}}\log$$ of both sides (the $$\boldsymbol{x}$$-independent evidence drops):

$$\nabla\log p_t(\boldsymbol{x}\mid c) = \textcolor{teal}{\nabla\log p_t(\boldsymbol{x})} + \textcolor{purple}{\nabla\log p_t(c\mid\boldsymbol{x})}.$$

Rearranging gives the **implicit classifier** — no separate model needed:

$$\textcolor{purple}{\nabla\log p_t(c\mid\boldsymbol{x})} = \nabla\log p_t(\boldsymbol{x}\mid c) - \nabla\log p_t(\boldsymbol{x}).$$

To strengthen the condition, **sharpen** the classifier by raising it to a power, $$p_t(c\mid\boldsymbol{x})^{w}$$, which scales its score by $$w$$. The guided score becomes

$$\tilde{\boldsymbol{s}} = \nabla\log p_t(\boldsymbol{x}) + (1+w)\big[\nabla\log p_t(\boldsymbol{x}\mid c) - \nabla\log p_t(\boldsymbol{x})\big].$$

Finally convert scores to noise predictions with the Day-7 identity $$\boldsymbol{s}=-\boldsymbol{\epsilon}/\sigma_t$$ (the $$-1/\sigma_t$$ cancels throughout), giving exactly the CFG rule

$$\tilde{\boldsymbol{\epsilon}} = (1+w)\,\boldsymbol{\epsilon}_\theta(\boldsymbol{x}_t,t,c) - w\,\boldsymbol{\epsilon}_\theta(\boldsymbol{x}_t,t,\varnothing).$$

So CFG is not a heuristic — it is Bayes' rule plus a sharpening exponent, expressed in noise-prediction coordinates.

## 2. Sampling as Solving a Differential Equation

### 2.1 Sampling is numerical integration

> Generating a sample means **numerically integrating** the reverse SDE or the probability-flow ODE from $$t=T$$ to $$t=0$$. Each step evaluates the network once — one **number of function evaluations (NFE)**.
{:.lead}

![Sampling runs a reverse-time process from noise back to data; numerically, this is integrating a differential equation.](/assets/figures/day08/pdm_reverse_sde.png)

On Day 7 we derived two dynamics with the right marginals — the reverse SDE and the PF-ODE. Turning either into an algorithm means choosing a **time discretization** $$T=t_0>t_1>\dots>t_N=0$$ and a **numerical scheme** to step between consecutive times. The cost is dominated by the number of network evaluations (NFEs), since the network (the score/denoiser) is by far the most expensive operation.

This reframing is liberating: the quality–speed trade-off becomes a classic **numerical-analysis** question. A more accurate integrator reaches the same quality in fewer steps. Everything in this lecture is an answer to "how do we integrate this ODE well in as few NFEs as possible?".

### 2.2 The probability-flow ODE from the Fokker–Planck equation

> Rewriting the FPE with the log-derivative trick yields a **Liouville equation** for an ODE with drift $$\tilde{\boldsymbol{\mu}} = \boldsymbol{f} - \tfrac12 g^2\boldsymbol{s}$$ — the probability-flow ODE, sharing marginals with the forward/reverse SDE.
{:.lead}

The forward SDE $$\mathrm{d}\boldsymbol{x}=\boldsymbol{f}\,\mathrm{d}t+g\,\mathrm{d}\boldsymbol{w}$$ evolves its density $$p_t$$ by the **Fokker–Planck equation (FPE)**. The goal is to find a *deterministic* ODE that pushes the **same** $$p_t$$. The whole derivation is one rewrite of the diffusion term, using the **log-derivative trick** $$\nabla p_t=p_t\boldsymbol{s}$$ with $$\boldsymbol{s}=\nabla\log p_t$$:

$$\begin{aligned}
\partial_t p_t
&= -\nabla\cdot\big(\textcolor{teal}{\boldsymbol{f}}\,p_t\big) + \tfrac12 g^2\,\Delta p_t & &\text{(Fokker–Planck)} \\
&= -\nabla\cdot\big(\textcolor{teal}{\boldsymbol{f}}\,p_t\big) + \tfrac12 g^2\,\nabla\cdot\big(\nabla p_t\big) & &\big(\Delta = \nabla\cdot\nabla\big) \\
&= -\nabla\cdot\big(\textcolor{teal}{\boldsymbol{f}}\,p_t\big) + \tfrac12 g^2\,\nabla\cdot\big(p_t\,\textcolor{purple}{\boldsymbol{s}}\big) & &\big(\nabla p_t = p_t\,\textcolor{purple}{\boldsymbol{s}}\big) \\
&= -\nabla\cdot\Big(\underbrace{\big[\,\textcolor{teal}{\boldsymbol{f}} - \tfrac12 g^2\,\textcolor{purple}{\boldsymbol{s}}\,\big]}_{\textcolor{orange}{\tilde{\boldsymbol{\mu}}}}\,p_t\Big) & &\text{(collect into one drift)}.
\end{aligned}$$

The last line is a **continuity (Liouville) equation** $$\partial_t p_t = -\nabla\cdot(\textcolor{orange}{\tilde{\boldsymbol{\mu}}}\,p_t)$$ with **no** second-order term — precisely the FPE of a noise-free process. Hence the marginals are transported by the deterministic **probability-flow ODE**

$$\boxed{\;\frac{\mathrm{d}\boldsymbol{x}}{\mathrm{d}t} = \textcolor{orange}{\tilde{\boldsymbol{\mu}}(\boldsymbol{x},t)} = \textcolor{teal}{\boldsymbol{f}(\boldsymbol{x},t)} - \tfrac12\, g(t)^2\,\textcolor{purple}{\boldsymbol{s}(\boldsymbol{x},t)}.\;}$$

Compare the **reverse SDE** drift $$\boldsymbol{f}-g^2\boldsymbol{s}$$ (Day 7): identical marginals $$\{p_t\}$$, but only **half** the score coefficient and no noise. The difference is exactly the half of the diffusion that the ODE converted into deterministic transport. This is why DDIM (Euler on the PF-ODE) and DDPM (the reverse SDE) **share one trained** $$\boldsymbol{s}_\theta$$ — they are two solvers for the same family of densities. Step-by-step algebra and the rigorous statement are in the optional block below.

<details class="optional-derivation" markdown="1">
<summary><strong>PF-ODE from the Fokker–Planck equation (full derivation)</strong> (optional — click to expand)</summary>

**Setup.** Forward SDE $$\mathrm{d}\boldsymbol{x}=\boldsymbol{f}(\boldsymbol{x},t)\,\mathrm{d}t+g(t)\,\mathrm{d}\boldsymbol{w}$$ with $$\boldsymbol{s}=\nabla_{\boldsymbol{x}}\log p_t$$.

**Step 1 — Fokker–Planck (Appendix B.1.3).**

$$\partial_t p_t = -\nabla\cdot(\boldsymbol{f}\,p_t) + \tfrac12 g(t)^2 \Delta p_t.$$

**Step 2 — product rule on the diffusion term.**

$$\partial_t p_t = -\nabla\cdot(\boldsymbol{f}\,p_t) + \tfrac12 g(t)^2\nabla\cdot\nabla p_t + \tfrac12 \nabla(g(t)^2)\cdot\nabla p_t.$$

For state-independent $$g(t)$$, the last term vanishes.

**Step 3 — log-derivative trick.** Use $$\nabla p_t = p_t\,\boldsymbol{s}$$:

$$\tfrac12 g(t)^2 \Delta p_t = \tfrac12 g(t)^2 \nabla\cdot(p_t\,\boldsymbol{s}) = \nabla\cdot\Big(\tfrac12 g(t)^2\,p_t\,\boldsymbol{s}\Big) - \tfrac12 g(t)^2\,p_t\,\nabla\cdot\boldsymbol{s}.$$

The $$\nabla\cdot\boldsymbol{s}$$ term cancels when combining all pieces (Principles D.2.6); equivalently, expand $$\nabla\cdot(\boldsymbol{f}p_t - \tfrac12 g^2 p_t \boldsymbol{s})$$ directly.

**Step 4 — Liouville form.** Factor out $$p_t$$:

$$\partial_t p_t = -\nabla\cdot\Big(p_t\,\underbrace{\Big(\boldsymbol{f} - \tfrac12 g(t)^2\,\boldsymbol{s}\Big)}_{\tilde{\boldsymbol{\mu}}(\boldsymbol{x},t)}\Big).$$

This is the FPE of an SDE with **zero diffusion** — the **Liouville equation** — hence the density is transported by the ODE

$$\mathrm{d}\boldsymbol{x} = \tilde{\boldsymbol{\mu}}(\boldsymbol{x},t)\,\mathrm{d}t = \big[\boldsymbol{f} - \tfrac12 g(t)^2\,\boldsymbol{s}(\boldsymbol{x},t)\big]\,\mathrm{d}t.$$

**Compare reverse SDE:** drift $$\boldsymbol{f}-g^2\boldsymbol{s}$$ (full score coefficient) plus noise $$g\,\mathrm{d}\bar{\boldsymbol{w}}$$. Same $$\{p_t\}$$, different sample paths.

*Reference:* Principles Appendix B.1.3, D.2.6; SDE course Lesson 2 §4; Song et al. (2021) Eq. (4.1.7).

</details>

<details class="optional-derivation" markdown="1">
<summary><strong>Appendix B — density evolution: change of variables → continuity → Fokker–Planck</strong> (optional — click to expand)</summary>

*Expanded from [Principles of Diffusion Models](https://arxiv.org/abs/2510.21890), Appendix B. Throughout, the forward SDE is $$\mathrm{d}\boldsymbol{x}=\boldsymbol{f}(\boldsymbol{x},t)\,\mathrm{d}t+g(t)\,\mathrm{d}\boldsymbol{w}$$ and the score is $$\boldsymbol{s}(\boldsymbol{x},t)=\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x})$$.*

**B.0 — The one idea: conservation of probability mass.** Every law below is the same statement — *probability is neither created nor destroyed, only transported* — written at three levels of generality: a single map (change of variables), a deterministic flow (continuity equation), and a flow plus noise (Fokker–Planck). Each is derived from the previous one.

**B.1 — Change of variables for one invertible map.** Let $$\Psi:\mathbb{R}^d\to\mathbb{R}^d$$ be a diffeomorphism and $$\boldsymbol{x}_1=\Psi(\boldsymbol{x}_0)$$ with $$\boldsymbol{x}_0\sim p_0$$. The mass in an infinitesimal box around $$\boldsymbol{x}_0$$ equals the mass in its image:

$$p_0(\boldsymbol{x}_0)\,\mathrm{d}\boldsymbol{x}_0 = p_1(\boldsymbol{x}_1)\,\mathrm{d}\boldsymbol{x}_1 .$$

The volume element transforms by the Jacobian $$\mathbf{J}_\Psi$$, with $$(\mathbf{J}_\Psi)_{ij}=\partial\Psi_i/\partial x_{0,j}$$, through $$\mathrm{d}\boldsymbol{x}_1=\left\lvert\det\mathbf{J}_\Psi\right\rvert\,\mathrm{d}\boldsymbol{x}_0$$. Solving for $$p_1$$ and using the inverse-function theorem $$\det(\partial\Psi^{-1}/\partial\boldsymbol{x}_1)=1/\det\mathbf{J}_\Psi$$,

$$p_1(\boldsymbol{x}_1)=p_0\!\big(\Psi^{-1}(\boldsymbol{x}_1)\big)\,\left\lvert\det\frac{\partial\Psi^{-1}}{\partial\boldsymbol{x}_1}\right\rvert =\frac{p_0\!\big(\Psi^{-1}(\boldsymbol{x}_1)\big)}{\left\lvert\det\mathbf{J}_\Psi\!\big(\Psi^{-1}(\boldsymbol{x}_1)\big)\right\rvert}.$$

Taking logs, with $$\boldsymbol{x}_0=\Psi^{-1}(\boldsymbol{x}_1)$$,

$$\log p_1(\boldsymbol{x}_1)=\log p_0(\boldsymbol{x}_0)-\log\left\lvert\det\mathbf{J}_\Psi(\boldsymbol{x}_0)\right\rvert .$$

**B.2 — Composition of maps (normalizing flows).** For a stack $$\Psi=\Psi_K\circ\cdots\circ\Psi_1$$ with intermediate states $$\boldsymbol{x}_k=\Psi_k(\boldsymbol{x}_{k-1})$$, the determinant chain rule turns the product of Jacobians into a sum of log-determinants:

$$\log p_K(\boldsymbol{x}_K)=\log p_0(\boldsymbol{x}_0)-\sum_{k=1}^{K}\log\left\lvert\det\frac{\partial\Psi_k}{\partial\boldsymbol{x}_{k-1}}\right\rvert .$$

This is exactly the training objective of a **normalizing flow**: maximize the data log-likelihood $$\log p_K(\boldsymbol{x}_K)$$ by composing invertible layers with cheap log-determinants.

**B.3 — Continuous limit → the continuity equation.** Replace the discrete stack by a deterministic flow $$\dot{\boldsymbol{x}}=\boldsymbol{f}(\boldsymbol{x},t)$$, i.e. the near-identity map $$\Psi_\delta(\boldsymbol{x})=\boldsymbol{x}+\delta\,\boldsymbol{f}(\boldsymbol{x},t)$$ over a small step $$\delta$$. Its Jacobian is

$$\mathbf{J}_{\Psi_\delta}=I+\delta\,\nabla\boldsymbol{f}+\mathcal{O}(\delta^2),\qquad (\nabla\boldsymbol{f})_{ij}=\frac{\partial f_i}{\partial x_j}.$$

By Jacobi's formula $$\det(I+\delta A)=1+\delta\,\mathrm{tr}(A)+\mathcal{O}(\delta^2)$$,

$$\log\det\mathbf{J}_{\Psi_\delta}=\delta\,\nabla\!\cdot\!\boldsymbol{f}+\mathcal{O}(\delta^2).$$

Insert this into the log change-of-variables formula along a trajectory and divide by $$\delta\to0$$ (the **instantaneous change of variables**, a.k.a. the continuous-normalizing-flow trace identity):

$$\frac{\mathrm{d}}{\mathrm{d}t}\log p_t(\boldsymbol{x}_t)=-\nabla\!\cdot\!\boldsymbol{f}(\boldsymbol{x}_t,t).$$

Now expand the total (material) derivative $$\frac{\mathrm{d}}{\mathrm{d}t}\log p_t=\partial_t\log p_t+\boldsymbol{f}\!\cdot\!\nabla\log p_t$$, multiply through by $$p_t$$, and use the product rule $$p_t\,\nabla\!\cdot\!\boldsymbol{f}+\boldsymbol{f}\!\cdot\!\nabla p_t=\nabla\!\cdot\!(p_t\boldsymbol{f})$$:

$$\partial_t p_t+\nabla\!\cdot\!(p_t\,\boldsymbol{f})=0 \qquad\text{(continuity / transport equation).}$$

**B.4 — Add noise → the Fokker–Planck equation.** Now the step is stochastic, $$\boldsymbol{x}_{t+\delta}=\boldsymbol{x}_t+\delta\,\boldsymbol{f}+g\sqrt{\delta}\,\boldsymbol{\epsilon}$$ with $$\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},I)$$, so the transition kernel is Gaussian, $$p_{t+\delta\mid t}(\boldsymbol{y}\mid\boldsymbol{x})=\mathcal{N}(\boldsymbol{y};\,\boldsymbol{x}+\delta\boldsymbol{f},\,\delta g^2 I)$$. Test against a smooth, compactly-supported $$\phi$$ and use Chapman–Kolmogorov:

$$\mathbb{E}[\phi(\boldsymbol{x}_{t+\delta})]=\iint \phi(\boldsymbol{y})\,p_{t+\delta\mid t}(\boldsymbol{y}\mid\boldsymbol{x})\,p_t(\boldsymbol{x})\,\mathrm{d}\boldsymbol{y}\,\mathrm{d}\boldsymbol{x}.$$

Taylor-expand $$\phi(\boldsymbol{y})$$ about $$\boldsymbol{x}$$ and take the Gaussian moments $$\mathbb{E}[\boldsymbol{y}-\boldsymbol{x}]=\delta\boldsymbol{f}$$ and $$\mathbb{E}[(\boldsymbol{y}-\boldsymbol{x})(\boldsymbol{y}-\boldsymbol{x})^{\top}]=\delta g^2 I+\mathcal{O}(\delta^2)$$:

$$\mathbb{E}[\phi(\boldsymbol{x}_{t+\delta})]-\mathbb{E}[\phi(\boldsymbol{x}_t)]=\delta\,\mathbb{E}\big[\boldsymbol{f}\!\cdot\!\nabla\phi+\tfrac12 g^2\Delta\phi\big]+\mathcal{O}(\delta^2).$$

Divide by $$\delta\to0$$ and write both sides as integrals against $$p_t$$; on the left $$\frac{\mathrm{d}}{\mathrm{d}t}\mathbb{E}[\phi]=\int\phi\,\partial_t p_t$$. Integrate by parts (the boundary terms vanish) to move the derivatives off $$\phi$$ and onto $$p_t$$:

$$\int\phi\,\partial_t p_t=\int\phi\Big[-\nabla\!\cdot\!(\boldsymbol{f}p_t)+\tfrac12 g^2\Delta p_t\Big].$$

Since $$\phi$$ is arbitrary, the integrands match:

$$\partial_t p_t=-\nabla\!\cdot\!(\boldsymbol{f}\,p_t)+\tfrac12 g(t)^2\,\Delta p_t \qquad\text{(Fokker–Planck).}$$

**B.5 — Score / probability-flow form.** Using $$\nabla p_t=p_t\boldsymbol{s}$$, rewrite the diffusion term as a divergence, $$\tfrac12 g^2\Delta p_t=\tfrac12 g^2\nabla\!\cdot\!(p_t\boldsymbol{s})=\nabla\!\cdot\!\big(\tfrac12 g^2 p_t\boldsymbol{s}\big)$$, so the FPE collapses into a *noise-free* continuity equation:

$$\partial_t p_t=-\nabla\!\cdot\!\Big(\big(\boldsymbol{f}-\tfrac12 g^2\boldsymbol{s}\big)\,p_t\Big).$$

The bracket is the **probability-flow ODE** velocity (Appendix D.3): at the level of *marginals*, adding noise to the SDE is equivalent to subtracting half the score from the drift and dropping the noise.

</details>

<details class="optional-derivation" markdown="1">
<summary><strong>Appendix D — proofs: score matching (ESM ↔ ISM ↔ DSM) and PF-ODE marginals</strong> (optional — click to expand)</summary>

*Expanded from Principles Appendix D (Props. 3.2.1, 3.3.1 / 4.3.1, 4.1.1).*

**D.1 — Explicit ↔ implicit score matching (Hyvärinen, 2005).** We want $$\boldsymbol{s}_\theta\approx\nabla\log p$$, but the data density $$p=p_{\text{data}}$$ is unknown. The explicit objective is

$$J_{\mathrm{ESM}}(\theta)=\tfrac12\,\mathbb{E}_{p}\big\lVert\boldsymbol{s}_\theta(\boldsymbol{x})-\nabla\log p(\boldsymbol{x})\big\rVert^2 .$$

Expand the square; the $$\lVert\nabla\log p\rVert^2$$ term is a $$\theta$$-independent constant:

$$J_{\mathrm{ESM}}=\tfrac12\,\mathbb{E}_p\lVert\boldsymbol{s}_\theta\rVert^2-\mathbb{E}_p\big[\boldsymbol{s}_\theta^{\top}\nabla\log p\big]+\text{const}.$$

The cross term becomes tractable after one integration by parts. With $$\nabla\log p=\nabla p/p$$,

$$\mathbb{E}_p\big[\boldsymbol{s}_\theta^{\top}\nabla\log p\big]=\int p\,\boldsymbol{s}_\theta^{\top}\frac{\nabla p}{p}=\int\boldsymbol{s}_\theta^{\top}\nabla p=-\int p\,(\nabla\!\cdot\!\boldsymbol{s}_\theta)=-\mathbb{E}_p[\nabla\!\cdot\!\boldsymbol{s}_\theta],$$

assuming $$p\,\boldsymbol{s}_\theta\to\mathbf{0}$$ at infinity. Hence the **implicit** (data-score-free) objective is

$$J_{\mathrm{ISM}}(\theta)=\mathbb{E}_p\Big[\tfrac12\lVert\boldsymbol{s}_\theta(\boldsymbol{x})\rVert^2+\nabla\!\cdot\!\boldsymbol{s}_\theta(\boldsymbol{x})\Big]+\text{const}.$$

The divergence $$\nabla\!\cdot\!\boldsymbol{s}_\theta=\mathrm{tr}(\partial\boldsymbol{s}_\theta/\partial\boldsymbol{x})$$ needs the Jacobian trace — $$\mathcal{O}(d)$$ backward passes, prohibitive in high dimension. Removing that cost is the whole point of denoising score matching.

**D.2 — Denoising score matching (Vincent, 2011).** Perturb the data with a Gaussian kernel, $$\tilde{\boldsymbol{x}}=\boldsymbol{x}+\sigma\boldsymbol{\epsilon}$$, giving the noisy marginal $$p_\sigma(\tilde{\boldsymbol{x}})=\int p(\boldsymbol{x})\,\mathcal{N}(\tilde{\boldsymbol{x}};\boldsymbol{x},\sigma^2 I)\,\mathrm{d}\boldsymbol{x}$$. Define the **denoising** objective against the *conditional* (closed-form) score:

$$J_{\mathrm{DSM}}(\theta)=\tfrac12\,\mathbb{E}_{p(\boldsymbol{x})\,p_\sigma(\tilde{\boldsymbol{x}}\mid\boldsymbol{x})}\big\lVert\boldsymbol{s}_\theta(\tilde{\boldsymbol{x}})-\nabla_{\tilde{\boldsymbol{x}}}\log p_\sigma(\tilde{\boldsymbol{x}}\mid\boldsymbol{x})\big\rVert^2 .$$

*Claim:* $$J_{\mathrm{DSM}}=J_{\mathrm{ESM},\,p_\sigma}+\text{const}$$, so both share the minimizer $$\boldsymbol{s}_\theta^\star=\nabla\log p_\sigma$$. Only the cross term depends on $$\theta$$; push the gradient through the mixture defining $$p_\sigma$$:

$$\mathbb{E}\big[\boldsymbol{s}_\theta^{\top}\nabla_{\tilde{\boldsymbol{x}}}\log p_\sigma(\tilde{\boldsymbol{x}}\mid\boldsymbol{x})\big]=\int\!\!\int p(\boldsymbol{x})\,\boldsymbol{s}_\theta(\tilde{\boldsymbol{x}})^{\top}\nabla_{\tilde{\boldsymbol{x}}}p_\sigma(\tilde{\boldsymbol{x}}\mid\boldsymbol{x})\,\mathrm{d}\boldsymbol{x}\,\mathrm{d}\tilde{\boldsymbol{x}}$$

$$=\int\boldsymbol{s}_\theta(\tilde{\boldsymbol{x}})^{\top}\nabla_{\tilde{\boldsymbol{x}}}\Big(\underbrace{\int p(\boldsymbol{x})\,p_\sigma(\tilde{\boldsymbol{x}}\mid\boldsymbol{x})\,\mathrm{d}\boldsymbol{x}}_{=\,p_\sigma(\tilde{\boldsymbol{x}})}\Big)\,\mathrm{d}\tilde{\boldsymbol{x}}=\mathbb{E}_{p_\sigma}\big[\boldsymbol{s}_\theta^{\top}\nabla\log p_\sigma(\tilde{\boldsymbol{x}})\big],$$

which is precisely the ESM cross term under $$p_\sigma$$. So the two objectives differ only by a $$\theta$$-independent constant. For the Gaussian kernel the conditional score is explicit,

$$\nabla_{\tilde{\boldsymbol{x}}}\log p_\sigma(\tilde{\boldsymbol{x}}\mid\boldsymbol{x})=-\frac{\tilde{\boldsymbol{x}}-\boldsymbol{x}}{\sigma^2}=-\frac{\boldsymbol{\epsilon}}{\sigma},$$

so DSM reduces to **noise prediction** $$\boldsymbol{s}_\theta\approx-\boldsymbol{\epsilon}/\sigma$$ with no Jacobian trace. Taking the conditional expectation recovers **Tweedie's formula**:

$$\nabla\log p_\sigma(\tilde{\boldsymbol{x}})=\mathbb{E}\big[-\boldsymbol{\epsilon}/\sigma\mid\tilde{\boldsymbol{x}}\big],\qquad \mathbb{E}[\boldsymbol{x}\mid\tilde{\boldsymbol{x}}]=\tilde{\boldsymbol{x}}+\sigma^2\,\nabla\log p_\sigma(\tilde{\boldsymbol{x}}).$$

The time-dependent version with $$\boldsymbol{x}_t=\alpha_t\boldsymbol{x}_0+\sigma_t\boldsymbol{\epsilon}$$ gives $$\nabla\log p_t(\boldsymbol{x}_t\mid\boldsymbol{x}_0)=-(\boldsymbol{x}_t-\alpha_t\boldsymbol{x}_0)/\sigma_t^2=-\boldsymbol{\epsilon}/\sigma_t$$, the conditional score regressed during training.

**D.3 — The probability-flow ODE preserves the SDE marginals (Prop. 4.1.1).** From Appendix B.5 the Fokker–Planck equation is *identically*

$$\partial_t p_t=-\nabla\!\cdot\!\big(\tilde{\boldsymbol{\mu}}\,p_t\big),\qquad \tilde{\boldsymbol{\mu}}(\boldsymbol{x},t)=\boldsymbol{f}-\tfrac12 g^2\boldsymbol{s}.$$

This is the continuity equation (Appendix B.3) of the **deterministic** ODE $$\dot{\boldsymbol{x}}=\tilde{\boldsymbol{\mu}}(\boldsymbol{x},t)$$ — zero diffusion. A continuity equation determines the marginals uniquely from $$p_0$$, so the SDE and the ODE, obeying the *same* equation, have the *same* time-$$t$$ marginals $$p_t$$. Only the **paths** differ: the SDE's are stochastic and mix, the ODE's are deterministic and never cross (which is what makes the ODE invertible and yields exact likelihoods).

The algebraic identity behind all of this — and behind the reverse-SDE drift derived in the time-reversal block above — is

$$\tfrac12 g^2\Delta p_t=\nabla\!\cdot\!\big(\tfrac12 g^2 p_t\,\boldsymbol{s}\big),$$

i.e. *a diffusion term equals an advection term driven by the score*. Splitting the full $$g^2\Delta$$ entirely into advection produces the reverse SDE drift $$\boldsymbol{f}-g^2\boldsymbol{s}$$ with noise $$g\,\mathrm{d}\bar{\boldsymbol{w}}$$ (Anderson); splitting only half produces the noise-free PF-ODE drift $$\boldsymbol{f}-\tfrac12 g^2\boldsymbol{s}$$. Same marginals, with a one-parameter family of stochastic-to-deterministic samplers in between (the SDE↔ODE "churn" of Karras et al., 2022).

</details>

### 2.3 DDIM as Euler on the probability-flow ODE

> **DDIM** is the deterministic sampler obtained by applying the **Euler method** to the probability-flow ODE. It is non-Markovian, so it can take large steps with the *same* trained model.
{:.lead}

![DDIM is exactly the Euler discretization of the probability-flow ODE.](/assets/figures/day08/pdm_ddim_euler.png)

Euler's method (Day 1) approximates $$\boldsymbol{x}(s)\approx\boldsymbol{x}(t) + (s-t)\,\dot{\boldsymbol{x}}(t)$$ using the slope at the start of the step. Applied to the PF-ODE $$\dot{\boldsymbol{x}} = \boldsymbol{f}-\tfrac12 g^2\boldsymbol{s}_\theta$$ it reproduces the **DDIM update** exactly. The cleanest way to see this uses the denoiser identity $$\boldsymbol{x}_t=\alpha_t\hat{\boldsymbol{x}}_0+\sigma_t\boldsymbol{\epsilon}_\theta$$: a single Euler step is equivalent to **freezing** the prediction $$(\hat{\boldsymbol{x}}_0,\boldsymbol{\epsilon}_\theta)$$ at $$t$$ and re-evaluating the forward rule at the next time $$s<t$$,

$$\begin{aligned}
\boldsymbol{x}_s
&= \alpha_s\,\textcolor{teal}{\hat{\boldsymbol{x}}_0} + \sigma_s\,\textcolor{purple}{\boldsymbol{\epsilon}_\theta(\boldsymbol{x}_t,t)} & &\text{(re-noise the frozen prediction to time }s\text{)} \\
&= \alpha_s\,\frac{\boldsymbol{x}_t-\sigma_t\,\textcolor{purple}{\boldsymbol{\epsilon}_\theta}}{\alpha_t} + \sigma_s\,\textcolor{purple}{\boldsymbol{\epsilon}_\theta} & &\big(\textcolor{teal}{\hat{\boldsymbol{x}}_0}=(\boldsymbol{x}_t-\sigma_t\boldsymbol{\epsilon}_\theta)/\alpha_t\big) \\
&= \frac{\alpha_s}{\alpha_t}\,\boldsymbol{x}_t + \alpha_s\Big(\frac{\sigma_s}{\alpha_s}-\frac{\sigma_t}{\alpha_t}\Big)\textcolor{purple}{\boldsymbol{\epsilon}_\theta} & &\text{(rearrange).}
\end{aligned}$$

Dividing the last line by $$\alpha_s$$ gives $$\boldsymbol{x}_s/\alpha_s = \boldsymbol{x}_t/\alpha_t + (\rho_s-\rho_t)\,\boldsymbol{\epsilon}_\theta$$ with $$\rho_t:=\sigma_t/\alpha_t$$ — literally an **Euler step in the log-SNR clock**, which is the modern view of DDIM (and the base case of DPM-Solver). The consequences are large:

- **Step skipping.** Because DDIM is non-Markovian (it depends on the marginal at $$t$$, not a Markov chain), we can use a coarse time grid — e.g. 20–50 steps instead of DDPM's 1000 — with no retraining.
- **Determinism.** With the noise fixed (an ODE, no $$\mathrm{d}\boldsymbol{w}$$), the map from initial noise to sample is deterministic and invertible — enabling latent interpolation, semantic editing, and exact reconstruction.

DDIM is **first-order**: its local error per step is $$O(\Delta t^2)$$, which is why very few steps still show artifacts. Higher-order solvers do better.

### 2.4 Discretization error and step count

> **Discretization error** is the gap between the exact ODE solution and its numerical approximation. It shrinks as steps increase and grows with trajectory curvature, accumulating along the path.
{:.lead}

![Sampling a 2-D distribution with the Score SDE; too few steps leaves visible structure errors.](/assets/figures/day08/pdm_score_sde_2d.png)

Two sources of error compound. **Local** error is made at each step (for Euler, $$O(\Delta t^2)$$ per step, $$O(\Delta t)$$ globally); it shrinks as we add steps. But more steps mean more NFEs — the very cost we want to avoid. **Curvature** is the other factor: the more the trajectory bends, the worse a straight-line (Euler) step approximates it.

This diagnosis points to exactly two remedies, which structure the rest of the day:

1. **Use a better integrator** — higher-order or exponential solvers that take the curvature into account (next section).
2. **Make the trajectory straighter** — so even a crude integrator suffices (rectified flow from Day 7, and flow maps below).

### 2.5 Stochastic versus deterministic samplers

> **SDE (stochastic)** samplers inject fresh noise each step and can self-correct errors; **ODE (deterministic)** samplers are noise-free and need fewer steps. Both share the same marginals $$p_t$$ by the Fokker–Planck equation.
{:.lead}

The reverse SDE and the PF-ODE produce the same distribution but behave differently as samplers:

- **SDE samplers** add noise at every step. That noise is self-correcting — errors made early can be "washed out" — so SDE samplers tend to win at **high** NFE budgets and produce slightly more diverse, detailed samples.
- **ODE samplers** are deterministic and smoother, dominating at **low** NFE budgets where every evaluation counts.

A useful hybrid (EDM's "churn") runs an ODE solver but injects a controlled amount of noise per step, interpolating between the two regimes. Underlying all of this is the **Fokker–Planck equation**, which guarantees that the deterministic and stochastic dynamics transport the *same* density $$p_t$$ — explore how a density flows under it:


<div class="interactive-viz" markdown="0">
  <iframe src="https://the-principles-of-diffusion-models.github.io/assets/fokker_planck.html" width="100%" height="780" loading="lazy" style="border:1px solid #ddd;border-radius:8px;" title="Fokker–Planck Equation"></iframe>
  <p style="font-size:0.85em;color:#888;margin-top:4px;">Interactive: <strong>Fokker–Planck Equation</strong> — <a href="https://the-principles-of-diffusion-models.github.io/assets/fokker_planck.html" target="_blank" rel="noopener">open in new tab</a>. Source: <em>The Principles of Diffusion Models</em>.</p>
</div>


## 3. Fast High-Order Solvers

### 3.1 Higher-order integration: Heun's method

> **Heun's method** is a second-order predictor–corrector: take an Euler step, evaluate the slope at the predicted endpoint, and step again with the **average** of the two slopes. Local error drops to $$O(\Delta t^3)$$.
{:.lead}

![A higher-order (Heun) solver stepped in log-SNR time tracks the trajectory far more accurately than Euler.](/assets/figures/day08/pdm_heun_logsnr.png)

Euler uses only the slope at the start of the interval, so it consistently misses the bend of a curved trajectory. Heun corrects this:

1. **Predict** with Euler: $$\tilde{\boldsymbol{x}} = \boldsymbol{x}_t - \Delta t\,\boldsymbol{d}(\boldsymbol{x}_t,t)$$.
2. **Correct** with the average slope: $$\boldsymbol{x}_{t-\Delta t} = \boldsymbol{x}_t - \tfrac{\Delta t}{2}\big[\boldsymbol{d}(\boldsymbol{x}_t,t) + \boldsymbol{d}(\tilde{\boldsymbol{x}},t-\Delta t)\big]$$.

This costs **2 NFEs per step** but is second-order ($$O(\Delta t^3)$$ local error), so it needs dramatically fewer steps for the same accuracy — a net win. Heun (with the schedule below) is the core of the widely used **EDM** sampler, which reaches excellent quality in ~20–40 NFEs. Compare Euler and Heun directly:


<div class="interactive-viz" markdown="0">
  <iframe src="https://the-principles-of-diffusion-models.github.io/assets/euler_vs_heun_solver.html" width="100%" height="760" loading="lazy" style="border:1px solid #ddd;border-radius:8px;" title="Euler vs Heun Solver"></iframe>
  <p style="font-size:0.85em;color:#888;margin-top:4px;">Interactive: <strong>Euler vs Heun Solver</strong> — <a href="https://the-principles-of-diffusion-models.github.io/assets/euler_vs_heun_solver.html" target="_blank" rel="noopener">open in new tab</a>. Source: <em>The Principles of Diffusion Models</em>.</p>
</div>


### 3.2 The right clock: log-SNR time

> Solver accuracy depends on the **time parameterization**. Stepping uniformly in **log-SNR** $$\lambda_t=\log(\alpha_t^2/\sigma_t^2)$$ (or EDM's $$\sigma$$ schedule) places steps where the trajectory changes fastest.
{:.lead}

An ODE can be re-parameterized by any monotonic change of the time variable, and the **choice matters enormously** for a discretized solver. Uniform steps in the native $$t$$ waste effort where the trajectory is nearly straight and under-resolve where it bends. The trajectory's "natural clock" is the **log signal-to-noise ratio** $$\lambda_t=\log(\alpha_t^2/\sigma_t^2)$$ from Day 6: stepping uniformly in $$\lambda$$ allocates steps in proportion to how much the sample actually moves.

EDM expresses the same insight through a carefully designed $$\sigma$$ (noise-level) schedule, concentrating steps at the small-noise end where detail is formed. The headline is that **a better schedule is free quality** — no retraining, no extra NFEs, just a smarter choice of where to place the steps you already take.

### 3.3 Exponential integrators: DPM-Solver and DEIS

> The PF-ODE has a **semilinear** form: a linear drift plus a nonlinear network term. **Exponential integrators** solve the linear part *exactly* (via an integrating factor) and only approximate the smooth nonlinear part.
{:.lead}

![Multistep exponential integrators (DEIS) reuse previous evaluations to achieve high order at low cost.](/assets/figures/day08/pdm_deis.png)

The diffusion ODE has a **semilinear** form — a linear term in $$\boldsymbol{x}$$ plus a term involving the network:

$$\dot{\boldsymbol{x}} = \textcolor{teal}{a(t)\,\boldsymbol{x}} + \textcolor{purple}{b(t)\,\boldsymbol{\epsilon}_\theta(\boldsymbol{x},t)}.$$

The linear part is exactly the kind of equation we solved on Day 1 with an **integrating factor** $$e^{-A(t)}$$, $$A(t)=\int_0^t a(u)\,\mathrm{d}u$$. Multiplying through collapses the linear term into a total derivative, and integrating from $$t$$ to $$s$$ gives **variation of constants**:

$$\begin{aligned}
\frac{\mathrm{d}}{\mathrm{d}t}\Big(e^{-A(t)}\boldsymbol{x}\Big)
&= e^{-A(t)}\big(\dot{\boldsymbol{x}} - \textcolor{teal}{a(t)\boldsymbol{x}}\big) = e^{-A(t)}\,\textcolor{purple}{b(t)\,\boldsymbol{\epsilon}_\theta} & &\text{(integrating factor)} \\
\Longrightarrow\quad \boldsymbol{x}_s &= \underbrace{e^{A(s)-A(t)}\,\boldsymbol{x}_t}_{\textcolor{teal}{\text{linear part: exact}}} \;+\; \underbrace{\int_t^s e^{A(s)-A(u)}\,b(u)\,\textcolor{purple}{\boldsymbol{\epsilon}_\theta(\boldsymbol{x}_u,u)}\,\mathrm{d}u}_{\textcolor{purple}{\text{network part: approximate}}}.
\end{aligned}$$

The first term is **exact** — no discretization error from the stiff linear drift. Only the smooth integral of $$\boldsymbol{\epsilon}_\theta$$ is approximated: freezing $$\boldsymbol{\epsilon}_\theta$$ at $$t$$ recovers DDIM (first order); a Taylor/multistep expansion of $$\boldsymbol{\epsilon}_\theta$$ in the log-SNR variable $$\lambda$$ gives the high-order solvers:

- **DPM-Solver** applies this exponential-integrator idea with high-order Taylor expansions of $$\boldsymbol{\epsilon}_\theta$$ along log-SNR time.
- **DEIS** uses a **multistep** variant, reusing network evaluations from previous steps (like Adams–Bashforth) to reach high order with roughly one NFE per step.

Together these reach high-quality samples in **10–20 NFEs**, an order of magnitude fewer than naive DDPM, and are the default fast samplers in many production systems.

## 4. Few-Step Sampling: Flow Maps and Distillation

### 4.1 The bottleneck: many function evaluations

> Even the best solvers need roughly $$10$$–$$50$$ NFEs. **Few-step sampling** aims for $$1$$–$$4$$ NFEs by learning to **jump** across time rather than integrate step by step.
{:.lead}

**Why this matters.** Interactive and real-time applications — drawing tools, video, on-device generation — cannot afford dozens of network passes per sample. No matter how clever the solver, integrating an ODE inherently requires multiple evaluations to follow a curved path accurately.

The shift in thinking is from *integration* to *amortization*: instead of repeatedly querying the slope and crawling along the trajectory, **learn a function that maps directly** from a point at one time to the corresponding point at another. We can train such a function by distilling a slow multi-step **teacher** into a fast **student**, or by enforcing self-consistency. Flow maps make this precise.

### 4.2 Flow maps

> A **flow map** $$\Phi_{s\to t}$$ sends a sample at time $$s$$ to its ODE solution at time $$t$$ in a single learned evaluation: $$\boldsymbol{x}_t = \Phi_{s\to t}(\boldsymbol{x}_s).$$
{:.lead}

![A flow map integrates the probability-flow ODE between two times in one step, rather than many small Euler steps.](/assets/figures/day08/pdm_flowmap.png)

The PF-ODE defines, for any pair of times $$(s,t)$$, an exact map taking $$\boldsymbol{x}_s$$ to $$\boldsymbol{x}_t$$ — the solution operator of the ODE. A numerical solver approximates this map by chaining many tiny steps; a **flow map** $$\Phi_{s\to t}$$ *learns it directly* so it can be applied in one shot. The special case $$\Phi_{T\to 0}$$ is a **one-step generator**: noise in, sample out. The remaining questions are how to parameterize $$\Phi$$ consistently across times and how to train it without simply storing a teacher's whole trajectory — answered by the semigroup/consistency structure. Compare flow-map model families:


<div class="interactive-viz" markdown="0">
  <iframe src="https://the-principles-of-diffusion-models.github.io/assets/flow_map_models.html" width="100%" height="780" loading="lazy" style="border:1px solid #ddd;border-radius:8px;" title="Flow Map Models Comparison"></iframe>
  <p style="font-size:0.85em;color:#888;margin-top:4px;">Interactive: <strong>Flow Map Models Comparison</strong> — <a href="https://the-principles-of-diffusion-models.github.io/assets/flow_map_models.html" target="_blank" rel="noopener">open in new tab</a>. Source: <em>The Principles of Diffusion Models</em>.</p>
</div>


### 4.3 The semigroup property

> Flow maps compose: stepping $$s\to u$$ then $$u\to t$$ equals stepping $$s\to t$$ directly. This **semigroup property** $$\Phi_{s\to t} = \Phi_{u\to t}\circ\Phi_{s\to u}$$ underlies self-consistent training.
{:.lead}

![The semigroup (consistency) property: composing partial jumps must equal the full jump.](/assets/figures/day08/pdm_flowmap_semigroup.png)

Because the flow map is the solution operator of a deterministic ODE, it must satisfy

$$\Phi_{s\to t} = \Phi_{u\to t}\circ\Phi_{s\to u}\quad\text{for any intermediate } u,$$

and in particular all points on a single ODE trajectory map to the **same** clean sample $$\boldsymbol{x}_0$$. This is the **consistency** condition. It is powerful because it is a *self-supervised* constraint: a network can be trained to satisfy it by penalizing the discrepancy between $$\Phi_{s\to t}(\boldsymbol{x}_s)$$ and $$\Phi_{u\to t}(\Phi_{s\to u}(\boldsymbol{x}_s))$$, **without** needing a precomputed teacher trajectory. Enforcing consistency is what lets a one-step map be learned stably.

### 4.4 Consistency models and distillation

> **Consistency models** train a network $$f_\theta(\boldsymbol{x}_t,t)\approx\boldsymbol{x}_0$$ that is consistent along trajectories, enabling one- or few-step generation. **Distillation** alternatively matches a multi-step teacher.
{:.lead}

![The flow-map timeline: a learned map jumps directly toward the data manifold, collapsing many steps into a few.](/assets/figures/day08/pdm_flowmap_timeline.png)

Two routes to a fast student:

- **Consistency training** uses the self-consistency condition above as the loss, learning $$f_\theta(\boldsymbol{x}_t,t)\approx\boldsymbol{x}_0$$ directly (from scratch or from a pretrained score model). Sampling is then one evaluation $$\boldsymbol{x}_0=f_\theta(\boldsymbol{x}_T,T)$$, optionally refined by a few alternating noise/denoise steps.
- **Distillation** trains the student to reproduce, in a few steps, what a slow many-step teacher produces — progressive distillation halves the step count repeatedly; adversarial and trajectory-matching variants push to 1–4 steps near teacher quality.

The trade-off is explicit: a small drop in sample quality for a $$10$$–$$1000\times$$ speedup. This is the active frontier that brings diffusion toward **real-time** generation — and it ties the whole week together, since it relies on the unified score/velocity model (Day 7), the ODE-as-sampler view, and the numerical-analysis toolkit of today.

## Checkpoint summary

Before moving to the practical, confirm you can:

- Explain conditional diffusion and why plain conditioning often under-uses the condition.
- Derive classifier-free guidance from Bayes' rule and the score↔noise identity, and explain the role of w.
- Argue that sampling is numerical integration and that DDIM is Euler on the probability-flow ODE.
- Describe the sources of discretization error and the two ways to reduce required NFEs.
- Contrast stochastic and deterministic samplers and state what the Fokker–Planck equation guarantees.
- Explain Heun's method, log-SNR time stepping, and exponential integrators (DPM-Solver/DEIS).
- Define a flow map and the semigroup/consistency property, and explain how consistency models reach one-step sampling.
- Looking ahead: diffusion fought the cost of *many sampling steps* (NFEs) with solvers and flow maps. Day 9 returns to the autoregressive family from the Day 6 taxonomy, where the analogous cost is *serial token-by-token decoding* — fought on Day 10 with the KV cache.
