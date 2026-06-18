---
layout: post
title: Day 8 - Guidance, Solvers, and Few-Step Sampling
image: /assets/img/sampling_space.png
accent_image: 
  background: url('/assets/img/sampling_space.png') center/cover
  overlay: false
accent_color: '#ccc'
theme_color: '#ccc'
description: >
  Controlling diffusion with guidance, sampling as ODE/SDE solving, fast high-order solvers, and one-step flow maps.
invert_sidebar: true
---

# Day 8 - Guidance, Solvers, and Few-Step Sampling

### Optional reading for this lesson
- [The Principles of Diffusion Models](https://arxiv.org/abs/2510.21890), Ch. 8–11
- [Ho & Salimans — Classifier-Free Diffusion Guidance (2022)](https://arxiv.org/abs/2207.12598)
- [Song et al. — Denoising Diffusion Implicit Models (DDIM, 2021)](https://arxiv.org/abs/2010.02502)
- [Karras et al. — Elucidating the Design Space of Diffusion Models (EDM, 2022)](https://arxiv.org/abs/2206.00364)
- [Song et al. — Consistency Models (2023)](https://arxiv.org/abs/2303.01469)

### [Slides](/assets/slides/day08.pdf)

### [Practical](/projects/day08-practical/)

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

![Guidance steers the sampling trajectory toward regions consistent with the condition (Principles Fig 8.1).](/assets/figures/day08/pdm_guidance.png)

By Bayes' rule, the conditional score decomposes into the unconditional score plus the gradient of a classifier's log-likelihood. So if we train a classifier $$p_\phi(c\mid\boldsymbol{x}_t)$$ **on noisy inputs** (it must operate at every noise level), we can steer an unconditional model:

$$\tilde{\boldsymbol{s}}(\boldsymbol{x}_t,t) = \nabla\log p_t(\boldsymbol{x}_t) + w\,\nabla_{\boldsymbol{x}_t}\log p_\phi(c\mid\boldsymbol{x}_t).$$

The weight $$w$$ controls how hard we push toward the class. The drawbacks are practical: you need a separate classifier trained on noised data, and its gradients can be brittle. Classifier-free guidance removes the external classifier entirely.

### 1.3 Classifier-free guidance

> **Classifier-free guidance (CFG)** trains a single network to be both conditional and unconditional (by randomly dropping $$c$$), then extrapolates at sampling time: $$\tilde{\boldsymbol{\epsilon}} = (1+w)\,\boldsymbol{\epsilon}_\theta(\boldsymbol{x}_t,t,c) - w\,\boldsymbol{\epsilon}_\theta(\boldsymbol{x}_t,t,\varnothing).$$
{:.lead}

![Classifier-free guidance interpolates/extrapolates between conditional and unconditional predictions to control fidelity (Principles Fig 8.2).](/assets/figures/day08/pdm_cfg.png)

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

![Sampling runs a reverse-time process from noise back to data; numerically, this is integrating a differential equation (Principles Fig 4.4).](/assets/figures/day08/pdm_reverse_sde.png)

On Day 7 we derived two dynamics with the right marginals — the reverse SDE and the PF-ODE. Turning either into an algorithm means choosing a **time discretization** $$T=t_0>t_1>\dots>t_N=0$$ and a **numerical scheme** to step between consecutive times. The cost is dominated by the number of network evaluations (NFEs), since the network (the score/denoiser) is by far the most expensive operation.

This reframing is liberating: the quality–speed trade-off becomes a classic **numerical-analysis** question. A more accurate integrator reaches the same quality in fewer steps. Everything in this lecture is an answer to "how do we integrate this ODE well in as few NFEs as possible?".

### 2.2 DDIM as Euler on the probability-flow ODE

> **DDIM** is the deterministic sampler obtained by applying the **Euler method** to the probability-flow ODE. It is non-Markovian, so it can take large steps with the *same* trained model.
{:.lead}

![DDIM is exactly the Euler discretization of the probability-flow ODE (Principles Fig 9.1).](/assets/figures/day08/pdm_ddim_euler.png)

Euler's method (Day 1) approximates $$\boldsymbol{x}(t-\Delta t)\approx\boldsymbol{x}(t) - \Delta t\,\dot{\boldsymbol{x}}(t)$$ using the slope at the start of the step. Applying this to the PF-ODE $$\dot{\boldsymbol{x}} = \boldsymbol{f}-\tfrac12 g^2\boldsymbol{s}_\theta$$ reproduces the DDIM update exactly. The consequences are large:

- **Step skipping.** Because DDIM is non-Markovian (it depends on the marginal at $$t$$, not a Markov chain), we can use a coarse time grid — e.g. 20–50 steps instead of DDPM's 1000 — with no retraining.
- **Determinism.** With the noise fixed (an ODE, no $$\mathrm{d}\boldsymbol{w}$$), the map from initial noise to sample is deterministic and invertible — enabling latent interpolation, semantic editing, and exact reconstruction.

DDIM is **first-order**: its local error per step is $$O(\Delta t^2)$$, which is why very few steps still show artifacts. Higher-order solvers do better.

### 2.3 Discretization error and step count

> **Discretization error** is the gap between the exact ODE solution and its numerical approximation. It shrinks as steps increase and grows with trajectory curvature, accumulating along the path.
{:.lead}

![Sampling a 2-D distribution with the Score SDE; too few steps leaves visible structure errors (Principles Fig 4.7).](/assets/figures/day08/pdm_score_sde_2d.png)

Two sources of error compound. **Local** error is made at each step (for Euler, $$O(\Delta t^2)$$ per step, $$O(\Delta t)$$ globally); it shrinks as we add steps. But more steps mean more NFEs — the very cost we want to avoid. **Curvature** is the other factor: the more the trajectory bends, the worse a straight-line (Euler) step approximates it.

This diagnosis points to exactly two remedies, which structure the rest of the day:

1. **Use a better integrator** — higher-order or exponential solvers that take the curvature into account (next section).
2. **Make the trajectory straighter** — so even a crude integrator suffices (rectified flow from Day 7, and flow maps below).

### 2.4 Stochastic versus deterministic samplers

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

![A higher-order (Heun) solver stepped in log-SNR time tracks the trajectory far more accurately than Euler (Principles Fig 9.3).](/assets/figures/day08/pdm_heun_logsnr.png)

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

![Multistep exponential integrators (DEIS) reuse previous evaluations to achieve high order at low cost (Principles Fig 9.2).](/assets/figures/day08/pdm_deis.png)

The diffusion ODE can be written as $$\dot{\boldsymbol{x}} = a(t)\,\boldsymbol{x} + b(t)\,\boldsymbol{\epsilon}_\theta(\boldsymbol{x},t)$$ — a **linear** term plus a term involving the network. The linear part is exactly the kind of equation we solved on Day 1 with an **integrating factor**, so we can integrate it in closed form and avoid the error a generic solver would make on it. Only the slowly varying network term $$\boldsymbol{\epsilon}_\theta$$ needs numerical approximation:

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

![A flow map integrates the probability-flow ODE between two times in one step, rather than many small Euler steps (Principles Fig 11.2).](/assets/figures/day08/pdm_flowmap.png)

The PF-ODE defines, for any pair of times $$(s,t)$$, an exact map taking $$\boldsymbol{x}_s$$ to $$\boldsymbol{x}_t$$ — the solution operator of the ODE. A numerical solver approximates this map by chaining many tiny steps; a **flow map** $$\Phi_{s\to t}$$ *learns it directly* so it can be applied in one shot. The special case $$\Phi_{T\to 0}$$ is a **one-step generator**: noise in, sample out. The remaining questions are how to parameterize $$\Phi$$ consistently across times and how to train it without simply storing a teacher's whole trajectory — answered by the semigroup/consistency structure. Compare flow-map model families:


<div class="interactive-viz" markdown="0">
  <iframe src="https://the-principles-of-diffusion-models.github.io/assets/flow_map_models.html" width="100%" height="780" loading="lazy" style="border:1px solid #ddd;border-radius:8px;" title="Flow Map Models Comparison"></iframe>
  <p style="font-size:0.85em;color:#888;margin-top:4px;">Interactive: <strong>Flow Map Models Comparison</strong> — <a href="https://the-principles-of-diffusion-models.github.io/assets/flow_map_models.html" target="_blank" rel="noopener">open in new tab</a>. Source: <em>The Principles of Diffusion Models</em>.</p>
</div>


### 4.3 The semigroup property

> Flow maps compose: stepping $$s\to u$$ then $$u\to t$$ equals stepping $$s\to t$$ directly. This **semigroup property** $$\Phi_{s\to t} = \Phi_{u\to t}\circ\Phi_{s\to u}$$ underlies self-consistent training.
{:.lead}

![The semigroup (consistency) property: composing partial jumps must equal the full jump (Principles Fig 11.3).](/assets/figures/day08/pdm_flowmap_semigroup.png)

Because the flow map is the solution operator of a deterministic ODE, it must satisfy

$$\Phi_{s\to t} = \Phi_{u\to t}\circ\Phi_{s\to u}\quad\text{for any intermediate } u,$$

and in particular all points on a single ODE trajectory map to the **same** clean sample $$\boldsymbol{x}_0$$. This is the **consistency** condition. It is powerful because it is a *self-supervised* constraint: a network can be trained to satisfy it by penalizing the discrepancy between $$\Phi_{s\to t}(\boldsymbol{x}_s)$$ and $$\Phi_{u\to t}(\Phi_{s\to u}(\boldsymbol{x}_s))$$, **without** needing a precomputed teacher trajectory. Enforcing consistency is what lets a one-step map be learned stably.

### 4.4 Consistency models and distillation

> **Consistency models** train a network $$f_\theta(\boldsymbol{x}_t,t)\approx\boldsymbol{x}_0$$ that is consistent along trajectories, enabling one- or few-step generation. **Distillation** alternatively matches a multi-step teacher.
{:.lead}

![The flow-map timeline: a learned map jumps directly toward the data manifold, collapsing many steps into a few (Principles Fig 11.1).](/assets/figures/day08/pdm_flowmap_timeline.png)

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
