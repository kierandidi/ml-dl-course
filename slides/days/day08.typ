#import "../lib.typ": *

#show: course-theme.with(title: [Guidance, Solvers & Fast Sampling], subtitle: [Day 8 | Aug 2026])

= Day 8: Guidance, Solvers & Fast Sampling

== Welcome

- *Guidance, Solvers & Fast Sampling* — Controlling and accelerating diffusion
- 3 hours lecture + practical
- Slides, notes, and code on the course site

== Outline

- Conditional Generation & Guidance
- Sampling = Solving a Differential Equation
- Fast High-Order Solvers
- Few-Step Sampling

= 1 · Conditional Generation & Guidance

== 1.1  Conditioning a Diffusion Model

- Goal: sample $p(x | c)$ — class, text prompt, image
- Train the denoiser with the condition: $epsilon_theta (x_t, t, c)$
- Conditional score $nabla log p_t (x | c)$ steers sampling
- But naive conditioning often *under-uses* the condition
- Guidance amplifies the influence of $c$

== 1.2  Classifier Guidance

- Bayes: $nabla log p(x|c) = nabla log p(x) + nabla log p(c|x)$
- Train a classifier $p_phi (c | x_t)$ on noisy inputs
- Push samples toward high $p(c|x)$ using its gradient
- Guidance scale $w$: $nabla log p + w nabla log p(c|x)$
- Needs a separate noisy-data classifier (a drawback)

== 1.2  Classifier Guidance

#align(center + horizon)[#image("/assets/figures/day08/pdm_guidance.png", width: 92%, height: 82%, fit: "contain")]

== 1.3  Classifier-Free Guidance

- Train one net for both conditional & unconditional (drop $c$)
- Combine at sampling time, no external classifier:
- $tilde(epsilon) = (1+w) epsilon_theta (x,t,c) - w epsilon_theta (x,t,nothing)$
- $w$ trades diversity (low) vs fidelity/prompt-match (high)
- The workhorse of text-to-image models

== 1.3  Classifier-Free Guidance

#align(center + horizon)[#image("/assets/figures/day08/pdm_cfg.png", width: 92%, height: 82%, fit: "contain")]

== 1.4  Derivation: Guidance from Bayes

- Bayes: $p(x|c) prop p(x) p(c|x)$
- Take $nabla_x log$: scores add
- Implicit classifier: $nabla log p(c|x) = nabla log p(x|c) - nabla log p(x)$
- Score form $arrow.r$ noise form via $s = -epsilon \\/ sigma_t$
- Sharpen: raise $p(c|x)$ to power $w arrow.r$ guidance scale

= 2 · Sampling = Solving a Differential Equation

== 2.1  Sampling Is Numerical Integration

- Day 7: sampling = integrate the reverse SDE or PF-ODE
- Each step needs one network eval (NFE) of the score
- Quality vs cost = solver accuracy vs number of steps
- Discretize time $T = t_0 > t_1 > dots.h > t_N = 0$
- Better solver $arrow.r$ same quality in fewer steps

== 2.1  Sampling Is Numerical Integration

#align(center + horizon)[#image("/assets/figures/day08/pdm_reverse_sde.png", width: 92%, height: 82%, fit: "contain")]

== 2.2  Derivation: PF-ODE from the FPE

- FPE: $partial_t p_t = - nabla dot (f p_t) + 1/2 g^2 Delta p_t$
- Log-derivative trick: $nabla p_t = p_t s$, $s = nabla log p_t$
- Factor $p_t$ out $arrow.r$ Liouville: $partial_t p_t = - nabla dot ( tilde(mu) p_t )$
- $tilde(mu) = f - 1/2 g^2 s$ $arrow.r$ PF-ODE (vs reverse SDE: $f - g^2 s$)
- Full derivation: optional notes block

== 2.2  Derivation: PF-ODE from the FPE

#align(center + horizon)[#image("/assets/figures/day08/pdm_ddim_euler.png", width: 92%, height: 82%, fit: "contain")]

== 2.3  DDIM as Euler on the PF-ODE

- DDIM = deterministic sampler = Euler on the PF-ODE
- Non-Markovian: skip steps without retraining
- Same trained model, far fewer steps than DDPM
- Deterministic $arrow.r$ reproducible, invertible, editable
- First-order: error per step $O(Delta t^2)$

== 2.3  DDIM as Euler on the PF-ODE

#align(center + horizon)[#image("/assets/figures/day08/pdm_score_sde_2d.png", width: 92%, height: 82%, fit: "contain")]

== 2.4  Discretization Error & Step Count

- Few steps + 1st-order $arrow.r$ visible artifacts
- Error accumulates over the trajectory
- More steps shrink error but cost more NFEs
- Curved trajectories are harder to integrate
- Two fixes: better solver, or straighter paths

== 2.5  Stochastic vs Deterministic Samplers

- SDE samplers inject noise each step (self-correcting)
- ODE samplers are deterministic (fast, fewer steps)
- SDE: better at high NFE; ODE: better at low NFE
- Churn: add a little noise to an ODE solver (EDM)
- Fokker-Planck: both share the same marginals $p_t$

== 2.5  Stochastic vs Deterministic Samplers

#align(center + horizon)[#image("/assets/figures/day08/pdm_heun_logsnr.png", width: 92%, height: 82%, fit: "contain")]

= 3 · Fast High-Order Solvers

== 3.1  Higher-Order: Heun's Method

- Euler uses the slope at the start of the step
- Heun averages start & end slopes (predictor-corrector)
- 2nd-order: error per step $O(Delta t^3)$
- 2 NFEs/step but far fewer steps overall
- Backbone of the EDM sampler

== 3.2  The Right Clock: log-SNR Time

- Solver accuracy depends on the time variable
- Step uniformly in log-SNR $lambda = log(alpha^2 \\/ sigma^2)$
- Spends steps where the trajectory bends most
- EDM uses a tailored $sigma$ schedule (same idea)
- Good schedule $arrow.r$ big quality gain for free

== 3.2  The Right Clock: log-SNR Time

#align(center + horizon)[#image("/assets/figures/day08/pdm_deis.png", width: 92%, height: 82%, fit: "contain")]

== 3.3  Exponential Integrators (DPM-Solver / DEIS)

- PF-ODE = linear drift + nonlinear network term
- Solve the linear part *exactly* (integrating factor, Day 1)
- Only approximate the smooth network part
- Multistep: reuse past evals for higher order
- 10-20 NFEs for high quality (DPM-Solver, DEIS)

= 4 · Few-Step Sampling

== 4.1  The Bottleneck: Many Function Evals

- Even good solvers need ~10-50 NFEs
- Real-time / interactive needs 1-4 steps
- Idea: learn to *jump* across time, not crawl
- Distill a slow teacher into a fast student
- Flow maps formalize the jump

== 4.1  The Bottleneck: Many Function Evals

#align(center + horizon)[#image("/assets/figures/day08/pdm_flowmap.png", width: 92%, height: 82%, fit: "contain")]

== 4.2  Flow Maps

- Flow map $Phi_(s arrow.r t)$: jump a sample from time $s$ to $t$
- Integrates the ODE in *one* learned step
- $x_t = Phi_(s arrow.r t)(x_s)$ for any $s, t$
- Generalizes the single-step generator
- Learn it by distillation or self-consistency

== 4.2  Flow Maps

#align(center + horizon)[#image("/assets/figures/day08/pdm_flowmap_semigroup.png", width: 92%, height: 82%, fit: "contain")]

== 4.3  The Semigroup Property

- Composing jumps = one bigger jump:
- $Phi_(s arrow.r t) = Phi_(u arrow.r t) circle.small Phi_(s arrow.r u)$
- Consistency: all points on a trajectory map to the same $x_0$
- Self-consistency gives a training signal w/o a teacher
- Enables 1-step generation

== 4.3  The Semigroup Property

#align(center + horizon)[#image("/assets/figures/day08/pdm_flowmap_timeline.png", width: 92%, height: 82%, fit: "contain")]

== 4.4  Consistency & Distillation

- Consistency models: train $f_theta (x_t, t) approx x_0$ for all $t$
- Distillation: match a multi-step teacher in few steps
- 1-4 step sampling, near teacher quality
- Trade a little quality for huge speedups
- Active frontier: real-time generative models

== Summary

- Day 8: *Guidance, Solvers & Fast Sampling*
- Controlling and accelerating diffusion
- Questions welcome — practical follows

== Questions?

#align(center + horizon)[
  #text(size: 44pt, weight: "bold", fill: course-primary)[Questions?]

  #v(1em)
  Practical session follows — see course site.
]
