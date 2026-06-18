#import "../lib.typ": *

#show: course-theme.with(title: [Score, SDEs & Flow Matching], subtitle: [Day 7 | Aug 2026])

= Day 7: Score, SDEs & Flow Matching

== Welcome

- *Score, SDEs & Flow Matching* — The continuous-time view of diffusion
- 3 hours lecture + practical
- Slides, notes, and code on the course site

== Outline

- The Score Function
- Learning the Score
- Sampling with the Score
- The Continuous-Time View
- Flow Matching
- One Model, Many Views

= 1 · The Score Function

== 1.1  What Is the Score?

- Score = gradient of log-density: $s(x) = nabla_x log p(x)$
- A *vector field* pointing toward higher probability
- Day 6 learned to denoise; today we learn this field
- Knowing the score is enough to *sample* (Langevin, SDE, ODE)
- Source: Principles of Diffusion Models, Ch. 3–6

== 1.1  What Is the Score?

#align(center + horizon)[#image("/assets/figures/day07/pdm_score_field.png", width: 92%, height: 82%, fit: "contain")]

== 1.2  Energy-Based Models

- Write $p_theta (x) = e^(-E_theta (x)) \\/ Z_theta$
- Normalizer $Z_theta = integral e^(-E_theta (x)) dif x$ is intractable
- Max-likelihood needs $nabla_theta log Z_theta$ — hard
- Idea: model the *score* and sidestep $Z$ entirely
- Flexible energy, but sampling needs MCMC

== 1.2  Energy-Based Models

#align(center + horizon)[#image("/assets/figures/day07/pdm_ebm_training.png", width: 92%, height: 82%, fit: "contain")]

== 1.3  Why the Score Avoids the Normalizer

- $log p_theta (x) = -E_theta (x) - log Z_theta$
- $Z_theta$ does not depend on $x$
- $arrow.r nabla_x log p_theta (x) = -nabla_x E_theta (x)$
- The intractable constant *vanishes* under $nabla_x$
- Learn the shape of $p$, not its normalization

= 2 · Learning the Score

== 2.1  Score Matching

- Fit $s_theta (x) approx nabla_x log p_"data" (x)$
- Naive loss needs the unknown true score
- Hyvärinen: integrate by parts to remove it
- $J = EE[1/2 \\|s_theta\\|^2 + "tr"(nabla_x s_theta)]$
- Trace (Jacobian) term is costly in high dimension

== 2.1  Score Matching

#align(center + horizon)[#image("/assets/figures/day07/pdm_score_matching.png", width: 92%, height: 82%, fit: "contain")]

== 2.2  Derivation: Implicit Score Matching

- Start: $1/2 EE_(p)\\|s_theta (x) - nabla log p(x)\\|^2$
- Expand the square; the cross term has $nabla log p$
- $EE_p [s_theta^T nabla log p] = EE_p[-nabla dot.op s_theta]$ (parts)
- Uses $p nabla log p = nabla p$ and vanishing boundary
- Leaves a loss in $s_theta$ alone (+ const)

== 2.3  Denoising Score Matching

- Add noise: $x_t = x_0 + sigma epsilon$, learn the *noisy* score
- Key identity: $nabla_(x_t) log p(x_t | x_0) = -(x_t - x_0)\\/sigma^2$
- Target is known in closed form $arrow.r$ no Jacobian trace
- $J_"DSM" = EE\\|s_theta (x_t) + (x_t - x_0)\\/sigma^2\\|^2$
- Score matching $=$ denoising — the Day 6 connection

== 2.3  Denoising Score Matching

#align(center + horizon)[#image("/assets/figures/day07/pdm_dsm_trick.png", width: 92%, height: 82%, fit: "contain")]

== 2.4  Multiple Noise Scales (NCSN)

- One noise level can't cover all of space
- Low noise: accurate near data, empty regions unseen
- High noise: fills space but blurs detail
- Train *one* network conditioned on noise level $sigma$
- Anneal from high to low noise when sampling

== 2.4  Multiple Noise Scales (NCSN)

#align(center + horizon)[#image("/assets/figures/day07/pdm_ncsn.png", width: 92%, height: 82%, fit: "contain")]

= 3 · Sampling with the Score

== 3.1  Langevin Dynamics

- Walk uphill in log-density, plus noise:
- $x_(k+1) = x_k + tau s_theta (x_k) + sqrt(2 tau) z_k$
- Stationary distribution is exactly $p(x)$
- Drift toward data + noise to explore
- Anneal $sigma$ (NCSN) for fast, stable mixing

== 3.1  Langevin Dynamics

#align(center + horizon)[#image("/assets/figures/day07/pdm_langevin.png", width: 92%, height: 82%, fit: "contain")]

= 4 · The Continuous-Time View

== 4.1  The Forward SDE

- Let steps $arrow.r$ 0: the noising chain becomes an SDE
- $dif x = f(x,t) dif t + g(t) dif w$ (drift + diffusion)
- VP-SDE corresponds to DDPM; VE-SDE to NCSN
- Marginal stays $p_t (x_t|x_0) = N(alpha_t x_0, sigma_t^2 I)$
- A whole *family* of distributions $p_t$, indexed by time

== 4.1  The Forward SDE

#align(center + horizon)[#image("/assets/figures/day07/pdm_forward_1d.png", width: 92%, height: 82%, fit: "contain")]

== 4.2  The Time-Dependent Score

- Now the score depends on time: $s(x,t) = nabla_x log p_t (x)$
- One network $s_theta (x, t)$ for all noise levels
- Smoothed at high $t$ (easy), sharp at low $t$ (hard)
- This is exactly the NCSN idea, in continuous time
- Trained by denoising score matching at each $t$

== 4.2  The Time-Dependent Score

#align(center + horizon)[#image("/assets/figures/day07/pdm_score_landscape.png", width: 92%, height: 82%, fit: "contain")]

== 4.3  Reverse SDE & Probability-Flow ODE

- Reverse SDE (Anderson): run time backward using the score
- $dif x = [f - g^2 nabla log p_t] dif t + g dif macron(w)$
- Probability-flow ODE: *same marginals*, no noise
- $dif x = [f - 1/2 g^2 nabla log p_t] dif t$
- SDE = stochastic sampler; ODE = deterministic + likelihood

== 4.3  Reverse SDE & Probability-Flow ODE

#align(center + horizon)[#image("/assets/figures/day07/pdm_three_dynamics.png", width: 92%, height: 82%, fit: "contain")]

== 4.4  Derivation: The DSM Objective

- Want $s_theta (x,t) approx nabla log p_t (x)$ (marginal)
- Marginal score = $EE[ nabla log p_t (x_t | x_0) | x_t]$
- So regress on the *conditional* score (known Gaussian)
- $nabla log p_t (x_t|x_0) = -(x_t - alpha_t x_0)\\/sigma_t^2 = -epsilon\\/sigma_t$
- $arrow.r$ predicting the score $=$ predicting the noise

= 5 · Flow Matching

== 5.1  Continuous Normalizing Flows

- Transport noise to data along an ODE: $dif x = v_theta (x,t) dif t$
- $v$ = a *velocity field* moving samples in time
- Density evolves by the continuity equation
- Old way (CNF): expensive max-likelihood training
- Flow matching: regress $v$ directly, simulation-free

== 5.1  Continuous Normalizing Flows

#align(center + horizon)[#image("/assets/figures/day07/pdm_nf.png", width: 92%, height: 82%, fit: "contain")]

== 5.2  Conditional Flow Matching

- Pick a simple per-sample path, e.g. $x_t = (1-t) x_0 + t x_1$
- Its velocity is known: $u_t (x | x_1) = x_1 - x_0$
- Regress $v_theta (x_t, t)$ onto this conditional velocity
- $J_"CFM" = EE\\|v_theta (x_t, t) - u_t (x|x_1)\\|^2$
- No simulation, no divergence term — just regression

== 5.2  Conditional Flow Matching

#align(center + horizon)[#image("/assets/figures/day07/pdm_cond_transition.png", width: 92%, height: 82%, fit: "contain")]

== 5.3  Conditional vs Marginal Velocity

- Many conditional paths overlap at a point $x_t$
- Marginal velocity = average of conditionals through $x_t$
- $v(x,t) = EE[u_t (x | x_1) | x_t = x]$
- CFM and marginal FM share the *same* gradient
- So regressing conditionals learns the marginal field

== 5.3  Conditional vs Marginal Velocity

#align(center + horizon)[#image("/assets/figures/day07/pdm_cond_vs_marginal.png", width: 92%, height: 82%, fit: "contain")]

== 5.4  Rectified Flow & Reflow

- Linear interpolation $arrow.r$ straight conditional paths
- But *marginal* trajectories can still be curved
- Curved paths need many ODE steps to integrate
- Reflow: retrain on (noise, sample) pairs $arrow.r$ straighter
- Straighter flow $arrow.r$ fewer steps, even one-step

== 5.4  Rectified Flow & Reflow

#align(center + horizon)[#image("/assets/figures/day07/pdm_curved_paths.png", width: 92%, height: 82%, fit: "contain")]

= 6 · One Model, Many Views

== 6.1  Four Equivalent Parameterizations

- Predict noise $epsilon$, data $x_0$, score $s$, or velocity $v$
- All related by the forward rule $x_t = alpha_t x_0 + sigma_t epsilon$
- $s = -epsilon \\/ sigma_t$ (score $=$ scaled noise)
- $v = alpha'_t x_0 + sigma'_t epsilon$ (flow-matching velocity)
- Same network, different target — pick for stability

== 6.1  Four Equivalent Parameterizations

#align(center + horizon)[#image("/assets/figures/day07/pdm_param_equiv.png", width: 92%, height: 82%, fit: "contain")]

== 6.2  The Unified Picture

- Variational (DDPM), score-SDE, and flow matching coincide
- All learn the same time-indexed field over $p_t$
- Sample with: ancestral, Langevin, reverse SDE, or ODE
- Score $arrow.l.r$ noise $arrow.l.r$ velocity are interchangeable
- Day 8: guidance, fast solvers, and few-step sampling

== 6.2  The Unified Picture

#align(center + horizon)[#image("/assets/figures/day07/pdm_unified.png", width: 92%, height: 82%, fit: "contain")]

== Summary

- Day 7: *Score, SDEs & Flow Matching*
- The continuous-time view of diffusion
- Questions welcome — practical follows

== Questions?

#align(center + horizon)[
  #text(size: 44pt, weight: "bold", fill: course-primary)[Questions?]

  #v(1em)
  Practical session follows — see course site.
]
