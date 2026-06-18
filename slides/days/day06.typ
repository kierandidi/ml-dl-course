#import "../lib.typ": *

#show: course-theme.with(title: [Generative Modeling & DDPM], subtitle: [Day 6 | Aug 2026])

= Day 6: Generative Modeling & DDPM

== Welcome

- *Generative Modeling & DDPM* — The variational view — from VAEs to diffusion
- 3 hours lecture + practical
- Slides, notes, and code on the course site

== Outline

- Deep Generative Modeling
- Variational Autoencoders
- The Forward (Noising) Process
- The Reverse (Denoising) Process
- Training DDPM
- From DDPM to SDEs (Outlook)

= 1 · Deep Generative Modeling

== 1.1  The Generative Goal

- Map a *simple* distribution (Gaussian noise) to a *complex* one (data)
- Sample $x tilde p_"data"$ we can only access through examples
- Diffusion: don't jump noise$arrow.r$data in one step — move in many small steps
- Forward = corrupt data into noise; reverse = learn to undo it
- Source: Principles of Diffusion Models, Ch. 1–2

== 1.1  The Generative Goal

#align(center + horizon)[#image("/assets/figures/day06/pdm_dgm_target.png", width: 92%, height: 82%, fit: "contain")]

== 1.2  Taxonomy of Models

- Likelihood-based: VAEs, normalizing flows, autoregressive, diffusion
- Implicit: GANs (no tractable $p(x)$)
- Trade-offs: sample quality vs exact likelihood vs sampling speed
- Diffusion = likelihood-based + high quality, but slow sampling

== 1.2  Taxonomy of Models

#align(center + horizon)[#image("/assets/figures/day06/pdm_dgm_zoo.png", width: 92%, height: 82%, fit: "contain")]

== 1.3  Latent-Variable Models

- Introduce latent $z$: $p_theta (x) = integral p_theta (x|z) p(z) dif z$
- Prior $p(z) = N(0, I)$ is easy to sample
- Marginal likelihood is intractable (integral over all $z$)
- Posterior $p_theta (z|x)$ also intractable $arrow.r$ variational inference

= 2 · Variational Autoencoders

== 2.1  The Evidence Lower Bound

- Encoder $q_phi (z|x)$ approximates the true posterior
- $log p_theta (x) >= EE_(q_phi)[log p_theta (x|z)] - D_"KL"(q_phi (z|x) || p(z))$
- Reconstruction term + regularization term
- Gap to $log p(x)$ is exactly $D_"KL"(q_phi (z|x) || p_theta (z|x)) >= 0$

== 2.1  The Evidence Lower Bound

#align(center + horizon)[#image("/assets/figures/day06/pdm_vae.png", width: 92%, height: 82%, fit: "contain")]

== 2.2  Derivation: ELBO in Three Lines

- Start: $log p_theta (x) = log integral p_theta (x,z) dif z$
- Multiply & divide by $q_phi$: $= log EE_(q_phi)[p_theta (x,z) \\/ q_phi (z|x)]$
- Jensen ($log$ concave): $>= EE_(q_phi)[log (p_theta (x,z) \\/ q_phi (z|x))]$
- Split the log: $= EE_(q_phi)[log p_theta (x|z)] - D_"KL"(q_phi || p(z))$
- Exact gap: $log p(x) - "ELBO" = D_"KL"(q_phi (z|x) || p_theta (z|x)) >= 0$
- Same algebra reused for the diffusion ELBO below

== 2.3  Reparameterization Trick

- Sample $z = mu_phi (x) + sigma_phi (x) dot.op epsilon$, $epsilon tilde N(0, I)$
- Moves randomness off the computation path $arrow.r$ low-variance gradients
- Backprop flows through $mu_phi$, $sigma_phi$
- DDPM = a *deep hierarchy* of these latents with a fixed encoder

= 3 · The Forward (Noising) Process

== 3.1  Unified Forward Rule

- $x_t = alpha_t x_0 + sigma_t epsilon$, $epsilon tilde N(0, I)$
- $p(x_t | x_0) = N(x_t; alpha_t x_0, sigma_t^2 I)$
- $alpha_t$ = signal kept; $sigma_t$ = noise added
- Signal-to-noise ratio $"SNR"(t) = alpha_t^2 / sigma_t^2$ decreases with $t$

== 3.1  Unified Forward Rule

#align(center + horizon)[#image("/assets/figures/day06/pdm_ddpm_forward.png", width: 92%, height: 82%, fit: "contain")]

== 3.2  Closed-Form Marginal

- Step kernel $q(x_t | x_(t-1)) = N(sqrt(1 - beta_t) x_(t-1), beta_t I)$
- Compose Gaussians: $x_t = sqrt(macron(alpha)_t) x_0 + sqrt(1 - macron(alpha)_t) epsilon$
- $macron(alpha)_t = product_(s=1)^t (1 - beta_s)$ so $alpha_t = sqrt(macron(alpha)_t)$
- Sample any $t$ in one shot — no need to simulate the chain

== 3.3  Derivation: Compose the Gaussians

- Write $a_t = 1 - beta_t$, so $x_t = sqrt(a_t) x_(t-1) + sqrt(1-a_t) epsilon_t$
- Substitute one step into the next (telescoping)
- Two independent Gaussians add $arrow.r$ variances add
- $a_t (1 - a_(t-1)) + (1 - a_t) = 1 - a_t a_(t-1)$
- Iterate to $x_0$: $x_t = sqrt(macron(alpha)_t) x_0 + sqrt(1 - macron(alpha)_t) epsilon$
- VP schedule: $alpha_t^2 + sigma_t^2 = 1$

== 3.4  Noise Schedules

- VP (DDPM/cosine): $alpha_t^2 + sigma_t^2 = 1$
- VE (EDM): $alpha_t = 1$, $sigma_t = t$
- Linear interpolation (flow matching): $alpha_t = 1-t$, $sigma_t = t$
- Interactive — compare schedules on the same image (see notes)

== 3.4  Noise Schedules

#align(center + horizon)[#image("/assets/figures/day06/pdm_ddpm_overview.png", width: 92%, height: 82%, fit: "contain")]

= 4 · The Reverse (Denoising) Process

== 4.1  Reverse as Denoising

- Start at $x_T tilde N(0, I)$, walk back to data
- Learn $p_theta (x_(t-1) | x_t)$ — one denoising step
- Oracle reverse kernel $p(x_(t-1)|x_t)$ is intractable...
- ...but $q(x_(t-1) | x_t, x_0)$ *is* Gaussian (condition on $x_0$)

== 4.1  Reverse as Denoising

#align(center + horizon)[#image("/assets/figures/day06/pdm_ddpm_reverse.png", width: 92%, height: 82%, fit: "contain")]

== 4.2  The True Posterior

- $q(x_(t-1) | x_t, x_0) = N(macron(mu)_t (x_t, x_0), macron(beta)_t I)$
- $macron(mu)_t = (sqrt(macron(alpha)_(t-1)) beta_t)/(1 - macron(alpha)_t) x_0 + (sqrt(1-beta_t)(1 - macron(alpha)_(t-1)))/(1 - macron(alpha)_t) x_t$
- $macron(beta)_t = (1 - macron(alpha)_(t-1))/(1 - macron(alpha)_t) beta_t$
- Conditioning trick: the secret sauce that makes training tractable

== 4.2  The True Posterior

#align(center + horizon)[#image("/assets/figures/day06/pdm_ddpm_conditioning.png", width: 92%, height: 82%, fit: "contain")]

== 4.3  Derivation: Gaussian Posterior

- Bayes: $q(x_(t-1)|x_t,x_0) prop q(x_t|x_(t-1)) q(x_(t-1)|x_0)$
- Both factors Gaussian $arrow.r$ collect quadratic & linear terms in $x_(t-1)$
- Precision: $1\\/macron(beta)_t = a_t\\/beta_t + 1\\/(1 - macron(alpha)_(t-1))$
- Variance: $macron(beta)_t = (1 - macron(alpha)_(t-1))\\/(1 - macron(alpha)_t) beta_t$
- Mean: $macron(mu)_t = (sqrt(macron(alpha)_(t-1)) beta_t)\\/(1-macron(alpha)_t) x_0 + (sqrt(a_t)(1-macron(alpha)_(t-1)))\\/(1-macron(alpha)_t) x_t$

== 4.4  Parameterizing the Reverse Step

- Match $p_theta (x_(t-1)|x_t) = N(mu_theta (x_t, t), macron(beta)_t I)$ to the posterior
- Predict $x_0$, the noise $epsilon$, or the velocity $v$ — equivalent
- $epsilon$-prediction: $mu_theta$ written via $epsilon_theta (x_t, t)$
- Network sees $(x_t, t)$ only — never the true $x_0$

= 5 · Training DDPM

== 5.1  Variational Bound to Simple Loss

- ELBO over the chain = $L_T + sum_t L_(t-1) + L_0$
- Each $L_(t-1) = D_"KL"(q(x_(t-1)|x_t,x_0) || p_theta (x_(t-1)|x_t))$
- KL of two Gaussians = weighted $||macron(mu)_t - mu_theta||^2$
- Substitute $epsilon$-parameterization $arrow.r$ clean noise-prediction loss

== 5.2  Derivation: KL to Noise Prediction

- Same-covariance Gaussians: $L_(t-1) = 1\\/(2 macron(beta)_t) ||macron(mu)_t - mu_theta||^2$
- Estimate $hat(x)_0 = (x_t - sqrt(1-macron(alpha)_t) epsilon_theta)\\/sqrt(macron(alpha)_t)$
- Substitute into $macron(mu)_t$ and $mu_theta$
- Prefactors $1\\/sqrt(a_t)$ and $x_t$ cancel — only noises remain
- $L_(t-1) prop ||epsilon - epsilon_theta (x_t,t)||^2$
- Set the weight to 1 $arrow.r L_"simple"$ (up-weights hard, high-noise steps)

== 5.3  The $epsilon$-Prediction Objective

- $L_"simple" = EE_(t, x_0, epsilon)[ ||epsilon - epsilon_theta (x_t, t)||^2 ]$
- $x_t = sqrt(macron(alpha)_t) x_0 + sqrt(1 - macron(alpha)_t) epsilon$
- Drops the awkward per-$t$ weights — works better in practice
- Tweedie links $epsilon$ to the score: $nabla log p_t (x_t) = -epsilon_theta \\/ sigma_t$ (Day 7)

== 5.4  Sampling: Denoise then Re-noise

- From $x_t$: predict $epsilon_theta$, estimate $hat(x)_0$, jump to posterior mean
- $x_(t-1) = 1\\/sqrt(a_t) (x_t - beta_t\\/sqrt(1-macron(alpha)_t) epsilon_theta) + sqrt(macron(beta)_t) z$
- Add a little fresh noise $z tilde N(0,I)$ (except at the last step)
- Repeat $T arrow.r 0$ — this is ancestral sampling
- Day 8: do this in far fewer steps with ODE/SDE solvers

== 5.4  Sampling: Denoise then Re-noise

#align(center + horizon)[#image("/assets/figures/day06/pdm_denoise_renoise.png", width: 92%, height: 82%, fit: "contain")]

= 6 · From DDPM to SDEs (Outlook)

== 6.1  Forward & Reverse as SDEs

- Take the step size to zero: the chain becomes a continuous-time SDE
- Forward: $dif x = f(x,t) dif t + g(t) dif w$ (drift + Brownian noise)
- Reverse: $dif x = [f - g^2 nabla log p_t (x)] dif t + g dif macron(w)$
- The score $nabla log p_t (x)$ replaces the unknown reverse drift
- Probability-flow ODE shares the same marginals (deterministic sampling)
- DDPM is the discretized variance-preserving SDE — full derivations in the SDE course

== 6.1  Forward & Reverse as SDEs

#align(center + horizon)[#image("/assets/figures/day06/sde_song_diffusion.png", width: 92%, height: 82%, fit: "contain")]

== 6.2  Simulating SDEs: Euler-Maruyama

- Discretize $[t_0, t]$ into steps of size $Delta t$
- Brownian increment $Delta beta(t_k) tilde N(0, Delta t I)$
- $x_(k+1) = x_k + b(x_k,t_k) Delta t + sigma(x_k,t_k) Delta beta(t_k)$
- Noise enters at scale $sqrt(Delta t)$ since $"Var" = Delta t$
- DDPM ancestral sampling = the VP special case of this scheme
- Reference: SDE course (Brownian motion, time reversal, Girsanov)

== 6.2  Simulating SDEs: Euler-Maruyama

#align(center + horizon)[#image("/assets/figures/day06/sde_euler_maruyama.png", width: 92%, height: 82%, fit: "contain")]

== Summary

- Day 6: *Generative Modeling & DDPM*
- The variational view — from VAEs to diffusion
- Questions welcome — practical follows

== Questions?

#align(center + horizon)[
  #text(size: 44pt, weight: "bold", fill: course-primary)[Questions?]

  #v(1em)
  Practical session follows — see course site.
]
