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

= Deep Generative Modeling

== The Generative Goal

- Map a *simple* distribution (Gaussian noise) to a *complex* one (data)
- Sample $x tilde p_"data"$ we can only access through examples
- Diffusion: don't jump noise$arrow.r$data in one step — move in many small steps
- Forward = corrupt data into noise; reverse = learn to undo it
- Source: Principles of Diffusion Models, Ch. 1–2

== The Generative Goal — illustration

#align(center)[#image("/assets/figures/day06/pdm_dgm_target.png", width: 80%)]

#text(size: 14pt, fill: gray)[Deep Generative Modeling — The Generative Goal (source: course materials)]

== Taxonomy of Models

- Likelihood-based: VAEs, normalizing flows, autoregressive, diffusion
- Implicit: GANs (no tractable $p(x)$)
- Trade-offs: sample quality vs exact likelihood vs sampling speed
- Diffusion = likelihood-based + high quality, but slow sampling

== Taxonomy of Models — illustration

#align(center)[#image("/assets/figures/day06/pdm_dgm_zoo.png", width: 80%)]

#text(size: 14pt, fill: gray)[Deep Generative Modeling — Taxonomy of Models (source: course materials)]

== Latent-Variable Models

- Introduce latent $z$: $p_theta (x) = integral p_theta (x|z) p(z) dif z$
- Prior $p(z) = N(0, I)$ is easy to sample
- Marginal likelihood is intractable (integral over all $z$)
- Posterior $p_theta (z|x)$ also intractable $arrow.r$ variational inference

= Variational Autoencoders

== The Evidence Lower Bound

- Encoder $q_phi (z|x)$ approximates the true posterior
- $log p_theta (x) >= EE_(q_phi)[log p_theta (x|z)] - D_"KL"(q_phi (z|x) || p(z))$
- Reconstruction term + regularization term
- Gap to $log p(x)$ is exactly $D_"KL"(q_phi (z|x) || p_theta (z|x)) >= 0$

== The Evidence Lower Bound — illustration

#align(center)[#image("/assets/figures/day06/pdm_vae.png", width: 80%)]

#text(size: 14pt, fill: gray)[Variational Autoencoders — The Evidence Lower Bound (source: course materials)]

== Reparameterization Trick

- Sample $z = mu_phi (x) + sigma_phi (x) dot.op epsilon$, $epsilon tilde N(0, I)$
- Moves randomness off the computation path $arrow.r$ low-variance gradients
- Backprop flows through $mu_phi$, $sigma_phi$
- DDPM = a *deep hierarchy* of these latents with a fixed encoder

= The Forward (Noising) Process

== Unified Forward Rule

- $x_t = alpha_t x_0 + sigma_t epsilon$, $epsilon tilde N(0, I)$
- $p(x_t | x_0) = N(x_t; alpha_t x_0, sigma_t^2 I)$
- $alpha_t$ = signal kept; $sigma_t$ = noise added
- Signal-to-noise ratio $"SNR"(t) = alpha_t^2 / sigma_t^2$ decreases with $t$

== Unified Forward Rule — illustration

#align(center)[#image("/assets/figures/day06/pdm_ddpm_forward.png", width: 80%)]

#text(size: 14pt, fill: gray)[The Forward (Noising) Process — Unified Forward Rule (source: course materials)]

== Closed-Form Marginal

- Step kernel $q(x_t | x_(t-1)) = N(sqrt(1 - beta_t) x_(t-1), beta_t I)$
- Compose Gaussians: $x_t = sqrt(macron(alpha)_t) x_0 + sqrt(1 - macron(alpha)_t) epsilon$
- $macron(alpha)_t = product_(s=1)^t (1 - beta_s)$ so $alpha_t = sqrt(macron(alpha)_t)$
- Sample any $t$ in one shot — no need to simulate the chain

== Noise Schedules

- VP (DDPM/cosine): $alpha_t^2 + sigma_t^2 = 1$
- VE (EDM): $alpha_t = 1$, $sigma_t = t$
- Linear interpolation (flow matching): $alpha_t = 1-t$, $sigma_t = t$
- Interactive — compare schedules on the same image (see notes)

== Noise Schedules — illustration

#align(center)[#image("/assets/figures/day06/pdm_ddpm_overview.png", width: 80%)]

#text(size: 14pt, fill: gray)[The Forward (Noising) Process — Noise Schedules (source: course materials)]

= The Reverse (Denoising) Process

== Reverse as Denoising

- Start at $x_T tilde N(0, I)$, walk back to data
- Learn $p_theta (x_(t-1) | x_t)$ — one denoising step
- Oracle reverse kernel $p(x_(t-1)|x_t)$ is intractable...
- ...but $q(x_(t-1) | x_t, x_0)$ *is* Gaussian (condition on $x_0$)

== Reverse as Denoising — illustration

#align(center)[#image("/assets/figures/day06/pdm_ddpm_reverse.png", width: 80%)]

#text(size: 14pt, fill: gray)[The Reverse (Denoising) Process — Reverse as Denoising (source: course materials)]

== The True Posterior

- $q(x_(t-1) | x_t, x_0) = N(macron(mu)_t (x_t, x_0), macron(beta)_t I)$
- $macron(mu)_t = (sqrt(macron(alpha)_(t-1)) beta_t)/(1 - macron(alpha)_t) x_0 + (sqrt(1-beta_t)(1 - macron(alpha)_(t-1)))/(1 - macron(alpha)_t) x_t$
- $macron(beta)_t = (1 - macron(alpha)_(t-1))/(1 - macron(alpha)_t) beta_t$
- Conditioning trick: the secret sauce that makes training tractable

== The True Posterior — illustration

#align(center)[#image("/assets/figures/day06/pdm_ddpm_conditioning.png", width: 80%)]

#text(size: 14pt, fill: gray)[The Reverse (Denoising) Process — The True Posterior (source: course materials)]

== Parameterizing the Reverse Step

- Match $p_theta (x_(t-1)|x_t) = N(mu_theta (x_t, t), macron(beta)_t I)$ to the posterior
- Predict $x_0$, the noise $epsilon$, or the velocity $v$ — equivalent
- $epsilon$-prediction: $mu_theta$ written via $epsilon_theta (x_t, t)$
- Network sees $(x_t, t)$ only — never the true $x_0$

= Training DDPM

== Variational Bound to Simple Loss

- ELBO over the chain = $L_T + sum_t L_(t-1) + L_0$
- Each $L_(t-1) = D_"KL"(q(x_(t-1)|x_t,x_0) || p_theta (x_(t-1)|x_t))$
- KL of two Gaussians = weighted $||macron(mu)_t - mu_theta||^2$
- Substitute $epsilon$-parameterization $arrow.r$ clean noise-prediction loss

== The $epsilon$-Prediction Objective

- $L_"simple" = EE_(t, x_0, epsilon)[ ||epsilon - epsilon_theta (x_t, t)||^2 ]$
- $x_t = sqrt(macron(alpha)_t) x_0 + sqrt(1 - macron(alpha)_t) epsilon$
- Drops the awkward per-$t$ weights — works better in practice
- Tweedie links $epsilon$ to the score: $nabla log p_t (x_t) = -epsilon_theta / sigma_t$ (Day 7)

== Sampling: Denoise then Re-noise

- From $x_t$: predict $epsilon_theta$, estimate $hat(x)_0$, jump to posterior mean
- Add a little fresh noise (except at the last step)
- Repeat $T arrow.r 0$ — this is ancestral sampling
- Day 8: do this in far fewer steps with ODE/SDE solvers

== Sampling: Denoise then Re-noise — illustration

#align(center)[#image("/assets/figures/day06/pdm_denoise_renoise.png", width: 80%)]

#text(size: 14pt, fill: gray)[Training DDPM — Sampling: Denoise then Re-noise (source: course materials)]

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
