---
layout: post
title: Day 6 - Generative Modeling & DDPM
image: /assets/img/lessons/day06.png
accent_image: 
  background: url('/assets/img/lessons/day06.png') center/cover
  overlay: false
accent_color: '#ccc'
theme_color: '#ccc'
description: >
  Deep generative modeling, VAEs and the ELBO, and the variational view of diffusion: the DDPM forward process, the reverse posterior, and the noise-prediction training objective — with interactive visualizations.
invert_sidebar: true
---

# Day 6 - Generative Modeling & DDPM

### Optional reading for this lesson
- [Lai, Song, Kim, Mitsufuji & Ermon — *The Principles of Diffusion Models*](https://arxiv.org/abs/2510.21890), Ch. 1–2
- [Ho, Jain & Abbeel — *Denoising Diffusion Probabilistic Models* (DDPM)](https://arxiv.org/abs/2006.11239)
- [Kingma & Welling — *Auto-Encoding Variational Bayes* (VAE)](https://arxiv.org/abs/1312.6114)
- [Interactive companion & teaching guide](https://the-principles-of-diffusion-models.github.io/)
- [Generative Modelling with SDEs — course notes (Brownian motion, Euler–Maruyama, time reversal, Girsanov)](https://kierandidi.github.io/) for the continuous-time derivations previewed at the end

### [Slides](/assets/slides/day06.pdf)

### Exercise

[Download the notebook](/notebooks/practicals/day06.ipynb) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kierandidi/ml-dl-course/blob/main/notebooks/practicals/day06.ipynb)

Week 2 turns to **generative modeling**: instead of predicting a label from an input, we want to model the data distribution itself and draw new samples from it. Today we build the **variational view of diffusion**. We start from variational autoencoders (VAEs) and the evidence lower bound, then show that a *denoising diffusion probabilistic model* (DDPM) is essentially a deep, fixed-encoder hierarchical VAE. We will derive the forward noising process and its closed-form marginal, the Gaussian reverse posterior that makes training tractable, and the surprisingly simple noise-prediction loss. Throughout we use the unified notation of *The Principles of Diffusion Models* ($x_t = \alpha_t x_0 + \sigma_t \epsilon$) so that everything connects cleanly to the score-based and flow-based views in Day 7.

* toc
{:toc}

## 1. Deep Generative Modeling

### 1.1 The generative goal and a taxonomy

> A **deep generative model** learns to turn samples from a simple reference distribution (e.g. $$\boldsymbol{z}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$$) into samples that look like data $$\boldsymbol{x}\sim p_{\text{data}}$$, while ideally also assigning a likelihood $$p_\theta(\boldsymbol{x})$$.
{:.lead}

We never see $$p_{\text{data}}$$ directly — only a finite set of examples (images, audio, molecules). The model is a procedure that maps the *simple side* (noise we can generate at will) to the *complex side* (structured data).

![The target of deep generative modeling: transport a simple distribution to the data distribution (Principles Fig 1.1)](/assets/figures/day06/pdm_dgm_target.png)

Generative models differ in **what they make tractable**:

- **Likelihood-based, exact**: normalizing flows and autoregressive models give exact $$\log p_\theta(\boldsymbol{x})$$ but constrain the architecture.
- **Likelihood-based, bounded**: VAEs and diffusion optimize a *lower bound* on $$\log p_\theta(\boldsymbol{x})$$.
- **Implicit**: GANs sample well but expose no tractable density.

![Computation graphs of prominent deep generative models (Principles Fig 1.2)](/assets/figures/day06/pdm_dgm_zoo.png)

Diffusion models sit in the second group: they are likelihood-based (so training is stable and principled) and produce state-of-the-art samples, at the cost of **iterative** sampling — the central tension we resolve in Day 8.

### 1.2 Latent-variable models and intractability

> A **latent-variable model** writes the data density as a marginal over a hidden variable $$\boldsymbol{z}$$: $$p_\theta(\boldsymbol{x}) = \int p_\theta(\boldsymbol{x}\mid\boldsymbol{z})\,p(\boldsymbol{z})\,\mathrm{d}\boldsymbol{z}.$$
{:.lead}

Choosing a simple prior $$p(\boldsymbol{z}) = \mathcal{N}(\mathbf{0},\mathbf{I})$$ and a flexible decoder $$p_\theta(\boldsymbol{x}\mid\boldsymbol{z})$$ makes *sampling* trivial (draw $$\boldsymbol{z}$$, push it through the decoder). But two quantities are **intractable**:

1. the **marginal likelihood** $$p_\theta(\boldsymbol{x})$$ — an integral over all $$\boldsymbol{z}$$;
2. the **posterior** $$p_\theta(\boldsymbol{z}\mid\boldsymbol{x}) = p_\theta(\boldsymbol{x}\mid\boldsymbol{z})p(\boldsymbol{z})/p_\theta(\boldsymbol{x})$$, which inherits the same intractable normalizer.

Variational inference sidesteps both by introducing an **approximate posterior** $$q_\phi(\boldsymbol{z}\mid\boldsymbol{x})$$ and optimizing a bound — the subject of the next section. Diffusion will take this idea to the extreme, using a *chain* of latents $$\boldsymbol{x}_1,\dots,\boldsymbol{x}_T$$ with a **fixed** (non-learned) approximate posterior.

## 2. Variational Autoencoders and the ELBO

### 2.1 Deriving the evidence lower bound

> The **evidence lower bound (ELBO)** is a tractable lower bound on the log-likelihood obtained with any approximate posterior $$q_\phi$$: $$\log p_\theta(\boldsymbol{x}) \ge \mathbb{E}_{q_\phi(\boldsymbol{z}\mid\boldsymbol{x})}[\log p_\theta(\boldsymbol{x}\mid\boldsymbol{z})] - D_{\mathrm{KL}}\!\big(q_\phi(\boldsymbol{z}\mid\boldsymbol{x})\,\Vert \,p(\boldsymbol{z})\big).$$
{:.lead}

Let us derive it carefully, because the *same* algebra reappears for diffusion. Start from the log-evidence and multiply-and-divide by $$q_\phi$$ inside an expectation:

$$\begin{aligned}
\log p_\theta(\boldsymbol{x})
&= \log \int p_\theta(\boldsymbol{x},\boldsymbol{z})\,\mathrm{d}\boldsymbol{z}
= \log \mathbb{E}_{q_\phi(\boldsymbol{z}\mid\boldsymbol{x})}\!\left[\frac{p_\theta(\boldsymbol{x},\boldsymbol{z})}{q_\phi(\boldsymbol{z}\mid\boldsymbol{x})}\right] \\
&\ge \mathbb{E}_{q_\phi(\boldsymbol{z}\mid\boldsymbol{x})}\!\left[\log \frac{p_\theta(\boldsymbol{x},\boldsymbol{z})}{q_\phi(\boldsymbol{z}\mid\boldsymbol{x})}\right] \qquad \textcolor{teal}{\text{(Jensen's inequality, } \log \text{ concave)}} \\
&= \mathbb{E}_{q_\phi}\big[\log p_\theta(\boldsymbol{x}\mid\boldsymbol{z})\big] - D_{\mathrm{KL}}\!\big(q_\phi(\boldsymbol{z}\mid\boldsymbol{x})\,\Vert \,p(\boldsymbol{z})\big) =: \mathcal{L}_{\text{ELBO}}.
\end{aligned}$$

The bound is **exact** when $$q_\phi(\boldsymbol{z}\mid\boldsymbol{x}) = p_\theta(\boldsymbol{z}\mid\boldsymbol{x})$$. Indeed, the gap is precisely a KL divergence:

$$\log p_\theta(\boldsymbol{x}) - \mathcal{L}_{\text{ELBO}} = \textcolor{purple}{D_{\mathrm{KL}}\!\big(q_\phi(\boldsymbol{z}\mid\boldsymbol{x})\,\Vert \,p_\theta(\boldsymbol{z}\mid\boldsymbol{x})\big)} \ge 0.$$

![A variational autoencoder: stochastic encoder, latent bottleneck, decoder (Principles Fig 2.1)](/assets/figures/day06/pdm_vae.png)

The two ELBO terms have clean interpretations: the first is a **reconstruction** term (the decoder should reproduce $$\boldsymbol{x}$$ from $$\boldsymbol{z}$$), the second is a **regularizer** pulling the encoder's posterior toward the prior.

### 2.2 The reparameterization trick

> To backpropagate through the sampling step, write the latent as a deterministic function of the parameters and an independent noise variable: $$\boldsymbol{z} = \boldsymbol{\mu}_\phi(\boldsymbol{x}) + \boldsymbol{\sigma}_\phi(\boldsymbol{x})\odot\boldsymbol{\epsilon},\quad \boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I}).$$
{:.lead}

Gradients of $$\mathbb{E}_{q_\phi}[\,f(\boldsymbol{z})\,]$$ cannot flow through a raw `sample` node. The reparameterization trick moves the randomness to an input $$\boldsymbol{\epsilon}$$ that does not depend on $$\phi$$, so

$$\nabla_\phi\, \mathbb{E}_{q_\phi}[f(\boldsymbol{z})] = \mathbb{E}_{\boldsymbol{\epsilon}}\big[\nabla_\phi\, f(\boldsymbol{\mu}_\phi + \boldsymbol{\sigma}_\phi \odot \boldsymbol{\epsilon})\big],$$

a low-variance, pathwise estimator (Day 1's "reparameterization gradient"). For Gaussian $$q$$ and $$p$$, the KL term is available in **closed form**:

$$D_{\mathrm{KL}}\big(\mathcal{N}(\boldsymbol{\mu},\mathrm{diag}\,\boldsymbol{\sigma}^2)\,\Vert \,\mathcal{N}(\mathbf{0},\mathbf{I})\big) = \tfrac{1}{2}\sum_i \big(\mu_i^2 + \sigma_i^2 - 1 - \log \sigma_i^2\big).$$

**Key idea for diffusion.** A DDPM is what you get if you (i) replace the single latent by a long chain $$\boldsymbol{x}_1,\dots,\boldsymbol{x}_T$$, (ii) **fix** the encoder to a simple Gaussian noising process instead of learning it, and (iii) learn only the decoder (the reverse/denoising steps). Everything below is the ELBO of that hierarchy.

## 3. The Forward (Noising) Process

### 3.1 The unified forward rule

> The **forward process** corrupts data into noise through a one-parameter family of Gaussians: $$p(\boldsymbol{x}_t\mid\boldsymbol{x}_0) = \mathcal{N}\big(\boldsymbol{x}_t;\,\alpha_t\boldsymbol{x}_0,\,\sigma_t^2\mathbf{I}\big),\quad\text{equivalently}\quad \boldsymbol{x}_t = \alpha_t \boldsymbol{x}_0 + \sigma_t\boldsymbol{\epsilon},\;\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I}).$$
{:.lead}

Here $$\alpha_t$$ is the **signal coefficient** (how much of $$\boldsymbol{x}_0$$ survives) and $$\sigma_t$$ is the **noise scale**. As $$t$$ grows, $$\alpha_t \to 0$$ and $$\sigma_t$$ grows, so $$\boldsymbol{x}_t$$ approaches a tractable prior (a Gaussian). A single scalar summarizes the corruption level, the **signal-to-noise ratio**

$$\mathrm{SNR}(t) = \frac{\alpha_t^2}{\sigma_t^2}, \qquad \lambda_t := \log \mathrm{SNR}(t) = \log\frac{\alpha_t^2}{\sigma_t^2},$$

which decreases monotonically from $$+\infty$$ (clean data) to $$-\infty$$ (pure noise). We will reuse $$\lambda_t$$ (log-SNR) as the natural "clock" for solvers in Day 8.

![The DDPM forward process: Gaussian noise is added step by step until the sample is indistinguishable from noise (Principles Fig 2.4)](/assets/figures/day06/pdm_ddpm_forward.png)

### 3.2 From step kernel to closed-form marginal

> For the DDPM Markov chain $$q(\boldsymbol{x}_t\mid\boldsymbol{x}_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}\,\boldsymbol{x}_{t-1},\,\beta_t\mathbf{I})$$, the $$t$$-step marginal collapses to a single Gaussian $$q(\boldsymbol{x}_t\mid\boldsymbol{x}_0) = \mathcal{N}\big(\sqrt{\bar\alpha_t}\,\boldsymbol{x}_0,\,(1-\bar\alpha_t)\mathbf{I}\big),\quad \bar\alpha_t = \prod_{s=1}^t (1-\beta_s).$$
{:.lead}

This is the property that makes diffusion trainable: we can jump to any noise level in one step. Let us prove it by induction, writing $$a_t := 1-\beta_t$$ so that $$\boldsymbol{x}_t = \sqrt{a_t}\,\boldsymbol{x}_{t-1} + \sqrt{1-a_t}\,\boldsymbol{\epsilon}_t$$ with independent $$\boldsymbol{\epsilon}_t\sim\mathcal{N}(\mathbf{0},\mathbf{I})$$. Substitute one step into the next:

$$\begin{aligned}
\boldsymbol{x}_t &= \sqrt{a_t}\,\big(\textcolor{blue}{\sqrt{a_{t-1}}\,\boldsymbol{x}_{t-2} + \sqrt{1-a_{t-1}}\,\boldsymbol{\epsilon}_{t-1}}\big) + \sqrt{1-a_t}\,\boldsymbol{\epsilon}_t \\
&= \sqrt{a_t a_{t-1}}\,\boldsymbol{x}_{t-2} + \underbrace{\sqrt{a_t(1-a_{t-1})}\,\boldsymbol{\epsilon}_{t-1} + \sqrt{1-a_t}\,\boldsymbol{\epsilon}_t}_{\textcolor{purple}{\text{sum of two independent Gaussians}}}.
\end{aligned}$$

The two noise terms are independent zero-mean Gaussians, so their sum is Gaussian with variance equal to the sum of variances:

$$\mathrm{Var} = a_t(1-a_{t-1}) + (1-a_t) = 1 - a_t a_{t-1}.$$

Hence $$\boldsymbol{x}_t = \sqrt{a_t a_{t-1}}\,\boldsymbol{x}_{t-2} + \sqrt{1 - a_t a_{t-1}}\,\bar{\boldsymbol{\epsilon}}$$ for a single $$\bar{\boldsymbol{\epsilon}}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$$. Iterating down to $$\boldsymbol{x}_0$$ gives

$$\boxed{\;\boldsymbol{x}_t = \sqrt{\bar\alpha_t}\,\boldsymbol{x}_0 + \sqrt{1-\bar\alpha_t}\,\boldsymbol{\epsilon},\qquad \bar\alpha_t = \prod_{s=1}^t a_s.\;}$$

Comparing with the unified rule, the DDPM schedule is the **variance-preserving (VP)** case

$$\alpha_t = \sqrt{\bar\alpha_t}, \qquad \sigma_t = \sqrt{1-\bar\alpha_t}, \qquad \alpha_t^2 + \sigma_t^2 = 1.$$

### 3.3 Noise schedules and the interactive explorer

> A **noise schedule** is the choice of curves $$(\alpha_t, \sigma_t)$$. The three standard families are **variance preserving** ($$\alpha_t^2+\sigma_t^2=1$$), **variance exploding** ($$\alpha_t=1,\ \sigma_t$$ growing), and **linear interpolation** ($$\alpha_t=1-t,\ \sigma_t=t$$).
{:.lead}

All three obey the *same* one-liner $$\boldsymbol{x}_t = \alpha_t\boldsymbol{x}_0 + \sigma_t\boldsymbol{\epsilon}$$; only the shapes of $$(\alpha_t,\sigma_t)$$ differ:

| Family | $$\alpha_t$$ | $$\sigma_t$$ | Used by |
|---|---|---|---|
| VP (linear-$$\beta$$) | $$\exp\!\big(-\tfrac12\!\int_0^t\!\beta(s)\,ds\big)$$ | $$\sqrt{1-\alpha_t^2}$$ | DDPM |
| VP (cosine) | $$\cos(\tfrac{\pi t}{2})$$ | $$\sin(\tfrac{\pi t}{2})$$ | improved DDPM |
| VE | $$1$$ | $$t$$ | EDM / NCSN |
| Linear interp. | $$1-t$$ | $$t$$ | flow matching / rectified flow |

The schedule controls **where the model spends capacity**: it determines how fast the SNR drops and therefore which noise levels dominate training. Use the widget to corrupt the same image under each schedule and watch how the SNR curve changes:


<div class="interactive-viz" markdown="0">
  <iframe src="https://the-principles-of-diffusion-models.github.io/assets/noise_schedule_explorer.html" width="100%" height="860" loading="lazy" style="border:1px solid #ddd;border-radius:8px;" title="Noise Schedule Explorer"></iframe>
  <p style="font-size:0.85em;color:#888;margin-top:4px;">Interactive: <strong>Noise Schedule Explorer</strong> — <a href="https://the-principles-of-diffusion-models.github.io/assets/noise_schedule_explorer.html" target="_blank" rel="noopener">open in new tab</a>. Source: <em>The Principles of Diffusion Models</em>.</p>
</div>


![DDPM as a fixed forward chain and a learned reverse chain (Principles Fig 2.3)](/assets/figures/day06/pdm_ddpm_overview.png)

## 4. The Reverse (Denoising) Process

### 4.1 Why we condition on the clean sample

> Sampling requires the **reverse kernel** $$p(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_t)$$, which is intractable. The trick is that the **conditioned** reverse kernel $$q(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_t,\boldsymbol{x}_0)$$ is an explicit Gaussian.
{:.lead}

Generation runs the chain backward: start from $$\boldsymbol{x}_T\sim\mathcal{N}(\mathbf{0},\mathbf{I})$$ and repeatedly sample $$\boldsymbol{x}_{t-1}\sim p_\theta(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_t)$$.

![The learned reverse process gradually removes noise to recover data (Principles Fig 2.5)](/assets/figures/day06/pdm_ddpm_reverse.png)

The marginal reverse kernel $$p(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_t)$$ would require integrating over all data — intractable. But by Bayes' rule, *conditioning on $$\boldsymbol{x}_0$$* turns it into a ratio of known Gaussians:

$$q(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_t,\boldsymbol{x}_0) = \frac{q(\boldsymbol{x}_t\mid\boldsymbol{x}_{t-1})\,q(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_0)}{q(\boldsymbol{x}_t\mid\boldsymbol{x}_0)}.$$

![The conditioning trick: conditioning on $x_0$ makes the reverse step a tractable Gaussian (Principles Fig 2.6)](/assets/figures/day06/pdm_ddpm_conditioning.png)

Use the interactive panel to see how conditioning on $$\boldsymbol{x}_0$$ pins down the otherwise-ambiguous reverse step:


<div class="interactive-viz" markdown="0">
  <iframe src="https://the-principles-of-diffusion-models.github.io/assets/ddpm_conditional_trick.html" width="100%" height="860" loading="lazy" style="border:1px solid #ddd;border-radius:8px;" title="DDPM Conditional Trick"></iframe>
  <p style="font-size:0.85em;color:#888;margin-top:4px;">Interactive: <strong>DDPM Conditional Trick</strong> — <a href="https://the-principles-of-diffusion-models.github.io/assets/ddpm_conditional_trick.html" target="_blank" rel="noopener">open in new tab</a>. Source: <em>The Principles of Diffusion Models</em>.</p>
</div>


### 4.2 Deriving the Gaussian posterior

> The conditioned reverse kernel is $$q(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_t,\boldsymbol{x}_0) = \mathcal{N}\big(\tilde{\boldsymbol{\mu}}_t(\boldsymbol{x}_t,\boldsymbol{x}_0),\,\tilde\beta_t\mathbf{I}\big)$$ with mean and variance given below.
{:.lead}

All three densities on the right are Gaussian in $$\boldsymbol{x}_{t-1}$$, so the product is Gaussian; we only need to collect the quadratic and linear terms in the exponent. Dropping terms that do not involve $$\boldsymbol{x}_{t-1}$$,

$$\begin{aligned}
-\log q(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_t,\boldsymbol{x}_0) &\;\overset{c}{=}\; \frac{\Vert \boldsymbol{x}_t - \sqrt{a_t}\,\boldsymbol{x}_{t-1}\Vert ^2}{2\beta_t} + \frac{\Vert \boldsymbol{x}_{t-1} - \sqrt{\bar\alpha_{t-1}}\,\boldsymbol{x}_0\Vert ^2}{2(1-\bar\alpha_{t-1})} \\
&\;\overset{c}{=}\; \frac{1}{2}\Big(\underbrace{\tfrac{a_t}{\beta_t} + \tfrac{1}{1-\bar\alpha_{t-1}}}_{\textcolor{blue}{1/\tilde\beta_t}}\Big)\Vert \boldsymbol{x}_{t-1}\Vert ^2 - \Big(\tfrac{\sqrt{a_t}}{\beta_t}\boldsymbol{x}_t + \tfrac{\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}}\boldsymbol{x}_0\Big)^{\!\top}\boldsymbol{x}_{t-1}.
\end{aligned}$$

Reading off a Gaussian $$\propto \exp(-\tfrac{1}{2\tilde\beta_t}\Vert \boldsymbol{x}_{t-1}-\tilde{\boldsymbol{\mu}}_t\Vert ^2)$$, the **precision** gives the variance and the **linear term** gives the mean. Using $$a_t = 1-\beta_t$$ and $$\bar\alpha_t = a_t\bar\alpha_{t-1}$$ to simplify,

$$\boxed{\;\tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\,\beta_t,\qquad \tilde{\boldsymbol{\mu}}_t(\boldsymbol{x}_t,\boldsymbol{x}_0) = \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\,\boldsymbol{x}_0 + \frac{\sqrt{a_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,\boldsymbol{x}_t.\;}$$

This is the **target** the reverse network must match. Note it depends on the (unknown at sampling time) $$\boldsymbol{x}_0$$ — handled next by predicting it.

### 4.3 Parameterizing the reverse step

> We set $$p_\theta(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_t) = \mathcal{N}\big(\boldsymbol{\mu}_\theta(\boldsymbol{x}_t,t),\,\tilde\beta_t\mathbf{I}\big)$$ and choose what the network predicts: the clean sample $$\boldsymbol{x}_0$$, the noise $$\boldsymbol{\epsilon}$$, or a velocity $$\boldsymbol{v}$$ — all equivalent.
{:.lead}

Since at sampling time we only have $$\boldsymbol{x}_t$$, we estimate $$\boldsymbol{x}_0$$ from it. From the forward marginal $$\boldsymbol{x}_t = \sqrt{\bar\alpha_t}\,\boldsymbol{x}_0 + \sqrt{1-\bar\alpha_t}\,\boldsymbol{\epsilon}$$ we can solve for $$\boldsymbol{x}_0$$ in terms of the noise:

$$\hat{\boldsymbol{x}}_0 = \frac{1}{\sqrt{\bar\alpha_t}}\big(\boldsymbol{x}_t - \sqrt{1-\bar\alpha_t}\,\textcolor{teal}{\boldsymbol{\epsilon}_\theta(\boldsymbol{x}_t,t)}\big).$$

Substituting this $$\hat{\boldsymbol{x}}_0$$ into $$\tilde{\boldsymbol{\mu}}_t$$ and simplifying yields the famous **noise-prediction mean**:

$$\boldsymbol{\mu}_\theta(\boldsymbol{x}_t,t) = \frac{1}{\sqrt{a_t}}\Big(\boldsymbol{x}_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\boldsymbol{\epsilon}_\theta(\boldsymbol{x}_t,t)\Big).$$

So predicting the noise $$\boldsymbol{\epsilon}$$ is equivalent to predicting $$\boldsymbol{x}_0$$, which is equivalent to predicting the posterior mean. Day 7 adds two more equivalent targets (the **score** $$\nabla\log p_t$$ and the **velocity** $$\boldsymbol{v}$$), and shows they are linear reparameterizations of one another.

## 5. Training and Sampling

### 5.1 From the variational bound to the simple loss

> Maximizing the ELBO of the diffusion hierarchy decomposes into per-step KL terms $$L_{t-1} = D_{\mathrm{KL}}\big(q(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_t,\boldsymbol{x}_0)\,\Vert \,p_\theta(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_t)\big)$$ that reduce to weighted squared errors.
{:.lead}

Applying the ELBO derivation to the joint $$q(\boldsymbol{x}_{1:T}\mid\boldsymbol{x}_0)$$ and the learned reverse chain $$p_\theta(\boldsymbol{x}_{0:T})$$ gives

$$-\log p_\theta(\boldsymbol{x}_0) \le \mathbb{E}_q\Big[\underbrace{D_{\mathrm{KL}}(q(\boldsymbol{x}_T\mid\boldsymbol{x}_0)\,\Vert \,p(\boldsymbol{x}_T))}_{L_T,\ \text{no params}} + \sum_{t>1}\underbrace{D_{\mathrm{KL}}(q(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_t,\boldsymbol{x}_0)\,\Vert \,p_\theta(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_t))}_{L_{t-1}} \underbrace{- \log p_\theta(\boldsymbol{x}_0\mid\boldsymbol{x}_1)}_{L_0}\Big].$$

Because both arguments of each $$L_{t-1}$$ are Gaussians with the **same** covariance $$\tilde\beta_t\mathbf{I}$$, the KL collapses to a scaled distance between means:

$$L_{t-1} \overset{c}{=} \frac{1}{2\tilde\beta_t}\,\big\Vert \tilde{\boldsymbol{\mu}}_t(\boldsymbol{x}_t,\boldsymbol{x}_0) - \boldsymbol{\mu}_\theta(\boldsymbol{x}_t,t)\big\Vert ^2.$$

Now substitute the two noise-prediction means from the previous section. The prefactors $$\tfrac{1}{\sqrt{a_t}}$$ and $$\boldsymbol{x}_t$$ **cancel**, leaving a difference of noises:

$$L_{t-1} \overset{c}{=} \textcolor{purple}{\frac{\beta_t^2}{2\tilde\beta_t\, a_t\,(1-\bar\alpha_t)}}\,\big\Vert \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\boldsymbol{x}_t,t)\big\Vert ^2,\qquad \boldsymbol{x}_t = \sqrt{\bar\alpha_t}\,\boldsymbol{x}_0 + \sqrt{1-\bar\alpha_t}\,\boldsymbol{\epsilon}.$$

### 5.2 The simple objective and ancestral sampling

> DDPM drops the per-step weight and trains the **simple noise-prediction loss** $$L_{\text{simple}} = \mathbb{E}_{t,\boldsymbol{x}_0,\boldsymbol{\epsilon}}\big[\,\Vert \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\boldsymbol{x}_t,t)\Vert ^2\big].$$
{:.lead}

Empirically, setting the awkward weight $$\textcolor{purple}{(\cdots)}$$ to $$1$$ improves sample quality — it up-weights the harder, higher-noise steps. The resulting training loop is remarkably simple:

1. draw $$\boldsymbol{x}_0\sim p_{\text{data}}$$, a timestep $$t\sim\mathcal{U}\{1,\dots,T\}$$, and $$\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$$;
2. form $$\boldsymbol{x}_t = \sqrt{\bar\alpha_t}\,\boldsymbol{x}_0 + \sqrt{1-\bar\alpha_t}\,\boldsymbol{\epsilon}$$;
3. take a gradient step on $$\Vert \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\boldsymbol{x}_t,t)\Vert ^2$$.

**Sampling** ("denoise then re-noise") runs the reverse chain: from $$\boldsymbol{x}_T\sim\mathcal{N}(\mathbf{0},\mathbf{I})$$, repeat for $$t=T,\dots,1$$

$$\boldsymbol{x}_{t-1} = \underbrace{\frac{1}{\sqrt{a_t}}\Big(\boldsymbol{x}_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\boldsymbol{\epsilon}_\theta(\boldsymbol{x}_t,t)\Big)}_{\textcolor{teal}{\text{denoise toward posterior mean}}} + \underbrace{\sqrt{\tilde\beta_t}\,\boldsymbol{z}}_{\textcolor{blue}{\text{re-noise},\ \boldsymbol{z}\sim\mathcal{N}(\mathbf{0},\mathbf{I})}},$$

with $$\boldsymbol{z}=\mathbf{0}$$ at the final step.

![The denoise-then-re-noise view of DDPM sampling (Principles Fig 2.7)](/assets/figures/day06/pdm_denoise_renoise.png)

This is **ancestral sampling** down the Markov chain — accurate but slow ($$T$$ network calls). The deep reason it works is the **change-of-variables / transport** picture: each step moves probability mass a little, and the whole chain transports the prior onto the data distribution. Explore that mass transport here:


<div class="interactive-viz" markdown="0">
  <iframe src="https://the-principles-of-diffusion-models.github.io/assets/cov_2d_map.html" width="100%" height="780" loading="lazy" style="border:1px solid #ddd;border-radius:8px;" title="Change-of-Variable 2D Map"></iframe>
  <p style="font-size:0.85em;color:#888;margin-top:4px;">Interactive: <strong>Change-of-Variable 2D Map</strong> — <a href="https://the-principles-of-diffusion-models.github.io/assets/cov_2d_map.html" target="_blank" rel="noopener">open in new tab</a>. Source: <em>The Principles of Diffusion Models</em>.</p>
</div>


**Outlook — the continuous-time (SDE) view.** Letting the step size go to zero turns the discrete chain into a **stochastic differential equation**. The forward and reverse processes become

$$\textcolor{blue}{\underbrace{\mathrm{d}\boldsymbol{x} = \boldsymbol{f}(\boldsymbol{x},t)\,\mathrm{d}t + g(t)\,\mathrm{d}\boldsymbol{w}}_{\text{forward (noising)}}}, \qquad \textcolor{purple}{\underbrace{\mathrm{d}\boldsymbol{x} = \big[\boldsymbol{f}(\boldsymbol{x},t) - g(t)^2\,\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x})\big]\mathrm{d}t + g(t)\,\mathrm{d}\bar{\boldsymbol{w}}}_{\text{reverse (denoising)}}},$$

where the **score** $$\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x})$$ plays the role our $$\boldsymbol{\epsilon}_\theta$$ learned. DDPM is exactly the discretization of the variance-preserving case of this SDE.

![Forward and reverse diffusion as SDEs, with the shared probability-flow ODE (Song et al., 2020; via the SDE course)](/assets/figures/day06/sde_song_diffusion.png)

To *simulate* such an SDE we use the simplest stochastic solver, **Euler–Maruyama** — the stochastic analogue of Euler's method from Day 1, with the noise entering at scale $$\sqrt{\Delta t}$$ (because a Brownian increment has variance $$\Delta t$$):

![The Euler–Maruyama discretization of an SDE (SDE course)](/assets/figures/day06/sde_euler_maruyama.png)

The full continuous-time derivations — Brownian motion, Itô calculus, the time-reversal formula, and Girsanov's theorem — are developed step by step in the companion [SDE course notes](https://kierandidi.github.io/). In **Day 7** we make this score/SDE view precise (score matching, flow matching); in **Day 8** we replace the $$T$$ tiny stochastic steps with a handful of ODE/SDE solver steps.

## Checkpoint summary

Before moving to the practical, confirm you can:

- Explain the generative goal and place diffusion in the taxonomy of generative models.
- Derive the ELBO and identify its reconstruction and KL terms; state the reparameterization trick.
- Derive the closed-form forward marginal $$\boldsymbol{x}_t=\sqrt{\bar\alpha_t}\boldsymbol{x}_0+\sqrt{1-\bar\alpha_t}\boldsymbol{\epsilon}$$ and relate it to $$(\alpha_t,\sigma_t)$$ and SNR.
- Compare VP, VE, and linear-interpolation noise schedules.
- Derive the Gaussian reverse posterior $$q(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_t,\boldsymbol{x}_0)$$ and explain the conditioning trick.
- Show how the KL training terms reduce to the simple noise-prediction loss, and write the ancestral sampler.
