"""Day 6 — Generative Modeling & DDPM (the variational view).

Sources: *The Principles of Diffusion Models* (Lai, Song, Kim, Mitsufuji,
Ermon, arXiv:2510.21890), Chapters 1–2; UCL x DeepMind L9–L11 (deep generative
models). Notation is unified with the Principles book:

    forward rule   x_t = alpha_t x_0 + sigma_t eps,   eps ~ N(0, I)
    marginal       p(x_t | x_0) = N(x_t; alpha_t x_0, sigma_t^2 I)
    VP schedule    alpha_t^2 + sigma_t^2 = 1
    log-SNR        lambda_t = log(alpha_t^2 / sigma_t^2)

All prose and derivations are written for this course (gaps in the book's
derivations are filled in step by step). Interactive widgets are referenced by
``{{viz:KEY}}`` markers expanded by generate_lectures.py.
"""

# Curated figures aligned to each content slide (in order). None => no figure slide.
FIGURES = [
    # Deep Generative Modeling
    "/assets/figures/day06/pdm_dgm_target.png",        # The generative goal
    "/assets/figures/day06/pdm_dgm_zoo.png",           # Taxonomy of models
    None,                                              # Latent-variable models
    # VAEs
    "/assets/figures/day06/pdm_vae.png",               # The ELBO
    None,                                              # Derivation: ELBO in three lines
    None,                                              # Reparameterization trick
    # Forward process
    "/assets/figures/day06/pdm_ddpm_forward.png",      # Unified forward rule
    None,                                              # Closed-form marginal
    None,                                              # Derivation: compose the Gaussians
    "/assets/figures/day06/pdm_ddpm_overview.png",     # Noise schedules
    # Reverse process
    "/assets/figures/day06/pdm_ddpm_reverse.png",      # Reverse as denoising
    "/assets/figures/day06/pdm_ddpm_conditioning.png", # The true posterior
    None,                                              # Derivation: Gaussian posterior
    None,                                              # Parameterizing reverse
    # Training
    None,                                              # Variational bound -> simple loss
    None,                                              # Derivation: KL -> noise prediction
    None,                                              # epsilon-prediction objective
    "/assets/figures/day06/pdm_denoise_renoise.png",   # Denoise-then-renoise sampling
    # Outlook: from DDPM to SDEs (figures from the SDE course)
    "/assets/figures/day06/sde_song_diffusion.png",    # Forward/reverse as SDEs
    "/assets/figures/day06/sde_euler_maruyama.png",    # Euler-Maruyama simulation
]

SLIDES = (
    "Generative Modeling & DDPM",
    "The variational view — from VAEs to diffusion",
    [
        (
            "Deep Generative Modeling",
            [
                (
                    "The Generative Goal",
                    [
                        "Map a *simple* distribution (Gaussian noise) to a *complex* one (data)",
                        "Sample $x tilde p_\"data\"$ we can only access through examples",
                        "Diffusion: don't jump noise$arrow.r$data in one step — move in many small steps",
                        "Forward = corrupt data into noise; reverse = learn to undo it",
                    ],
                ),
                (
                    "Taxonomy of Models",
                    [
                        "Likelihood-based: VAEs, normalizing flows, autoregressive, diffusion",
                        "Implicit: GANs (no tractable $p(x)$)",
                        "Trade-offs: sample quality vs exact likelihood vs sampling speed",
                        "Diffusion = likelihood-based + high quality, but slow sampling",
                    ],
                ),
                (
                    "Latent-Variable Models",
                    [
                        "Introduce latent $z$: $p_theta (x) = integral p_theta (x|z) p(z) dif z$",
                        "Prior $p(z) = N(0, I)$ is easy to sample",
                        "Marginal likelihood is intractable (integral over all $z$)",
                        "Posterior $p_theta (z|x)$ also intractable $arrow.r$ variational inference",
                    ],
                ),
            ],
        ),
        (
            "Variational Autoencoders",
            [
                (
                    "The Evidence Lower Bound",
                    [
                        "Encoder $q_phi (z|x)$ approximates the true posterior",
                        "$log p_theta (x) >= EE_(q_phi)[log p_theta (x|z)] - D_\"KL\"(q_phi (z|x) || p(z))$",
                        "Reconstruction term + regularization term",
                        "Gap to $log p(x)$ is exactly $D_\"KL\"(q_phi (z|x) || p_theta (z|x)) >= 0$",
                    ],
                ),
                (
                    "Derivation: ELBO in Three Lines",
                    [
                        "Start: $log p_theta (x) = log integral p_theta (x,z) dif z$",
                        "Multiply & divide by $q_phi$: $= log EE_(q_phi)[p_theta (x,z) \\/ q_phi (z|x)]$",
                        "Jensen ($log$ concave): $>= EE_(q_phi)[log (p_theta (x,z) \\/ q_phi (z|x))]$",
                        "Split the log: $= EE_(q_phi)[log p_theta (x|z)] - D_\"KL\"(q_phi || p(z))$",
                        "Exact gap: $log p(x) - \"ELBO\" = D_\"KL\"(q_phi (z|x) || p_theta (z|x)) >= 0$",
                        "Same algebra reused for the diffusion ELBO below",
                    ],
                ),
                (
                    "Reparameterization Trick",
                    [
                        "Sample $z = mu_phi (x) + sigma_phi (x) dot.op epsilon$, $epsilon tilde N(0, I)$",
                        "Moves randomness off the computation path $arrow.r$ low-variance gradients",
                        "Backprop flows through $mu_phi$, $sigma_phi$",
                        "DDPM = a *deep hierarchy* of these latents with a fixed encoder",
                    ],
                ),
            ],
        ),
        (
            "The Forward (Noising) Process",
            [
                (
                    "Unified Forward Rule",
                    [
                        "$x_t = alpha_t x_0 + sigma_t epsilon$, $epsilon tilde N(0, I)$",
                        "$p(x_t | x_0) = N(x_t; alpha_t x_0, sigma_t^2 I)$",
                        "$alpha_t$ = signal kept; $sigma_t$ = noise added",
                        "Signal-to-noise ratio $\"SNR\"(t) = alpha_t^2 / sigma_t^2$ decreases with $t$",
                    ],
                ),
                (
                    "Closed-Form Marginal",
                    [
                        "Step kernel $q(x_t | x_(t-1)) = N(sqrt(1 - beta_t) x_(t-1), beta_t I)$",
                        "Compose Gaussians: $x_t = sqrt(macron(alpha)_t) x_0 + sqrt(1 - macron(alpha)_t) epsilon$",
                        "$macron(alpha)_t = product_(s=1)^t (1 - beta_s)$ so $alpha_t = sqrt(macron(alpha)_t)$",
                        "Sample any $t$ in one shot — no need to simulate the chain",
                    ],
                ),
                (
                    "Derivation: Compose the Gaussians",
                    [
                        "Write $a_t = 1 - beta_t$, so $x_t = sqrt(a_t) x_(t-1) + sqrt(1-a_t) epsilon_t$",
                        "Substitute one step into the next (telescoping)",
                        "Two independent Gaussians add $arrow.r$ variances add",
                        "$a_t (1 - a_(t-1)) + (1 - a_t) = 1 - a_t a_(t-1)$",
                        "Iterate to $x_0$: $x_t = sqrt(macron(alpha)_t) x_0 + sqrt(1 - macron(alpha)_t) epsilon$",
                        "VP schedule: $alpha_t^2 + sigma_t^2 = 1$",
                    ],
                ),
                (
                    "Noise Schedules",
                    [
                        "VP (DDPM/cosine): $alpha_t^2 + sigma_t^2 = 1$",
                        "VE (EDM): $alpha_t = 1$, $sigma_t = t$",
                        "Linear interpolation (flow matching): $alpha_t = 1-t$, $sigma_t = t$",
                        "Interactive — compare schedules on the same image (see notes)",
                    ],
                ),
            ],
        ),
        (
            "The Reverse (Denoising) Process",
            [
                (
                    "Reverse as Denoising",
                    [
                        "Start at $x_T tilde N(0, I)$, walk back to data",
                        "Learn $p_theta (x_(t-1) | x_t)$ — one denoising step",
                        "Oracle reverse kernel $p(x_(t-1)|x_t)$ is intractable...",
                        "...but $q(x_(t-1) | x_t, x_0)$ *is* Gaussian (condition on $x_0$)",
                    ],
                ),
                (
                    "The True Posterior",
                    [
                        "$q(x_(t-1) | x_t, x_0) = N(macron(mu)_t (x_t, x_0), macron(beta)_t I)$",
                        "$macron(mu)_t = (sqrt(macron(alpha)_(t-1)) beta_t)/(1 - macron(alpha)_t) x_0 + (sqrt(1-beta_t)(1 - macron(alpha)_(t-1)))/(1 - macron(alpha)_t) x_t$",
                        "$macron(beta)_t = (1 - macron(alpha)_(t-1))/(1 - macron(alpha)_t) beta_t$",
                        "Conditioning trick: the secret sauce that makes training tractable",
                    ],
                ),
                (
                    "Derivation: Gaussian Posterior",
                    [
                        "Bayes: $q(x_(t-1)|x_t,x_0) prop q(x_t|x_(t-1)) q(x_(t-1)|x_0)$",
                        "Both factors Gaussian $arrow.r$ collect quadratic & linear terms in $x_(t-1)$",
                        "Precision: $1\\/macron(beta)_t = a_t\\/beta_t + 1\\/(1 - macron(alpha)_(t-1))$",
                        "Variance: $macron(beta)_t = (1 - macron(alpha)_(t-1))\\/(1 - macron(alpha)_t) beta_t$",
                        "Mean: $macron(mu)_t = (sqrt(macron(alpha)_(t-1)) beta_t)\\/(1-macron(alpha)_t) x_0 + (sqrt(a_t)(1-macron(alpha)_(t-1)))\\/(1-macron(alpha)_t) x_t$",
                    ],
                ),
                (
                    "Parameterizing the Reverse Step",
                    [
                        "Match $p_theta (x_(t-1)|x_t) = N(mu_theta (x_t, t), macron(beta)_t I)$ to the posterior",
                        "Predict $x_0$, the noise $epsilon$, or the velocity $v$ — equivalent",
                        "$epsilon$-prediction: $mu_theta$ written via $epsilon_theta (x_t, t)$",
                        "Network sees $(x_t, t)$ only — never the true $x_0$",
                    ],
                ),
            ],
        ),
        (
            "Training DDPM",
            [
                (
                    "Variational Bound to Simple Loss",
                    [
                        "ELBO over the chain = $L_T + sum_t L_(t-1) + L_0$",
                        "Each $L_(t-1) = D_\"KL\"(q(x_(t-1)|x_t,x_0) || p_theta (x_(t-1)|x_t))$",
                        "KL of two Gaussians = weighted $||macron(mu)_t - mu_theta||^2$",
                        "Substitute $epsilon$-parameterization $arrow.r$ clean noise-prediction loss",
                    ],
                ),
                (
                    "Derivation: KL to Noise Prediction",
                    [
                        "Same-covariance Gaussians: $L_(t-1) = 1\\/(2 macron(beta)_t) ||macron(mu)_t - mu_theta||^2$",
                        "Estimate $hat(x)_0 = (x_t - sqrt(1-macron(alpha)_t) epsilon_theta)\\/sqrt(macron(alpha)_t)$",
                        "Substitute into $macron(mu)_t$ and $mu_theta$",
                        "Prefactors $1\\/sqrt(a_t)$ and $x_t$ cancel — only noises remain",
                        "$L_(t-1) prop ||epsilon - epsilon_theta (x_t,t)||^2$",
                        "Set the weight to 1 $arrow.r L_\"simple\"$ (up-weights hard, high-noise steps)",
                    ],
                ),
                (
                    "The $epsilon$-Prediction Objective",
                    [
                        "$L_\"simple\" = EE_(t, x_0, epsilon)[ ||epsilon - epsilon_theta (x_t, t)||^2 ]$",
                        "$x_t = sqrt(macron(alpha)_t) x_0 + sqrt(1 - macron(alpha)_t) epsilon$",
                        "Drops the awkward per-$t$ weights — works better in practice",
                        "Tweedie links $epsilon$ to the score: $nabla log p_t (x_t) = -epsilon_theta \\/ sigma_t$ (Day 7)",
                    ],
                ),
                (
                    "Sampling: Denoise then Re-noise",
                    [
                        "From $x_t$: predict $epsilon_theta$, estimate $hat(x)_0$, jump to posterior mean",
                        "$x_(t-1) = 1\\/sqrt(a_t) (x_t - beta_t\\/sqrt(1-macron(alpha)_t) epsilon_theta) + sqrt(macron(beta)_t) z$",
                        "Add a little fresh noise $z tilde N(0,I)$ (except at the last step)",
                        "Repeat $T arrow.r 0$ — this is ancestral sampling",
                        "Day 8: do this in far fewer steps with ODE/SDE solvers",
                    ],
                ),
            ],
        ),
        (
            "From DDPM to SDEs (Outlook)",
            [
                (
                    "Forward & Reverse as SDEs",
                    [
                        "Take the step size to zero: the chain becomes a continuous-time SDE",
                        "Forward: $dif x = f(x,t) dif t + g(t) dif w$ (drift + Brownian noise)",
                        "Reverse: $dif x = [f - g^2 nabla log p_t (x)] dif t + g dif macron(w)$",
                        "The score $nabla log p_t (x)$ replaces the unknown reverse drift",
                        "Probability-flow ODE shares the same marginals (deterministic sampling)",
                        "DDPM is the discretized variance-preserving SDE — full derivations on Day 7",
                    ],
                ),
                (
                    "Simulating SDEs: Euler-Maruyama",
                    [
                        "Discretize $[t_0, t]$ into steps of size $Delta t$",
                        "Brownian increment $Delta beta(t_k) tilde N(0, Delta t I)$",
                        "$x_(k+1) = x_k + b(x_k,t_k) Delta t + sigma(x_k,t_k) Delta beta(t_k)$",
                        "Noise enters at scale $sqrt(Delta t)$ since $\"Var\" = Delta t$",
                        "DDPM ancestral sampling = the VP special case of this scheme",
                        "Continuous-time tools: Brownian motion, time reversal, Girsanov",
                    ],
                ),
            ],
        ),
    ],
)

# ---------------------------------------------------------------------------
# Lecture notes (detailed, with worked derivations and embedded widgets).
LECTURE = {
    "day": 6,
    "slug": "generative-modeling-diffusion",
    "title": "Generative Modeling & DDPM",
    "description": (
        "Deep generative modeling, VAEs and the ELBO, and the variational view of "
        "diffusion: the DDPM forward process, the reverse posterior, and the "
        "noise-prediction training objective — with interactive visualizations."
    ),
    "reading": [
        "[Lai, Song, Kim, Mitsufuji & Ermon — *The Principles of Diffusion Models*](https://arxiv.org/abs/2510.21890), Ch. 1–2",
        "[Ho, Jain & Abbeel — *Denoising Diffusion Probabilistic Models* (DDPM)](https://arxiv.org/abs/2006.11239)",
        "[Kingma & Welling — *Auto-Encoding Variational Bayes* (VAE)](https://arxiv.org/abs/1312.6114)",
        "[Interactive companion & teaching guide](https://the-principles-of-diffusion-models.github.io/)",
        "[Generative Modelling with SDEs — course notes (Brownian motion, Euler–Maruyama, time reversal, Girsanov)](https://kierandidi.github.io/) for the continuous-time derivations previewed at the end",
    ],
    "intro": (
        "Week 2 turns to **generative modeling**: instead of predicting a label from an input, "
        "we want to model the data distribution itself and draw new samples from it. Today we "
        "build the **variational view of diffusion**. We start from variational autoencoders "
        "(VAEs) and the evidence lower bound, then show that a *denoising diffusion probabilistic "
        "model* (DDPM) is essentially a deep, fixed-encoder hierarchical VAE. We will derive the "
        "forward noising process and its closed-form marginal, the Gaussian reverse posterior that "
        "makes training tractable, and the surprisingly simple noise-prediction loss. "
        "Throughout we use a unified notation "
        "($x_t = \\alpha_t x_0 + \\sigma_t \\epsilon$) so that everything connects cleanly to the "
        "score-based and flow-based views in Day 7."
    ),
    "sections": [
        {
            "title": "Deep Generative Modeling",
            "subsections": [
                {
                    "heading": "The generative goal and a taxonomy",
                    "definition": (
                        "A **deep generative model** learns to turn samples from a simple reference "
                        "distribution (e.g. $$\\boldsymbol{z}\\sim\\mathcal{N}(\\mathbf{0},\\mathbf{I})$$) "
                        "into samples that look like data $$\\boldsymbol{x}\\sim p_{\\text{data}}$$, "
                        "while ideally also assigning a likelihood $$p_\\theta(\\boldsymbol{x})$$."
                    ),
                    "body": """We never see $$p_{\\text{data}}$$ directly — only a finite set of examples (images, audio, molecules). The model is a procedure that maps the *simple side* (noise we can generate at will) to the *complex side* (structured data).

![The target of deep generative modeling: transport a simple distribution to the data distribution](/assets/figures/day06/pdm_dgm_target.png)

Generative models differ in **what they make tractable**:

- **Likelihood-based, exact**: normalizing flows and autoregressive models give exact $$\\log p_\\theta(\\boldsymbol{x})$$ but constrain the architecture.
- **Likelihood-based, bounded**: VAEs and diffusion optimize a *lower bound* on $$\\log p_\\theta(\\boldsymbol{x})$$.
- **Implicit**: GANs sample well but expose no tractable density.

![Computation graphs of prominent deep generative models](/assets/figures/day06/pdm_dgm_zoo.png)

Diffusion models sit in the second group: they are likelihood-based (so training is stable and principled) and produce state-of-the-art samples, at the cost of **iterative** sampling — the central tension we resolve in Day 8.""",
                },
                {
                    "heading": "Latent-variable models and intractability",
                    "definition": (
                        "A **latent-variable model** writes the data density as a marginal over a "
                        "hidden variable $$\\boldsymbol{z}$$: "
                        "$$p_\\theta(\\boldsymbol{x}) = \\int p_\\theta(\\boldsymbol{x}\\mid\\boldsymbol{z})\\,p(\\boldsymbol{z})\\,\\mathrm{d}\\boldsymbol{z}.$$"
                    ),
                    "body": """Choosing a simple prior $$p(\\boldsymbol{z}) = \\mathcal{N}(\\mathbf{0},\\mathbf{I})$$ and a flexible decoder $$p_\\theta(\\boldsymbol{x}\\mid\\boldsymbol{z})$$ makes *sampling* trivial (draw $$\\boldsymbol{z}$$, push it through the decoder). But two quantities are **intractable**:

1. the **marginal likelihood** $$p_\\theta(\\boldsymbol{x})$$ — an integral over all $$\\boldsymbol{z}$$;
2. the **posterior** $$p_\\theta(\\boldsymbol{z}\\mid\\boldsymbol{x}) = p_\\theta(\\boldsymbol{x}\\mid\\boldsymbol{z})p(\\boldsymbol{z})/p_\\theta(\\boldsymbol{x})$$, which inherits the same intractable normalizer.

Variational inference sidesteps both by introducing an **approximate posterior** $$q_\\phi(\\boldsymbol{z}\\mid\\boldsymbol{x})$$ and optimizing a bound — the subject of the next section. Diffusion will take this idea to the extreme, using a *chain* of latents $$\\boldsymbol{x}_1,\\dots,\\boldsymbol{x}_T$$ with a **fixed** (non-learned) approximate posterior.""",
                },
            ],
        },
        {
            "title": "Variational Autoencoders and the ELBO",
            "subsections": [
                {
                    "heading": "Deriving the evidence lower bound",
                    "definition": (
                        "The **evidence lower bound (ELBO)** is a tractable lower bound on the "
                        "log-likelihood obtained with any approximate posterior $$q_\\phi$$: "
                        "$$\\log p_\\theta(\\boldsymbol{x}) \\ge "
                        "\\mathbb{E}_{q_\\phi(\\boldsymbol{z}\\mid\\boldsymbol{x})}"
                        "[\\log p_\\theta(\\boldsymbol{x}\\mid\\boldsymbol{z})] - "
                        "D_{\\mathrm{KL}}\\!\\big(q_\\phi(\\boldsymbol{z}\\mid\\boldsymbol{x})\\,\\|\\,p(\\boldsymbol{z})\\big).$$"
                    ),
                    "body": """Let us derive it carefully, because the *same* algebra reappears for diffusion. Start from the log-evidence and multiply-and-divide by $$q_\\phi$$ inside an expectation:

$$\\begin{aligned}
\\log p_\\theta(\\boldsymbol{x})
&= \\log \\int p_\\theta(\\boldsymbol{x},\\boldsymbol{z})\\,\\mathrm{d}\\boldsymbol{z}
= \\log \\mathbb{E}_{q_\\phi(\\boldsymbol{z}\\mid\\boldsymbol{x})}\\!\\left[\\frac{p_\\theta(\\boldsymbol{x},\\boldsymbol{z})}{q_\\phi(\\boldsymbol{z}\\mid\\boldsymbol{x})}\\right] \\\\
&\\ge \\mathbb{E}_{q_\\phi(\\boldsymbol{z}\\mid\\boldsymbol{x})}\\!\\left[\\log \\frac{p_\\theta(\\boldsymbol{x},\\boldsymbol{z})}{q_\\phi(\\boldsymbol{z}\\mid\\boldsymbol{x})}\\right] \\qquad \\textcolor{teal}{\\text{(Jensen's inequality, } \\log \\text{ concave)}} \\\\
&= \\mathbb{E}_{q_\\phi}\\big[\\log p_\\theta(\\boldsymbol{x}\\mid\\boldsymbol{z})\\big] - D_{\\mathrm{KL}}\\!\\big(q_\\phi(\\boldsymbol{z}\\mid\\boldsymbol{x})\\,\\|\\,p(\\boldsymbol{z})\\big) =: \\mathcal{L}_{\\text{ELBO}}.
\\end{aligned}$$

The bound is **exact** when $$q_\\phi(\\boldsymbol{z}\\mid\\boldsymbol{x}) = p_\\theta(\\boldsymbol{z}\\mid\\boldsymbol{x})$$. Indeed, the gap is precisely a KL divergence:

$$\\log p_\\theta(\\boldsymbol{x}) - \\mathcal{L}_{\\text{ELBO}} = \\textcolor{purple}{D_{\\mathrm{KL}}\\!\\big(q_\\phi(\\boldsymbol{z}\\mid\\boldsymbol{x})\\,\\|\\,p_\\theta(\\boldsymbol{z}\\mid\\boldsymbol{x})\\big)} \\ge 0.$$

![A variational autoencoder: stochastic encoder, latent bottleneck, decoder](/assets/figures/day06/pdm_vae.png)

The two ELBO terms have clean interpretations: the first is a **reconstruction** term (the decoder should reproduce $$\\boldsymbol{x}$$ from $$\\boldsymbol{z}$$), the second is a **regularizer** pulling the encoder's posterior toward the prior.""",
                },
                {
                    "heading": "The reparameterization trick",
                    "definition": (
                        "To backpropagate through the sampling step, write the latent as a "
                        "deterministic function of the parameters and an independent noise variable: "
                        "$$\\boldsymbol{z} = \\boldsymbol{\\mu}_\\phi(\\boldsymbol{x}) + "
                        "\\boldsymbol{\\sigma}_\\phi(\\boldsymbol{x})\\odot\\boldsymbol{\\epsilon},\\quad "
                        "\\boldsymbol{\\epsilon}\\sim\\mathcal{N}(\\mathbf{0},\\mathbf{I}).$$"
                    ),
                    "body": """Gradients of $$\\mathbb{E}_{q_\\phi}[\\,f(\\boldsymbol{z})\\,]$$ cannot flow through a raw `sample` node. The reparameterization trick moves the randomness to an input $$\\boldsymbol{\\epsilon}$$ that does not depend on $$\\phi$$, so

$$\\nabla_\\phi\\, \\mathbb{E}_{q_\\phi}[f(\\boldsymbol{z})] = \\mathbb{E}_{\\boldsymbol{\\epsilon}}\\big[\\nabla_\\phi\\, f(\\boldsymbol{\\mu}_\\phi + \\boldsymbol{\\sigma}_\\phi \\odot \\boldsymbol{\\epsilon})\\big],$$

a low-variance, pathwise estimator (Day 1's "reparameterization gradient"). For Gaussian $$q$$ and $$p$$ the KL term is available in **closed form**. Taking $$q=\\mathcal{N}(\\boldsymbol{\\mu},\\mathrm{diag}\\,\\boldsymbol{\\sigma}^2)$$ and $$p=\\mathcal{N}(\\mathbf{0},\\mathbf{I})$$ (so the dimensions decouple), write the KL as an expectation of the log-ratio of densities and use $$\\mathbb{E}_q[(z_i-\\mu_i)^2]=\\sigma_i^2$$, $$\\mathbb{E}_q[z_i^2]=\\mu_i^2+\\sigma_i^2$$:

$$\\begin{aligned}
D_{\\mathrm{KL}}\\big(q\\,\\|\\,p\\big)
&= \\mathbb{E}_{q}\\!\\big[\\log q(\\boldsymbol{z}) - \\log p(\\boldsymbol{z})\\big] \\\\
&= \\sum_i \\mathbb{E}_{q}\\!\\Big[\\,\\underbrace{-\\tfrac12\\log(2\\pi\\sigma_i^2) - \\tfrac{(z_i-\\mu_i)^2}{2\\sigma_i^2}}_{\\textcolor{teal}{\\log q}} \\;+\\; \\underbrace{\\tfrac12\\log(2\\pi) + \\tfrac{z_i^2}{2}}_{\\textcolor{purple}{-\\log p}}\\Big] \\\\
&= \\sum_i \\Big(-\\tfrac12\\log\\sigma_i^2 - \\tfrac12 + \\tfrac12(\\mu_i^2+\\sigma_i^2)\\Big)
= \\tfrac{1}{2}\\sum_i \\big(\\mu_i^2 + \\sigma_i^2 - 1 - \\log \\sigma_i^2\\big).
\\end{aligned}$$

**Key idea for diffusion.** A DDPM is what you get if you (i) replace the single latent by a long chain $$\\boldsymbol{x}_1,\\dots,\\boldsymbol{x}_T$$, (ii) **fix** the encoder to a simple Gaussian noising process instead of learning it, and (iii) learn only the decoder (the reverse/denoising steps). Everything below is the ELBO of that hierarchy.""",
                },
            ],
        },
        {
            "title": "The Forward (Noising) Process",
            "subsections": [
                {
                    "heading": "The unified forward rule",
                    "definition": (
                        "The **forward process** corrupts data into noise through a one-parameter "
                        "family of Gaussians: "
                        "$$p(\\boldsymbol{x}_t\\mid\\boldsymbol{x}_0) = "
                        "\\mathcal{N}\\big(\\boldsymbol{x}_t;\\,\\alpha_t\\boldsymbol{x}_0,\\,\\sigma_t^2\\mathbf{I}\\big),"
                        "\\quad\\text{equivalently}\\quad "
                        "\\boldsymbol{x}_t = \\alpha_t \\boldsymbol{x}_0 + \\sigma_t\\boldsymbol{\\epsilon},\\;\\boldsymbol{\\epsilon}\\sim\\mathcal{N}(\\mathbf{0},\\mathbf{I}).$$"
                    ),
                    "body": """Here $$\\alpha_t$$ is the **signal coefficient** (how much of $$\\boldsymbol{x}_0$$ survives) and $$\\sigma_t$$ is the **noise scale**. As $$t$$ grows, $$\\alpha_t \\to 0$$ and $$\\sigma_t$$ grows, so $$\\boldsymbol{x}_t$$ approaches a tractable prior (a Gaussian). A single scalar summarizes the corruption level, the **signal-to-noise ratio**

$$\\mathrm{SNR}(t) = \\frac{\\alpha_t^2}{\\sigma_t^2}, \\qquad \\lambda_t := \\log \\mathrm{SNR}(t) = \\log\\frac{\\alpha_t^2}{\\sigma_t^2},$$

which decreases monotonically from $$+\\infty$$ (clean data) to $$-\\infty$$ (pure noise). We will reuse $$\\lambda_t$$ (log-SNR) as the natural "clock" for solvers in Day 8.

![The DDPM forward process: Gaussian noise is added step by step until the sample is indistinguishable from noise](/assets/figures/day06/pdm_ddpm_forward.png)""",
                },
                {
                    "heading": "From step kernel to closed-form marginal",
                    "definition": (
                        "For the DDPM Markov chain $$q(\\boldsymbol{x}_t\\mid\\boldsymbol{x}_{t-1}) = "
                        "\\mathcal{N}(\\sqrt{1-\\beta_t}\\,\\boldsymbol{x}_{t-1},\\,\\beta_t\\mathbf{I})$$, "
                        "the $$t$$-step marginal collapses to a single Gaussian "
                        "$$q(\\boldsymbol{x}_t\\mid\\boldsymbol{x}_0) = "
                        "\\mathcal{N}\\big(\\sqrt{\\bar\\alpha_t}\\,\\boldsymbol{x}_0,\\,(1-\\bar\\alpha_t)\\mathbf{I}\\big),"
                        "\\quad \\bar\\alpha_t = \\prod_{s=1}^t (1-\\beta_s).$$"
                    ),
                    "body": """This is the property that makes diffusion trainable: we can jump to any noise level in one step. Let us prove it by induction, writing $$a_t := 1-\\beta_t$$ so that $$\\boldsymbol{x}_t = \\sqrt{a_t}\\,\\boldsymbol{x}_{t-1} + \\sqrt{1-a_t}\\,\\boldsymbol{\\epsilon}_t$$ with independent $$\\boldsymbol{\\epsilon}_t\\sim\\mathcal{N}(\\mathbf{0},\\mathbf{I})$$. Substitute one step into the next:

$$\\begin{aligned}
\\boldsymbol{x}_t &= \\sqrt{a_t}\\,\\big(\\textcolor{blue}{\\sqrt{a_{t-1}}\\,\\boldsymbol{x}_{t-2} + \\sqrt{1-a_{t-1}}\\,\\boldsymbol{\\epsilon}_{t-1}}\\big) + \\sqrt{1-a_t}\\,\\boldsymbol{\\epsilon}_t \\\\
&= \\sqrt{a_t a_{t-1}}\\,\\boldsymbol{x}_{t-2} + \\underbrace{\\sqrt{a_t(1-a_{t-1})}\\,\\boldsymbol{\\epsilon}_{t-1} + \\sqrt{1-a_t}\\,\\boldsymbol{\\epsilon}_t}_{\\textcolor{purple}{\\text{sum of two independent Gaussians}}}.
\\end{aligned}$$

The two noise terms are independent zero-mean Gaussians, so their sum is Gaussian with variance equal to the sum of variances:

$$\\mathrm{Var} = a_t(1-a_{t-1}) + (1-a_t) = 1 - a_t a_{t-1}.$$

Hence $$\\boldsymbol{x}_t = \\sqrt{a_t a_{t-1}}\\,\\boldsymbol{x}_{t-2} + \\sqrt{1 - a_t a_{t-1}}\\,\\bar{\\boldsymbol{\\epsilon}}$$ for a single $$\\bar{\\boldsymbol{\\epsilon}}\\sim\\mathcal{N}(\\mathbf{0},\\mathbf{I})$$. Iterating down to $$\\boldsymbol{x}_0$$ gives

$$\\boxed{\\;\\boldsymbol{x}_t = \\sqrt{\\bar\\alpha_t}\\,\\boldsymbol{x}_0 + \\sqrt{1-\\bar\\alpha_t}\\,\\boldsymbol{\\epsilon},\\qquad \\bar\\alpha_t = \\prod_{s=1}^t a_s.\\;}$$

Comparing with the unified rule, the DDPM schedule is the **variance-preserving (VP)** case

$$\\alpha_t = \\sqrt{\\bar\\alpha_t}, \\qquad \\sigma_t = \\sqrt{1-\\bar\\alpha_t}, \\qquad \\alpha_t^2 + \\sigma_t^2 = 1.$$""",
                },
                {
                    "heading": "Noise schedules and the interactive explorer",
                    "definition": (
                        "A **noise schedule** is the choice of curves $$(\\alpha_t, \\sigma_t)$$. The "
                        "three standard families are **variance preserving** ($$\\alpha_t^2+\\sigma_t^2=1$$), "
                        "**variance exploding** ($$\\alpha_t=1,\\ \\sigma_t$$ growing), and **linear "
                        "interpolation** ($$\\alpha_t=1-t,\\ \\sigma_t=t$$)."
                    ),
                    "body": """All three obey the *same* one-liner $$\\boldsymbol{x}_t = \\alpha_t\\boldsymbol{x}_0 + \\sigma_t\\boldsymbol{\\epsilon}$$; only the shapes of $$(\\alpha_t,\\sigma_t)$$ differ:

| Family | $$\\alpha_t$$ | $$\\sigma_t$$ | Used by |
|---|---|---|---|
| VP (linear-$$\\beta$$) | $$\\exp\\!\\big(-\\tfrac12\\!\\int_0^t\\!\\beta(s)\\,ds\\big)$$ | $$\\sqrt{1-\\alpha_t^2}$$ | DDPM |
| VP (cosine) | $$\\cos(\\tfrac{\\pi t}{2})$$ | $$\\sin(\\tfrac{\\pi t}{2})$$ | improved DDPM |
| VE | $$1$$ | $$t$$ | EDM / NCSN |
| Linear interp. | $$1-t$$ | $$t$$ | flow matching / rectified flow |

The schedule controls **where the model spends capacity**: it determines how fast the SNR drops and therefore which noise levels dominate training. Use the widget to corrupt the same image under each schedule and watch how the SNR curve changes:

{{viz:noise_schedule_explorer}}

![DDPM as a fixed forward chain and a learned reverse chain](/assets/figures/day06/pdm_ddpm_overview.png)""",
                },
            ],
        },
        {
            "title": "The Reverse (Denoising) Process",
            "subsections": [
                {
                    "heading": "Why we condition on the clean sample",
                    "definition": (
                        "Sampling requires the **reverse kernel** $$p(\\boldsymbol{x}_{t-1}\\mid\\boldsymbol{x}_t)$$, "
                        "which is intractable. The trick is that the **conditioned** reverse kernel "
                        "$$q(\\boldsymbol{x}_{t-1}\\mid\\boldsymbol{x}_t,\\boldsymbol{x}_0)$$ is an "
                        "explicit Gaussian."
                    ),
                    "body": """Generation runs the chain backward: start from $$\\boldsymbol{x}_T\\sim\\mathcal{N}(\\mathbf{0},\\mathbf{I})$$ and repeatedly sample $$\\boldsymbol{x}_{t-1}\\sim p_\\theta(\\boldsymbol{x}_{t-1}\\mid\\boldsymbol{x}_t)$$.

![The learned reverse process gradually removes noise to recover data](/assets/figures/day06/pdm_ddpm_reverse.png)

The marginal reverse kernel $$p(\\boldsymbol{x}_{t-1}\\mid\\boldsymbol{x}_t)$$ would require integrating over all data — intractable. But by Bayes' rule, *conditioning on $$\\boldsymbol{x}_0$$* turns it into a ratio of known Gaussians:

$$q(\\boldsymbol{x}_{t-1}\\mid\\boldsymbol{x}_t,\\boldsymbol{x}_0) = \\frac{q(\\boldsymbol{x}_t\\mid\\boldsymbol{x}_{t-1})\\,q(\\boldsymbol{x}_{t-1}\\mid\\boldsymbol{x}_0)}{q(\\boldsymbol{x}_t\\mid\\boldsymbol{x}_0)}.$$

![The conditioning trick: conditioning on $x_0$ makes the reverse step a tractable Gaussian](/assets/figures/day06/pdm_ddpm_conditioning.png)

Use the interactive panel to see how conditioning on $$\\boldsymbol{x}_0$$ pins down the otherwise-ambiguous reverse step:

{{viz:ddpm_conditional_trick}}""",
                },
                {
                    "heading": "Deriving the Gaussian posterior",
                    "definition": (
                        "The conditioned reverse kernel is "
                        "$$q(\\boldsymbol{x}_{t-1}\\mid\\boldsymbol{x}_t,\\boldsymbol{x}_0) = "
                        "\\mathcal{N}\\big(\\tilde{\\boldsymbol{\\mu}}_t(\\boldsymbol{x}_t,\\boldsymbol{x}_0),\\,\\tilde\\beta_t\\mathbf{I}\\big)$$ "
                        "with mean and variance given below."
                    ),
                    "body": """All three densities on the right are Gaussian in $$\\boldsymbol{x}_{t-1}$$, so the product is Gaussian; we only need to collect the quadratic and linear terms in the exponent. Dropping terms that do not involve $$\\boldsymbol{x}_{t-1}$$,

$$\\begin{aligned}
-\\log q(\\boldsymbol{x}_{t-1}\\mid\\boldsymbol{x}_t,\\boldsymbol{x}_0) &\\;\\overset{c}{=}\\; \\frac{\\|\\boldsymbol{x}_t - \\sqrt{a_t}\\,\\boldsymbol{x}_{t-1}\\|^2}{2\\beta_t} + \\frac{\\|\\boldsymbol{x}_{t-1} - \\sqrt{\\bar\\alpha_{t-1}}\\,\\boldsymbol{x}_0\\|^2}{2(1-\\bar\\alpha_{t-1})} \\\\
&\\;\\overset{c}{=}\\; \\frac{1}{2}\\Big(\\underbrace{\\tfrac{a_t}{\\beta_t} + \\tfrac{1}{1-\\bar\\alpha_{t-1}}}_{\\textcolor{blue}{1/\\tilde\\beta_t}}\\Big)\\|\\boldsymbol{x}_{t-1}\\|^2 - \\Big(\\tfrac{\\sqrt{a_t}}{\\beta_t}\\boldsymbol{x}_t + \\tfrac{\\sqrt{\\bar\\alpha_{t-1}}}{1-\\bar\\alpha_{t-1}}\\boldsymbol{x}_0\\Big)^{\\!\\top}\\boldsymbol{x}_{t-1}.
\\end{aligned}$$

Reading off a Gaussian $$\\propto \\exp(-\\tfrac{1}{2\\tilde\\beta_t}\\|\\boldsymbol{x}_{t-1}-\\tilde{\\boldsymbol{\\mu}}_t\\|^2)$$, the **precision** gives the variance and the **linear term** gives the mean. Using $$a_t = 1-\\beta_t$$ and $$\\bar\\alpha_t = a_t\\bar\\alpha_{t-1}$$ to simplify,

$$\\boxed{\\;\\tilde\\beta_t = \\frac{1-\\bar\\alpha_{t-1}}{1-\\bar\\alpha_t}\\,\\beta_t,\\qquad \\tilde{\\boldsymbol{\\mu}}_t(\\boldsymbol{x}_t,\\boldsymbol{x}_0) = \\frac{\\sqrt{\\bar\\alpha_{t-1}}\\,\\beta_t}{1-\\bar\\alpha_t}\\,\\boldsymbol{x}_0 + \\frac{\\sqrt{a_t}\\,(1-\\bar\\alpha_{t-1})}{1-\\bar\\alpha_t}\\,\\boldsymbol{x}_t.\\;}$$

This is the **target** the reverse network must match. Note it depends on the (unknown at sampling time) $$\\boldsymbol{x}_0$$ — handled next by predicting it.""",
                },
                {
                    "heading": "Parameterizing the reverse step",
                    "definition": (
                        "We set $$p_\\theta(\\boldsymbol{x}_{t-1}\\mid\\boldsymbol{x}_t) = "
                        "\\mathcal{N}\\big(\\boldsymbol{\\mu}_\\theta(\\boldsymbol{x}_t,t),\\,\\tilde\\beta_t\\mathbf{I}\\big)$$ "
                        "and choose what the network predicts: the clean sample $$\\boldsymbol{x}_0$$, the "
                        "noise $$\\boldsymbol{\\epsilon}$$, or a velocity $$\\boldsymbol{v}$$ — all equivalent."
                    ),
                    "body": """Since at sampling time we only have $$\\boldsymbol{x}_t$$, we estimate $$\\boldsymbol{x}_0$$ from it. From the forward marginal $$\\boldsymbol{x}_t = \\sqrt{\\bar\\alpha_t}\\,\\boldsymbol{x}_0 + \\sqrt{1-\\bar\\alpha_t}\\,\\boldsymbol{\\epsilon}$$ we can solve for $$\\boldsymbol{x}_0$$ in terms of the noise:

$$\\hat{\\boldsymbol{x}}_0 = \\frac{1}{\\sqrt{\\bar\\alpha_t}}\\big(\\boldsymbol{x}_t - \\sqrt{1-\\bar\\alpha_t}\\,\\textcolor{teal}{\\boldsymbol{\\epsilon}_\\theta(\\boldsymbol{x}_t,t)}\\big).$$

Substituting this $$\\hat{\\boldsymbol{x}}_0$$ into $$\\tilde{\\boldsymbol{\\mu}}_t$$ and simplifying yields the famous **noise-prediction mean**:

$$\\boldsymbol{\\mu}_\\theta(\\boldsymbol{x}_t,t) = \\frac{1}{\\sqrt{a_t}}\\Big(\\boldsymbol{x}_t - \\frac{\\beta_t}{\\sqrt{1-\\bar\\alpha_t}}\\,\\boldsymbol{\\epsilon}_\\theta(\\boldsymbol{x}_t,t)\\Big).$$

So predicting the noise $$\\boldsymbol{\\epsilon}$$ is equivalent to predicting $$\\boldsymbol{x}_0$$, which is equivalent to predicting the posterior mean. Day 7 adds two more equivalent targets (the **score** $$\\nabla\\log p_t$$ and the **velocity** $$\\boldsymbol{v}$$), and shows they are linear reparameterizations of one another.""",
                },
            ],
        },
        {
            "title": "Training and Sampling",
            "subsections": [
                {
                    "heading": "From the variational bound to the simple loss",
                    "definition": (
                        "Maximizing the ELBO of the diffusion hierarchy decomposes into per-step KL "
                        "terms $$L_{t-1} = D_{\\mathrm{KL}}\\big(q(\\boldsymbol{x}_{t-1}\\mid\\boldsymbol{x}_t,\\boldsymbol{x}_0)\\,\\|\\,p_\\theta(\\boldsymbol{x}_{t-1}\\mid\\boldsymbol{x}_t)\\big)$$ "
                        "that reduce to weighted squared errors."
                    ),
                    "body": """Applying the ELBO derivation to the joint $$q(\\boldsymbol{x}_{1:T}\\mid\\boldsymbol{x}_0)$$ and the learned reverse chain $$p_\\theta(\\boldsymbol{x}_{0:T})$$ gives

$$-\\log p_\\theta(\\boldsymbol{x}_0) \\le \\mathbb{E}_q\\Big[\\underbrace{D_{\\mathrm{KL}}(q(\\boldsymbol{x}_T\\mid\\boldsymbol{x}_0)\\,\\|\\,p(\\boldsymbol{x}_T))}_{L_T,\\ \\text{no params}} + \\sum_{t>1}\\underbrace{D_{\\mathrm{KL}}(q(\\boldsymbol{x}_{t-1}\\mid\\boldsymbol{x}_t,\\boldsymbol{x}_0)\\,\\|\\,p_\\theta(\\boldsymbol{x}_{t-1}\\mid\\boldsymbol{x}_t))}_{L_{t-1}} \\underbrace{- \\log p_\\theta(\\boldsymbol{x}_0\\mid\\boldsymbol{x}_1)}_{L_0}\\Big].$$

Because both arguments of each $$L_{t-1}$$ are Gaussians with the **same** covariance $$\\tilde\\beta_t\\mathbf{I}$$, the KL collapses to a scaled distance between means:

$$L_{t-1} \\overset{c}{=} \\frac{1}{2\\tilde\\beta_t}\\,\\big\\|\\tilde{\\boldsymbol{\\mu}}_t(\\boldsymbol{x}_t,\\boldsymbol{x}_0) - \\boldsymbol{\\mu}_\\theta(\\boldsymbol{x}_t,t)\\big\\|^2.$$

Now substitute the two noise-prediction means. The target mean comes from putting the *true* noise $$\\boldsymbol{\\epsilon}$$ (via $$\\boldsymbol{x}_0=(\\boldsymbol{x}_t-\\sqrt{1-\\bar\\alpha_t}\\,\\boldsymbol{\\epsilon})/\\sqrt{\\bar\\alpha_t}$$) into $$\\tilde{\\boldsymbol{\\mu}}_t$$, and the model mean uses $$\\boldsymbol{\\epsilon}_\\theta$$; both have the **same** $$\\tfrac{1}{\\sqrt{a_t}}\\boldsymbol{x}_t$$ piece:

$$\\tilde{\\boldsymbol{\\mu}}_t = \\tfrac{1}{\\sqrt{a_t}}\\Big(\\boldsymbol{x}_t - \\tfrac{\\beta_t}{\\sqrt{1-\\bar\\alpha_t}}\\,\\textcolor{teal}{\\boldsymbol{\\epsilon}}\\Big), \\qquad \\boldsymbol{\\mu}_\\theta = \\tfrac{1}{\\sqrt{a_t}}\\Big(\\boldsymbol{x}_t - \\tfrac{\\beta_t}{\\sqrt{1-\\bar\\alpha_t}}\\,\\textcolor{purple}{\\boldsymbol{\\epsilon}_\\theta}\\Big).$$

Subtracting, the $$\\tfrac{1}{\\sqrt{a_t}}\\boldsymbol{x}_t$$ terms **cancel** and only the noises survive:

$$\\begin{aligned}
\\tilde{\\boldsymbol{\\mu}}_t - \\boldsymbol{\\mu}_\\theta
&= \\frac{\\beta_t}{\\sqrt{a_t}\\,\\sqrt{1-\\bar\\alpha_t}}\\,\\big(\\textcolor{purple}{\\boldsymbol{\\epsilon}_\\theta} - \\textcolor{teal}{\\boldsymbol{\\epsilon}}\\big), \\\\
L_{t-1} = \\frac{1}{2\\tilde\\beta_t}\\big\\|\\tilde{\\boldsymbol{\\mu}}_t - \\boldsymbol{\\mu}_\\theta\\big\\|^2
&\\overset{c}{=} \\textcolor{purple}{\\frac{\\beta_t^2}{2\\tilde\\beta_t\\, a_t\\,(1-\\bar\\alpha_t)}}\\,\\big\\|\\boldsymbol{\\epsilon} - \\boldsymbol{\\epsilon}_\\theta(\\boldsymbol{x}_t,t)\\big\\|^2,
\\end{aligned}$$

with $$\\boldsymbol{x}_t = \\sqrt{\\bar\\alpha_t}\\,\\boldsymbol{x}_0 + \\sqrt{1-\\bar\\alpha_t}\\,\\boldsymbol{\\epsilon}$$.""",
                },
                {
                    "heading": "The simple objective and ancestral sampling",
                    "definition": (
                        "DDPM drops the per-step weight and trains the **simple noise-prediction loss** "
                        "$$L_{\\text{simple}} = \\mathbb{E}_{t,\\boldsymbol{x}_0,\\boldsymbol{\\epsilon}}"
                        "\\big[\\,\\|\\boldsymbol{\\epsilon} - \\boldsymbol{\\epsilon}_\\theta(\\boldsymbol{x}_t,t)\\|^2\\big].$$"
                    ),
                    "body": """Empirically, setting the awkward weight $$\\textcolor{purple}{(\\cdots)}$$ to $$1$$ improves sample quality — it up-weights the harder, higher-noise steps. The resulting training loop is remarkably simple:

1. draw $$\\boldsymbol{x}_0\\sim p_{\\text{data}}$$, a timestep $$t\\sim\\mathcal{U}\\{1,\\dots,T\\}$$, and $$\\boldsymbol{\\epsilon}\\sim\\mathcal{N}(\\mathbf{0},\\mathbf{I})$$;
2. form $$\\boldsymbol{x}_t = \\sqrt{\\bar\\alpha_t}\\,\\boldsymbol{x}_0 + \\sqrt{1-\\bar\\alpha_t}\\,\\boldsymbol{\\epsilon}$$;
3. take a gradient step on $$\\|\\boldsymbol{\\epsilon} - \\boldsymbol{\\epsilon}_\\theta(\\boldsymbol{x}_t,t)\\|^2$$.

**Sampling** ("denoise then re-noise") runs the reverse chain: from $$\\boldsymbol{x}_T\\sim\\mathcal{N}(\\mathbf{0},\\mathbf{I})$$, repeat for $$t=T,\\dots,1$$

$$\\boldsymbol{x}_{t-1} = \\underbrace{\\frac{1}{\\sqrt{a_t}}\\Big(\\boldsymbol{x}_t - \\frac{\\beta_t}{\\sqrt{1-\\bar\\alpha_t}}\\,\\boldsymbol{\\epsilon}_\\theta(\\boldsymbol{x}_t,t)\\Big)}_{\\textcolor{teal}{\\text{denoise toward posterior mean}}} + \\underbrace{\\sqrt{\\tilde\\beta_t}\\,\\boldsymbol{z}}_{\\textcolor{blue}{\\text{re-noise},\\ \\boldsymbol{z}\\sim\\mathcal{N}(\\mathbf{0},\\mathbf{I})}},$$

with $$\\boldsymbol{z}=\\mathbf{0}$$ at the final step.

![The denoise-then-re-noise view of DDPM sampling](/assets/figures/day06/pdm_denoise_renoise.png)

This is **ancestral sampling** down the Markov chain — accurate but slow ($$T$$ network calls). The deep reason it works is the **change-of-variables / transport** picture: each step moves probability mass a little, and the whole chain transports the prior onto the data distribution. Explore that mass transport here:

{{viz:cov_2d_map}}

**Outlook — the continuous-time (SDE) view.** Letting the step size go to zero turns the discrete chain into a **stochastic differential equation**. The forward and reverse processes become

$$\\textcolor{blue}{\\underbrace{\\mathrm{d}\\boldsymbol{x} = \\boldsymbol{f}(\\boldsymbol{x},t)\\,\\mathrm{d}t + g(t)\\,\\mathrm{d}\\boldsymbol{w}}_{\\text{forward (noising)}}}, \\qquad \\textcolor{purple}{\\underbrace{\\mathrm{d}\\boldsymbol{x} = \\big[\\boldsymbol{f}(\\boldsymbol{x},t) - g(t)^2\\,\\nabla_{\\boldsymbol{x}}\\log p_t(\\boldsymbol{x})\\big]\\mathrm{d}t + g(t)\\,\\mathrm{d}\\bar{\\boldsymbol{w}}}_{\\text{reverse (denoising)}}},$$

where the **score** $$\\nabla_{\\boldsymbol{x}}\\log p_t(\\boldsymbol{x})$$ plays the role our $$\\boldsymbol{\\epsilon}_\\theta$$ learned. DDPM is exactly the discretization of the variance-preserving case of this SDE.

![Forward and reverse diffusion as SDEs, with the shared probability-flow ODE](/assets/figures/day06/sde_song_diffusion.png)

To *simulate* such an SDE we use the simplest stochastic solver, **Euler–Maruyama** — the stochastic analogue of Euler's method from Day 1, with the noise entering at scale $$\\sqrt{\\Delta t}$$ (because a Brownian increment has variance $$\\Delta t$$):

![The Euler–Maruyama discretization of an SDE](/assets/figures/day06/sde_euler_maruyama.png)

The full continuous-time derivations — Brownian motion, Itô calculus, the time-reversal formula, and Girsanov's theorem — are developed step by step in the optional deep dives. In **Day 7** we make this score/SDE view precise (score matching, flow matching); in **Day 8** we replace the $$T$$ tiny stochastic steps with a handful of ODE/SDE solver steps.""",
                },
            ],
        },
    ],
    "checkpoint": [
        "Explain the generative goal and place diffusion in the taxonomy of generative models.",
        "Derive the ELBO and identify its reconstruction and KL terms; state the reparameterization trick.",
        "Derive the closed-form forward marginal $$\\boldsymbol{x}_t=\\sqrt{\\bar\\alpha_t}\\boldsymbol{x}_0+\\sqrt{1-\\bar\\alpha_t}\\boldsymbol{\\epsilon}$$ and relate it to $$(\\alpha_t,\\sigma_t)$$ and SNR.",
        "Compare VP, VE, and linear-interpolation noise schedules.",
        "Derive the Gaussian reverse posterior $$q(\\boldsymbol{x}_{t-1}\\mid\\boldsymbol{x}_t,\\boldsymbol{x}_0)$$ and explain the conditioning trick.",
        "Show how the KL training terms reduce to the simple noise-prediction loss, and write the ancestral sampler.",
    ],
}
