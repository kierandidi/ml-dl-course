"""Day 7 — Score-Based Models, SDEs, and Flow Matching.

Sources: *The Principles of Diffusion Models* (Lai, Song, Kim, Mitsufuji, Ermon,
arXiv:2510.21890), Chapters 3–6; Song et al., "Score-Based Generative Modeling
through SDEs" (2021); Lipman et al., "Flow Matching for Generative Modeling"
(2023). Notation is unified with the Principles book and with Day 6:

    forward rule   x_t = alpha_t x_0 + sigma_t eps,   eps ~ N(0, I)
    marginal       p_t(x_t | x_0) = N(alpha_t x_0, sigma_t^2 I)
    score          s(x, t) = grad_x log p_t(x)

All prose and derivations are written for this course; gaps in the book's
derivations are filled in step by step, with color-coding in the notes.
Interactive widgets are referenced by ``{{viz:KEY}}`` markers expanded by
generate_lectures.py.
"""

from diffusion_appendix import DAY07_OPTIONAL

# Curated figures aligned to each content slide (in order). None => no figure slide.
FIGURES = [
    # The score function
    "/assets/figures/day07/pdm_score_field.png",        # What is the score
    "/assets/figures/day07/pdm_ebm_training.png",       # Energy-based models
    None,                                               # Why the score avoids the normalizer
    # Learning the score
    "/assets/figures/day07/pdm_score_matching.png",     # Score matching
    None,                                               # Derivation: implicit score matching
    "/assets/figures/day07/pdm_dsm_trick.png",          # Denoising score matching
    "/assets/figures/day07/pdm_ncsn.png",               # Multiple noise scales (NCSN)
    # Sampling
    "/assets/figures/day07/pdm_langevin.png",           # Langevin dynamics
    # Continuous-time view
    "/assets/figures/day07/pdm_forward_1d.png",         # The forward SDE
    "/assets/figures/day07/pdm_score_landscape.png",    # Time-dependent score
    "/assets/figures/day07/pdm_three_dynamics.png",     # Reverse SDE & PF-ODE
    None,                                               # Derivation: DSM objective
    # Flow matching
    "/assets/figures/day07/pdm_nf.png",                 # Continuous normalizing flows
    "/assets/figures/day07/pdm_cond_transition.png",    # Conditional flow matching
    "/assets/figures/day07/pdm_cond_vs_marginal.png",   # Conditional vs marginal velocity
    "/assets/figures/day07/pdm_curved_paths.png",       # Rectified flow & reflow
    # Unified view
    "/assets/figures/day07/pdm_param_equiv.png",        # Four equivalent parameterizations
    "/assets/figures/day07/pdm_unified.png",            # The unified picture
]

SLIDES = (
    "Score, SDEs & Flow Matching",
    "The continuous-time view of diffusion",
    [
        (
            "The Score Function",
            [
                (
                    "What Is the Score?",
                    [
                        "Score = gradient of log-density: $s(x) = nabla_x log p(x)$",
                        "A *vector field* pointing toward higher probability",
                        "Day 6 learned to denoise; today we learn this field",
                        "Knowing the score is enough to *sample* (Langevin, SDE, ODE)",
                        "Source: Principles of Diffusion Models, Ch. 3–6",
                    ],
                ),
                (
                    "Energy-Based Models",
                    [
                        "Write $p_theta (x) = e^(-E_theta (x)) \\/ Z_theta$",
                        "Normalizer $Z_theta = integral e^(-E_theta (x)) dif x$ is intractable",
                        "Max-likelihood needs $nabla_theta log Z_theta$ — hard",
                        "Idea: model the *score* and sidestep $Z$ entirely",
                        "Flexible energy, but sampling needs MCMC",
                    ],
                ),
                (
                    "Why the Score Avoids the Normalizer",
                    [
                        "$log p_theta (x) = -E_theta (x) - log Z_theta$",
                        "$Z_theta$ does not depend on $x$",
                        "$arrow.r nabla_x log p_theta (x) = -nabla_x E_theta (x)$",
                        "The intractable constant *vanishes* under $nabla_x$",
                        "Learn the shape of $p$, not its normalization",
                    ],
                ),
            ],
        ),
        (
            "Learning the Score",
            [
                (
                    "Score Matching",
                    [
                        "Fit $s_theta (x) approx nabla_x log p_\"data\" (x)$",
                        "Naive loss needs the unknown true score",
                        "Hyvärinen: integrate by parts to remove it",
                        "$J = EE[1/2 \\|s_theta\\|^2 + \"tr\"(nabla_x s_theta)]$",
                        "Trace (Jacobian) term is costly in high dimension",
                    ],
                ),
                (
                    "Derivation: Implicit Score Matching",
                    [
                        "Start: $1/2 EE_(p)\\|s_theta (x) - nabla log p(x)\\|^2$",
                        "Expand the square; the cross term has $nabla log p$",
                        "$EE_p [s_theta^T nabla log p] = EE_p[-nabla dot.op s_theta]$ (parts)",
                        "Uses $p nabla log p = nabla p$ and vanishing boundary",
                        "Leaves a loss in $s_theta$ alone (+ const)",
                    ],
                ),
                (
                    "Denoising Score Matching",
                    [
                        "Add noise: $x_t = x_0 + sigma epsilon$, learn the *noisy* score",
                        "Key identity: $nabla_(x_t) log p(x_t | x_0) = -(x_t - x_0)\\/sigma^2$",
                        "Target is known in closed form $arrow.r$ no Jacobian trace",
                        "$J_\"DSM\" = EE\\|s_theta (x_t) + (x_t - x_0)\\/sigma^2\\|^2$",
                        "Score matching $=$ denoising — the Day 6 connection",
                    ],
                ),
                (
                    "Multiple Noise Scales (NCSN)",
                    [
                        "One noise level can't cover all of space",
                        "Low noise: accurate near data, empty regions unseen",
                        "High noise: fills space but blurs detail",
                        "Train *one* network conditioned on noise level $sigma$",
                        "Anneal from high to low noise when sampling",
                    ],
                ),
            ],
        ),
        (
            "Sampling with the Score",
            [
                (
                    "Langevin Dynamics",
                    [
                        "Walk uphill in log-density, plus noise:",
                        "$x_(k+1) = x_k + tau s_theta (x_k) + sqrt(2 tau) z_k$",
                        "Stationary distribution is exactly $p(x)$",
                        "Drift toward data + noise to explore",
                        "Anneal $sigma$ (NCSN) for fast, stable mixing",
                    ],
                ),
            ],
        ),
        (
            "The Continuous-Time View",
            [
                (
                    "The Forward SDE",
                    [
                        "Let steps $arrow.r$ 0: the noising chain becomes an SDE",
                        "$dif x = f(x,t) dif t + g(t) dif w$ (drift + diffusion)",
                        "VP-SDE corresponds to DDPM; VE-SDE to NCSN",
                        "Marginal stays $p_t (x_t|x_0) = N(alpha_t x_0, sigma_t^2 I)$",
                        "A whole *family* of distributions $p_t$, indexed by time",
                    ],
                ),
                (
                    "The Time-Dependent Score",
                    [
                        "Now the score depends on time: $s(x,t) = nabla_x log p_t (x)$",
                        "One network $s_theta (x, t)$ for all noise levels",
                        "Smoothed at high $t$ (easy), sharp at low $t$ (hard)",
                        "This is exactly the NCSN idea, in continuous time",
                        "Trained by denoising score matching at each $t$",
                    ],
                ),
                (
                    "Heuristic: Reversing an SDE",
                    [
                        "Chain rule: $p_(t,delta)(x,y) = p_(t+delta|t)(y|x) p_t(x) = p_(t|t+delta)(x|y) p_(t+delta)(y)$",
                        "Forward step (Euler): $p_(t+delta|t)(y|x) = N(y | x + f delta, delta g^2)$",
                        "Taylor on $log p_t$ around $y$: factor $exp((x-y)^T s(y,t))$ with $s = nabla log p_t$",
                        "Complete the square $arrow.r$ reverse drift $f - g^2 s$",
                        "Full step-by-step derivation: optional notes block",
                    ],
                ),
                (
                    "Reverse SDE & Probability-Flow ODE",
                    [
                        "Anderson (1982): rigorous reverse-time SDE with score term",
                        "$dif x = [f - g^2 s] dif t + g dif macron(w)$, $s = nabla log p_t approx s_theta$",
                        "Probability-flow ODE: *same marginals*, no noise",
                        "$dif x = [f - 1/2 g^2 s] dif t$",
                        "SDE = stochastic sampler; ODE = deterministic + likelihood",
                        "PF-ODE from FPE: Day 8 (optional derivation block)",
                    ],
                ),
                (
                    "Derivation: The DSM Objective",
                    [
                        "Want $s_theta (x,t) approx nabla log p_t (x)$ (marginal)",
                        "Marginal score = $EE[ nabla log p_t (x_t | x_0) | x_t]$",
                        "So regress on the *conditional* score (known Gaussian)",
                        "$nabla log p_t (x_t|x_0) = -(x_t - alpha_t x_0)\\/sigma_t^2 = -epsilon\\/sigma_t$",
                        "$arrow.r$ predicting the score $=$ predicting the noise",
                    ],
                ),
            ],
        ),
        (
            "Flow Matching",
            [
                (
                    "Continuous Normalizing Flows",
                    [
                        "Transport noise to data along an ODE: $dif x = v_theta (x,t) dif t$",
                        "$v$ = a *velocity field* moving samples in time",
                        "Density evolves by the continuity equation",
                        "Old way (CNF): expensive max-likelihood training",
                        "Flow matching: regress $v$ directly, simulation-free",
                    ],
                ),
                (
                    "Conditional Flow Matching",
                    [
                        "Pick a simple per-sample path, e.g. $x_t = (1-t) x_0 + t x_1$",
                        "Its velocity is known: $u_t (x | x_1) = x_1 - x_0$",
                        "Regress $v_theta (x_t, t)$ onto this conditional velocity",
                        "$J_\"CFM\" = EE\\|v_theta (x_t, t) - u_t (x|x_1)\\|^2$",
                        "No simulation, no divergence term — just regression",
                    ],
                ),
                (
                    "Conditional vs Marginal Velocity",
                    [
                        "Many conditional paths overlap at a point $x_t$",
                        "Marginal velocity = average of conditionals through $x_t$",
                        "$v(x,t) = EE[u_t (x | x_1) | x_t = x]$",
                        "CFM and marginal FM share the *same* gradient",
                        "So regressing conditionals learns the marginal field",
                    ],
                ),
                (
                    "Rectified Flow & Reflow",
                    [
                        "Linear interpolation $arrow.r$ straight conditional paths",
                        "But *marginal* trajectories can still be curved",
                        "Curved paths need many ODE steps to integrate",
                        "Reflow: retrain on (noise, sample) pairs $arrow.r$ straighter",
                        "Straighter flow $arrow.r$ fewer steps, even one-step",
                    ],
                ),
            ],
        ),
        (
            "One Model, Many Views",
            [
                (
                    "Four Equivalent Parameterizations",
                    [
                        "Predict noise $epsilon$, data $x_0$, score $s$, or velocity $v$",
                        "All related by the forward rule $x_t = alpha_t x_0 + sigma_t epsilon$",
                        "$s = -epsilon \\/ sigma_t$ (score $=$ scaled noise)",
                        "$v = alpha'_t x_0 + sigma'_t epsilon$ (flow-matching velocity)",
                        "Same network, different target — pick for stability",
                    ],
                ),
                (
                    "The Unified Picture",
                    [
                        "Variational (DDPM), score-SDE, and flow matching coincide",
                        "All learn the same time-indexed field over $p_t$",
                        "Sample with: ancestral, Langevin, reverse SDE, or ODE",
                        "Score $arrow.l.r$ noise $arrow.l.r$ velocity are interchangeable",
                        "Day 8: guidance, fast solvers, and few-step sampling",
                    ],
                ),
            ],
        ),
    ],
)

LECTURE = {
    "day": 7,
    "slug": "score-sde-flow-matching",
    "title": "Score-Based Models, SDEs, and Flow Matching",
    "description": "The continuous-time view of diffusion: score functions, the score SDE and probability-flow ODE, and flow matching.",
    "reading": [
        "[The Principles of Diffusion Models](https://arxiv.org/abs/2510.21890), Ch. 3–6; Appendices B–D (optional deep dives in notes)",
        "[Song et al. — Score-Based Generative Modeling through SDEs (2021)](https://arxiv.org/abs/2011.13456)",
        "[Lipman et al. — Flow Matching for Generative Modeling (2023)](https://arxiv.org/abs/2210.02747)",
        "[Interactive companion — The Principles of Diffusion Models](https://the-principles-of-diffusion-models.github.io/)",
        "[SDE course — Lesson 2: FPE, time reversal (Nelson/Anderson), DSM proof, PF-ODE](https://kierandidi.github.io/) (Generative Modelling with SDEs)",
        "[Ludwig Winkler — Simple sketch of the reverse SDE](https://ludwigwinkler.github.io/blog/SimpleReverseSDE/)",
    ],
    "intro": (
        "Day 6 built diffusion from the variational, discrete-time DDPM viewpoint. Today we take the "
        "continuous-time view that unifies modern generative modeling. The central object is the "
        "**score** — the gradient of the log-density — which we can learn by denoising and use to sample. "
        "Letting the number of noising steps go to infinity turns the forward chain into a stochastic "
        "differential equation (SDE), whose time-reversal and deterministic *probability-flow ODE* give "
        "two ways to generate data. Flow matching arrives at the same place from a different door, "
        "regressing a velocity field directly. We finish by showing that DDPM, score-SDE, and flow "
        "matching are three views of one model."
    ),
    "sections": [
        {
            "title": "The Score Function",
            "subsections": [
                {
                    "heading": "The score as a vector field",
                    "definition": (
                        "The **(Stein) score** of a density is the gradient of its log: "
                        "$$\\boldsymbol{s}(\\boldsymbol{x}) = \\nabla_{\\boldsymbol{x}}\\log p(\\boldsymbol{x}).$$ "
                        "It is a vector field that, at every point, points in the direction of steepest "
                        "increase of probability."
                    ),
                    "body": """**Why this matters.** On Day 6 we trained a network to *denoise*. Today we reveal what that network is really learning — the score — and why the score is exactly the quantity you need to turn noise into data. Note this is **not** the "score" of maximum-likelihood statistics (the gradient w.r.t. *parameters*); here the gradient is w.r.t. the *input* $$\\boldsymbol{x}$$.

![The score field of a distribution: arrows point toward regions of higher probability density (Principles Fig 3.2).](/assets/figures/day07/pdm_score_field.png)

Geometrically, the score is a map $$\\boldsymbol{x}\\mapsto\\nabla\\log p(\\boldsymbol{x})$$ that, like a gravitational field, pulls points toward the modes of the distribution and is zero at stationary points. If we know this field, we can "flow" or "diffuse" random points into samples from $$p$$ — no normalized density required. Explore how the field relates to the density:

{{viz:score_landscape}}

The local arrows tell you which way to nudge a sample to make it more probable; the global structure tells you where the mass is.

{{viz:score_global_vs_local}}""",
                },
                {
                    "heading": "Energy-based models and the normalizer problem",
                    "definition": (
                        "An **energy-based model (EBM)** writes "
                        "$$p_\\theta(\\boldsymbol{x}) = \\frac{e^{-E_\\theta(\\boldsymbol{x})}}{Z_\\theta},"
                        "\\qquad Z_\\theta = \\int e^{-E_\\theta(\\boldsymbol{x})}\\,\\mathrm{d}\\boldsymbol{x},$$ "
                        "where the **partition function** $$Z_\\theta$$ is generally intractable."
                    ),
                    "body": """![Training an energy-based model shapes the energy landscape so that data sits in low-energy basins (Principles Fig 3.1).](/assets/figures/day07/pdm_ebm_training.png)

EBMs are maximally flexible: any nonnegative function defines a distribution after normalization. The catch is precisely that normalization. Maximum-likelihood training needs $$\\nabla_\\theta\\log p_\\theta = -\\nabla_\\theta E_\\theta - \\nabla_\\theta\\log Z_\\theta$$, and the second term involves an expectation under the model that requires expensive MCMC to estimate.

The escape route motivates the whole day: instead of modeling the normalized density, model its **score**. As we show next, the score is blind to $$Z_\\theta$$.""",
                },
                {
                    "heading": "Why the score sidesteps the normalizer",
                    "definition": (
                        "Because the partition function is constant in $$\\boldsymbol{x}$$, it disappears "
                        "under the gradient: "
                        "$$\\nabla_{\\boldsymbol{x}}\\log p_\\theta(\\boldsymbol{x}) = "
                        "-\\nabla_{\\boldsymbol{x}} E_\\theta(\\boldsymbol{x}).$$"
                    ),
                    "body": """Take logs of the EBM and differentiate with respect to the input:

$$\\log p_\\theta(\\boldsymbol{x}) = -E_\\theta(\\boldsymbol{x}) - \\log Z_\\theta \\;\\;\\Longrightarrow\\;\\; \\nabla_{\\boldsymbol{x}}\\log p_\\theta(\\boldsymbol{x}) = -\\nabla_{\\boldsymbol{x}} E_\\theta(\\boldsymbol{x}) - \\underbrace{\\nabla_{\\boldsymbol{x}}\\log Z_\\theta}_{=\\,\\mathbf{0}}.$$

The term $$\\log Z_\\theta$$ is a number — it does not depend on $$\\boldsymbol{x}$$ — so its gradient is exactly zero. The intractable obstacle simply **vanishes**. This is the key insight of score-based modeling: by working with $$\\nabla_{\\boldsymbol{x}}\\log p$$ we learn the *shape* of the distribution (where mass concentrates and how it falls off) without ever computing how much total mass there is. The remaining question is how to fit a model to a score we cannot observe directly.""",
                },
            ],
        },
        {
            "title": "Learning the Score",
            "subsections": [
                {
                    "heading": "Score matching",
                    "definition": (
                        "**Score matching** fits $$\\boldsymbol{s}_\\theta(\\boldsymbol{x})\\approx"
                        "\\nabla\\log p_{\\text{data}}(\\boldsymbol{x})$$ by minimizing the expected squared "
                        "difference between the model and the true score."
                    ),
                    "body": """![Score matching learns a vector field that agrees with the data score across space (Principles Fig 3.4).](/assets/figures/day07/pdm_score_matching.png)

The obvious objective,

$$J_{\\text{ESM}}(\\theta) = \\tfrac12\\,\\mathbb{E}_{p_{\\text{data}}}\\big\\|\\boldsymbol{s}_\\theta(\\boldsymbol{x}) - \\nabla\\log p_{\\text{data}}(\\boldsymbol{x})\\big\\|^2,$$

is useless as written because it contains the unknown true score $$\\nabla\\log p_{\\text{data}}$$. Hyvärinen's remarkable result is that this objective can be rewritten — via integration by parts — into an equivalent one that depends only on $$\\boldsymbol{s}_\\theta$$ and its derivatives. We derive it next.""",
                },
                {
                    "heading": "Derivation: implicit score matching",
                    "definition": (
                        "Integration by parts converts the intractable objective into "
                        "$$J(\\theta) = \\mathbb{E}_{p_{\\text{data}}}\\!\\left[\\tfrac12\\|\\boldsymbol{s}_\\theta(\\boldsymbol{x})\\|^2 "
                        "+ \\operatorname{tr}\\!\\big(\\nabla_{\\boldsymbol{x}}\\boldsymbol{s}_\\theta(\\boldsymbol{x})\\big)\\right] + \\text{const}.$$"
                    ),
                    "body": """Expand the square in the explicit objective:

$$J_{\\text{ESM}} = \\mathbb{E}_{p}\\Big[\\tfrac12\\|\\boldsymbol{s}_\\theta\\|^2 - \\textcolor{teal}{\\boldsymbol{s}_\\theta^{\\top}\\nabla\\log p} + \\tfrac12\\|\\nabla\\log p\\|^2\\Big].$$

The last term does not involve $$\\theta$$ (a constant). The middle (cross) term is the problem; rewrite it using $$p\\,\\nabla\\log p = \\nabla p$$:

$$\\textcolor{teal}{\\mathbb{E}_{p}\\big[\\boldsymbol{s}_\\theta^{\\top}\\nabla\\log p\\big]} = \\int p(\\boldsymbol{x})\\,\\boldsymbol{s}_\\theta(\\boldsymbol{x})^{\\top}\\frac{\\nabla p(\\boldsymbol{x})}{p(\\boldsymbol{x})}\\,\\mathrm{d}\\boldsymbol{x} = \\int \\boldsymbol{s}_\\theta(\\boldsymbol{x})^{\\top}\\nabla p(\\boldsymbol{x})\\,\\mathrm{d}\\boldsymbol{x}.$$

Now integrate by parts (component-wise), assuming $$p(\\boldsymbol{x})\\,\\boldsymbol{s}_\\theta(\\boldsymbol{x})\\to\\mathbf{0}$$ as $$\\|\\boldsymbol{x}\\|\\to\\infty$$ so the boundary term drops:

$$\\int \\boldsymbol{s}_\\theta^{\\top}\\nabla p\\,\\mathrm{d}\\boldsymbol{x} = -\\int p\\,(\\nabla\\cdot\\boldsymbol{s}_\\theta)\\,\\mathrm{d}\\boldsymbol{x} = -\\,\\mathbb{E}_{p}\\big[\\operatorname{tr}(\\nabla_{\\boldsymbol{x}}\\boldsymbol{s}_\\theta)\\big].$$

Substituting back gives the **implicit score matching** objective

$$J(\\theta) = \\mathbb{E}_{p_{\\text{data}}}\\!\\left[\\tfrac12\\|\\boldsymbol{s}_\\theta(\\boldsymbol{x})\\|^2 + \\textcolor{purple}{\\operatorname{tr}\\!\\big(\\nabla_{\\boldsymbol{x}}\\boldsymbol{s}_\\theta(\\boldsymbol{x})\\big)}\\right] + \\text{const},$$

which no longer references the true score. The price is the **trace of the Jacobian** $$\\operatorname{tr}(\\nabla\\boldsymbol{s}_\\theta)$$ — a sum of $$d$$ second derivatives that costs $$d$$ backward passes in $$d$$ dimensions. For images ($$d\\sim10^5$$) this is prohibitive, which is why we turn to denoising.""",
                },
                {
                    "heading": "Denoising score matching",
                    "definition": (
                        "**Denoising score matching (DSM)** perturbs the data with Gaussian noise and "
                        "fits the score of the *noisy* density. The target is then available in closed form: "
                        "for $$\\boldsymbol{x}_t = \\boldsymbol{x}_0 + \\sigma\\boldsymbol{\\epsilon}$$, "
                        "$$\\nabla_{\\boldsymbol{x}_t}\\log p(\\boldsymbol{x}_t\\mid\\boldsymbol{x}_0) = "
                        "-\\frac{\\boldsymbol{x}_t-\\boldsymbol{x}_0}{\\sigma^2} = -\\frac{\\boldsymbol{\\epsilon}}{\\sigma}.$$"
                    ),
                    "body": """![Denoising score matching: the score of the noised distribution points from the noisy sample back toward the clean data (Principles Fig 4.6).](/assets/figures/day07/pdm_dsm_trick.png)

The perturbation kernel is Gaussian, $$p(\\boldsymbol{x}_t\\mid\\boldsymbol{x}_0)=\\mathcal{N}(\\boldsymbol{x}_0,\\sigma^2 I)$$, whose log-gradient we can write down exactly (differentiate $$-\\|\\boldsymbol{x}_t-\\boldsymbol{x}_0\\|^2/2\\sigma^2$$). Vincent's identity shows that matching the *conditional* score also matches the *marginal* noisy score, so the trainable objective is just a regression:

$$J_{\\text{DSM}}(\\theta) = \\mathbb{E}_{\\boldsymbol{x}_0,\\boldsymbol{\\epsilon}}\\left\\|\\boldsymbol{s}_\\theta(\\boldsymbol{x}_t) + \\frac{\\boldsymbol{x}_t-\\boldsymbol{x}_0}{\\sigma^2}\\right\\|^2 = \\mathbb{E}\\left\\|\\boldsymbol{s}_\\theta(\\boldsymbol{x}_t) + \\frac{\\boldsymbol{\\epsilon}}{\\sigma}\\right\\|^2.$$

No Jacobian trace, no MCMC — just predict the (scaled) noise that was added. This is **exactly** the DDPM noise-prediction loss of Day 6 in disguise: learning the score *is* learning to denoise, with $$\\boldsymbol{s}_\\theta = -\\boldsymbol{\\epsilon}_\\theta/\\sigma$$. Build intuition for the equivalence:

{{viz:denoising_score_matching}}""",
                },
                {
                    "heading": "Multiple noise scales (NCSN)",
                    "definition": (
                        "A single noise level cannot teach the score everywhere. **Noise-Conditional "
                        "Score Networks (NCSN)** train one network "
                        "$$\\boldsymbol{s}_\\theta(\\boldsymbol{x},\\sigma)$$ across a range of noise "
                        "scales $$\\sigma$$."
                    ),
                    "body": """![Training across many noise scales covers both the data manifold and the empty regions far from it (Principles Fig 3.7).](/assets/figures/day07/pdm_ncsn.png)

The difficulty with DSM at a single $$\\sigma$$ is a coverage problem:

- **Small $$\\sigma$$**: the noisy samples hug the data manifold, so the score is accurate *near* the data but never trained in the vast empty regions where sampling actually starts.
- **Large $$\\sigma$$**: noise fills the whole space (good coverage) but smears out the fine structure of $$p_{\\text{data}}$$.

The fix is to use **many** noise scales and a single network conditioned on $$\\sigma$$. At sampling time we **anneal** from large $$\\sigma$$ (which moves us into the right region from anywhere) down to small $$\\sigma$$ (which refines detail). Taking the number of scales to infinity is precisely the continuous-time SDE we build below.""",
                },
            ],
        },
        {
            "title": "Sampling with the Score",
            "subsections": [
                {
                    "heading": "Langevin dynamics",
                    "definition": (
                        "**Langevin dynamics** turns a score into samples by repeatedly stepping uphill "
                        "in log-density and injecting noise: "
                        "$$\\boldsymbol{x}_{k+1} = \\boldsymbol{x}_k + \\tau\\,\\boldsymbol{s}_\\theta(\\boldsymbol{x}_k) "
                        "+ \\sqrt{2\\tau}\\,\\boldsymbol{z}_k,\\quad \\boldsymbol{z}_k\\sim\\mathcal{N}(\\mathbf{0},I).$$"
                    ),
                    "body": """![Langevin sampling: noisy gradient ascent on the log-density converges to samples from $$p$$ (Principles Fig 3.3).](/assets/figures/day07/pdm_langevin.png)

The update has two parts: a **drift** $$\\tau\\,\\boldsymbol{s}_\\theta$$ that climbs toward high-probability regions, and a **diffusion** $$\\sqrt{2\\tau}\\,\\boldsymbol{z}_k$$ that keeps the chain exploring rather than collapsing onto a single mode. Under mild conditions, as $$\\tau\\to0$$ and the number of steps grows, the iterates converge in distribution to $$p$$ — the noise is exactly calibrated so that $$p$$ is the **stationary distribution**.

Pure Langevin mixes slowly between far-apart modes. Combining it with the NCSN annealing schedule — run Langevin at large $$\\sigma$$, then progressively smaller — gives **annealed Langevin dynamics**, the original score-based sampler. The continuous-time limit of this annealed process is the reverse SDE we meet next.""",
                },
            ],
        },
        {
            "title": "The Continuous-Time View",
            "subsections": [
                {
                    "heading": "The forward SDE",
                    "definition": (
                        "Taking the noising chain to infinitely many infinitesimal steps yields a "
                        "**stochastic differential equation** "
                        "$$\\mathrm{d}\\boldsymbol{x} = \\boldsymbol{f}(\\boldsymbol{x},t)\\,\\mathrm{d}t + "
                        "g(t)\\,\\mathrm{d}\\boldsymbol{w},$$ with **drift** $$\\boldsymbol{f}$$ and "
                        "**diffusion** $$g$$, driven by a Wiener process $$\\boldsymbol{w}$$."
                    ),
                    "body": """![The forward process as a continuous-time SDE: a 1-D density smoothly transported from data to noise (Principles Fig 4.3).](/assets/figures/day07/pdm_forward_1d.png)

The discrete DDPM update $$\\boldsymbol{x}_t=\\sqrt{1-\\beta_t}\\,\\boldsymbol{x}_{t-1}+\\sqrt{\\beta_t}\\,\\boldsymbol{\\epsilon}$$ is the Euler–Maruyama discretization (Day 1) of an SDE. Two standard families:

- **Variance-Preserving (VP) SDE** — the continuous limit of DDPM, with $$\\boldsymbol{f}=-\\tfrac12\\beta(t)\\boldsymbol{x}$$ and $$g=\\sqrt{\\beta(t)}$$; keeps total variance bounded.
- **Variance-Exploding (VE) SDE** — the continuous limit of NCSN, with $$\\boldsymbol{f}=\\mathbf{0}$$ and growing $$g$$; variance blows up as noise is added.

Crucially the marginal at each time stays Gaussian in the conditioning variable, $$p_t(\\boldsymbol{x}_t\\mid\\boldsymbol{x}_0)=\\mathcal{N}(\\alpha_t\\boldsymbol{x}_0,\\sigma_t^2 I)$$, so the unified $$\\boldsymbol{x}_t=\\alpha_t\\boldsymbol{x}_0+\\sigma_t\\boldsymbol{\\epsilon}$$ notation from Day 6 carries over unchanged — the SDE just describes how $$\\alpha_t,\\sigma_t$$ evolve.""",
                },
                {
                    "heading": "The time-dependent score",
                    "definition": (
                        "Each forward time defines a distribution $$p_t$$, so the score becomes "
                        "**time-dependent**: $$\\boldsymbol{s}(\\boldsymbol{x},t) = "
                        "\\nabla_{\\boldsymbol{x}}\\log p_t(\\boldsymbol{x}).$$ A single network "
                        "$$\\boldsymbol{s}_\\theta(\\boldsymbol{x},t)$$ learns it for all $$t$$."
                    ),
                    "body": """![The score landscape changes with time: smooth and easy at high noise, sharp and detailed at low noise (Principles Fig 4.1).](/assets/figures/day07/pdm_score_landscape.png)

This is the NCSN idea recast in continuous time: instead of a discrete set of noise levels, there is a continuum $$t\\in[0,T]$$, and the network is conditioned on $$t$$. At large $$t$$ the density $$p_t$$ is a broad, smooth Gaussian whose score is easy to learn; at small $$t$$ it concentrates on the data manifold and the score becomes sharp and hard. Training is denoising score matching at every time:

$$J(\\theta) = \\mathbb{E}_{t}\\,\\lambda(t)\\,\\mathbb{E}_{\\boldsymbol{x}_0,\\boldsymbol{\\epsilon}}\\left\\|\\boldsymbol{s}_\\theta(\\boldsymbol{x}_t,t) + \\frac{\\boldsymbol{\\epsilon}}{\\sigma_t}\\right\\|^2,$$

with a time-weighting $$\\lambda(t)$$. One network, trained once, gives the score at every noise level — everything we need to run the dynamics below.""",
                },
                {
                    "heading": "Heuristic: reversing an SDE",
                    "definition": (
                        "To reverse the forward SDE, factor the joint density with the chain rule "
                        "and approximate forward transitions by Euler–Maruyama; a Taylor expansion "
                        "of $$\\log p_t$$ reveals the **score** $$\\boldsymbol{s}=\\nabla\\log p_t$$ "
                        "in the reverse drift $$\\boldsymbol{f}-g^2\\boldsymbol{s}$$."
                    ),
                    "body": """The forward SDE (Principles notation)

$$\\mathrm{d}\\boldsymbol{x} = \\boldsymbol{f}(\\boldsymbol{x},t)\\,\\mathrm{d}t + g(t)\\,\\mathrm{d}\\boldsymbol{w}$$

has Gaussian Euler–Maruyama transitions $$p_{t+\\delta\\mid t}(\\boldsymbol{y}\\mid\\boldsymbol{x})=\\mathcal{N}\\big(\\boldsymbol{y};\\,\\boldsymbol{x}+\\boldsymbol{f}\\delta,\\,g^2\\delta\\, I\\big)$$ over a small step $$\\delta$$. We want the **reverse** kernel $$p_{t\\mid t+\\delta}(\\boldsymbol{x}\\mid\\boldsymbol{y})$$. Bayes' rule gives it as a ratio, and we collect everything into one exponent (writing $$\\overset{c}{=}$$ for "equal up to an additive constant in $$\\boldsymbol{x}$$" and $$\\boldsymbol{u}:=\\boldsymbol{x}-\\boldsymbol{y}=O(\\sqrt\\delta)$$):

$$\\begin{aligned}
\\log p_{t\\mid t+\\delta}(\\boldsymbol{x}\\mid\\boldsymbol{y})
&\\;\\overset{c}{=}\\; \\underbrace{-\\frac{\\lVert \\boldsymbol{y}-\\boldsymbol{x}-\\boldsymbol{f}\\delta\\rVert^2}{2g^2\\delta}}_{\\textcolor{blue}{\\text{forward EM kernel}}} \\;+\\; \\underbrace{\\log p_t(\\boldsymbol{x}) - \\log p_{t+\\delta}(\\boldsymbol{y})}_{\\textcolor{teal}{\\text{Bayes ratio}}} \\\\
&\\;\\overset{c}{=}\\; -\\frac{\\lVert \\boldsymbol{u}+\\boldsymbol{f}\\delta\\rVert^2}{2g^2\\delta} \\;+\\; \\boldsymbol{u}^{\\top}\\textcolor{purple}{\\boldsymbol{s}} \\;+\\; O(\\delta) & &\\textcolor{teal}{\\big(\\text{Taylor: }\\log p_t(\\boldsymbol{x})-\\log p_{t+\\delta}(\\boldsymbol{y})=\\boldsymbol{u}^{\\top}\\boldsymbol{s}+O(\\delta)\\big)} \\\\
&\\;\\overset{c}{=}\\; -\\frac{\\lVert \\boldsymbol{u}\\rVert^2}{2g^2\\delta} + \\boldsymbol{u}^{\\top}\\Big(\\textcolor{purple}{\\boldsymbol{s}}-\\frac{\\boldsymbol{f}}{g^2}\\Big) & &\\text{(expand the square, drop }O(\\delta)\\text{)} \\\\
&\\;\\overset{c}{=}\\; -\\frac{1}{2g^2\\delta}\\big\\lVert \\boldsymbol{u} - \\big(\\textcolor{purple}{g^2\\boldsymbol{s}}-\\textcolor{blue}{\\boldsymbol{f}}\\big)\\delta\\big\\rVert^2 & &\\text{(complete the square in }\\boldsymbol{u}\\text{)}.
\\end{aligned}$$

The reverse kernel is therefore again Gaussian, $$p_{t\\mid t+\\delta}(\\boldsymbol{x}\\mid\\boldsymbol{y})=\\mathcal{N}\\big(\\boldsymbol{x};\\,\\boldsymbol{y}-[\\boldsymbol{f}-g^2\\boldsymbol{s}]\\delta,\\;g^2\\delta\\,I\\big)$$, i.e. one **backward** Euler–Maruyama step

$$\\boldsymbol{x} \\;=\\; \\boldsymbol{y} - \\big[\\,\\textcolor{blue}{\\boldsymbol{f}(\\boldsymbol{y},t)} - \\textcolor{purple}{g^2\\,\\boldsymbol{s}(\\boldsymbol{y},t)}\\,\\big]\\delta \\;+\\; g\\sqrt{\\delta}\\,\\boldsymbol{z},\\qquad \\boldsymbol{z}\\sim\\mathcal{N}(\\mathbf{0},I).$$

Reading off the drift, the reverse process moves with $$\\boldsymbol{f}-g^2\\boldsymbol{s}$$ — the **score** appears exactly where DSM trains it. Letting $$\\delta\\to0$$ turns this into the reverse-time SDE, and at sampling time we replace $$\\boldsymbol{s}$$ with the learned $$\\boldsymbol{s}_\\theta(\\boldsymbol{x},t)$$. The fully rigorous version (Anderson's theorem) is next; an even longer step-by-step expansion (Winkler; SDE course §2.1) is in the optional block below.""",
                    "optional": [DAY07_OPTIONAL[0]],
                },
                {
                    "heading": "Anderson's reverse-time SDE",
                    "definition": (
                        "Anderson (1982): the time-reversal of "
                        "$$\\mathrm{d}\\boldsymbol{x} = \\boldsymbol{f}\\,\\mathrm{d}t + g\\,\\mathrm{d}\\boldsymbol{w}$$ "
                        "is $$\\mathrm{d}\\boldsymbol{x} = [\\boldsymbol{f}-g^2\\boldsymbol{s}]\\,\\mathrm{d}t + g\\,\\mathrm{d}\\bar{\\boldsymbol{w}}$$ "
                        "with $$\\boldsymbol{s}=\\nabla\\log p_t$$ — the rigorous form of the heuristic above."
                    ),
                    "body": """The heuristic above is made rigorous by **matching Fokker–Planck equations (FPE)** — Anderson's (1982) argument. Recall the forward marginals $$p_t$$ obey the FPE $$\\partial_t p_t = -\\nabla\\cdot(\\boldsymbol{f}\\,p_t) + \\tfrac12 g^2\\,\\Delta p_t$$. Run time **backward** with $$\\tau:=T-t$$ and let $$q_\\tau:=p_{T-\\tau}$$ be the reverse marginals. We look for a forward-in-$$\\tau$$ SDE $$\\mathrm{d}\\boldsymbol{x}=\\boldsymbol{b}\\,\\mathrm{d}\\tau+g\\,\\mathrm{d}\\boldsymbol{w}_\\tau$$ whose own FPE, $$\\partial_\\tau q_\\tau=-\\nabla\\cdot(\\boldsymbol{b}\\,q_\\tau)+\\tfrac12 g^2\\Delta q_\\tau$$, reproduces $$q_\\tau$$. Differentiate $$q_\\tau$$ and substitute the forward FPE (chain rule $$\\partial_\\tau q_\\tau=-\\partial_t p_t$$):

$$\\begin{aligned}
\\partial_\\tau q_\\tau
&= -\\,\\partial_t p_t\\big|_{t=T-\\tau}
= \\nabla\\cdot\\big(\\textcolor{teal}{\\boldsymbol{f}}\\,q_\\tau\\big) - \\tfrac12 g^2\\,\\Delta q_\\tau & &\\text{(forward FPE, flip sign of }\\partial_t\\text{)} \\\\
&= \\nabla\\cdot\\big(\\boldsymbol{f}\\,q_\\tau\\big) - \\underbrace{g^2\\,\\Delta q_\\tau}_{=\\,g^2\\nabla\\cdot(q_\\tau\\textcolor{purple}{\\boldsymbol{s}})} + \\tfrac12 g^2\\,\\Delta q_\\tau & &\\big(\\text{split }-\\tfrac12 = -1+\\tfrac12\\big) \\\\
&= \\nabla\\cdot\\Big(\\big[\\textcolor{teal}{\\boldsymbol{f}} - g^2\\textcolor{purple}{\\boldsymbol{s}}\\big]q_\\tau\\Big) + \\tfrac12 g^2\\,\\Delta q_\\tau & &\\big(\\nabla q_\\tau = q_\\tau\\,\\textcolor{purple}{\\boldsymbol{s}},\\ \\textcolor{purple}{\\boldsymbol{s}}=\\nabla\\log q_\\tau\\big).
\\end{aligned}$$

Comparing with the target FPE $$\\partial_\\tau q_\\tau=-\\nabla\\cdot(\\boldsymbol{b}\\,q_\\tau)+\\tfrac12 g^2\\Delta q_\\tau$$ identifies the reverse drift $$\\boldsymbol{b}=\\textcolor{purple}{g^2\\boldsymbol{s}}-\\textcolor{teal}{\\boldsymbol{f}}$$. Undoing $$\\tau=T-t$$ (so $$\\mathrm{d}\\tau=-\\mathrm{d}t$$) writes this **same** process in the original time as

$$\\boxed{\\;\\mathrm{d}\\boldsymbol{x} = \\big[\\,\\textcolor{teal}{\\boldsymbol{f}(\\boldsymbol{x},t)} - g(t)^2\\,\\textcolor{purple}{\\boldsymbol{s}(\\boldsymbol{x},t)}\\,\\big]\\,\\mathrm{d}t + g(t)\\,\\mathrm{d}\\bar{\\boldsymbol{w}}\\;}$$

— exactly the reverse-time SDE, with $$\\bar{\\boldsymbol{w}}$$ a reverse Wiener process. The crucial step is purely algebraic: converting **one of the two halves** of the diffusion term into a drift via $$\\nabla q=q\\,\\boldsymbol{s}$$, which is where the score enters. Running this SDE from $$\\boldsymbol{x}_T\\sim\\mathcal{N}(\\mathbf{0},I)$$ down to $$t=0$$ (with $$\\boldsymbol{s}\\to\\boldsymbol{s}_\\theta$$) generates samples from $$p_{\\text{data}}$$. Two conventions appear in the literature (reverse Wiener vs. $$t\\mapsto T-t$$); they agree on the score term $$g^2\\boldsymbol{s}$$. The reverse SDE generalizes annealed Langevin dynamics and DDPM ancestral sampling; the textbook proof via forward/backward Kolmogorov equations is in the optional block below.""",
                    "optional": [DAY07_OPTIONAL[1]],
                },
                {
                    "heading": "The probability-flow ODE (same marginals, no noise)",
                    "definition": (
                        "Every SDE admits a deterministic **probability-flow ODE** with the *same* "
                        "marginals $$\\{p_t\\}$$: "
                        "$$\\mathrm{d}\\boldsymbol{x} = \\big[\\boldsymbol{f}-\\tfrac12 g^2\\boldsymbol{s}\\big]\\mathrm{d}t.$$ "
                        "The $$\\tfrac12$$ factor and absence of noise follow from the FPE (Day 8)."
                    ),
                    "body": """![Three dynamics with identical time marginals: the forward SDE, the reverse SDE, and the probability-flow ODE (Principles Fig 4.5).](/assets/figures/day07/pdm_three_dynamics.png)

**Where does the $$\\tfrac12$$ come from?** Take the *same* forward Fokker–Planck equation, but this time convert the **entire** diffusion term into a transport (drift) term using $$\\nabla p_t=p_t\\boldsymbol{s}$$:

$$\\begin{aligned}
\\partial_t p_t
&= -\\nabla\\cdot\\big(\\textcolor{teal}{\\boldsymbol{f}}\\,p_t\\big) + \\tfrac12 g^2\\,\\Delta p_t & &\\text{(forward FPE)} \\\\
&= -\\nabla\\cdot\\big(\\boldsymbol{f}\\,p_t\\big) + \\tfrac12 g^2\\,\\nabla\\cdot\\big(p_t\\,\\textcolor{purple}{\\boldsymbol{s}}\\big) & &\\big(\\Delta p_t=\\nabla\\cdot(p_t\\textcolor{purple}{\\boldsymbol{s}})\\big) \\\\
&= -\\nabla\\cdot\\Big(\\underbrace{\\big[\\textcolor{teal}{\\boldsymbol{f}} - \\tfrac12 g^2\\textcolor{purple}{\\boldsymbol{s}}\\big]}_{\\textcolor{orange}{\\tilde{\\boldsymbol{\\mu}}}}\\,p_t\\Big) & &\\text{(continuity equation, no }\\Delta\\text{)}.
\\end{aligned}$$

A density obeying $$\\partial_t p_t = -\\nabla\\cdot(\\textcolor{orange}{\\tilde{\\boldsymbol{\\mu}}}\\,p_t)$$ with **no** second-order term is transported deterministically by the **probability-flow ODE**

$$\\boxed{\\;\\dot{\\boldsymbol{x}} = \\textcolor{orange}{\\tilde{\\boldsymbol{\\mu}}(\\boldsymbol{x},t)} = \\textcolor{teal}{\\boldsymbol{f}(\\boldsymbol{x},t)} - \\tfrac12\\, g(t)^2\\,\\textcolor{purple}{\\boldsymbol{s}(\\boldsymbol{x},t)}.\\;}$$

Contrast the two reverse dynamics: Anderson moved the *whole* $$g^2\\boldsymbol{s}$$ into the drift and **kept** the noise; the PF-ODE moves only *half* of it and **drops** the noise. Both share the marginals $$\\{p_t\\}$$ — the missing $$\\tfrac12 g^2\\boldsymbol{s}$$ is exactly the deterministic drift that mimics, in expectation, the diffusion it replaced.

The reverse SDE is a **stochastic** sampler; the **probability-flow ODE** is deterministic (no $$\\mathrm{d}\\bar{\\boldsymbol{w}}$$). Practical uses: reproducible sampling, exact likelihoods (CNF), fast ODE solvers (Day 8).

Compare all three dynamics:

{{viz:score_sde_three_dynamics}}""",
                },
                {
                    "heading": "The denoising score matching objective",
                    "definition": (
                        "The marginal score equals the conditional expectation of the closed-form "
                        "conditional score: $$\\nabla\\log p_t(\\boldsymbol{x}_t) = "
                        "\\mathbb{E}[\\nabla\\log p_t(\\boldsymbol{x}_t\\mid\\boldsymbol{x}_0)\\,\\big|\\,\\boldsymbol{x}_t]$$. "
                        "Hence regressing on $$-\\boldsymbol{\\epsilon}/\\sigma_t$$ trains the marginal score."
                    ),
                    "body": """The conditional density $$p_t(\\boldsymbol{x}_t\\mid\\boldsymbol{x}_0)=\\mathcal{N}(\\alpha_t\\boldsymbol{x}_0,\\sigma_t^2 I)$$ is Gaussian, so with $$\\boldsymbol{x}_t=\\alpha_t\\boldsymbol{x}_0+\\sigma_t\\boldsymbol{\\epsilon}$$ its score is available in closed form:

$$\\nabla_{\\boldsymbol{x}_t}\\log p_t(\\boldsymbol{x}_t\\mid\\boldsymbol{x}_0) = -\\frac{\\boldsymbol{x}_t-\\alpha_t\\boldsymbol{x}_0}{\\sigma_t^2} = \\textcolor{teal}{-\\frac{\\boldsymbol{\\epsilon}}{\\sigma_t}}.$$

Why does regressing on this *conditional* target recover the *marginal* score? For any squared-error problem the minimizer is the **conditional mean** of the target. Minimizing pointwise at fixed $$\\boldsymbol{x}_t$$,

$$\\begin{aligned}
\\boldsymbol{s}_\\theta^\\star(\\boldsymbol{x}_t,t)
&= \\arg\\min_{\\boldsymbol{s}}\\;\\mathbb{E}_{\\boldsymbol{x}_0\\mid\\boldsymbol{x}_t}\\big\\lVert \\boldsymbol{s} - \\textcolor{teal}{\\nabla\\log p_t(\\boldsymbol{x}_t\\mid\\boldsymbol{x}_0)}\\big\\rVert^2 \\\\
&= \\mathbb{E}\\big[\\,\\textcolor{teal}{\\nabla\\log p_t(\\boldsymbol{x}_t\\mid\\boldsymbol{x}_0)}\\,\\big|\\,\\boldsymbol{x}_t\\big] & &\\text{(minimizer = conditional mean)} \\\\
&= \\frac{1}{p_t(\\boldsymbol{x}_t)}\\int \\nabla_{\\boldsymbol{x}_t}p_t(\\boldsymbol{x}_t\\mid\\boldsymbol{x}_0)\\,p(\\boldsymbol{x}_0)\\,\\mathrm{d}\\boldsymbol{x}_0 & &\\big(p_t(\\boldsymbol{x}_t\\mid\\boldsymbol{x}_0)\\,\\nabla\\log(\\cdot)=\\nabla p_t(\\cdot\\mid\\boldsymbol{x}_0)\\big)\\\\
&= \\frac{\\nabla_{\\boldsymbol{x}_t} p_t(\\boldsymbol{x}_t)}{p_t(\\boldsymbol{x}_t)} = \\textcolor{purple}{\\nabla\\log p_t(\\boldsymbol{x}_t)} & &\\big(\\text{swap }\\nabla,\\textstyle\\int;\\ p_t(\\boldsymbol{x}_t)=\\!\\int p_t(\\boldsymbol{x}_t\\mid\\boldsymbol{x}_0)p(\\boldsymbol{x}_0)\\big).
\\end{aligned}$$

So the network trained on the cheap conditional target converges to the true marginal score $$\\textcolor{purple}{\\nabla\\log p_t}$$ — and since the target is $$-\\boldsymbol{\\epsilon}/\\sigma_t$$, **predicting the score is predicting the noise** ($$\\boldsymbol{s}_\\theta=-\\boldsymbol{\\epsilon}_\\theta/\\sigma_t$$, Day 6). The same identity underlies **Tweedie's formula**; full details in the optional block below.""",
                    "optional": [DAY07_OPTIONAL[2]] + DAY07_OPTIONAL[3:],
                },
            ],
        },
        {
            "title": "Flow Matching",
            "subsections": [
                {
                    "heading": "Continuous normalizing flows and velocity fields",
                    "definition": (
                        "A **continuous normalizing flow (CNF)** transports samples by an ODE "
                        "$$\\mathrm{d}\\boldsymbol{x} = \\boldsymbol{v}_\\theta(\\boldsymbol{x},t)\\,\\mathrm{d}t,$$ "
                        "whose **velocity field** $$\\boldsymbol{v}$$ moves a base distribution into the data "
                        "distribution as $$t$$ goes from $$0$$ to $$1$$."
                    ),
                    "body": """![A normalizing flow transports a simple base density to the data density along a learned path (Principles Fig 5.2).](/assets/figures/day07/pdm_nf.png)

If samples move with velocity $$\\boldsymbol{v}$$, their density evolves by the **continuity equation** $$\\partial_t p_t + \\nabla\\cdot(p_t\\,\\boldsymbol{v}_t)=0$$ (conservation of probability mass). Classical CNFs trained this velocity by maximizing likelihood, which requires integrating the ODE — and its divergence — during training. That is slow and unstable.

**Flow matching** asks a simpler question: can we *regress* the velocity field directly, without simulating the ODE? The answer is yes, provided we choose the paths cleverly.""",
                },
                {
                    "heading": "Conditional flow matching",
                    "definition": (
                        "**Conditional flow matching (CFM)** specifies a simple per-example path from noise "
                        "to a data point and regresses the network onto its known velocity. For the linear "
                        "path $$\\boldsymbol{x}_t=(1-t)\\boldsymbol{x}_0+t\\,\\boldsymbol{x}_1$$, the velocity "
                        "is the constant $$\\boldsymbol{u}_t=\\boldsymbol{x}_1-\\boldsymbol{x}_0.$$"
                    ),
                    "body": """![A conditional transition path interpolates a single noise sample to a single data sample; its velocity is known in closed form (Principles Fig 5.5).](/assets/figures/day07/pdm_cond_transition.png)

Pick a base sample $$\\boldsymbol{x}_0\\sim\\mathcal{N}(\\mathbf{0},I)$$ and a data sample $$\\boldsymbol{x}_1\\sim p_{\\text{data}}$$, and connect them with a simple path — for the straight line, $$\\boldsymbol{x}_t=(1-t)\\boldsymbol{x}_0+t\\boldsymbol{x}_1$$, the velocity is simply $$\\dot{\\boldsymbol{x}}_t=\\boldsymbol{x}_1-\\boldsymbol{x}_0$$. The objective is a plain regression:

$$J_{\\text{CFM}}(\\theta) = \\mathbb{E}_{t,\\boldsymbol{x}_0,\\boldsymbol{x}_1}\\big\\|\\boldsymbol{v}_\\theta(\\boldsymbol{x}_t,t) - (\\boldsymbol{x}_1-\\boldsymbol{x}_0)\\big\\|^2.$$

No ODE simulation during training, no divergence term — just sample a pair, sample a time, and regress. This is the appeal of flow matching: the stability of a supervised regression with the flexibility of a CNF.""",
                },
                {
                    "heading": "Conditional versus marginal velocity",
                    "definition": (
                        "Although we regress on per-sample (conditional) velocities, the network learns the "
                        "**marginal** velocity, which is their conditional average through each point: "
                        "$$\\boldsymbol{v}(\\boldsymbol{x},t) = "
                        "\\mathbb{E}\\big[\\boldsymbol{u}_t(\\boldsymbol{x}\\mid\\boldsymbol{x}_1)\\,\\big|\\,\\boldsymbol{x}_t=\\boldsymbol{x}\\big].$$"
                    ),
                    "body": """![Many conditional paths cross any given point; the marginal velocity is their average, and that is what the network learns (Principles Fig 5.6).](/assets/figures/day07/pdm_cond_vs_marginal.png)

A subtlety: many different $$(\\boldsymbol{x}_0,\\boldsymbol{x}_1)$$ pairs route their straight-line paths through the *same* point $$\\boldsymbol{x}_t=\\boldsymbol{x}$$, each with its own velocity. The single-valued field the network must learn is their **average**. Because the minimizer of the squared-error CFM loss at each point is the conditional mean of the target, the network automatically converges to this marginal velocity:

$$\\nabla_\\theta J_{\\text{CFM}} = \\nabla_\\theta J_{\\text{FM}} \\quad\\Longrightarrow\\quad \\boldsymbol{v}_\\theta^\\star(\\boldsymbol{x},t) = \\mathbb{E}\\big[\\boldsymbol{u}_t(\\boldsymbol{x}\\mid\\boldsymbol{x}_1)\\mid\\boldsymbol{x}_t=\\boldsymbol{x}\\big].$$

The two objectives differ only by a constant, so optimizing the tractable conditional one is equivalent to optimizing the intractable marginal one. This is the exact analogue of the score-matching argument above. See how conditional and marginal fields relate:

{{viz:conditional_vs_marginal_velocity}}

{{viz:conditional_vs_marginal_paths}}""",
                },
                {
                    "heading": "Rectified flow and reflow",
                    "definition": (
                        "**Rectified flow** uses straight-line conditional paths; even so, the learned "
                        "**marginal** trajectories are generally curved. **Reflow** retrains on the model's "
                        "own (noise, sample) pairs to straighten them, enabling few-step sampling."
                    ),
                    "body": """![Marginal ODE trajectories are curved even when conditional paths are straight, which forces many integration steps (Principles Fig 5.9).](/assets/figures/day07/pdm_curved_paths.png)

The number of function evaluations a deterministic sampler needs is governed by how *curved* the marginal trajectories are: a straight path can be integrated exactly in one Euler step, while a curved one needs many. Reflow is an elegant fix: generate pairs $$(\\boldsymbol{x}_0,\\boldsymbol{x}_1=\\text{ODE}(\\boldsymbol{x}_0))$$ from the trained model, then *retrain* flow matching using those coupled pairs as the new conditional endpoints. The resulting flow is **straighter**, and iterating drives it toward straight marginal paths that can be sampled in very few — even one — steps. This is one of the routes to the fast samplers of Day 8.""",
                },
            ],
        },
        {
            "title": "One Model, Many Views",
            "subsections": [
                {
                    "heading": "Four equivalent parameterizations",
                    "definition": (
                        "Given the forward rule $$\\boldsymbol{x}_t=\\alpha_t\\boldsymbol{x}_0+\\sigma_t"
                        "\\boldsymbol{\\epsilon}$$, predicting the **noise** $$\\boldsymbol{\\epsilon}$$, the "
                        "**data** $$\\boldsymbol{x}_0$$, the **score** $$\\boldsymbol{s}$$, or the "
                        "**velocity** $$\\boldsymbol{v}$$ are all linearly interchangeable."
                    ),
                    "body": """![The four common prediction targets are equivalent reparameterizations of the same network output (Principles Fig 6.1).](/assets/figures/day07/pdm_param_equiv.png)

From the single relation $$\\boldsymbol{x}_t=\\alpha_t\\boldsymbol{x}_0+\\sigma_t\\boldsymbol{\\epsilon}$$ we can solve for any target in terms of any other:

$$\\boldsymbol{s}(\\boldsymbol{x}_t,t) = -\\frac{\\boldsymbol{\\epsilon}}{\\sigma_t},\\qquad \\boldsymbol{x}_0 = \\frac{\\boldsymbol{x}_t-\\sigma_t\\boldsymbol{\\epsilon}}{\\alpha_t},\\qquad \\boldsymbol{v} = \\dot\\alpha_t\\,\\boldsymbol{x}_0 + \\dot\\sigma_t\\,\\boldsymbol{\\epsilon}.$$

So a single network can be trained to output any one of them, and the others follow by algebra. The choice is purely practical: **$$\\boldsymbol{\\epsilon}$$-prediction** is stable at high noise, **$$\\boldsymbol{x}_0$$-prediction** at low noise, and **$$\\boldsymbol{v}$$-prediction** balances both and is popular for distillation. Explore the equivalence:

{{viz:four_predictions}}

{{viz:ddpm_prediction_equiv}}""",
                },
                {
                    "heading": "The unified picture",
                    "definition": (
                        "The variational (DDPM), score-based SDE, and flow-matching views all learn the "
                        "**same** time-indexed field over the same family of marginals $$p_t$$, and differ "
                        "only in parameterization and sampler."
                    ),
                    "body": """![The variational, score-SDE, and flow-matching formulations are three views of one underlying model (Principles Fig 6.2).](/assets/figures/day07/pdm_unified.png)

Stepping back, every approach this week shares the same skeleton:

1. a **forward process** $$\\boldsymbol{x}_t=\\alpha_t\\boldsymbol{x}_0+\\sigma_t\\boldsymbol{\\epsilon}$$ that interpolates data and noise;
2. a **network** trained by a denoising/regression objective to predict noise / data / score / velocity (all equivalent);
3. a **sampler** that runs a dynamics backward — ancestral (DDPM), annealed Langevin, the reverse SDE, or the probability-flow ODE.

DDPM, score-SDE, and flow matching are not competitors but different coordinates on the same object. This unification is what makes the field so powerful: an advance in one view (a better objective, a better sampler) transfers immediately to the others. **Day 8** exploits exactly this, using the deterministic ODE view to build guidance, high-order solvers, and few-step samplers.""",
                },
            ],
        },
    ],
    "checkpoint": [
        "Define the score and explain why it sidesteps the intractable normalizer of an energy-based model.",
        "Derive implicit score matching by integration by parts and say why the Jacobian-trace term is costly.",
        "State the denoising-score-matching identity and explain why learning the score equals learning to denoise.",
        "Write the Langevin update and explain the roles of the drift and noise terms.",
        "Explain the forward SDE, the reverse SDE, and the probability-flow ODE, and what each is good for.",
        "Derive that the marginal score is the conditional mean of the (Gaussian) conditional score.",
        "Set up the conditional flow-matching loss and argue it learns the marginal velocity field.",
        "Convert between the noise, data, score, and velocity parameterizations and justify the unified view.",
    ],
}
