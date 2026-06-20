"""Day 8 — Guidance, Solvers, and Few-Step Sampling.

Sources: *The Principles of Diffusion Models* (Lai, Song, Kim, Mitsufuji,
Ermon, arXiv:2510.21890), Chapters 8–11; Ho & Salimans, "Classifier-Free
Diffusion Guidance" (2022); Song et al., "Denoising Diffusion Implicit Models"
(DDIM, 2021); Karras et al., "Elucidating the Design Space..." (EDM, 2022);
Song et al., "Consistency Models" (2023). Notation is unified with Days 6-7:

    forward rule   x_t = alpha_t x_0 + sigma_t eps,   eps ~ N(0, I)
    score          s(x, t) = grad_x log p_t(x)
    PF-ODE         dx = [f - 1/2 g^2 s] dt

All prose and derivations are written for this course; interactive widgets are
referenced by ``{{viz:KEY}}`` markers expanded by generate_lectures.py.
"""

from diffusion_appendix import DAY08_OPTIONAL

# Curated figures aligned to each content slide (in order). None => no figure slide.
FIGURES = [
    # Conditional generation & guidance
    None,                                               # Conditioning a diffusion model
    "/assets/figures/day08/pdm_guidance.png",           # Classifier guidance
    "/assets/figures/day08/pdm_cfg.png",                # Classifier-free guidance
    None,                                               # Derivation: guidance from Bayes
    # Sampling = solving an ODE/SDE
    "/assets/figures/day08/pdm_reverse_sde.png",        # Sampling is numerical integration
    "/assets/figures/day08/pdm_ddim_euler.png",         # DDIM = Euler on the PF-ODE
    "/assets/figures/day08/pdm_score_sde_2d.png",       # Discretization error & step count
    None,                                               # Stochastic vs deterministic
    # Fast high-order solvers
    "/assets/figures/day08/pdm_heun_logsnr.png",        # Higher-order: Heun
    None,                                               # The right clock: log-SNR time
    "/assets/figures/day08/pdm_deis.png",               # Exponential integrators
    # Few-step sampling
    None,                                               # The bottleneck: many NFEs
    "/assets/figures/day08/pdm_flowmap.png",            # Flow maps
    "/assets/figures/day08/pdm_flowmap_semigroup.png",  # The semigroup property
    "/assets/figures/day08/pdm_flowmap_timeline.png",   # Consistency & distillation
]

SLIDES = (
    "Guidance, Solvers & Fast Sampling",
    "Controlling and accelerating diffusion",
    [
        (
            "Conditional Generation & Guidance",
            [
                (
                    "Conditioning a Diffusion Model",
                    [
                        "Goal: sample $p(x | c)$ — class, text prompt, image",
                        "Train the denoiser with the condition: $epsilon_theta (x_t, t, c)$",
                        "Conditional score $nabla log p_t (x | c)$ steers sampling",
                        "But naive conditioning often *under-uses* the condition",
                        "Guidance amplifies the influence of $c$",
                    ],
                ),
                (
                    "Classifier Guidance",
                    [
                        "Bayes: $nabla log p(x|c) = nabla log p(x) + nabla log p(c|x)$",
                        "Train a classifier $p_phi (c | x_t)$ on noisy inputs",
                        "Push samples toward high $p(c|x)$ using its gradient",
                        "Guidance scale $w$: $nabla log p + w nabla log p(c|x)$",
                        "Needs a separate noisy-data classifier (a drawback)",
                    ],
                ),
                (
                    "Classifier-Free Guidance",
                    [
                        "Train one net for both conditional & unconditional (drop $c$)",
                        "Combine at sampling time, no external classifier:",
                        "$tilde(epsilon) = (1+w) epsilon_theta (x,t,c) - w epsilon_theta (x,t,nothing)$",
                        "$w$ trades diversity (low) vs fidelity/prompt-match (high)",
                        "The workhorse of text-to-image models",
                    ],
                ),
                (
                    "Derivation: Guidance from Bayes",
                    [
                        "Bayes: $p(x|c) prop p(x) p(c|x)$",
                        "Take $nabla_x log$: scores add",
                        "Implicit classifier: $nabla log p(c|x) = nabla log p(x|c) - nabla log p(x)$",
                        "Score form $arrow.r$ noise form via $s = -epsilon \\/ sigma_t$",
                        "Sharpen: raise $p(c|x)$ to power $w arrow.r$ guidance scale",
                    ],
                ),
            ],
        ),
        (
            "Sampling = Solving a Differential Equation",
            [
                (
                    "Sampling Is Numerical Integration",
                    [
                        "Day 7: sampling = integrate the reverse SDE or PF-ODE",
                        "Each step needs one network eval (NFE) of the score",
                        "Quality vs cost = solver accuracy vs number of steps",
                        "Discretize time $T = t_0 > t_1 > dots.h > t_N = 0$",
                        "Better solver $arrow.r$ same quality in fewer steps",
                    ],
                ),
                (
                    "Derivation: PF-ODE from the FPE",
                    [
                        "FPE: $partial_t p_t = - nabla dot (f p_t) + 1/2 g^2 Delta p_t$",
                        "Log-derivative trick: $nabla p_t = p_t s$, $s = nabla log p_t$",
                        "Factor $p_t$ out $arrow.r$ Liouville: $partial_t p_t = - nabla dot ( tilde(mu) p_t )$",
                        "$tilde(mu) = f - 1/2 g^2 s$ $arrow.r$ PF-ODE (vs reverse SDE: $f - g^2 s$)",
                        "Full derivation: optional notes block",
                    ],
                ),
                (
                    "DDIM as Euler on the PF-ODE",
                    [
                        "DDIM = deterministic sampler = Euler on the PF-ODE",
                        "Non-Markovian: skip steps without retraining",
                        "Same trained model, far fewer steps than DDPM",
                        "Deterministic $arrow.r$ reproducible, invertible, editable",
                        "First-order: error per step $O(Delta t^2)$",
                    ],
                ),
                (
                    "Discretization Error & Step Count",
                    [
                        "Few steps + 1st-order $arrow.r$ visible artifacts",
                        "Error accumulates over the trajectory",
                        "More steps shrink error but cost more NFEs",
                        "Curved trajectories are harder to integrate",
                        "Two fixes: better solver, or straighter paths",
                    ],
                ),
                (
                    "Stochastic vs Deterministic Samplers",
                    [
                        "SDE samplers inject noise each step (self-correcting)",
                        "ODE samplers are deterministic (fast, fewer steps)",
                        "SDE: better at high NFE; ODE: better at low NFE",
                        "Churn: add a little noise to an ODE solver (EDM)",
                        "Fokker-Planck: both share the same marginals $p_t$",
                    ],
                ),
            ],
        ),
        (
            "Fast High-Order Solvers",
            [
                (
                    "Higher-Order: Heun's Method",
                    [
                        "Euler uses the slope at the start of the step",
                        "Heun averages start & end slopes (predictor-corrector)",
                        "2nd-order: error per step $O(Delta t^3)$",
                        "2 NFEs/step but far fewer steps overall",
                        "Backbone of the EDM sampler",
                    ],
                ),
                (
                    "The Right Clock: log-SNR Time",
                    [
                        "Solver accuracy depends on the time variable",
                        "Step uniformly in log-SNR $lambda = log(alpha^2 \\/ sigma^2)$",
                        "Spends steps where the trajectory bends most",
                        "EDM uses a tailored $sigma$ schedule (same idea)",
                        "Good schedule $arrow.r$ big quality gain for free",
                    ],
                ),
                (
                    "Exponential Integrators (DPM-Solver / DEIS)",
                    [
                        "PF-ODE = linear drift + nonlinear network term",
                        "Solve the linear part *exactly* (integrating factor, Day 1)",
                        "Only approximate the smooth network part",
                        "Multistep: reuse past evals for higher order",
                        "10-20 NFEs for high quality (DPM-Solver, DEIS)",
                    ],
                ),
            ],
        ),
        (
            "Few-Step Sampling",
            [
                (
                    "The Bottleneck: Many Function Evals",
                    [
                        "Even good solvers need ~10-50 NFEs",
                        "Real-time / interactive needs 1-4 steps",
                        "Idea: learn to *jump* across time, not crawl",
                        "Distill a slow teacher into a fast student",
                        "Flow maps formalize the jump",
                    ],
                ),
                (
                    "Flow Maps",
                    [
                        "Flow map $Phi_(s arrow.r t)$: jump a sample from time $s$ to $t$",
                        "Integrates the ODE in *one* learned step",
                        "$x_t = Phi_(s arrow.r t)(x_s)$ for any $s, t$",
                        "Generalizes the single-step generator",
                        "Learn it by distillation or self-consistency",
                    ],
                ),
                (
                    "The Semigroup Property",
                    [
                        "Composing jumps = one bigger jump:",
                        "$Phi_(s arrow.r t) = Phi_(u arrow.r t) circle.small Phi_(s arrow.r u)$",
                        "Consistency: all points on a trajectory map to the same $x_0$",
                        "Self-consistency gives a training signal w/o a teacher",
                        "Enables 1-step generation",
                    ],
                ),
                (
                    "Consistency & Distillation",
                    [
                        "Consistency models: train $f_theta (x_t, t) approx x_0$ for all $t$",
                        "Distillation: match a multi-step teacher in few steps",
                        "1-4 step sampling, near teacher quality",
                        "Trade a little quality for huge speedups",
                        "Active frontier: real-time generative models",
                    ],
                ),
            ],
        ),
    ],
)

LECTURE = {
    "day": 8,
    "slug": "guidance-solvers-fast-sampling",
    "title": "Guidance, Solvers, and Few-Step Sampling",
    "description": "Controlling diffusion with guidance, sampling as ODE/SDE solving, fast high-order solvers, and one-step flow maps.",
    "reading": [
        "[The Principles of Diffusion Models](https://arxiv.org/abs/2510.21890), Ch. 8–11; Appendix B (FPE), D.2.6 (PF-ODE proof)",
        "[Ho & Salimans — Classifier-Free Diffusion Guidance (2022)](https://arxiv.org/abs/2207.12598)",
        "[Song et al. — Denoising Diffusion Implicit Models (DDIM, 2021)](https://arxiv.org/abs/2010.02502)",
        "[Karras et al. — Elucidating the Design Space of Diffusion Models (EDM, 2022)](https://arxiv.org/abs/2206.00364)",
        "[Song et al. — Consistency Models (2023)](https://arxiv.org/abs/2303.01469)",
        "[SDE course — Lesson 2 §4: PF-ODE derivation from the FPE](https://kierandidi.github.io/) (Generative Modelling with SDEs)",
    ],
    "intro": (
        "Days 6–7 built diffusion models and showed that sampling means running a learned dynamics "
        "backward in time. Today we make them **useful and fast**. First, **guidance** lets us steer "
        "generation toward a condition — a class label or a text prompt — and trade diversity for "
        "fidelity. Second, we recognize sampling as **numerical integration** of the probability-flow "
        "ODE (or reverse SDE), which lets us import centuries of numerical-analysis wisdom: DDIM is just "
        "Euler's method, and high-order and exponential integrators reach high quality in a handful of "
        "steps. Finally, **flow maps** and **consistency models** collapse the trajectory into a single "
        "learned jump, pushing toward real-time, few-step generation."
    ),
    "sections": [
        {
            "title": "Conditional Generation and Guidance",
            "subsections": [
                {
                    "heading": "Conditioning a diffusion model",
                    "definition": (
                        "A **conditional** diffusion model samples $$p(\\boldsymbol{x}\\mid c)$$ for a "
                        "condition $$c$$ (class, text, image) by training a condition-aware denoiser "
                        "$$\\boldsymbol{\\epsilon}_\\theta(\\boldsymbol{x}_t,t,c)$$, equivalently a "
                        "conditional score $$\\nabla\\log p_t(\\boldsymbol{x}\\mid c)$$."
                    ),
                    "body": """**Why this matters.** Unconditional generation is a curiosity; the applications people care about — text-to-image, class-conditional synthesis, inpainting, super-resolution — are all *conditional*. The simplest recipe is to feed the condition $$c$$ to the network alongside $$(\\boldsymbol{x}_t,t)$$ and train exactly as before. Then the learned score is the conditional score, and any sampler from Day 7 produces samples from $$p(\\boldsymbol{x}\\mid c)$$.

In practice this "plain" conditioning often **under-uses** the condition: samples are diverse but only loosely match the prompt. We want a knob that trades some diversity for much stronger adherence to $$c$$. That knob is **guidance**, and it falls straight out of Bayes' rule on scores.""",
                },
                {
                    "heading": "Classifier guidance",
                    "definition": (
                        "**Classifier guidance** adds the gradient of a noise-aware classifier to the "
                        "score: $$\\nabla\\log p_t(\\boldsymbol{x}\\mid c) = \\nabla\\log p_t(\\boldsymbol{x}) "
                        "+ \\nabla\\log p_t(c\\mid\\boldsymbol{x}),$$ scaled by a guidance weight $$w$$."
                    ),
                    "body": """![Guidance steers the sampling trajectory toward regions consistent with the condition.](/assets/figures/day08/pdm_guidance.png)

By Bayes' rule, the conditional score decomposes into the unconditional score plus the gradient of a classifier's log-likelihood. So if we train a classifier $$p_\\phi(c\\mid\\boldsymbol{x}_t)$$ **on noisy inputs** (it must operate at every noise level), we can steer an unconditional model:

$$\\tilde{\\boldsymbol{s}}(\\boldsymbol{x}_t,t) = \\nabla\\log p_t(\\boldsymbol{x}_t) + w\\,\\nabla_{\\boldsymbol{x}_t}\\log p_\\phi(c\\mid\\boldsymbol{x}_t).$$

The weight $$w$$ controls how hard we push toward the class. The drawbacks are practical: you need a separate classifier trained on noised data, and its gradients can be brittle. Classifier-free guidance removes the external classifier entirely.""",
                },
                {
                    "heading": "Classifier-free guidance",
                    "definition": (
                        "**Classifier-free guidance (CFG)** trains a single network to be both conditional "
                        "and unconditional (by randomly dropping $$c$$), then extrapolates at sampling time: "
                        "$$\\tilde{\\boldsymbol{\\epsilon}} = (1+w)\\,\\boldsymbol{\\epsilon}_\\theta(\\boldsymbol{x}_t,t,c) "
                        "- w\\,\\boldsymbol{\\epsilon}_\\theta(\\boldsymbol{x}_t,t,\\varnothing).$$"
                    ),
                    "body": """![Classifier-free guidance interpolates/extrapolates between conditional and unconditional predictions to control fidelity.](/assets/figures/day08/pdm_cfg.png)

The trick is to avoid a separate classifier by noting that its gradient equals the *difference* of conditional and unconditional scores (next subsection). During training we replace $$c$$ with a null token $$\\varnothing$$ a fraction of the time, so one network learns both predictions. At sampling we extrapolate **away** from the unconditional prediction and **toward** the conditional one:

$$\\tilde{\\boldsymbol{\\epsilon}}_\\theta = \\boldsymbol{\\epsilon}_\\theta(\\boldsymbol{x}_t,t,\\varnothing) + (1+w)\\big[\\boldsymbol{\\epsilon}_\\theta(\\boldsymbol{x}_t,t,c) - \\boldsymbol{\\epsilon}_\\theta(\\boldsymbol{x}_t,t,\\varnothing)\\big].$$

The guidance scale $$w$$ is the single most important sampling knob in modern text-to-image models: $$w=0$$ is plain conditioning (diverse, loosely matched), while larger $$w$$ sharpens prompt adherence and image fidelity at the cost of diversity (and, if pushed too far, saturation artifacts).""",
                },
                {
                    "heading": "Derivation: guidance from Bayes' rule",
                    "definition": (
                        "Guidance is Bayes' rule differentiated. Sharpening the implicit classifier by an "
                        "exponent $$w$$ yields the guided score, and the score↔noise identity converts it "
                        "to the CFG noise formula."
                    ),
                    "body": """Start from Bayes' rule at noise level $$t$$, $$p_t(\\boldsymbol{x}\\mid c)\\propto p_t(\\boldsymbol{x})\\,p_t(c\\mid\\boldsymbol{x})$$, and take $$\\nabla_{\\boldsymbol{x}}\\log$$ of both sides (the $$\\boldsymbol{x}$$-independent evidence drops):

$$\\nabla\\log p_t(\\boldsymbol{x}\\mid c) = \\textcolor{teal}{\\nabla\\log p_t(\\boldsymbol{x})} + \\textcolor{purple}{\\nabla\\log p_t(c\\mid\\boldsymbol{x})}.$$

Rearranging gives the **implicit classifier** — no separate model needed:

$$\\textcolor{purple}{\\nabla\\log p_t(c\\mid\\boldsymbol{x})} = \\nabla\\log p_t(\\boldsymbol{x}\\mid c) - \\nabla\\log p_t(\\boldsymbol{x}).$$

To strengthen the condition, **sharpen** the classifier by raising it to a power, $$p_t(c\\mid\\boldsymbol{x})^{w}$$, which scales its score by $$w$$. The guided score becomes

$$\\tilde{\\boldsymbol{s}} = \\nabla\\log p_t(\\boldsymbol{x}) + (1+w)\\big[\\nabla\\log p_t(\\boldsymbol{x}\\mid c) - \\nabla\\log p_t(\\boldsymbol{x})\\big].$$

Finally convert scores to noise predictions with the Day-7 identity $$\\boldsymbol{s}=-\\boldsymbol{\\epsilon}/\\sigma_t$$ (the $$-1/\\sigma_t$$ cancels throughout), giving exactly the CFG rule

$$\\tilde{\\boldsymbol{\\epsilon}} = (1+w)\\,\\boldsymbol{\\epsilon}_\\theta(\\boldsymbol{x}_t,t,c) - w\\,\\boldsymbol{\\epsilon}_\\theta(\\boldsymbol{x}_t,t,\\varnothing).$$

So CFG is not a heuristic — it is Bayes' rule plus a sharpening exponent, expressed in noise-prediction coordinates.""",
                },
            ],
        },
        {
            "title": "Sampling as Solving a Differential Equation",
            "subsections": [
                {
                    "heading": "Sampling is numerical integration",
                    "definition": (
                        "Generating a sample means **numerically integrating** the reverse SDE or the "
                        "probability-flow ODE from $$t=T$$ to $$t=0$$. Each step evaluates the network "
                        "once — one **number of function evaluations (NFE)**."
                    ),
                    "body": """![Sampling runs a reverse-time process from noise back to data; numerically, this is integrating a differential equation.](/assets/figures/day08/pdm_reverse_sde.png)

On Day 7 we derived two dynamics with the right marginals — the reverse SDE and the PF-ODE. Turning either into an algorithm means choosing a **time discretization** $$T=t_0>t_1>\\dots>t_N=0$$ and a **numerical scheme** to step between consecutive times. The cost is dominated by the number of network evaluations (NFEs), since the network (the score/denoiser) is by far the most expensive operation.

This reframing is liberating: the quality–speed trade-off becomes a classic **numerical-analysis** question. A more accurate integrator reaches the same quality in fewer steps. Everything in this lecture is an answer to "how do we integrate this ODE well in as few NFEs as possible?".""",
                },
                {
                    "heading": "The probability-flow ODE from the Fokker–Planck equation",
                    "definition": (
                        "Rewriting the FPE with the log-derivative trick yields a **Liouville equation** "
                        "for an ODE with drift $$\\tilde{\\boldsymbol{\\mu}} = \\boldsymbol{f} - \\tfrac12 g^2\\boldsymbol{s}$$ "
                        "— the probability-flow ODE, sharing marginals with the forward/reverse SDE."
                    ),
                    "body": """The forward SDE $$\\mathrm{d}\\boldsymbol{x}=\\boldsymbol{f}\\,\\mathrm{d}t+g\\,\\mathrm{d}\\boldsymbol{w}$$ evolves its density $$p_t$$ by the **Fokker–Planck equation (FPE)**. The goal is to find a *deterministic* ODE that pushes the **same** $$p_t$$. The whole derivation is one rewrite of the diffusion term, using the **log-derivative trick** $$\\nabla p_t=p_t\\boldsymbol{s}$$ with $$\\boldsymbol{s}=\\nabla\\log p_t$$:

$$\\begin{aligned}
\\partial_t p_t
&= -\\nabla\\cdot\\big(\\textcolor{teal}{\\boldsymbol{f}}\\,p_t\\big) + \\tfrac12 g^2\\,\\Delta p_t & &\\text{(Fokker–Planck)} \\\\
&= -\\nabla\\cdot\\big(\\textcolor{teal}{\\boldsymbol{f}}\\,p_t\\big) + \\tfrac12 g^2\\,\\nabla\\cdot\\big(\\nabla p_t\\big) & &\\big(\\Delta = \\nabla\\cdot\\nabla\\big) \\\\
&= -\\nabla\\cdot\\big(\\textcolor{teal}{\\boldsymbol{f}}\\,p_t\\big) + \\tfrac12 g^2\\,\\nabla\\cdot\\big(p_t\\,\\textcolor{purple}{\\boldsymbol{s}}\\big) & &\\big(\\nabla p_t = p_t\\,\\textcolor{purple}{\\boldsymbol{s}}\\big) \\\\
&= -\\nabla\\cdot\\Big(\\underbrace{\\big[\\,\\textcolor{teal}{\\boldsymbol{f}} - \\tfrac12 g^2\\,\\textcolor{purple}{\\boldsymbol{s}}\\,\\big]}_{\\textcolor{orange}{\\tilde{\\boldsymbol{\\mu}}}}\\,p_t\\Big) & &\\text{(collect into one drift)}.
\\end{aligned}$$

The last line is a **continuity (Liouville) equation** $$\\partial_t p_t = -\\nabla\\cdot(\\textcolor{orange}{\\tilde{\\boldsymbol{\\mu}}}\\,p_t)$$ with **no** second-order term — precisely the FPE of a noise-free process. Hence the marginals are transported by the deterministic **probability-flow ODE**

$$\\boxed{\\;\\frac{\\mathrm{d}\\boldsymbol{x}}{\\mathrm{d}t} = \\textcolor{orange}{\\tilde{\\boldsymbol{\\mu}}(\\boldsymbol{x},t)} = \\textcolor{teal}{\\boldsymbol{f}(\\boldsymbol{x},t)} - \\tfrac12\\, g(t)^2\\,\\textcolor{purple}{\\boldsymbol{s}(\\boldsymbol{x},t)}.\\;}$$

Compare the **reverse SDE** drift $$\\boldsymbol{f}-g^2\\boldsymbol{s}$$ (Day 7): identical marginals $$\\{p_t\\}$$, but only **half** the score coefficient and no noise. The difference is exactly the half of the diffusion that the ODE converted into deterministic transport. This is why DDIM (Euler on the PF-ODE) and DDPM (the reverse SDE) **share one trained** $$\\boldsymbol{s}_\\theta$$ — they are two solvers for the same family of densities. Step-by-step algebra and the rigorous statement are in the optional block below.""",
                    "optional": DAY08_OPTIONAL,
                },
                {
                    "heading": "DDIM as Euler on the probability-flow ODE",
                    "definition": (
                        "**DDIM** is the deterministic sampler obtained by applying the **Euler method** to "
                        "the probability-flow ODE. It is non-Markovian, so it can take large steps with the "
                        "*same* trained model."
                    ),
                    "body": """![DDIM is exactly the Euler discretization of the probability-flow ODE.](/assets/figures/day08/pdm_ddim_euler.png)

Euler's method (Day 1) approximates $$\\boldsymbol{x}(s)\\approx\\boldsymbol{x}(t) + (s-t)\\,\\dot{\\boldsymbol{x}}(t)$$ using the slope at the start of the step. Applied to the PF-ODE $$\\dot{\\boldsymbol{x}} = \\boldsymbol{f}-\\tfrac12 g^2\\boldsymbol{s}_\\theta$$ it reproduces the **DDIM update** exactly. The cleanest way to see this uses the denoiser identity $$\\boldsymbol{x}_t=\\alpha_t\\hat{\\boldsymbol{x}}_0+\\sigma_t\\boldsymbol{\\epsilon}_\\theta$$: a single Euler step is equivalent to **freezing** the prediction $$(\\hat{\\boldsymbol{x}}_0,\\boldsymbol{\\epsilon}_\\theta)$$ at $$t$$ and re-evaluating the forward rule at the next time $$s<t$$,

$$\\begin{aligned}
\\boldsymbol{x}_s
&= \\alpha_s\\,\\textcolor{teal}{\\hat{\\boldsymbol{x}}_0} + \\sigma_s\\,\\textcolor{purple}{\\boldsymbol{\\epsilon}_\\theta(\\boldsymbol{x}_t,t)} & &\\text{(re-noise the frozen prediction to time }s\\text{)} \\\\
&= \\alpha_s\\,\\frac{\\boldsymbol{x}_t-\\sigma_t\\,\\textcolor{purple}{\\boldsymbol{\\epsilon}_\\theta}}{\\alpha_t} + \\sigma_s\\,\\textcolor{purple}{\\boldsymbol{\\epsilon}_\\theta} & &\\big(\\textcolor{teal}{\\hat{\\boldsymbol{x}}_0}=(\\boldsymbol{x}_t-\\sigma_t\\boldsymbol{\\epsilon}_\\theta)/\\alpha_t\\big) \\\\
&= \\frac{\\alpha_s}{\\alpha_t}\\,\\boldsymbol{x}_t + \\alpha_s\\Big(\\frac{\\sigma_s}{\\alpha_s}-\\frac{\\sigma_t}{\\alpha_t}\\Big)\\textcolor{purple}{\\boldsymbol{\\epsilon}_\\theta} & &\\text{(rearrange).}
\\end{aligned}$$

Dividing the last line by $$\\alpha_s$$ gives $$\\boldsymbol{x}_s/\\alpha_s = \\boldsymbol{x}_t/\\alpha_t + (\\rho_s-\\rho_t)\\,\\boldsymbol{\\epsilon}_\\theta$$ with $$\\rho_t:=\\sigma_t/\\alpha_t$$ — literally an **Euler step in the log-SNR clock**, which is the modern view of DDIM (and the base case of DPM-Solver). The consequences are large:

- **Step skipping.** Because DDIM is non-Markovian (it depends on the marginal at $$t$$, not a Markov chain), we can use a coarse time grid — e.g. 20–50 steps instead of DDPM's 1000 — with no retraining.
- **Determinism.** With the noise fixed (an ODE, no $$\\mathrm{d}\\boldsymbol{w}$$), the map from initial noise to sample is deterministic and invertible — enabling latent interpolation, semantic editing, and exact reconstruction.

DDIM is **first-order**: its local error per step is $$O(\\Delta t^2)$$, which is why very few steps still show artifacts. Higher-order solvers do better.""",
                },
                {
                    "heading": "Discretization error and step count",
                    "definition": (
                        "**Discretization error** is the gap between the exact ODE solution and its "
                        "numerical approximation. It shrinks as steps increase and grows with trajectory "
                        "curvature, accumulating along the path."
                    ),
                    "body": """![Sampling a 2-D distribution with the Score SDE; too few steps leaves visible structure errors.](/assets/figures/day08/pdm_score_sde_2d.png)

Two sources of error compound. **Local** error is made at each step (for Euler, $$O(\\Delta t^2)$$ per step, $$O(\\Delta t)$$ globally); it shrinks as we add steps. But more steps mean more NFEs — the very cost we want to avoid. **Curvature** is the other factor: the more the trajectory bends, the worse a straight-line (Euler) step approximates it.

This diagnosis points to exactly two remedies, which structure the rest of the day:

1. **Use a better integrator** — higher-order or exponential solvers that take the curvature into account (next section).
2. **Make the trajectory straighter** — so even a crude integrator suffices (rectified flow from Day 7, and flow maps below).""",
                },
                {
                    "heading": "Stochastic versus deterministic samplers",
                    "definition": (
                        "**SDE (stochastic)** samplers inject fresh noise each step and can self-correct "
                        "errors; **ODE (deterministic)** samplers are noise-free and need fewer steps. "
                        "Both share the same marginals $$p_t$$ by the Fokker–Planck equation."
                    ),
                    "body": """The reverse SDE and the PF-ODE produce the same distribution but behave differently as samplers:

- **SDE samplers** add noise at every step. That noise is self-correcting — errors made early can be "washed out" — so SDE samplers tend to win at **high** NFE budgets and produce slightly more diverse, detailed samples.
- **ODE samplers** are deterministic and smoother, dominating at **low** NFE budgets where every evaluation counts.

A useful hybrid (EDM's "churn") runs an ODE solver but injects a controlled amount of noise per step, interpolating between the two regimes. Underlying all of this is the **Fokker–Planck equation**, which guarantees that the deterministic and stochastic dynamics transport the *same* density $$p_t$$ — explore how a density flows under it:

{{viz:fokker_planck}}""",
                },
            ],
        },
        {
            "title": "Fast High-Order Solvers",
            "subsections": [
                {
                    "heading": "Higher-order integration: Heun's method",
                    "definition": (
                        "**Heun's method** is a second-order predictor–corrector: take an Euler step, "
                        "evaluate the slope at the predicted endpoint, and step again with the **average** "
                        "of the two slopes. Local error drops to $$O(\\Delta t^3)$$."
                    ),
                    "body": """![A higher-order (Heun) solver stepped in log-SNR time tracks the trajectory far more accurately than Euler.](/assets/figures/day08/pdm_heun_logsnr.png)

Euler uses only the slope at the start of the interval, so it consistently misses the bend of a curved trajectory. Heun corrects this:

1. **Predict** with Euler: $$\\tilde{\\boldsymbol{x}} = \\boldsymbol{x}_t - \\Delta t\\,\\boldsymbol{d}(\\boldsymbol{x}_t,t)$$.
2. **Correct** with the average slope: $$\\boldsymbol{x}_{t-\\Delta t} = \\boldsymbol{x}_t - \\tfrac{\\Delta t}{2}\\big[\\boldsymbol{d}(\\boldsymbol{x}_t,t) + \\boldsymbol{d}(\\tilde{\\boldsymbol{x}},t-\\Delta t)\\big]$$.

This costs **2 NFEs per step** but is second-order ($$O(\\Delta t^3)$$ local error), so it needs dramatically fewer steps for the same accuracy — a net win. Heun (with the schedule below) is the core of the widely used **EDM** sampler, which reaches excellent quality in ~20–40 NFEs. Compare Euler and Heun directly:

{{viz:euler_vs_heun_solver}}""",
                },
                {
                    "heading": "The right clock: log-SNR time",
                    "definition": (
                        "Solver accuracy depends on the **time parameterization**. Stepping uniformly in "
                        "**log-SNR** $$\\lambda_t=\\log(\\alpha_t^2/\\sigma_t^2)$$ (or EDM's $$\\sigma$$ "
                        "schedule) places steps where the trajectory changes fastest."
                    ),
                    "body": """An ODE can be re-parameterized by any monotonic change of the time variable, and the **choice matters enormously** for a discretized solver. Uniform steps in the native $$t$$ waste effort where the trajectory is nearly straight and under-resolve where it bends. The trajectory's "natural clock" is the **log signal-to-noise ratio** $$\\lambda_t=\\log(\\alpha_t^2/\\sigma_t^2)$$ from Day 6: stepping uniformly in $$\\lambda$$ allocates steps in proportion to how much the sample actually moves.

EDM expresses the same insight through a carefully designed $$\\sigma$$ (noise-level) schedule, concentrating steps at the small-noise end where detail is formed. The headline is that **a better schedule is free quality** — no retraining, no extra NFEs, just a smarter choice of where to place the steps you already take.""",
                },
                {
                    "heading": "Exponential integrators: DPM-Solver and DEIS",
                    "definition": (
                        "The PF-ODE has a **semilinear** form: a linear drift plus a nonlinear network "
                        "term. **Exponential integrators** solve the linear part *exactly* (via an "
                        "integrating factor) and only approximate the smooth nonlinear part."
                    ),
                    "body": """![Multistep exponential integrators (DEIS) reuse previous evaluations to achieve high order at low cost.](/assets/figures/day08/pdm_deis.png)

The diffusion ODE has a **semilinear** form — a linear term in $$\\boldsymbol{x}$$ plus a term involving the network:

$$\\dot{\\boldsymbol{x}} = \\textcolor{teal}{a(t)\\,\\boldsymbol{x}} + \\textcolor{purple}{b(t)\\,\\boldsymbol{\\epsilon}_\\theta(\\boldsymbol{x},t)}.$$

The linear part is exactly the kind of equation we solved on Day 1 with an **integrating factor** $$e^{-A(t)}$$, $$A(t)=\\int_0^t a(u)\\,\\mathrm{d}u$$. Multiplying through collapses the linear term into a total derivative, and integrating from $$t$$ to $$s$$ gives **variation of constants**:

$$\\begin{aligned}
\\frac{\\mathrm{d}}{\\mathrm{d}t}\\Big(e^{-A(t)}\\boldsymbol{x}\\Big)
&= e^{-A(t)}\\big(\\dot{\\boldsymbol{x}} - \\textcolor{teal}{a(t)\\boldsymbol{x}}\\big) = e^{-A(t)}\\,\\textcolor{purple}{b(t)\\,\\boldsymbol{\\epsilon}_\\theta} & &\\text{(integrating factor)} \\\\
\\Longrightarrow\\quad \\boldsymbol{x}_s &= \\underbrace{e^{A(s)-A(t)}\\,\\boldsymbol{x}_t}_{\\textcolor{teal}{\\text{linear part: exact}}} \\;+\\; \\underbrace{\\int_t^s e^{A(s)-A(u)}\\,b(u)\\,\\textcolor{purple}{\\boldsymbol{\\epsilon}_\\theta(\\boldsymbol{x}_u,u)}\\,\\mathrm{d}u}_{\\textcolor{purple}{\\text{network part: approximate}}}.
\\end{aligned}$$

The first term is **exact** — no discretization error from the stiff linear drift. Only the smooth integral of $$\\boldsymbol{\\epsilon}_\\theta$$ is approximated: freezing $$\\boldsymbol{\\epsilon}_\\theta$$ at $$t$$ recovers DDIM (first order); a Taylor/multistep expansion of $$\\boldsymbol{\\epsilon}_\\theta$$ in the log-SNR variable $$\\lambda$$ gives the high-order solvers:

- **DPM-Solver** applies this exponential-integrator idea with high-order Taylor expansions of $$\\boldsymbol{\\epsilon}_\\theta$$ along log-SNR time.
- **DEIS** uses a **multistep** variant, reusing network evaluations from previous steps (like Adams–Bashforth) to reach high order with roughly one NFE per step.

Together these reach high-quality samples in **10–20 NFEs**, an order of magnitude fewer than naive DDPM, and are the default fast samplers in many production systems.""",
                },
            ],
        },
        {
            "title": "Few-Step Sampling: Flow Maps and Distillation",
            "subsections": [
                {
                    "heading": "The bottleneck: many function evaluations",
                    "definition": (
                        "Even the best solvers need roughly $$10$$–$$50$$ NFEs. **Few-step sampling** aims "
                        "for $$1$$–$$4$$ NFEs by learning to **jump** across time rather than integrate "
                        "step by step."
                    ),
                    "body": """**Why this matters.** Interactive and real-time applications — drawing tools, video, on-device generation — cannot afford dozens of network passes per sample. No matter how clever the solver, integrating an ODE inherently requires multiple evaluations to follow a curved path accurately.

The shift in thinking is from *integration* to *amortization*: instead of repeatedly querying the slope and crawling along the trajectory, **learn a function that maps directly** from a point at one time to the corresponding point at another. We can train such a function by distilling a slow multi-step **teacher** into a fast **student**, or by enforcing self-consistency. Flow maps make this precise.""",
                },
                {
                    "heading": "Flow maps",
                    "definition": (
                        "A **flow map** $$\\Phi_{s\\to t}$$ sends a sample at time $$s$$ to its ODE solution "
                        "at time $$t$$ in a single learned evaluation: "
                        "$$\\boldsymbol{x}_t = \\Phi_{s\\to t}(\\boldsymbol{x}_s).$$"
                    ),
                    "body": """![A flow map integrates the probability-flow ODE between two times in one step, rather than many small Euler steps.](/assets/figures/day08/pdm_flowmap.png)

The PF-ODE defines, for any pair of times $$(s,t)$$, an exact map taking $$\\boldsymbol{x}_s$$ to $$\\boldsymbol{x}_t$$ — the solution operator of the ODE. A numerical solver approximates this map by chaining many tiny steps; a **flow map** $$\\Phi_{s\\to t}$$ *learns it directly* so it can be applied in one shot. The special case $$\\Phi_{T\\to 0}$$ is a **one-step generator**: noise in, sample out. The remaining questions are how to parameterize $$\\Phi$$ consistently across times and how to train it without simply storing a teacher's whole trajectory — answered by the semigroup/consistency structure. Compare flow-map model families:

{{viz:flow_map_models}}""",
                },
                {
                    "heading": "The semigroup property",
                    "definition": (
                        "Flow maps compose: stepping $$s\\to u$$ then $$u\\to t$$ equals stepping $$s\\to t$$ "
                        "directly. This **semigroup property** "
                        "$$\\Phi_{s\\to t} = \\Phi_{u\\to t}\\circ\\Phi_{s\\to u}$$ underlies self-consistent "
                        "training."
                    ),
                    "body": """![The semigroup (consistency) property: composing partial jumps must equal the full jump.](/assets/figures/day08/pdm_flowmap_semigroup.png)

Because the flow map is the solution operator of a deterministic ODE, it must satisfy

$$\\Phi_{s\\to t} = \\Phi_{u\\to t}\\circ\\Phi_{s\\to u}\\quad\\text{for any intermediate } u,$$

and in particular all points on a single ODE trajectory map to the **same** clean sample $$\\boldsymbol{x}_0$$. This is the **consistency** condition. It is powerful because it is a *self-supervised* constraint: a network can be trained to satisfy it by penalizing the discrepancy between $$\\Phi_{s\\to t}(\\boldsymbol{x}_s)$$ and $$\\Phi_{u\\to t}(\\Phi_{s\\to u}(\\boldsymbol{x}_s))$$, **without** needing a precomputed teacher trajectory. Enforcing consistency is what lets a one-step map be learned stably.""",
                },
                {
                    "heading": "Consistency models and distillation",
                    "definition": (
                        "**Consistency models** train a network $$f_\\theta(\\boldsymbol{x}_t,t)\\approx"
                        "\\boldsymbol{x}_0$$ that is consistent along trajectories, enabling one- or few-step "
                        "generation. **Distillation** alternatively matches a multi-step teacher."
                    ),
                    "body": """![The flow-map timeline: a learned map jumps directly toward the data manifold, collapsing many steps into a few.](/assets/figures/day08/pdm_flowmap_timeline.png)

Two routes to a fast student:

- **Consistency training** uses the self-consistency condition above as the loss, learning $$f_\\theta(\\boldsymbol{x}_t,t)\\approx\\boldsymbol{x}_0$$ directly (from scratch or from a pretrained score model). Sampling is then one evaluation $$\\boldsymbol{x}_0=f_\\theta(\\boldsymbol{x}_T,T)$$, optionally refined by a few alternating noise/denoise steps.
- **Distillation** trains the student to reproduce, in a few steps, what a slow many-step teacher produces — progressive distillation halves the step count repeatedly; adversarial and trajectory-matching variants push to 1–4 steps near teacher quality.

The trade-off is explicit: a small drop in sample quality for a $$10$$–$$1000\\times$$ speedup. This is the active frontier that brings diffusion toward **real-time** generation — and it ties the whole week together, since it relies on the unified score/velocity model (Day 7), the ODE-as-sampler view, and the numerical-analysis toolkit of today.""",
                },
            ],
        },
    ],
    "checkpoint": [
        "Explain conditional diffusion and why plain conditioning often under-uses the condition.",
        "Derive classifier-free guidance from Bayes' rule and the score↔noise identity, and explain the role of w.",
        "Argue that sampling is numerical integration and that DDIM is Euler on the probability-flow ODE.",
        "Describe the sources of discretization error and the two ways to reduce required NFEs.",
        "Contrast stochastic and deterministic samplers and state what the Fokker–Planck equation guarantees.",
        "Explain Heun's method, log-SNR time stepping, and exponential integrators (DPM-Solver/DEIS).",
        "Define a flow map and the semigroup/consistency property, and explain how consistency models reach one-step sampling.",
        "Looking ahead: diffusion fought the cost of *many sampling steps* (NFEs) with solvers and flow maps. Day 9 returns to the autoregressive family from the Day 6 taxonomy, where the analogous cost is *serial token-by-token decoding* — fought on Day 10 with the KV cache.",
    ],
}
