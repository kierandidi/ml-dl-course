#!/usr/bin/env python3
"""Generate week-2 lecture posts (days 6–10) under lectures/_posts/."""
from __future__ import annotations

import re
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
POSTS = ROOT / "lectures" / "_posts"

# (day, date, slug, short title, description, optional reading lines, body markdown)
LECTURES: list[tuple[int, str, str, str, str, list[str], str]] = [
    (
        6,
        "2026-08-24",
        "day06-generative-modeling",
        "Generative Modeling",
        "KL divergence, ELBO, MLE/MAP, and model families from VAEs to energy-based models.",
        [
            "[The Principles of Diffusion Models](https://arxiv.org/abs/2510.21890) — Ch. 1–2 (variational view)",
            "[Kingma & Welling — Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)",
            "[Rezende & Mohamed — Normalizing Flows](https://arxiv.org/abs/1505.05770)",
        ],
        dedent(
            r"""
            Week 2 opens with the probabilistic language of generative modeling: how we compare distributions,
            how we optimize latent-variable models, and how explicit density models differ from implicit samplers.

            ## 1. Likelihoods, MLE, and MAP

            > **Definition (generative model).** A generative model specifies a distribution \(p_\theta(\mathbf{x})\)
            > over data \(\mathbf{x} \in \mathcal{X}\), either directly or via latent variables
            > \(\mathbf{z}\) and a joint \(p_\theta(\mathbf{x}, \mathbf{z})\).
            {:.lead}

            Given i.i.d. samples \(\{\mathbf{x}^{(i)}\}_{i=1}^N\) from an unknown data distribution \(p_{\mathrm{data}}\),
            **maximum likelihood estimation (MLE)** chooses parameters that maximize the empirical log-likelihood:

            $$
            \hat{\theta}_{\mathrm{MLE}} = \arg\max_\theta \frac{1}{N}\sum_{i=1}^N \log p_\theta(\mathbf{x}^{(i)}).
            $$

            When we have a prior \(p(\theta)\) over parameters, **maximum a posteriori (MAP)** adds a regularizer:

            $$
            \hat{\theta}_{\mathrm{MAP}} = \arg\max_\theta \left[ \sum_{i=1}^N \log p_\theta(\mathbf{x}^{(i)}) + \log p(\theta) \right].
            $$

            For deep generative models the likelihood is often intractable; we then optimize surrogates (ELBO, score
            matching, flow-matching losses) that remain consistent with likelihood principles.

            ![Generative modeling overview](/assets/figures/day06/pdf0_page000.png)
            *Figure: generative modeling landscape (slides).*

            ### 1.1 KL divergence

            > **Definition.** For distributions \(p\) and \(q\) on the same space, the Kullback–Leibler divergence is
            > \(D_{\mathrm{KL}}(p \| q) = \mathbb{E}_{\mathbf{x}\sim p}\left[\log \frac{p(\mathbf{x})}{q(\mathbf{x})}\right]\).
            {:.lead}

            Properties used throughout the course:

            - \(D_{\mathrm{KL}}(p \| q) \ge 0\) with equality iff \(p = q\) (a.e.).
            - Not symmetric: \(D_{\mathrm{KL}}(p \| q) \neq D_{\mathrm{KL}}(q \| p)\) in general.
            - **Information inequality:** minimizing \(D_{\mathrm{KL}}(p_{\mathrm{data}} \| p_\theta)\) is equivalent to MLE when \(p_\theta\) is flexible enough.

            For latent-variable models with joint \(p_\theta(\mathbf{x}, \mathbf{z}) = p_\theta(\mathbf{z})\,p_\theta(\mathbf{x}\mid\mathbf{z})\),

            $$
            \log p_\theta(\mathbf{x}) = \log \int p_\theta(\mathbf{x}, \mathbf{z})\,d\mathbf{z},
            $$

            which is typically intractable for neural encoders/decoders.

            ## 2. ELBO and variational inference

            Introduce an approximate posterior \(q_\phi(\mathbf{z}\mid\mathbf{x})\). For any fixed \(\mathbf{x}\),

            $$
            \log p_\theta(\mathbf{x}) = \mathcal{L}(\theta, \phi; \mathbf{x}) + D_{\mathrm{KL}}\!\left(q_\phi(\mathbf{z}\mid\mathbf{x}) \,\|\, p_\theta(\mathbf{z}\mid\mathbf{x})\right),
            $$

            where the **evidence lower bound (ELBO)** is

            $$
            \mathcal{L}(\theta, \phi; \mathbf{x}) =
            \mathbb{E}_{q_\phi(\mathbf{z}\mid\mathbf{x})}\!\left[\log p_\theta(\mathbf{x}\mid\mathbf{z})\right]
            - D_{\mathrm{KL}}\!\left(q_\phi(\mathbf{z}\mid\mathbf{x}) \,\|\, p_\theta(\mathbf{z})\right).
            $$

            Since \(D_{\mathrm{KL}} \ge 0\), we have \(\mathcal{L}(\theta,\phi;\mathbf{x}) \le \log p_\theta(\mathbf{x})\).
            **Maximizing the ELBO** tightens the bound and improves the marginal likelihood proxy.

            ![ELBO decomposition](/assets/figures/day06/pdf0_page005.png)
            *Figure: variational bound on log-likelihood.*

            ### 2.1 Amortized inference

            In a **VAE**, \(q_\phi(\mathbf{z}\mid\mathbf{x}) = \mathcal{N}(\mathbf{z}; \boldsymbol{\mu}_\phi(\mathbf{x}), \mathrm{diag}(\boldsymbol{\sigma}_\phi^2(\mathbf{x})))\)
            and \(p_\theta(\mathbf{x}\mid\mathbf{z})\) is a decoder. Training maximizes

            $$
            \mathbb{E}_{\mathbf{x}\sim p_{\mathrm{data}}}\left[\mathcal{L}(\theta,\phi;\mathbf{x})\right],
            $$

            using the reparameterization trick \(\mathbf{z} = \boldsymbol{\mu}_\phi(\mathbf{x}) + \boldsymbol{\sigma}_\phi(\mathbf{x}) \odot \boldsymbol{\epsilon}\),
            \(\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0}, \mathbf{I})\).

            ## 3. Explicit vs implicit generative models

            | Family | Tractable density? | Sample | Examples |
            |--------|-------------------|--------|----------|
            | **Explicit** | Yes (up to constant) | MCMC or exact | Flows, autoregressive, some EBMs |
            | **Implicit** | No closed form | Forward pass | GANs, diffusion (learn score/velocity) |
            | **Latent explicit** | Marginal hard | Encode–decode | VAE, hierarchical VAE |
            | **Energy-based** | \(p(\mathbf{x}) \propto e^{-E(\mathbf{x})}\) | MCMC / Langevin | Hopfield, modern EBMs |

            > **Explicit model:** you can evaluate \(p_\theta(\mathbf{x})\) (or \(\log p_\theta(\mathbf{x})\)) up to normalization for a single \(\mathbf{x}\).
            > **Implicit model:** you can sample \(\mathbf{x}\sim p_\theta\) without a normalized density.
            {:.lead}

            ![Model taxonomy](/assets/figures/day06/pdf0_page010.png)
            *Figure: where common architectures sit in the taxonomy.*

            ### 3.1 Normalizing flows

            A flow is an invertible map \(\mathbf{f}_\theta: \mathbb{R}^d \to \mathbb{R}^d\) with tractable Jacobian determinant:

            $$
            p_\theta(\mathbf{x}) = p_0(\mathbf{f}_\theta^{-1}(\mathbf{x}))
            \left|\det \frac{\partial \mathbf{f}_\theta^{-1}(\mathbf{x})}{\partial \mathbf{x}}\right|.
            $$

            Composition of coupling layers yields expressive densities and exact MLE training.

            ### 3.2 Energy-based models

            An **EBM** defines \(p_\theta(\mathbf{x}) = \frac{1}{Z(\theta)} e^{-E_\theta(\mathbf{x})}\) with partition function
            \(Z(\theta) = \int e^{-E_\theta(\mathbf{x})}\,d\mathbf{x}\). Training often uses contrastive objectives or score matching
            because \(Z(\theta)\) is intractable in high dimensions.

            ## 4. VAEs, flows, and EBMs in practice

            **VAEs** trade exact likelihood for fast amortized inference; blurry samples can result from Gaussian decoders and ELBO gap.

            **Flows** give exact likelihoods but architectural constraints (invertibility, dimension preservation).

            **EBMs** model unnormalized densities and connect naturally to **score functions**
            \(\nabla_{\mathbf{x}} \log p(\mathbf{x}) = -\nabla_{\mathbf{x}} E(\mathbf{x})\), foreshadowing diffusion training.

            ![VAE and flow sketches](/assets/figures/day06/pdf1_page000.png)
            *Figure: encoder–decoder vs invertible layers.*

            ### 4.1 Design checklist

            1. Do you need exact $$\log p(\mathbf{x})$$? → flows or autoregressive models.
            2. Do you need fast latent codes? → VAE or VAE + diffusion in latent space (LDM).
            3. Do you only need high-quality samples? → diffusion, flows, or GAN-style objectives.

            ### 4.2 GANs as implicit competitors

            Generative adversarial networks optimize a minimax game between generator $$G_\theta$$ and discriminator $$D_\phi$$:

            $$
            \min_\theta \max_\phi \;
            \mathbb{E}_{\mathbf{x}\sim p_{\mathrm{data}}}[\log D_\phi(\mathbf{x})]
            + \mathbb{E}_{\mathbf{z}\sim p_0}[\log(1 - D_\phi(G_\theta(\mathbf{z})))].
            $$

            There is no tractable global density unless combined with additional constraints; mode collapse and training
            instability motivated diffusion and flow alternatives covered in days 7–8.

            ![Energy landscape](/assets/figures/day06/pdf1_page010.png)
            *Figure: implicit vs explicit sampling geometry.*

            ## Checkpoint summary

            - **MLE/MAP** fit parameters to data (with optional priors).
            - **KL** measures how one distribution diverges from another; ELBO = log-likelihood minus posterior KL gap.
            - **Explicit** models expose densities; **implicit** models expose samplers or scores.
            - **VAE / flows / EBMs** are the three classical families we extend in days 7–10.
            """
        ).strip(),
    ),
    (
        7,
        "2026-08-25",
        "day07-training-diffusion-flow",
        "Training Diffusion and Flow Models",
        "Forward noising, score matching, and the flow-matching training objective.",
        [
            "[MIT 6.S184 — Flow Matching and Diffusion (2026)](https://diffusion.csail.mit.edu/2026/index.html)",
            "[Song et al. — Score-Based Generative Modeling](https://arxiv.org/abs/2011.13456)",
            "[Lipman et al. — Flow Matching](https://arxiv.org/abs/2210.02747)",
        ],
        dedent(
            r"""
            Diffusion and flow models learn **time-dependent vector fields** or **scores** that transport a simple prior
            to the data distribution. This lecture focuses on **training objectives**, not sampling.

            ## 1. Forward (noising) process

            > **Definition (forward SDE, informal).** A forward process gradually adds noise:
            > \(d\mathbf{x}_t = \mathbf{f}(\mathbf{x}_t, t)\,dt + g(t)\,d\mathbf{w}_t\), \(t \in [0, T]\),
            > with \(\mathbf{x}_0 \sim p_{\mathrm{data}}\) and \(\mathbf{x}_T \approx \mathcal{N}(\mathbf{0}, \mathbf{I})\).
            {:.lead}

            A common **variance-preserving (VP)** discrete schedule uses

            $$
            q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}\!\left(\mathbf{x}_t;\, \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0,\, (1-\bar{\alpha}_t)\mathbf{I}\right),
            $$

            with noise schedule \(\beta_t\) and \(\bar{\alpha}_t = \prod_{s=1}^{t}(1-\beta_s)\).

            The forward kernel tells us how corrupted data looks at time \(t\); learning reverses this corruption.

            ![Forward noising](/assets/figures/day07/pdf0_page000.png)
            *Figure: data → noise continuum.*

            ### 1.1 Conditional means and scores

            From the Gaussian kernel,

            $$
            \mathbb{E}[\mathbf{x}_t \mid \mathbf{x}_0] = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0,
            \qquad
            \nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t \mid \mathbf{x}_0) = -\frac{\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0}{1-\bar{\alpha}_t}.
            $$

            Denoising networks often predict noise \(\boldsymbol{\epsilon}\), score, or velocity depending on parameterization.

            ## 2. Score matching

            > **Score function.** \(s(\mathbf{x}) = \nabla_{\mathbf{x}} \log p(\mathbf{x})\) points toward high-density regions.
            {:.lead}

            **Denoising score matching (DSM):** train \(s_\theta(\mathbf{x}_t, t)\) to match the score of the noisy distribution
            \(q(\mathbf{x}_t)\). With sampled pairs \((\mathbf{x}_0, \mathbf{x}_t \sim q(\mathbf{x}_t\mid\mathbf{x}_0))\),

            $$
            \mathcal{L}_{\mathrm{DSM}}(\theta) =
            \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_t}\left[
            \lambda(t)\,\left\| s_\theta(\mathbf{x}_t, t) - \nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t \mid \mathbf{x}_0) \right\|_2^2
            \right].
            $$

            Weight \(\lambda(t)\) balances signal across noise levels. Predicting \(\boldsymbol{\epsilon}\) is an equivalent
            reparameterization widely used in DDPM implementations.

            ![Score matching](/assets/figures/day07/pdf0_page008.png)
            *Figure: denoising target at each noise level.*

            ### 2.1 Connection to EBMs

            If \(p(\mathbf{x}) \propto e^{-E(\mathbf{x})}\), then \(s(\mathbf{x}) = -\nabla_{\mathbf{x}} E(\mathbf{x})\).
            Score-based diffusion avoids estimating the partition function \(Z\) directly.

            ## 3. Flow matching objective

            **Flow matching** learns a velocity field \(\mathbf{v}_\theta(\mathbf{x}, t)\) such that the ODE

            $$
            \frac{d\mathbf{x}_t}{dt} = \mathbf{v}_\theta(\mathbf{x}_t, t)
            $$

            transports a prior \(p_0\) to \(p_{\mathrm{data}}\). Construct a **probability path**
            \(p_t(\mathbf{x})\) with marginals \(p_0\) and \(p_1 = p_{\mathrm{data}}\).

            > **Conditional flow matching.** Sample \(\mathbf{x}_1 \sim p_{\mathrm{data}}\), \(\mathbf{x}_0 \sim p_0\),
            > build a path \(\mathbf{x}_t\) (e.g. linear interpolation \(\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1\)),
            > and regress \(\mathbf{v}_\theta(\mathbf{x}_t, t)\) onto the path velocity \(\dot{\mathbf{x}}_t\).
            {:.lead}

            A typical loss:

            $$
            \mathcal{L}_{\mathrm{FM}}(\theta) =
            \mathbb{E}_{t \sim \mathcal{U}(0,1),\, \mathbf{x}_0,\, \mathbf{x}_1,\, \mathbf{x}_t}
            \left\| \mathbf{v}_\theta(\mathbf{x}_t, t) - \dot{\mathbf{x}}_t \right\|_2^2.
            $$

            **Rectified flows** and **optimal transport** paths reduce curvature and improve sample efficiency at inference.

            ![Flow matching paths](/assets/figures/day07/pdf1_page010.png)
            *Figure: coupling noise to data along a path.*

            ### 3.1 Diffusion as a special flow

            Diffusion SDEs induce a family of marginals \(p_t\); the **probability flow ODE** uses a velocity field derived from
            the score so that marginals match the SDE. Training can therefore be viewed in a unified **flow / score** picture.

            ## 4. Training loop in practice

            1. Sample minibatch \(\mathbf{x}_0 \sim p_{\mathrm{data}}\).
            2. Sample time \(t\) (continuous or discrete index).
            3. Form \(\mathbf{x}_t\) via forward kernel or path sampler.
            4. Compute loss (DSM, \(\boldsymbol{\epsilon}\)-prediction, or flow matching).
            5. Backprop through U-Net / DiT backbone.

            ![Training pipeline](/assets/figures/day07/pdf1_page020.png)
            *Figure: denoiser architecture and time conditioning.*

            ### 4.1 Engineering notes

            - **Time embedding:** sinusoidal or learned embeddings injected into every block.
            - **EMA weights:** exponential moving average of parameters stabilizes sampling metrics.
            - **Class conditioning:** auxiliary label $$y$$ enters via embedding or cross-attention (day 8: guidance).

            ### 4.2 Loss weighting and continuous time

            In continuous-time VP-SDEs, the DSM weight $$\lambda(t)$$ compensates for shrinking signal-to-noise ratio as
            $$t \to T$$. A practical recipe:

            $$
            \lambda(t) \propto \mathbb{E}\left[\left\|\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t \mid \mathbf{x}_0)\right\|_2^2\right]^{-1}
            $$

            so each noise level contributes equally to gradients. Discrete DDPM uses uniform $$t \sim \mathcal{U}\{1,\ldots,T\}$$ with
            optional importance sampling on hard timesteps.

            ### 4.3 Bridging diffusion and flow training

            Given score $$s_\theta(\mathbf{x}, t) \approx \nabla_{\mathbf{x}} \log p_t(\mathbf{x})$$, the probability-flow velocity can be written

            $$
            \mathbf{v}^*(\mathbf{x}, t) = \mathbf{f}(\mathbf{x}, t) - \tfrac{1}{2} g(t)^2 s_\theta(\mathbf{x}, t),
            $$

            allowing a single network to support both DSM pretraining and flow-matching fine-tuning on rectified paths.

            ![Discrete vs continuous time](/assets/figures/day07/pdf0_page016.png)
            *Figure: timestep schedules and signal-to-noise ratio.*

            ## Checkpoint summary

            - **Forward process** defines $$q(\mathbf{x}_t\mid\mathbf{x}_0)$$ and noisy scores.
            - **Score matching** trains \(\nabla_{\mathbf{x}} \log p_t(\mathbf{x})\) without normalized densities.
            - **Flow matching** trains velocity fields along explicit paths between prior and data.
            - Diffusion training is the discrete-time score-matching special case of this broader picture.
            """
        ).strip(),
    ),
    (
        8,
        "2026-08-26",
        "day08-diffusion-flow-inference",
        "Inference for Diffusion and Flow Models",
        "Reverse SDEs, probability-flow ODEs, Euler–Maruyama, and classifier-free guidance.",
        [
            "[Song et al. — Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456)",
            "[Ho et al. — Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)",
            "[Ho & Salimans — Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)",
        ],
        dedent(
            r"""
            Training gives a score or velocity field; **inference** integrates dynamics backward in time to produce samples.
            We compare **stochastic** (SDE) and **deterministic** (ODE) samplers and how guidance steers generation.

            ## 1. Reverse SDE

            > **Time reversal (informal).** If forward dynamics add noise according to an SDE, the reverse-time SDE
            > removes noise using the **score** of the marginal \(p_t(\mathbf{x})\).
            {:.lead}

            For a forward Itô SDE

            $$
            d\mathbf{x}_t = \mathbf{f}(\mathbf{x}_t, t)\,dt + g(t)\,d\mathbf{w}_t,
            $$

            the reverse SDE (Anderson, 1982) involves \(\nabla_{\mathbf{x}} \log p_t(\mathbf{x}_t)\):

            $$
            d\mathbf{x}_t = \left[\mathbf{f}(\mathbf{x}_t, t) - g(t)^2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x}_t)\right] dt + g(t)\,d\bar{\mathbf{w}}_t,
            $$

            run from \(t=T\) down to \(0\). In practice \(s_\theta(\mathbf{x}, t) \approx \nabla_{\mathbf{x}} \log p_t(\mathbf{x})\).

            ![Reverse-time dynamics](/assets/figures/day08/pdf0_page025.png)
            *Figure: stochastic denoising trajectory.*

            ### 1.1 Discrete DDPM updates

            DDPM defines a Markov reverse kernel \(p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)\) parameterized via
            predicted noise. Each step is a Gaussian whose mean depends on \(s_\theta\) or \(\boldsymbol{\epsilon}_\theta\).

            ## 2. Probability flow ODE

            > **Probability flow ODE.** There exists an ODE with the **same marginals** \(p_t\) as the forward SDE but
            > without injected noise; it uses the score to cancel the diffusion term.
            {:.lead}

            A common form (VP parameterization):

            $$
            \frac{d\mathbf{x}_t}{dt} = \mathbf{f}(\mathbf{x}_t, t) - \tfrac{1}{2} g(t)^2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x}_t).
            $$

            **Deterministic sampling** integrates this ODE from noise to data—often with fewer steps than the SDE when using
            high-order solvers (Heun, DPM-Solver, etc.).

            ![ODE vs SDE marginals](/assets/figures/day08/pdf1_page020.png)
            *Figure: same \(p_t\), different sample paths.*

            ### 2.1 When to prefer ODE vs SDE

            - **ODE:** reproducible samples, fast solvers, good FID with few steps.
            - **SDE:** extra stochasticity can improve diversity; closer to training noise process.

            ## 3. Euler–Maruyama discretization

            To simulate

            $$
            d\mathbf{x}_t = \mathbf{a}(\mathbf{x}_t, t)\,dt + g(t)\,d\mathbf{w}_t
            $$

            on a grid \(0 = t_0 < t_1 < \cdots < t_K = T\), **Euler–Maruyama** uses

            $$
            \mathbf{x}_{n+1} = \mathbf{x}_n + \mathbf{a}(\mathbf{x}_n, t_n)\,\Delta t_n + g(t_n)\sqrt{\Delta t_n}\,\boldsymbol{\zeta}_n,
            \qquad \boldsymbol{\zeta}_n \sim \mathcal{N}(\mathbf{0}, \mathbf{I}).
            $$

            For the **reverse** SDE, replace \(\nabla_{\mathbf{x}} \log p_t\) with \(s_\theta(\mathbf{x}, t)\) and step **backward**
            (\(\Delta t < 0\)). Step size controls quality–speed trade-off.

            ![Numerical integration](/assets/figures/day08/pdf1_page040.png)
            *Figure: discretization error vs step count.*

            ### 3.1 Flow-model inference

            Given learned \(\mathbf{v}_\theta(\mathbf{x}, t)\), integrate

            $$
            \mathbf{x}_{n+1} = \mathbf{x}_n + \mathbf{v}_\theta(\mathbf{x}_n, t_n)\,\Delta t_n
            $$

            from \(t=0\) (prior) to \(t=1\) (data). Same integrators apply; no Brownian term unless you add stochasticity.

            ## 4. Classifier-free guidance

            Conditional models use \(s_\theta(\mathbf{x}, t, y)\) or \(\boldsymbol{\epsilon}_\theta(\mathbf{x}, t, y)\).
            **Classifier guidance** perturbs scores with \(\nabla_{\mathbf{x}} \log p(y\mid\mathbf{x}_t)\)—requires a separate classifier.

            **Classifier-free guidance (CFG)** trains with random dropout of \(y\), then at sample time mixes conditional and unconditional predictions:

            $$
            \tilde{\boldsymbol{\epsilon}}_\theta(\mathbf{x}, t, y) =
            \boldsymbol{\epsilon}_\theta(\mathbf{x}, t, \varnothing)
            + w\left(\boldsymbol{\epsilon}_\theta(\mathbf{x}, t, y) - \boldsymbol{\epsilon}_\theta(\mathbf{x}, t, \varnothing)\right),
            $$

            where \(w \ge 1\) is the **guidance scale**. Larger \(w\) increases adherence to \(y\) but can reduce diversity or stability.

            ![Guidance effect](/assets/figures/day08/pdf1_page060.png)
            *Figure: text/image conditioning strength.*

            ### 4.1 Practical guidance tips

            - Train with 10–20% null labels for $$\varnothing$$.
            - Tune $$w$$ per task; too high → oversaturated or mode-collapsed outputs.
            - CFG applies equally to flow velocities by mixing $$\mathbf{v}_\theta(\mathbf{x}, t, y)$$.

            ### 4.2 Step schedulers and distillation

            **DDIM** and related samplers skip Markovian noise injection by integrating a non-Markovian generative process
            consistent with the same training objective—often 10–50 steps instead of hundreds.

            **Consistency models** and **progressive distillation** train a student that maps $$\mathbf{x}_t \to \mathbf{x}_{t-\Delta}$$
            in one shot, amortizing solver cost at deployment.

            ### 4.3 Error analysis for Euler–Maruyama

            For step size $$\Delta t$$, local weak error is $$\mathcal{O}(\Delta t)$$ under smooth coefficients; global error accumulates
            over $$K = T/\Delta t$$ steps. Adaptive solvers monitor drift norms and shrink $$\Delta t$$ when score magnitudes spike—common
            near $$t \approx 0$$ where data detail is recovered.

            ![Sampler comparison](/assets/figures/day08/pdf0_page035.png)
            *Figure: SDE, ODE, and distilled few-step samplers.*

            ## Checkpoint summary

            - **Reverse SDE** = forward drift correction minus score term + noise.
            - **Probability flow ODE** shares marginals with deterministic integration.
            - **Euler–Maruyama** discretizes SDEs; flow models use the ODE limit.
            - **CFG** trades off fidelity to conditioning vs sample diversity via scale \(w\).
            """
        ).strip(),
    ),
    (
        9,
        "2026-08-27",
        "day09-autoregressive-llms",
        "Autoregressive Language Models",
        "Encoder–decoder vs decoder-only stacks, training with cross-entropy, RoPE, and the transformer block.",
        [
            "[Vaswani et al. — Attention Is All You Need](https://arxiv.org/abs/1706.03762)",
            "[Gordić — Inside the Transformer: The Life of a Token](https://www.aleksagordic.com/blog/transformer)",
            "[Su et al. — RoFormer / RoPE](https://arxiv.org/abs/2104.09864)",
        ],
        dedent(
            r"""
            Large language models are **autoregressive**: they factorize sequences with causal conditioning. We map architectural
            families, the standard training loop, and the mathematical core of modern decoder-only transformers.

            ## 1. Model families: encoder, decoder, and hybrids

            > **Autoregressive factorization.**
            > \(p(\mathbf{x}) = \prod_{i=1}^{L} p(x_i \mid x_{<i})\) with causal masking so token \(i\) never sees the future.
            {:.lead}

            | Architecture | Attention | Typical use |
            |--------------|-----------|-------------|
            | **Encoder-only** | Bidirectional | Classification, embeddings (BERT) |
            | **Decoder-only** | Causal | GPT-style LMs, chat models |
            | **Encoder–decoder** | Cross + causal | Translation, summarization (T5) |

            **Encoder–decoder** maps input tokens to memory with a bidirectional encoder; the decoder attends to memory with
            cross-attention and to past outputs with a causal mask.

            **Decoder-only** treats prompt and completion as one sequence—today's default for general-purpose LLMs.

            ![Architecture families](/assets/figures/day09/pdf0_page025.png)
            *Figure: where information flows in each stack.*

            ### 1.1 Causal mask

            For positions \(i, j\), attention logits satisfy \(A_{ij} = -\infty\) when \(j > i\). After softmax, token \(i\)
            only aggregates keys/values from positions \(\le i\).

            ## 2. Training loop and cross-entropy

            Given tokenized sequence \(\mathbf{x} = (x_1, \ldots, x_L)\), the model outputs logits \(\mathbf{z}_i \in \mathbb{R}^{|\mathcal{V}|}\)
            for each position. **Next-token prediction** maximizes

            $$
            \mathcal{L}(\theta) = -\sum_{i=1}^{L} \log p_\theta(x_i \mid x_{<i})
            = -\sum_{i=1}^{L} \log \mathrm{softmax}(\mathbf{z}_i)_{x_i}.
            $$

            This is **multiclass cross-entropy** averaged over non-masked positions (padding masked out in the loss).

            > **Teacher forcing.** During training, the model always conditions on ground-truth prefixes \(x_{<i}\),
            > not its own previous predictions—stable gradients, train/inference mismatch handled at decode time (day 10).
            {:.lead}

            ![Training batch](/assets/figures/day09/pdf1_page020.png)
            *Figure: packed sequences and label shift by one.*

            ### 2.1 Optimization stack

            - AdamW with weight decay on matrices (not biases/LayerNorm).
            - Learning-rate warmup + cosine decay.
            - Gradient clipping and mixed-precision (bf16/fp16).

            ## 3. RoPE positional embeddings

            **Sinusoidal** absolute positions add to embeddings; **RoPE (rotary position embedding)** encodes relative position
            in attention by rotating query/key pairs in 2D subspaces.

            For head dimension pairs \((2k, 2k+1)\) and position \(m\),

            $$
            \mathrm{RoPE}(\mathbf{q}, m) =
            \begin{pmatrix}
            \cos m\theta_k & -\sin m\theta_k \\
            \sin m\theta_k & \cos m\theta_k
            \end{pmatrix}
            \begin{pmatrix} q_{2k} \\ q_{2k+1} \end{pmatrix},
            \qquad \theta_k = 10000^{-2k/d_{\mathrm{head}}}.
            $$

            Attention score \(\langle \mathrm{RoPE}(\mathbf{q}, m), \mathrm{RoPE}(\mathbf{k}, n)\rangle\) depends on \(m-n\),
            improving length extrapolation (YaRN scales frequencies for very long contexts).

            ![RoPE intuition](/assets/figures/day09/pdf1_page030.png)
            *Figure: relative position via rotation.*

            ## 4. Transformer block internals

            A **pre-norm** decoder block (schematic):

            $$
            \mathbf{h}' = \mathbf{h} + \mathrm{MHA}(\mathrm{LN}(\mathbf{h})),\qquad
            \mathbf{h}'' = \mathbf{h}' + \mathrm{MLP}(\mathrm{LN}(\mathbf{h}')).
            $$

            **Multi-head attention (MHA):**

            $$
            \mathrm{Attention}(Q,K,V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_{\mathrm{head}}}}\right) V,
            $$

            with \(Q = XW_Q\), \(K = XW_K\), \(V = XW_V\), split into heads. **GQA/MQA** share key/value heads to cut memory at inference.

            **MLP (SwiGLU / GeGLU):** gated feed-forward expands dimension (e.g. \(4\times\)) then projects back.

            ### 4.1 Life of a token (forward pass)

            1. Embed token IDs → vectors.
            2. For each layer: causal self-attention + MLP with residuals.
            3. Final linear **lm_head** → logits; softmax for loss or sampling.

            Stack depth $$L$$, hidden size $$d_{\mathrm{model}}$$, and head count set capacity; FLOPs scale roughly $$\mathcal{O}(L\, d_{\mathrm{model}}^2)$$ per token.

            ### 4.2 Parameter and FLOP scaling (sketch)

            Per layer, attention matrices contribute $$\approx 4 d_{\mathrm{model}}^2$$ parameters (Q,K,V,O projections) and
            MLP another $$\approx 8 d_{\mathrm{model}}^2$$ with expansion ratio 4. Total parameters scale
            $$\mathcal{O}(L_{\mathrm{layers}}\, d_{\mathrm{model}}^2)$$—the basis for Chinchilla-style compute–data trade-offs.

            ### 4.3 Tokenization and packing

            Subword tokenizers (BPE, SentencePiece) map bytes/characters to a vocabulary $$\mathcal{V}$$.
            **Document packing** concatenates multiple examples with attention masks so padding waste drops in training batches.

            ![Transformer block](/assets/figures/day09/pdf0_page035.png)
            *Figure: residual stream through attention and MLP.*

            ## Checkpoint summary

            - **Decoder-only** LMs dominate general text generation; encoder–decoder remains strong for fixed input→output tasks.
            - Training = sum of cross-entropies on next tokens with causal masking.
            - **RoPE** bakes relative position into attention via rotations.
            - A block = LN + attention + LN + MLP with residuals; depth stacks identical blocks with distinct weights.
            """
        ).strip(),
    ),
    (
        10,
        "2026-08-28",
        "day10-ar-inference",
        "Autoregressive Inference",
        "KV caching, temperature, nucleus (top-p) sampling, and batched decoding.",
        [
            "[Attention Is All You Need](https://arxiv.org/abs/1706.03762) — §3 (complexity)",
            "[Holtzman et al. — The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)",
            "[Gordić — KV cache discussion](https://www.aleksagordic.com/blog/transformer)",
        ],
        dedent(
            r"""
            Inference generates one token at a time. Efficiency hinges on **KV caching**, while **temperature** and **top-p**
            shape output quality. Production systems batch requests under memory and latency constraints.

            ## 1. Autoregressive decoding

            At step \(t\), we have prefix \(x_{\le t}\). The model outputs distribution \(p_\theta(x_{t+1}\mid x_{\le t})\).
            We sample or argmax \(x_{t+1}\), append, and repeat until EOS or max length.

            > **Exposure bias.** Training uses teacher forcing; inference feeds the model its own samples—errors compound.
            > Mitigations: scheduled sampling, distillation, RL fine-tuning (out of scope here).
            {:.lead}

            ![Decoding loop](/assets/figures/day10/pdf0_page000.png)
            *Figure: single-token extension per step.*

            ### 1.1 Complexity without cache

            Recomputing keys/values for all past tokens each step costs \(\mathcal{O}(t)\) attention per step and
            \(\mathcal{O}(L^2)\) over full length \(L\)—prohibitive for long contexts.

            ## 2. KV cache

            For each layer \(\ell\) and head, projections produce queries, keys, values. **Keys and values for past tokens are fixed**
            under causal attention, so we store them.

            > **KV cache memory (per layer, rough).**
            > \(2 \times L_{\mathrm{heads}} \times L_{\mathrm{seq}} \times d_{\mathrm{head}} \times \texttt{bytes\_per\_elem}\)
            > for keys plus values, times number of layers.
            {:.lead}

            Let \(C_{\mathrm{KV}}\) denote cached tensors. At step \(t\),

            $$
            K^{(\ell)} = \big[ K^{(\ell)}_{\mathrm{cache}} \;\|\; k^{(\ell)}_t \big], \qquad
            V^{(\ell)} = \big[ V^{(\ell)}_{\mathrm{cache}} \;\|\; v^{(\ell)}_t \big],
            $$

            and only \(q^{(\ell)}_t\) attends to the concatenated length-\(t\) sequence. Per-step cost becomes \(\mathcal{O}(t)\)
            attention per layer instead of recomputing from scratch.

            ![KV cache growth](/assets/figures/day10/pdf0_page005.png)
            *Figure: cache size grows linearly with context.*

            ### 2.1 Multi-request batching

            Batching \(B\) sequences pads to a common length (or uses ragged/paged attention). Cache is indexed per sequence;
            **PagedAttention** stores KV blocks in non-contiguous pages to reduce fragmentation on GPUs.

            ## 3. Temperature and top-p sampling

            Logits \(\mathbf{z}\) become probabilities

            $$
            p_i = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}, \qquad \tau > 0 \;\text{(temperature)}.
            $$

            - \(\tau \to 0^+\): distribution sharpens → greedy / near-argmax behavior.
            - \(\tau = 1\): training-scale probabilities.
            - \(\tau > 1\): flatter, more random outputs.

            **Top-p (nucleus) sampling:** sort probabilities \(p_{(1)} \ge p_{(2)} \ge \cdots\) and keep the smallest set
            \(V_p \subset \mathcal{V}\) such that \(\sum_{i \in V_p} p_i \ge p_{\mathrm{cut}}\) (e.g. \(p_{\mathrm{cut}}=0.9\)).
            Renormalize and sample within \(V_p\). This adapts the support size to model confidence.

            ![Sampling trade-offs](/assets/figures/day10/pdf1_page010.png)
            *Figure: temperature vs diversity.*

            ### 3.1 Other decoding knobs

            - **Top-k:** restrict to \(k\) largest logits before softmax.
            - **Repetition penalty:** down-weight logits of tokens already generated.
            - **Stop sequences:** user-defined EOS strings.

            ## 4. Batching, throughput, and serving

            **Static batching** waits until \(B\) requests fill—simple but increases latency.

            **Continuous batching** admits new prompts as others finish generations; improves GPU utilization in serving systems.

            Metrics:

            - **Time-to-first-token (TTFT):** prefill processes the prompt (large matmul, fills KV cache).
            - **Inter-token latency:** decode steps with cache—memory-bandwidth bound.

            $$
            \text{Throughput} \approx \frac{B}{\text{latency per step}} \quad\text{(tokens/s, rough)}.
            $$

            Quantization (INT8/INT4 weights, FP8 KV) trades accuracy for larger batch or longer context within the same VRAM.

            ![Serving stack](/assets/figures/day10/pdf1_page000.png)
            *Figure: prefill vs decode phases.*

            ### 4.1 Practical checklist

            1. Enable KV cache for all decode paths.
            2. Tune $$\tau$$ and top-p jointly on a validation prompt set.
            3. Cap `max_new_tokens` and monitor cache memory $$\propto L \times d_{\mathrm{model}} \times L_{\mathrm{layers}}$$.

            ### 4.2 Speculative decoding

            A **draft model** (small, fast) proposes $$k$$ tokens; the **target model** verifies them in parallel with one forward pass.
            Accepted prefixes advance without $$k$$ serial steps—latency drops when draft and target distributions align.

            ### 4.3 Beam search (brief)

            Beam search keeps $$B$$ highest-probability partial hypotheses. Score accumulation uses length normalization

            $$
            \frac{1}{t^\alpha} \sum_{i=1}^{t} \log p_\theta(x_i \mid x_{<i}),
            $$

            with $$\alpha \in [0,1]$$ to penalize overly short outputs. Used in translation; less common in open-ended chat where sampling dominates.

            ![Batched inference](/assets/figures/day10/pdf0_page010.png)
            *Figure: prefill batch vs decode batch on GPU.*

            ## Checkpoint summary

            - **Autoregressive inference** extends the sequence one token at a time.
            - **KV cache** avoids recomputing past keys/values; memory grows with context length.
            - **Temperature** and **top-p** control randomness vs coherence.
            - **Batching + paging** maximize hardware utilization in real deployments.
            """
        ).strip(),
    ),
]


def front_matter(day: int, title: str, description: str) -> str:
    return f"""---
layout: post
title: Day {day} - {title}
image: /assets/img/sampling_space.png
accent_image: 
  background: url('/assets/img/sampling_space.png') center/cover
  overlay: false
accent_color: '#ccc'
theme_color: '#ccc'
description: >
  {description}
invert_sidebar: true
---
"""


def body_header(day: int, title: str, reading: list[str]) -> str:
    nn = f"{day:02d}"
    lines = [
        f"# Day {day} - {title}",
        "",
        f"### [Slides](/assets/slides/day{nn}.pdf)",
        "",
        f"### [Practical](/projects/day{nn}-practical/)",
        "",
        "### Optional reading for this lesson",
    ]
    lines.extend(f"- {r}" for r in reading)
    lines.extend(["", "* toc", "{:toc}", "", ""])
    return "\n".join(lines)


def normalize_math(text: str) -> str:
    """Use $$ delimiters for KaTeX (Hydejack convention)."""
    return re.sub(r"\\\((.+?)\\\)", r"$$\1$$", text, flags=re.DOTALL)


def write_post(day: int, date: str, slug: str, title: str, description: str, reading: list[str], body: str) -> Path:
    POSTS.mkdir(parents=True, exist_ok=True)
    path = POSTS / f"{date}-{slug}.md"
    body = normalize_math(body)
    content = front_matter(day, title, description) + "\n" + body_header(day, title, reading) + body + "\n"
    path.write_text(content, encoding="utf-8")
    return path


def main() -> None:
    written: list[Path] = []
    for day, date, slug, title, description, reading, body in LECTURES:
        path = write_post(day, date, slug, title, description, reading, body)
        n_lines = len(path.read_text(encoding="utf-8").splitlines())
        print(f"Wrote {path.relative_to(ROOT)} ({n_lines} lines)")
        written.append(path)
    print(f"\nGenerated {len(written)} posts in {POSTS.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
