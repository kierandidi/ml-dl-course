"""Optional SDE / diffusion derivations for Days 7–8.

Notation matches the main course (Days 6–7, *Principles of Diffusion Models*):

    x_t = alpha_t x_0 + sigma_t eps
    s(x,t) = grad_x log p_t(x)  ~  s_theta(x,t)
    forward SDE:  dx = f(x,t) dt + g(t) dw
    reverse SDE:  dx = [f - g^2 s] dt + g d w_bar
    PF-ODE:       dx = [f - 1/2 g^2 s] dt

VP-SDE (DDPM): f = -1/2 beta(t) x,  g = sqrt(beta(t))  <->  alpha_t, sigma_t.
"""

# ---------------------------------------------------------------------------
# Day 7 — course derivations (expanded, step-by-step)
# ---------------------------------------------------------------------------

OPT_REVERSE_SDE_HEURISTIC = """**Setup (Principles / Day 6–7 notation).** Forward SDE

$$\\mathrm{d}\\boldsymbol{x} = \\boldsymbol{f}(\\boldsymbol{x},t)\\,\\mathrm{d}t + g(t)\\,\\mathrm{d}\\boldsymbol{w}, \\qquad \\boldsymbol{s}(\\boldsymbol{x},t)=\\nabla_{\\boldsymbol{x}}\\log p_t(\\boldsymbol{x}).$$

Discretise with Euler–Maruyama over $$\\delta>0$$: if $$\\boldsymbol{x}$$ is the state at time $$t$$ and $$\\boldsymbol{y}$$ at $$t+\\delta$$,

$$\\boldsymbol{y} \\approx \\boldsymbol{x} + \\boldsymbol{f}(\\boldsymbol{x},t)\\,\\delta + g(t)\\sqrt{\\delta}\\,\\boldsymbol{\\epsilon}, \\qquad \\boldsymbol{\\epsilon}\\sim\\mathcal{N}(\\mathbf{0},I).$$

Hence the **forward transition kernel** is Gaussian:

$$p_{t+\\delta\\mid t}(\\boldsymbol{y}\\mid\\boldsymbol{x}) = \\mathcal{N}\\!\\Big(\\boldsymbol{y}\\;\\Big|\\;\\boldsymbol{x}+\\boldsymbol{f}(\\boldsymbol{x},t)\\,\\delta,\\;\\delta\\,g(t)^2 I\\Big).$$

**Step 1 — chain rule.** For exact transition densities,

$$p_{t,t+\\delta}(\\boldsymbol{x},\\boldsymbol{y}) = p_{t+\\delta\\mid t}(\\boldsymbol{y}\\mid\\boldsymbol{x})\\,p_t(\\boldsymbol{x}) = p_{t\\mid t+\\delta}(\\boldsymbol{x}\\mid\\boldsymbol{y})\\,p_{t+\\delta}(\\boldsymbol{y}).$$

Rearranging,

$$p_{t\\mid t+\\delta}(\\boldsymbol{x}\\mid\\boldsymbol{y}) = p_{t+\\delta\\mid t}(\\boldsymbol{y}\\mid\\boldsymbol{x})\\,\\frac{p_t(\\boldsymbol{x})}{p_{t+\\delta}(\\boldsymbol{y})}.$$

**Step 2 — Taylor expansion of the log-density.** Expand $$\\log p_t$$ around $$\\boldsymbol{y}$$ (Winkler / SDE course §2.1):

$$\\log p_t(\\boldsymbol{x}) - \\log p_{t+\\delta}(\\boldsymbol{y}) = (\\boldsymbol{x}-\\boldsymbol{y})^{\\top}\\nabla_{\\boldsymbol{y}}\\log p_t(\\boldsymbol{y}) + \\mathcal{O}(\\|\\boldsymbol{x}-\\boldsymbol{y}\\|^2).$$

Under a standard Lipschitz bound on $$\\log p_t - \\log p_{t+\\delta}$$, the $$p_{t+\\delta}(\\boldsymbol{y})$$ factor in the denominator can be absorbed into the remainder as $$\\mathcal{O}(\\delta^2)$$ when $$\\delta\\to 0$$. Thus

$$\\frac{p_t(\\boldsymbol{x})}{p_{t+\\delta}(\\boldsymbol{y})} \\approx \\exp\\big((\\boldsymbol{x}-\\boldsymbol{y})^{\\top}\\boldsymbol{s}(\\boldsymbol{y},t) + \\mathcal{O}(\\delta^2)\\big).$$

**Step 3 — complete the square.** Multiply the forward Gaussian kernel by this exponential. Writing $$\\boldsymbol{\\mu}_+ = \\boldsymbol{x}+\\boldsymbol{f}(\\boldsymbol{x},t)\\delta$$ and $$\\sigma^2 = \\delta\\,g(t)^2$$,

$$p_{t\\mid t+\\delta}(\\boldsymbol{x}\\mid\\boldsymbol{y}) \\propto \\exp\\!\\Big(-\\frac{\\|\\boldsymbol{x}-\\boldsymbol{\\mu}_+\\|^2}{2\\sigma^2} + (\\boldsymbol{x}-\\boldsymbol{y})^{\\top}\\boldsymbol{s}(\\boldsymbol{y},t)\\Big).$$

Complete the square in $$\\boldsymbol{x}$$ (expand, collect linear and quadratic terms). The linear term picks up a contribution from $$\\boldsymbol{s}$$; the resulting Gaussian has mean shifted by $$+\\,g(t)^2\\,\\boldsymbol{s}(\\boldsymbol{y},t)\\,\\delta$$ relative to the naive time-reversal of the forward drift.

**Step 4 — continuous-time limit.** Reversing time ($$t\\mapsto T-t$$) and taking $$\\delta\\to 0$$ yields the **reverse-time SDE**

$$\\mathrm{d}\\boldsymbol{x} = \\big[\\boldsymbol{f}(\\boldsymbol{x},t) - g(t)^2\\,\\boldsymbol{s}(\\boldsymbol{x},t)\\big]\\,\\mathrm{d}t + g(t)\\,\\mathrm{d}\\bar{\\boldsymbol{w}}.$$

The score appears because reversing a diffusion requires knowing how density changes in space — the quantity we train with denoising score matching. See [Winkler's sketch](https://ludwigwinkler.github.io/blog/SimpleReverseSDE/) and Anderson (1982) for the rigorous treatment below."""

OPT_ANDERSON_REVERSE = """**Goal.** Show that the time-reversal of

$$\\mathrm{d}\\boldsymbol{X}_t = \\boldsymbol{f}(\\boldsymbol{X}_t,t)\\,\\mathrm{d}t + g(t)\\,\\mathrm{d}\\boldsymbol{w}$$

is

$$\\mathrm{d}\\boldsymbol{X}_t = \\big[\\boldsymbol{f}(\\boldsymbol{X}_t,t) - g(t)^2\\,\\boldsymbol{s}(\\boldsymbol{X}_t,t)\\big]\\,\\mathrm{d}t + g(t)\\,\\mathrm{d}\\bar{\\boldsymbol{w}},$$

with the **same marginals** $$\\{p_t\\}$$ when run backward from $$t=T$$ to $$0$$.

**Step 1 — joint density.** Write the joint $$p_{s,t}(\\boldsymbol{x}_s,\\boldsymbol{x}_t)=p_{s\\mid t}(\\boldsymbol{x}_s\\mid\\boldsymbol{x}_t)\\,p_t(\\boldsymbol{x}_t)$$ for $$s<t$$.

**Step 2 — forward Kolmogorov equation.** The transition density $$p_{s\\mid t}(\\boldsymbol{x}_s\\mid\\boldsymbol{x}_t)$$ (as a function of $$\\boldsymbol{x}_t,t$$) satisfies the **backward Kolmogorov** equation, which encodes the forward SDE drift $$\\boldsymbol{f}$$ and diffusion $$g$$.

**Step 3 — differentiate the joint in $$t$$.** Using $$\\partial_t p_t = \\partial_t[p_{s\\mid t}\\,p_t]$$ and substituting the backward equation for $$p_{s\\mid t}$$ plus the **Fokker–Planck** equation for $$p_t$$,

$$\\partial_t p_{s,t} = \\int \\Big[\\text{(backward)}\\,p_{s\\mid t}\\,p_t + p_{s\\mid t}\\,\\text{(FPE)}\\,p_t\\Big]\\,\\mathrm{d}\\boldsymbol{x}_t.$$

**Step 4 — complete the square.** After integrating by parts in $$\\boldsymbol{x}_t$$, the terms involving $$\\nabla p_t$$ combine into a single **score** term $$g(t)^2\\nabla\\log p_t$$ multiplying $$p_{s,t}$$. The remaining terms match the FPE of an SDE with drift $$\\boldsymbol{f}-g^2\\boldsymbol{s}$$ and diffusion $$g$$.

**Step 5 — read off the reverse SDE.** The reverse-time process (with $$\\mathrm{d}\\bar{\\boldsymbol{w}}$$) has the stated drift. At sampling time replace $$\\boldsymbol{s}$$ with $$\\boldsymbol{s}_\\theta(\\boldsymbol{x},t)$$.

*Reference:* Anderson (1982); SDE course [Lesson 2 §2.2](https://kierandidi.github.io/); Song et al. (2021) Eq. (4.1.6)."""

OPT_DSM_MARGINAL = """**Claim (Proposition 4.3.1, Principles).** The minimizer of

$$J(\\theta)=\\mathbb{E}_{t,\\boldsymbol{x}_0,\\boldsymbol{\\epsilon}}\\Big\\|\\boldsymbol{s}_\\theta(\\boldsymbol{x}_t,t) - \\nabla_{\\boldsymbol{x}_t}\\log p_t(\\boldsymbol{x}_t\\mid\\boldsymbol{x}_0)\\Big\\|^2$$

satisfies $$\\boldsymbol{s}_\\theta^\\star(\\boldsymbol{x}_t,t)=\\nabla_{\\boldsymbol{x}_t}\\log p_t(\\boldsymbol{x}_t)$$ for a.e. $$\\boldsymbol{x}_t\\sim p_t$$.

**Step 1 — regression identity.** For any target $$\\boldsymbol{g}(\\boldsymbol{x}_t,\\boldsymbol{x}_0)$$,

$$\\arg\\min_{\\boldsymbol{s}}\\,\\mathbb{E}\\|\\boldsymbol{s}(\\boldsymbol{x}_t,t)-\\boldsymbol{g}\\|^2 = \\mathbb{E}[\\boldsymbol{g}\\mid\\boldsymbol{x}_t].$$

Set $$\\boldsymbol{g}=\\nabla_{\\boldsymbol{x}_t}\\log p_t(\\boldsymbol{x}_t\\mid\\boldsymbol{x}_0)$$.

**Step 2 — posterior average.** Write $$p_t(\\boldsymbol{x}_t)=\\int p_t(\\boldsymbol{x}_t\\mid\\boldsymbol{x}_0)p_{\\text{data}}(\\boldsymbol{x}_0)\\,\\mathrm{d}\\boldsymbol{x}_0$$ and differentiate:

$$\\nabla_{\\boldsymbol{x}_t}\\log p_t(\\boldsymbol{x}_t) = \\frac{\\int \\nabla_{\\boldsymbol{x}_t}p_t(\\boldsymbol{x}_t\\mid\\boldsymbol{x}_0)\\,p_{\\text{data}}(\\boldsymbol{x}_0)\\,\\mathrm{d}\\boldsymbol{x}_0}{p_t(\\boldsymbol{x}_t)}.$$

Use $$\\nabla p_t(\\cdot\\mid\\boldsymbol{x}_0)=p_t(\\cdot\\mid\\boldsymbol{x}_0)\\,\\nabla\\log p_t(\\cdot\\mid\\boldsymbol{x}_0)$$ and recognise $$p_t(\\boldsymbol{x}_0\\mid\\boldsymbol{x}_t)$$:

$$\\nabla_{\\boldsymbol{x}_t}\\log p_t(\\boldsymbol{x}_t) = \\mathbb{E}_{\\boldsymbol{x}_0\\mid\\boldsymbol{x}_t}\\big[\\nabla_{\\boldsymbol{x}_t}\\log p_t(\\boldsymbol{x}_t\\mid\\boldsymbol{x}_0)\\big].$$

**Step 3 — Gaussian conditional score.** With $$\\boldsymbol{x}_t=\\alpha_t\\boldsymbol{x}_0+\\sigma_t\\boldsymbol{\\epsilon}$$,

$$\\nabla_{\\boldsymbol{x}_t}\\log p_t(\\boldsymbol{x}_t\\mid\\boldsymbol{x}_0) = -\\frac{\\boldsymbol{x}_t-\\alpha_t\\boldsymbol{x}_0}{\\sigma_t^2} = -\\frac{\\boldsymbol{\\epsilon}}{\\sigma_t}.$$

**Conclusion.** Regressing on the known conditional score trains the **marginal** score; predicting $$\\boldsymbol{s}_\\theta\\approx -\\boldsymbol{\\epsilon}/\\sigma_t$$ is predicting the noise (Day 6). Related: **Tweedie's formula** $$\\mathbb{E}[\\boldsymbol{x}_0\\mid\\boldsymbol{x}_t]=(\\boldsymbol{x}_t+\\sigma_t^2\\boldsymbol{s})/\\alpha_t$$."""

OPT_PF_ODE_FPE = """**Setup.** Forward SDE $$\\mathrm{d}\\boldsymbol{x}=\\boldsymbol{f}(\\boldsymbol{x},t)\\,\\mathrm{d}t+g(t)\\,\\mathrm{d}\\boldsymbol{w}$$ with $$\\boldsymbol{s}=\\nabla_{\\boldsymbol{x}}\\log p_t$$.

**Step 1 — Fokker–Planck (Appendix B.1.3).**

$$\\partial_t p_t = -\\nabla\\cdot(\\boldsymbol{f}\\,p_t) + \\tfrac12 g(t)^2 \\Delta p_t.$$

**Step 2 — product rule on the diffusion term.**

$$\\partial_t p_t = -\\nabla\\cdot(\\boldsymbol{f}\\,p_t) + \\tfrac12 g(t)^2\\nabla\\cdot\\nabla p_t + \\tfrac12 \\nabla(g(t)^2)\\cdot\\nabla p_t.$$

For state-independent $$g(t)$$, the last term vanishes.

**Step 3 — log-derivative trick.** Use $$\\nabla p_t = p_t\\,\\boldsymbol{s}$$:

$$\\tfrac12 g(t)^2 \\Delta p_t = \\tfrac12 g(t)^2 \\nabla\\cdot(p_t\\,\\boldsymbol{s}) = \\nabla\\cdot\\Big(\\tfrac12 g(t)^2\\,p_t\\,\\boldsymbol{s}\\Big) - \\tfrac12 g(t)^2\\,p_t\\,\\nabla\\cdot\\boldsymbol{s}.$$

The $$\\nabla\\cdot\\boldsymbol{s}$$ term cancels when combining all pieces (Principles D.2.6); equivalently, expand $$\\nabla\\cdot(\\boldsymbol{f}p_t - \\tfrac12 g^2 p_t \\boldsymbol{s})$$ directly.

**Step 4 — Liouville form.** Factor out $$p_t$$:

$$\\partial_t p_t = -\\nabla\\cdot\\Big(p_t\\,\\underbrace{\\Big(\\boldsymbol{f} - \\tfrac12 g(t)^2\\,\\boldsymbol{s}\\Big)}_{\\tilde{\\boldsymbol{\\mu}}(\\boldsymbol{x},t)}\\Big).$$

This is the FPE of an SDE with **zero diffusion** — the **Liouville equation** — hence the density is transported by the ODE

$$\\mathrm{d}\\boldsymbol{x} = \\tilde{\\boldsymbol{\\mu}}(\\boldsymbol{x},t)\\,\\mathrm{d}t = \\big[\\boldsymbol{f} - \\tfrac12 g(t)^2\\,\\boldsymbol{s}(\\boldsymbol{x},t)\\big]\\,\\mathrm{d}t.$$

**Compare reverse SDE:** drift $$\\boldsymbol{f}-g^2\\boldsymbol{s}$$ (full score coefficient) plus noise $$g\\,\\mathrm{d}\\bar{\\boldsymbol{w}}$$. Same $$\\{p_t\\}$$, different sample paths.

*Reference:* Principles Appendix B.1.3, D.2.6; SDE course Lesson 2 §4; Song et al. (2021) Eq. (4.1.7)."""

# ---------------------------------------------------------------------------
# Principles book appendices (condensed for students)
# ---------------------------------------------------------------------------

OPT_APPENDIX_B_DENSITY = r"""*Expanded from [Principles of Diffusion Models](https://arxiv.org/abs/2510.21890), Appendix B. Throughout, the forward SDE is $$\mathrm{d}\boldsymbol{x}=\boldsymbol{f}(\boldsymbol{x},t)\,\mathrm{d}t+g(t)\,\mathrm{d}\boldsymbol{w}$$ and the score is $$\boldsymbol{s}(\boldsymbol{x},t)=\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x})$$.*

**B.0 — The one idea: conservation of probability mass.** Every law below is the same statement — *probability is neither created nor destroyed, only transported* — written at three levels of generality: a single map (change of variables), a deterministic flow (continuity equation), and a flow plus noise (Fokker–Planck). Each is derived from the previous one.

**B.1 — Change of variables for one invertible map.** Let $$\Psi:\mathbb{R}^d\to\mathbb{R}^d$$ be a diffeomorphism and $$\boldsymbol{x}_1=\Psi(\boldsymbol{x}_0)$$ with $$\boldsymbol{x}_0\sim p_0$$. The mass in an infinitesimal box around $$\boldsymbol{x}_0$$ equals the mass in its image:

$$p_0(\boldsymbol{x}_0)\,\mathrm{d}\boldsymbol{x}_0 = p_1(\boldsymbol{x}_1)\,\mathrm{d}\boldsymbol{x}_1 .$$

The volume element transforms by the Jacobian $$\mathbf{J}_\Psi$$, with $$(\mathbf{J}_\Psi)_{ij}=\partial\Psi_i/\partial x_{0,j}$$, through $$\mathrm{d}\boldsymbol{x}_1=\left\lvert\det\mathbf{J}_\Psi\right\rvert\,\mathrm{d}\boldsymbol{x}_0$$. Solving for $$p_1$$ and using the inverse-function theorem $$\det(\partial\Psi^{-1}/\partial\boldsymbol{x}_1)=1/\det\mathbf{J}_\Psi$$,

$$p_1(\boldsymbol{x}_1)=p_0\!\big(\Psi^{-1}(\boldsymbol{x}_1)\big)\,\left\lvert\det\frac{\partial\Psi^{-1}}{\partial\boldsymbol{x}_1}\right\rvert =\frac{p_0\!\big(\Psi^{-1}(\boldsymbol{x}_1)\big)}{\left\lvert\det\mathbf{J}_\Psi\!\big(\Psi^{-1}(\boldsymbol{x}_1)\big)\right\rvert}.$$

Taking logs, with $$\boldsymbol{x}_0=\Psi^{-1}(\boldsymbol{x}_1)$$,

$$\log p_1(\boldsymbol{x}_1)=\log p_0(\boldsymbol{x}_0)-\log\left\lvert\det\mathbf{J}_\Psi(\boldsymbol{x}_0)\right\rvert .$$

**B.2 — Composition of maps (normalizing flows).** For a stack $$\Psi=\Psi_K\circ\cdots\circ\Psi_1$$ with intermediate states $$\boldsymbol{x}_k=\Psi_k(\boldsymbol{x}_{k-1})$$, the determinant chain rule turns the product of Jacobians into a sum of log-determinants:

$$\log p_K(\boldsymbol{x}_K)=\log p_0(\boldsymbol{x}_0)-\sum_{k=1}^{K}\log\left\lvert\det\frac{\partial\Psi_k}{\partial\boldsymbol{x}_{k-1}}\right\rvert .$$

This is exactly the training objective of a **normalizing flow**: maximize the data log-likelihood $$\log p_K(\boldsymbol{x}_K)$$ by composing invertible layers with cheap log-determinants.

**B.3 — Continuous limit → the continuity equation.** Replace the discrete stack by a deterministic flow $$\dot{\boldsymbol{x}}=\boldsymbol{f}(\boldsymbol{x},t)$$, i.e. the near-identity map $$\Psi_\delta(\boldsymbol{x})=\boldsymbol{x}+\delta\,\boldsymbol{f}(\boldsymbol{x},t)$$ over a small step $$\delta$$. Its Jacobian is

$$\mathbf{J}_{\Psi_\delta}=I+\delta\,\nabla\boldsymbol{f}+\mathcal{O}(\delta^2),\qquad (\nabla\boldsymbol{f})_{ij}=\frac{\partial f_i}{\partial x_j}.$$

By Jacobi's formula $$\det(I+\delta A)=1+\delta\,\mathrm{tr}(A)+\mathcal{O}(\delta^2)$$,

$$\log\det\mathbf{J}_{\Psi_\delta}=\delta\,\nabla\!\cdot\!\boldsymbol{f}+\mathcal{O}(\delta^2).$$

Insert this into the log change-of-variables formula along a trajectory and divide by $$\delta\to0$$ (the **instantaneous change of variables**, a.k.a. the continuous-normalizing-flow trace identity):

$$\frac{\mathrm{d}}{\mathrm{d}t}\log p_t(\boldsymbol{x}_t)=-\nabla\!\cdot\!\boldsymbol{f}(\boldsymbol{x}_t,t).$$

Now expand the total (material) derivative $$\frac{\mathrm{d}}{\mathrm{d}t}\log p_t=\partial_t\log p_t+\boldsymbol{f}\!\cdot\!\nabla\log p_t$$, multiply through by $$p_t$$, and use the product rule $$p_t\,\nabla\!\cdot\!\boldsymbol{f}+\boldsymbol{f}\!\cdot\!\nabla p_t=\nabla\!\cdot\!(p_t\boldsymbol{f})$$:

$$\partial_t p_t+\nabla\!\cdot\!(p_t\,\boldsymbol{f})=0 \qquad\text{(continuity / transport equation).}$$

**B.4 — Add noise → the Fokker–Planck equation.** Now the step is stochastic, $$\boldsymbol{x}_{t+\delta}=\boldsymbol{x}_t+\delta\,\boldsymbol{f}+g\sqrt{\delta}\,\boldsymbol{\epsilon}$$ with $$\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},I)$$, so the transition kernel is Gaussian, $$p_{t+\delta\mid t}(\boldsymbol{y}\mid\boldsymbol{x})=\mathcal{N}(\boldsymbol{y};\,\boldsymbol{x}+\delta\boldsymbol{f},\,\delta g^2 I)$$. Test against a smooth, compactly-supported $$\phi$$ and use Chapman–Kolmogorov:

$$\mathbb{E}[\phi(\boldsymbol{x}_{t+\delta})]=\iint \phi(\boldsymbol{y})\,p_{t+\delta\mid t}(\boldsymbol{y}\mid\boldsymbol{x})\,p_t(\boldsymbol{x})\,\mathrm{d}\boldsymbol{y}\,\mathrm{d}\boldsymbol{x}.$$

Taylor-expand $$\phi(\boldsymbol{y})$$ about $$\boldsymbol{x}$$ and take the Gaussian moments $$\mathbb{E}[\boldsymbol{y}-\boldsymbol{x}]=\delta\boldsymbol{f}$$ and $$\mathbb{E}[(\boldsymbol{y}-\boldsymbol{x})(\boldsymbol{y}-\boldsymbol{x})^{\top}]=\delta g^2 I+\mathcal{O}(\delta^2)$$:

$$\mathbb{E}[\phi(\boldsymbol{x}_{t+\delta})]-\mathbb{E}[\phi(\boldsymbol{x}_t)]=\delta\,\mathbb{E}\big[\boldsymbol{f}\!\cdot\!\nabla\phi+\tfrac12 g^2\Delta\phi\big]+\mathcal{O}(\delta^2).$$

Divide by $$\delta\to0$$ and write both sides as integrals against $$p_t$$; on the left $$\frac{\mathrm{d}}{\mathrm{d}t}\mathbb{E}[\phi]=\int\phi\,\partial_t p_t$$. Integrate by parts (the boundary terms vanish) to move the derivatives off $$\phi$$ and onto $$p_t$$:

$$\int\phi\,\partial_t p_t=\int\phi\Big[-\nabla\!\cdot\!(\boldsymbol{f}p_t)+\tfrac12 g^2\Delta p_t\Big].$$

Since $$\phi$$ is arbitrary, the integrands match:

$$\partial_t p_t=-\nabla\!\cdot\!(\boldsymbol{f}\,p_t)+\tfrac12 g(t)^2\,\Delta p_t \qquad\text{(Fokker–Planck).}$$

**B.5 — Score / probability-flow form.** Using $$\nabla p_t=p_t\boldsymbol{s}$$, rewrite the diffusion term as a divergence, $$\tfrac12 g^2\Delta p_t=\tfrac12 g^2\nabla\!\cdot\!(p_t\boldsymbol{s})=\nabla\!\cdot\!\big(\tfrac12 g^2 p_t\boldsymbol{s}\big)$$, so the FPE collapses into a *noise-free* continuity equation:

$$\partial_t p_t=-\nabla\!\cdot\!\Big(\big(\boldsymbol{f}-\tfrac12 g^2\boldsymbol{s}\big)\,p_t\Big).$$

The bracket is the **probability-flow ODE** velocity (Appendix D.3): at the level of *marginals*, adding noise to the SDE is equivalent to subtracting half the score from the drift and dropping the noise."""

OPT_APPENDIX_C_ITO_GIRSANOV = r"""*Expanded from Principles Appendix C.*

**C.1 — Itô's formula (the stochastic chain rule).** Let $$h(\boldsymbol{x},t)$$ be smooth and $$\mathrm{d}\boldsymbol{x}=\boldsymbol{f}\,\mathrm{d}t+g\,\mathrm{d}\boldsymbol{w}$$. Taylor-expand to second order — *second order matters here*, because a Brownian increment scales like $$\sqrt{\mathrm{d}t}$$, not $$\mathrm{d}t$$:

$$\mathrm{d}h=\partial_t h\,\mathrm{d}t+\nabla h^{\top}\mathrm{d}\boldsymbol{x}+\tfrac12\,\mathrm{d}\boldsymbol{x}^{\top}(\nabla^2 h)\,\mathrm{d}\boldsymbol{x}+\cdots$$

Apply the Itô multiplication table $$\mathrm{d}t^2=0$$, $$\mathrm{d}t\,\mathrm{d}w_i=0$$, $$\mathrm{d}w_i\,\mathrm{d}w_j=\delta_{ij}\,\mathrm{d}t$$. The quadratic form keeps only the Brownian part, $$\mathrm{d}\boldsymbol{x}^{\top}(\nabla^2 h)\,\mathrm{d}\boldsymbol{x}=g^2\sum_i\partial_i^2 h\,\mathrm{d}t=g^2\Delta h\,\mathrm{d}t$$, giving

$$\mathrm{d}h=\Big(\partial_t h+\boldsymbol{f}\!\cdot\!\nabla h+\tfrac12 g^2\Delta h\Big)\mathrm{d}t+g\,\nabla h^{\top}\mathrm{d}\boldsymbol{w}.$$

The extra $$\tfrac12 g^2\Delta h$$ relative to ordinary calculus is the Itô correction.

**C.2 — Fokker–Planck from Itô (forward Kolmogorov).** Define the **generator** $$\mathcal{A}h=\boldsymbol{f}\!\cdot\!\nabla h+\tfrac12 g^2\Delta h$$. Taking expectations in C.1 annihilates the martingale $$\mathrm{d}\boldsymbol{w}$$ term:

$$\frac{\mathrm{d}}{\mathrm{d}t}\,\mathbb{E}[h(\boldsymbol{x}_t)]=\mathbb{E}[\mathcal{A}h]=\int (\mathcal{A}h)\,p_t\,\mathrm{d}\boldsymbol{x}.$$

But also $$\frac{\mathrm{d}}{\mathrm{d}t}\mathbb{E}[h]=\int h\,\partial_t p_t$$. Equate and integrate by parts to move $$\mathcal{A}$$ onto $$p_t$$ via its **adjoint** $$\mathcal{A}^{\dagger}$$:

$$\int h\,\partial_t p_t=\int h\,\mathcal{A}^{\dagger}p_t,\qquad \mathcal{A}^{\dagger}p=-\nabla\!\cdot\!(\boldsymbol{f}p)+\tfrac12 g^2\Delta p.$$

Since $$h$$ is arbitrary, $$\partial_t p_t=\mathcal{A}^{\dagger}p_t$$ — the Fokker–Planck equation again, now obtained independently of Appendix B.

**C.3 — Girsanov's theorem (reweighting paths).** Consider two diffusions with the **same** noise coefficient $$g$$ but different drifts: $$\boldsymbol{f}$$ under measure $$\mathbb{P}$$ and $$\boldsymbol{f}+g\,\boldsymbol{u}$$ under measure $$\mathbb{Q}$$. Girsanov's theorem states that the two path measures are mutually absolutely continuous, with Radon–Nikodym derivative (an exponential / Doléans-Dade martingale)

$$\frac{\mathrm{d}\mathbb{Q}}{\mathrm{d}\mathbb{P}}\bigg|_{[0,T]}=\exp\!\Big(\int_0^T\boldsymbol{u}^{\top}\mathrm{d}\boldsymbol{w}-\tfrac12\int_0^T\lVert\boldsymbol{u}\rVert^2\,\mathrm{d}t\Big),$$

valid under Novikov's condition $$\mathbb{E}\exp\!\big(\tfrac12\int_0^T\lVert\boldsymbol{u}\rVert^2\,\mathrm{d}t\big)<\infty$$. Equivalently, $$\tilde{\boldsymbol{w}}_t=\boldsymbol{w}_t-\int_0^t\boldsymbol{u}\,\mathrm{d}s$$ is a $$\mathbb{Q}$$-Brownian motion, so under $$\mathbb{Q}$$ the process has the shifted drift $$\boldsymbol{f}+g\boldsymbol{u}$$.

Taking $$\log$$ and expectation yields the **KL divergence between path measures**:

$$\mathrm{KL}(\mathbb{P}\,\Vert\,\mathbb{Q})=\mathbb{E}_{\mathbb{P}}\Big[\log\tfrac{\mathrm{d}\mathbb{P}}{\mathrm{d}\mathbb{Q}}\Big]=\tfrac12\,\mathbb{E}_{\mathbb{P}}\!\int_0^T\lVert\boldsymbol{u}\rVert^2\,\mathrm{d}t .$$

**Why this matters for diffusion.** Matching a model SDE (drift built from $$\boldsymbol{s}_\theta$$) to the true reverse SDE (drift built from $$\boldsymbol{s}$$) corresponds to $$g\boldsymbol{u}=g^2(\boldsymbol{s}-\boldsymbol{s}_\theta)$$, so the path-space KL becomes

$$\mathrm{KL}=\tfrac12\int_0^T g(t)^2\,\mathbb{E}\big\lVert\boldsymbol{s}(\boldsymbol{x}_t,t)-\boldsymbol{s}_\theta(\boldsymbol{x}_t,t)\big\rVert^2\,\mathrm{d}t .$$

In words: the **weighted score-matching loss is exactly the KL between the data and model generative processes** — the continuous-time ELBO for diffusion models (Song et al., 2021)."""

OPT_APPENDIX_D_PROOFS = r"""*Expanded from Principles Appendix D (Props. 3.2.1, 3.3.1 / 4.3.1, 4.1.1).*

**D.1 — Explicit ↔ implicit score matching (Hyvärinen, 2005).** We want $$\boldsymbol{s}_\theta\approx\nabla\log p$$, but the data density $$p=p_{\text{data}}$$ is unknown. The explicit objective is

$$J_{\mathrm{ESM}}(\theta)=\tfrac12\,\mathbb{E}_{p}\big\lVert\boldsymbol{s}_\theta(\boldsymbol{x})-\nabla\log p(\boldsymbol{x})\big\rVert^2 .$$

Expand the square; the $$\lVert\nabla\log p\rVert^2$$ term is a $$\theta$$-independent constant:

$$J_{\mathrm{ESM}}=\tfrac12\,\mathbb{E}_p\lVert\boldsymbol{s}_\theta\rVert^2-\mathbb{E}_p\big[\boldsymbol{s}_\theta^{\top}\nabla\log p\big]+\text{const}.$$

The cross term becomes tractable after one integration by parts. With $$\nabla\log p=\nabla p/p$$,

$$\mathbb{E}_p\big[\boldsymbol{s}_\theta^{\top}\nabla\log p\big]=\int p\,\boldsymbol{s}_\theta^{\top}\frac{\nabla p}{p}=\int\boldsymbol{s}_\theta^{\top}\nabla p=-\int p\,(\nabla\!\cdot\!\boldsymbol{s}_\theta)=-\mathbb{E}_p[\nabla\!\cdot\!\boldsymbol{s}_\theta],$$

assuming $$p\,\boldsymbol{s}_\theta\to\mathbf{0}$$ at infinity. Hence the **implicit** (data-score-free) objective is

$$J_{\mathrm{ISM}}(\theta)=\mathbb{E}_p\Big[\tfrac12\lVert\boldsymbol{s}_\theta(\boldsymbol{x})\rVert^2+\nabla\!\cdot\!\boldsymbol{s}_\theta(\boldsymbol{x})\Big]+\text{const}.$$

The divergence $$\nabla\!\cdot\!\boldsymbol{s}_\theta=\mathrm{tr}(\partial\boldsymbol{s}_\theta/\partial\boldsymbol{x})$$ needs the Jacobian trace — $$\mathcal{O}(d)$$ backward passes, prohibitive in high dimension. Removing that cost is the whole point of denoising score matching.

**D.2 — Denoising score matching (Vincent, 2011).** Perturb the data with a Gaussian kernel, $$\tilde{\boldsymbol{x}}=\boldsymbol{x}+\sigma\boldsymbol{\epsilon}$$, giving the noisy marginal $$p_\sigma(\tilde{\boldsymbol{x}})=\int p(\boldsymbol{x})\,\mathcal{N}(\tilde{\boldsymbol{x}};\boldsymbol{x},\sigma^2 I)\,\mathrm{d}\boldsymbol{x}$$. Define the **denoising** objective against the *conditional* (closed-form) score:

$$J_{\mathrm{DSM}}(\theta)=\tfrac12\,\mathbb{E}_{p(\boldsymbol{x})\,p_\sigma(\tilde{\boldsymbol{x}}\mid\boldsymbol{x})}\big\lVert\boldsymbol{s}_\theta(\tilde{\boldsymbol{x}})-\nabla_{\tilde{\boldsymbol{x}}}\log p_\sigma(\tilde{\boldsymbol{x}}\mid\boldsymbol{x})\big\rVert^2 .$$

*Claim:* $$J_{\mathrm{DSM}}=J_{\mathrm{ESM},\,p_\sigma}+\text{const}$$, so both share the minimizer $$\boldsymbol{s}_\theta^\star=\nabla\log p_\sigma$$. Only the cross term depends on $$\theta$$; push the gradient through the mixture defining $$p_\sigma$$:

$$\mathbb{E}\big[\boldsymbol{s}_\theta^{\top}\nabla_{\tilde{\boldsymbol{x}}}\log p_\sigma(\tilde{\boldsymbol{x}}\mid\boldsymbol{x})\big]=\int\!\!\int p(\boldsymbol{x})\,\boldsymbol{s}_\theta(\tilde{\boldsymbol{x}})^{\top}\nabla_{\tilde{\boldsymbol{x}}}p_\sigma(\tilde{\boldsymbol{x}}\mid\boldsymbol{x})\,\mathrm{d}\boldsymbol{x}\,\mathrm{d}\tilde{\boldsymbol{x}}$$

$$=\int\boldsymbol{s}_\theta(\tilde{\boldsymbol{x}})^{\top}\nabla_{\tilde{\boldsymbol{x}}}\Big(\underbrace{\int p(\boldsymbol{x})\,p_\sigma(\tilde{\boldsymbol{x}}\mid\boldsymbol{x})\,\mathrm{d}\boldsymbol{x}}_{=\,p_\sigma(\tilde{\boldsymbol{x}})}\Big)\,\mathrm{d}\tilde{\boldsymbol{x}}=\mathbb{E}_{p_\sigma}\big[\boldsymbol{s}_\theta^{\top}\nabla\log p_\sigma(\tilde{\boldsymbol{x}})\big],$$

which is precisely the ESM cross term under $$p_\sigma$$. So the two objectives differ only by a $$\theta$$-independent constant. For the Gaussian kernel the conditional score is explicit,

$$\nabla_{\tilde{\boldsymbol{x}}}\log p_\sigma(\tilde{\boldsymbol{x}}\mid\boldsymbol{x})=-\frac{\tilde{\boldsymbol{x}}-\boldsymbol{x}}{\sigma^2}=-\frac{\boldsymbol{\epsilon}}{\sigma},$$

so DSM reduces to **noise prediction** $$\boldsymbol{s}_\theta\approx-\boldsymbol{\epsilon}/\sigma$$ with no Jacobian trace. Taking the conditional expectation recovers **Tweedie's formula**:

$$\nabla\log p_\sigma(\tilde{\boldsymbol{x}})=\mathbb{E}\big[-\boldsymbol{\epsilon}/\sigma\mid\tilde{\boldsymbol{x}}\big],\qquad \mathbb{E}[\boldsymbol{x}\mid\tilde{\boldsymbol{x}}]=\tilde{\boldsymbol{x}}+\sigma^2\,\nabla\log p_\sigma(\tilde{\boldsymbol{x}}).$$

The time-dependent version with $$\boldsymbol{x}_t=\alpha_t\boldsymbol{x}_0+\sigma_t\boldsymbol{\epsilon}$$ gives $$\nabla\log p_t(\boldsymbol{x}_t\mid\boldsymbol{x}_0)=-(\boldsymbol{x}_t-\alpha_t\boldsymbol{x}_0)/\sigma_t^2=-\boldsymbol{\epsilon}/\sigma_t$$, the conditional score regressed during training.

**D.3 — The probability-flow ODE preserves the SDE marginals (Prop. 4.1.1).** From Appendix B.5 the Fokker–Planck equation is *identically*

$$\partial_t p_t=-\nabla\!\cdot\!\big(\tilde{\boldsymbol{\mu}}\,p_t\big),\qquad \tilde{\boldsymbol{\mu}}(\boldsymbol{x},t)=\boldsymbol{f}-\tfrac12 g^2\boldsymbol{s}.$$

This is the continuity equation (Appendix B.3) of the **deterministic** ODE $$\dot{\boldsymbol{x}}=\tilde{\boldsymbol{\mu}}(\boldsymbol{x},t)$$ — zero diffusion. A continuity equation determines the marginals uniquely from $$p_0$$, so the SDE and the ODE, obeying the *same* equation, have the *same* time-$$t$$ marginals $$p_t$$. Only the **paths** differ: the SDE's are stochastic and mix, the ODE's are deterministic and never cross (which is what makes the ODE invertible and yields exact likelihoods).

The algebraic identity behind all of this — and behind the reverse-SDE drift derived in the time-reversal block above — is

$$\tfrac12 g^2\Delta p_t=\nabla\!\cdot\!\big(\tfrac12 g^2 p_t\,\boldsymbol{s}\big),$$

i.e. *a diffusion term equals an advection term driven by the score*. Splitting the full $$g^2\Delta$$ entirely into advection produces the reverse SDE drift $$\boldsymbol{f}-g^2\boldsymbol{s}$$ with noise $$g\,\mathrm{d}\bar{\boldsymbol{w}}$$ (Anderson); splitting only half produces the noise-free PF-ODE drift $$\boldsymbol{f}-\tfrac12 g^2\boldsymbol{s}$$. Same marginals, with a one-parameter family of stochastic-to-deterministic samplers in between (the SDE↔ODE "churn" of Karras et al., 2022)."""

# Bundles for each lecture day
DAY07_OPTIONAL = [
    ("Heuristic reverse SDE (chain rule + Euler–Maruyama + Taylor)", OPT_REVERSE_SDE_HEURISTIC),
    ("Anderson's reverse-time SDE (sketch)", OPT_ANDERSON_REVERSE),
    ("DSM objective: conditional → marginal score (full proof)", OPT_DSM_MARGINAL),
    ("Appendix B — density evolution: change of variables → continuity → Fokker–Planck", OPT_APPENDIX_B_DENSITY),
    ("Appendix C — Itô's formula and Girsanov's theorem", OPT_APPENDIX_C_ITO_GIRSANOV),
    ("Appendix D — proofs: score matching (ESM ↔ ISM ↔ DSM) and PF-ODE marginals", OPT_APPENDIX_D_PROOFS),
]

DAY08_OPTIONAL = [
    ("PF-ODE from the Fokker–Planck equation (full derivation)", OPT_PF_ODE_FPE),
    ("Appendix B — density evolution: change of variables → continuity → Fokker–Planck", OPT_APPENDIX_B_DENSITY),
    ("Appendix D — proofs: score matching (ESM ↔ ISM ↔ DSM) and PF-ODE marginals", OPT_APPENDIX_D_PROOFS),
]
