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

OPT_APPENDIX_B_DENSITY = """*Condensed from [Principles of Diffusion Models](https://arxiv.org/abs/2510.21890), Appendix B.*

**B.1 — Change of variables → continuity → FPE.**

1. **Single bijection** $$\\boldsymbol{x}_1=\\Psi(\\boldsymbol{x}_0)$$: $$p_1(\\boldsymbol{x}_1)=p_0(\\Psi^{-1}(\\boldsymbol{x}_1))\\,|\\det\\partial\\Psi^{-1}/\\partial\\boldsymbol{x}_1|$$.
2. **Composition** of maps: log-density accumulates $$-\\sum_k \\log|\\det\\partial\\Psi_k/\\partial\\boldsymbol{x}_{k-1}|$$ (normalizing flows, Eq. B.1.2).
3. **Continuous limit** $$\\boldsymbol{x}_{t+\\delta}=\\boldsymbol{x}_t+\\delta\\,\\boldsymbol{f}(\\boldsymbol{x}_t,t)$$: Jacobian $$\\det(I+\\delta\\nabla\\boldsymbol{f})=1+\\delta\\,\\nabla\\cdot\\boldsymbol{f}+\\mathcal{O}(\\delta^2)$$ gives the **continuity equation**

$$\\partial_t p_t + \\nabla\\cdot(p_t\\,\\boldsymbol{f}) = 0.$$

4. **Add noise** $$\\mathrm{d}\\boldsymbol{x}=\\boldsymbol{f}\\,\\mathrm{d}t+g(t)\\,\\mathrm{d}\\boldsymbol{w}$$: spreading term $$\\tfrac12 g(t)^2\\Delta p_t$$ yields the **Fokker–Planck equation**

$$\\partial_t p_t = -\\nabla\\cdot(\\boldsymbol{f}\\,p_t) + \\tfrac12 g(t)^2 \\Delta p_t = -\\nabla\\cdot\\Big(\\big(\\boldsymbol{f}-\\tfrac12 g(t)^2\\,\\boldsymbol{s}\\big)\\,p_t\\Big).$$

**B.2 — Intuition.** In a small box, mass changes only through net flux $$\\boldsymbol{j}=p_t\\boldsymbol{v}$$; conservation $$\\partial_t p_t + \\nabla\\cdot\\boldsymbol{j}=0$$ is the continuity equation. The divergence theorem upgrades the box argument to arbitrary control volumes."""

OPT_APPENDIX_C_ITO_GIRSANOV = """*Condensed from Principles Appendix C.*

**C.1 — Itô's formula (chain rule for SDEs).** For smooth $$h(\\boldsymbol{x},t)$$ and $$\\mathrm{d}\\boldsymbol{x}=\\boldsymbol{f}\\,\\mathrm{d}t+g\\,\\mathrm{d}\\boldsymbol{w}$$,

$$\\mathrm{d}h = \\Big(\\partial_t h + \\nabla h^{\\top}\\boldsymbol{f} + \\tfrac12 g^2 \\Delta h\\Big)\\mathrm{d}t + g\\,\\nabla h^{\\top}\\mathrm{d}\\boldsymbol{w}.$$

Key rule: $$(\\mathrm{d}\\boldsymbol{w})^2 = \\mathrm{d}t$$, so second-order terms survive (unlike ordinary calculus). **C.1.4** uses Itô's formula on $$h=\\log p_t$$ to derive the Fokker–Planck equation.

**C.2 — Girsanov's theorem.** Two SDEs with the same diffusion $$g$$ but drifts $$\\boldsymbol{f}$$ and $$\\boldsymbol{f}+g\\,\\boldsymbol{u}$$ assign different path probabilities. The **likelihood ratio** (Radon–Nikodym derivative) on a path $$\\boldsymbol{x}_{[0,T]}$$ is

$$\\frac{\\mathrm{d}\\mathbb{Q}}{\\mathrm{d}\\mathbb{P}} = \\exp\\Big(\\int_0^T \\boldsymbol{u}^{\\top}\\mathrm{d}\\boldsymbol{w} - \\tfrac12\\int_0^T \\|\\boldsymbol{u}\\|^2\\,\\mathrm{d}t\\Big).$$

For the reverse SDE, $$\\boldsymbol{u}=-g\\,\\boldsymbol{s}$$ — the score reweights forward paths to reverse paths. This explains why score matching implicitly performs **likelihood-based** training (Song et al., 2021)."""

OPT_APPENDIX_D_PROOFS = """*Selected proofs from Principles Appendix D.*

**D.2.1 — Score matching via integration by parts (Prop. 3.2.1).** Expand

$$\\tfrac12\\mathbb{E}\\|\\boldsymbol{s}_\\theta-\\boldsymbol{s}\\|^2 = \\tfrac12\\mathbb{E}\\|\\boldsymbol{s}_\\theta\\|^2 - \\mathbb{E}[\\boldsymbol{s}_\\theta^{\\top}\\boldsymbol{s}] + \\text{const}.$$

Use $$\\boldsymbol{s}=\\nabla\\log p_{\\text{data}}$$ and $$\\mathbb{E}[\\boldsymbol{s}_\\theta^{\\top}\\nabla p/p]=-\\mathbb{E}[\\nabla\\cdot\\boldsymbol{s}_\\theta]$$ (integration by parts, vanishing boundary) to obtain the tractable objective with Jacobian trace.

**D.2.2 — Denoising score matching (Prop. 3.3.1 / 4.3.1).** Add noise $$\\tilde{\\boldsymbol{x}}=\\boldsymbol{x}+\\sigma\\boldsymbol{\\epsilon}$$; the cross term becomes an expectation under $$p_\\sigma(\\tilde{\\boldsymbol{x}}\\mid\\boldsymbol{x})$$, which integrates by parts to remove $$\\nabla\\log p_{\\text{data}}$$. The minimizer is the **marginal** noisy score — the DSM proof in the block above.

**D.2.6 — PF-ODE shares marginals (Prop. 4.1.1).** Part 1: verify that $$\\tilde{\\boldsymbol{\\mu}}=\\boldsymbol{f}-\\tfrac12 g^2\\boldsymbol{s}$$ reproduces the FPE. Part 2: show the reverse SDE with drift $$\\boldsymbol{f}-g^2\\boldsymbol{s}$$ has the same $$p_t$$ when time is reversed — connecting Anderson, PF-ODE, and FPE in one proof chain."""

# Bundles for each lecture day
DAY07_OPTIONAL = [
    ("Heuristic reverse SDE (chain rule + Euler–Maruyama + Taylor)", OPT_REVERSE_SDE_HEURISTIC),
    ("Anderson's reverse-time SDE (sketch)", OPT_ANDERSON_REVERSE),
    ("DSM objective: conditional → marginal score (full proof)", OPT_DSM_MARGINAL),
    ("Appendix B — density evolution and the Fokker–Planck equation", OPT_APPENDIX_B_DENSITY),
    ("Appendix C — Itô's formula and Girsanov's theorem", OPT_APPENDIX_C_ITO_GIRSANOV),
    ("Appendix D — score matching and PF-ODE proofs", OPT_APPENDIX_D_PROOFS),
]

DAY08_OPTIONAL = [
    ("PF-ODE from the Fokker–Planck equation (full derivation)", OPT_PF_ODE_FPE),
    ("Appendix B — density evolution (continuity → FPE)", OPT_APPENDIX_B_DENSITY),
    ("Appendix D.2.6 — PF-ODE and reverse SDE share marginals", OPT_APPENDIX_D_PROOFS),
]
