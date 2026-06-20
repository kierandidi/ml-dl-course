#!/usr/bin/env python3
"""Generate rich Jekyll practical project pages with math and figures."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ROOT / "_projects"
FIGURES = ROOT / "assets" / "figures"


# Curated figure picks for days with hand-cropped source figures.
FIGURE_OVERRIDES: dict[int, list[str]] = {
    1: [
        "/assets/figures/day01/mml_gradient.png",
        "/assets/figures/day01/mml_projection.png",
        "/assets/figures/day01/mml_gaussian.png",
        "/assets/figures/day01/ode_vectorfield.png",
    ],
    2: [
        "/assets/figures/day02/ml_taxonomy.png",
        "/assets/figures/day02/mml_linear_regression.png",
        "/assets/figures/day02/mml_pca_illustration.png",
        "/assets/figures/day02/mml_svm_margin.png",
    ],
    9: [
        "/assets/figures/day09/llmks_architecture.png",
        "/assets/figures/day09/llmks_attention_dict.png",
        "/assets/figures/day09/llmks_rope.png",
        "/assets/figures/day09/llmks_training.png",
    ],
    10: [
        "/assets/figures/day10/llmks_generation.png",
        "/assets/figures/day10/llmks_kvcache.png",
        "/assets/figures/day10/llmks_sampling.png",
        "/assets/figures/day10/llmks_kv_bottleneck.png",
    ],
}


def figs(day: int, n: int = 4) -> list[str]:
    if day in FIGURE_OVERRIDES:
        return FIGURE_OVERRIDES[day][:n]
    d = FIGURES / f"day{day:02d}"
    if not d.is_dir():
        return []
    return [f"/assets/figures/day{day:02d}/{p.name}" for p in sorted(d.glob("*.png"))[:n]]


def img_block(path: str, caption: str) -> str:
    return f"""![{caption}]({path})
*{caption}*
"""


LECTURE_URLS = {
    1: "/blog/lectures/2026/08/17/day01-math-foundations/",
    2: "/blog/lectures/2026/08/18/day02-statistical-learning/",
    3: "/blog/lectures/2026/08/19/day03-deep-neural-networks/",
    4: "/blog/lectures/2026/08/20/day04-convolutional-networks/",
    5: "/blog/lectures/2026/08/21/day05-rnns-and-transformers/",
    6: "/blog/lectures/2026/08/24/day06-generative-modeling/",
    7: "/blog/lectures/2026/08/25/day07-training-diffusion-flow/",
    8: "/blog/lectures/2026/08/26/day08-diffusion-flow-inference/",
    9: "/blog/lectures/2026/08/25/day09-autoregressive-llms/",
    10: "/blog/lectures/2026/08/26/day10-ar-inference/",
}

PAGES: list[dict] = [
    {
        "file": "day01-practical.md",
        "title": "Day 1 Exercise — Math Foundations",
        "caption": "Gradients, probability, and MLE",
        "date": "17-08-2026",
        "desc": "Numerical gradients, Gaussian MLE, and probability exercises with worked derivations.",
        "body": """
## Learning objectives

- Implement finite-difference gradients and compare to symbolic answers
- Derive and code MLE for Gaussian mean/variance
- Connect calculus to optimization used in ML

## Key derivations

**Gradient of a quadratic.** For $$f(w) = \\frac{1}{2} w^T A w - b^T w$$ with symmetric $$A$$,

$$\\nabla_w f(w) = A w - b.$$

**Gaussian MLE.** For $$x_1,\\ldots,x_n \\sim \\mathcal{N}(\\mu, \\sigma^2)$$,

$$\\hat{\\mu}_{\\mathrm{MLE}} = \\frac{1}{n}\\sum_i x_i, \\qquad
\\hat{\\sigma}^2_{\\mathrm{MLE}} = \\frac{1}{n}\\sum_i (x_i - \\hat{\\mu})^2.$$

**Multivariate MLE.** For $$\\mathbf{x} \\sim \\mathcal{N}(\\boldsymbol{\\mu}, \\Sigma)$$,

$$\\hat{\\boldsymbol{\\mu}} = \\frac{1}{n}\\sum_i \\mathbf{x}^{(i)}, \\qquad
\\hat{\\Sigma} = \\frac{1}{n}\\sum_i (\\mathbf{x}^{(i)} - \\hat{\\boldsymbol{\\mu}})(\\mathbf{x}^{(i)} - \\hat{\\boldsymbol{\\mu}})^T.$$

## Exercises (notebook)

1. Numerical vs analytic gradient on $$f(x,y) = x^2 + xy + y^2$$
2. MLE for 1D Gaussian — plot likelihood surface
3. Bivariate Gaussian: estimate $$\\boldsymbol{\\mu}, \\Sigma$$ and verify via `scipy.stats`
4. **Reflection:** where do non-convex objectives appear later in the course?

## Notebook

Open [`notebooks/practicals/day01.ipynb`](/notebooks/practicals/day01.ipynb) (also in the GitHub repo).
""",
        "day": 1,
    },
    {
        "file": "day02-practical.md",
        "title": "Day 2 Exercise — Statistical Learning",
        "caption": "Regression, PCA, GMM, and SVM",
        "date": "18-08-2026",
        "desc": "MML Ch. 8–12 exercises: OLS/ridge, PCA, GMM with EM, and SVM on synthetic data.",
        "body": """
## Learning objectives

- Fit linear and ridge regression; relate OLS to Gaussian MLE
- Run PCA for visualization and reconstruction; report explained variance
- Fit a 2-component GMM with EM; interpret responsibilities
- Train a linear SVM; compare margin to logistic regression

## Key derivations

**OLS / MLE.**

$$\\hat{\\boldsymbol{\\theta}} = (\\Phi^{\\top}\\Phi)^{-1}\\Phi^{\\top}\\mathbf{y}.$$

**Ridge regression.**

$$\\hat{\\boldsymbol{\\theta}} = (\\Phi^{\\top}\\Phi + \\lambda \\mathbf{I})^{-1}\\Phi^{\\top}\\mathbf{y}.$$

**PCA (top-$M$ eigenvectors of sample covariance).**

$$\\mathbf{S} = \\frac{1}{N}\\sum_n \\tilde{\\mathbf{x}}_n \\tilde{\\mathbf{x}}_n^{\\top}, \\quad \\mathbf{z}_n = \\mathbf{B}^{\\top}\\tilde{\\mathbf{x}}_n.$$

**GMM responsibility (E-step).**

$$r_{nk} = \\frac{\\pi_k\\,\\mathcal{N}(\\mathbf{x}_n\\mid\\boldsymbol{\\mu}_k, \\boldsymbol{\\Sigma}_k)}{\\sum_j \\pi_j\\,\\mathcal{N}(\\mathbf{x}_n\\mid\\boldsymbol{\\mu}_j, \\boldsymbol{\\Sigma}_j)}.$$

## Exercises

1. **Regression** — polynomial features on 1D data; plot train/validation MSE vs degree and vs ridge $$\\lambda$$
2. **PCA** — project 2D/3D data to 2 components; plot reconstruction error vs $$M$$
3. **GMM** — fit $$K=2$$ on a bimodal 1D dataset; plot fitted density and responsibilities
4. **SVM** — linearly separable 2D points; plot decision boundary and support vectors; sweep soft-margin $$C$$
5. **Reflection** — where does each method sit in the ML taxonomy (supervised vs unsupervised)?

## Notebook

[`notebooks/practicals/day02.ipynb`](/notebooks/practicals/day02.ipynb)
""",
        "day": 2,
    },
    {
        "file": "day03-practical.md",
        "title": "Day 3 Exercise — Deep Neural Networks",
        "caption": "MLP and autograd",
        "date": "19-08-2026",
        "desc": "Train an MLP on MNIST; verify backprop with PyTorch autograd.",
        "body": """
## Learning objectives

- Build a multi-layer perceptron in PyTorch
- Verify gradients with `torch.autograd.gradcheck`
- Compare SGD and Adam on convergence speed

## Key derivations

**Backprop for a layer.** With $$\\mathbf{z} = \\mathbf{W}\\mathbf{h} + \\mathbf{b}$$, $$\\mathbf{h}' = g(\\mathbf{z})$$,

$$\\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{z}} = \\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{h}'} \\odot g'(\\mathbf{z}), \\quad
\\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{W}} = \\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{z}} \\mathbf{h}^T.$$

**Adam update.**

$$m_t = \\beta_1 m_{t-1} + (1-\\beta_1) g_t, \\quad
v_t = \\beta_2 v_{t-1} + (1-\\beta_2) g_t^2, \\quad
\\theta_{t+1} = \\theta_t - \\eta \\frac{\\hat{m}_t}{\\sqrt{\\hat{v}_t} + \\epsilon}.$$

## Exercises

1. Train 2-layer MLP on MNIST — report test accuracy
2. `gradcheck` on a single linear+ReLU block
3. Plot train loss for SGD vs Adam (same architecture)
4. **Reflection:** why ReLU helps optimization vs sigmoid

## Notebook

[`notebooks/practicals/day03.ipynb`](/notebooks/practicals/day03.ipynb)
""",
        "day": 3,
    },
    {
        "file": "day04-practical.md",
        "title": "Day 4 Exercise — Convolutional Networks",
        "caption": "Fashion-MNIST CNN",
        "date": "20-08-2026",
        "desc": "Implement a CNN for image classification; visualize filters and receptive fields.",
        "body": """
## Learning objectives

- Implement Conv2d → ReLU → Pool → Linear head
- Understand parameter sharing and translation equivariance
- Visualize first-layer convolutional filters

## Key ideas

**2D convolution.**

$$(f * k)[i,j] = \\sum_{u,v} f[i-u,j-v]\\, k[u,v].$$

**Receptive field** grows with depth: RF $$\\approx 1 + \\sum_\\ell (k_\\ell - 1) \\prod_{m<\\ell} s_m$$.

**Cross-entropy for $$K$$ classes.**

$$\\mathcal{L} = -\\sum_{k=1}^K y_k \\log \\hat{p}_k.$$

## Exercises

1. CNN on Fashion-MNIST — target $$>88\\%$$ test accuracy
2. Plot training curves (loss/accuracy)
3. Visualize 8 first-layer filters
4. **Reflection:** what inductive bias does convolution encode?

## Notebook

[`notebooks/practicals/day04.ipynb`](/notebooks/practicals/day04.ipynb)
""",
        "day": 4,
    },
    {
        "file": "day05-practical.md",
        "title": "Day 5 Exercise — Sequences & Attention",
        "caption": "RNN and attention",
        "date": "21-08-2026",
        "desc": "Sentiment RNN and manual scaled dot-product attention on short sequences.",
        "body": """
## Learning objectives

- Train an LSTM/GRU for sequence classification
- Implement scaled dot-product attention by hand
- Compare sequential vs parallel computation

## Key derivations

**Scaled dot-product attention.**

$$\\mathrm{Attention}(Q,K,V) = \\mathrm{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right) V.$$

**LSTM cell (schematic).** Gates $$f,i,o$$ control forget, input, and output; state $$c_t$$ carries long-range memory.

**Causal mask:** $$A_{ij} = -\\infty$$ for $$j > i$$ so position $$i$$ cannot attend to the future.

## Exercises

1. IMDB/binary sentiment with embedding + LSTM
2. Manual attention on length-$$T$$ toy sequences — heatmap
3. Count parameters: RNN vs 1-layer transformer block
4. **Reflection:** why transformers replaced RNNs at scale

## Notebook

[`notebooks/practicals/day05.ipynb`](/notebooks/practicals/day05.ipynb)
""",
        "day": 5,
    },
    {
        "file": "day06-practical.md",
        "title": "Day 6 Exercise — Generative Modeling",
        "caption": "KL, ELBO, and VAE",
        "date": "24-08-2026",
        "desc": "Derive ELBO, compute KL for Gaussians, train a 1D VAE.",
        "body": """
## Learning objectives

- Compute $$D_{\\mathrm{KL}}(q\\|p)$$ for univariate Gaussians in closed form
- Derive the ELBO and implement a 1D Gaussian VAE
- Classify model families (explicit vs implicit)

## Key derivations

**KL between Gaussians** $$\\mathcal{N}(\\mu_0,\\sigma_0^2)$$ and $$\\mathcal{N}(\\mu_1,\\sigma_1^2)$$:

$$D_{\\mathrm{KL}} = \\log\\frac{\\sigma_1}{\\sigma_0} + \\frac{\\sigma_0^2 + (\\mu_0-\\mu_1)^2}{2\\sigma_1^2} - \\frac{1}{2}.$$

**ELBO.**

$$\\log p(x) \\geq \\mathbb{E}_{q(z\\mid x)}[\\log p(x\\mid z)] - D_{\\mathrm{KL}}(q(z\\mid x)\\,\\|\\,p(z)).$$

**VAE loss** = reconstruction + KL to prior $$\\mathcal{N}(0,I)$$.

## Exercises

1. Pen-and-paper KL (then verify numerically)
2. Implement ELBO for factorized Gaussian $$q(z\\mid x)$$
3. Train 1D VAE on synthetic mixture — plot samples
4. **Reflection:** explicit vs implicit — one example each

## Notebook

[`notebooks/practicals/day06.ipynb`](/notebooks/practicals/day06.ipynb)
""",
        "day": 6,
    },
    {
        "file": "day07-practical.md",
        "title": "Day 7 Exercise — Flow & Diffusion Training",
        "caption": "Flow matching and denoising",
        "date": "25-08-2026",
        "desc": "1D flow matching and denoising score targets (MIT Lab 2 inspired).",
        "body": """
## Learning objectives

- Define a probability path $$p_t(x)$$ from noise to data
- Train a velocity field / score network with simple regression
- Relate flow matching to diffusion denoising objectives

## Key derivations

**Flow matching objective.**

$$\\mathcal{L}_{\\mathrm{FM}} = \\mathbb{E}_{t,x_0,x_1}\\big[ \\| v_\\theta(x_t, t) - (x_1 - x_0) \\|^2 \\big]$$

for appropriate interpolant $$x_t$$.

**Denoising score matching.**

$$\\mathcal{L} = \\mathbb{E}_{t,x_0,\\epsilon}\\big[ \\| s_\\theta(x_t, t) + \\sigma_t^{-1}\\epsilon \\|^2 \\big], \\quad x_t = x_0 + \\sigma_t \\epsilon.$$

## Exercises

1. Sample 1D interpolants $$x_t = (1-t)x_0 + t x_1$$
2. Train MLP $$v_\\theta(x,t)$$ on synthetic 1D data
3. Plot learned field vs ground truth
4. **Reflection:** three views of diffusion (variational, score, flow)

## Notebook

[`notebooks/practicals/day07.ipynb`](/notebooks/practicals/day07.ipynb)
""",
        "day": 7,
    },
    {
        "file": "day08-practical.md",
        "title": "Day 8 Exercise — SDE vs ODE Sampling",
        "caption": "Simulate generative dynamics",
        "date": "26-08-2026",
        "desc": "Euler–Maruyama vs probability-flow ODE on 2D moons (MIT Lab 1 inspired).",
        "body": """
## Learning objectives

- Simulate ODEs and SDEs with Euler schemes
- Compare stochastic vs deterministic sampling paths
- Implement a few-step reverse diffusion sampler on 2D data

## Key derivations

**Euler–Maruyama.**

$$X_{t+\\Delta t} = X_t + u_t(X_t)\\Delta t + \\sigma_t \\sqrt{\\Delta t}\\, \\xi, \\quad \\xi \\sim \\mathcal{N}(0,I).$$

**Probability-flow ODE** (schematic): replace noise term with a drift correction using score $$\\nabla \\log p_t(x)$$.

**Reverse diffusion (DDPM-style, few steps):**

$$x_{t-\\Delta} \\approx x_t + \\tfrac{1}{2}\\beta_t \\nabla \\log p_t(x_t) + \\text{noise term}.$$

## Exercises

1. Integrate 2D ODE — plot trajectories
2. SDE with additive noise — ensemble of paths
3. 10-step sampler from noise to data cloud
4. **Reflection:** when is stochastic sampling preferred?

## Notebook

[`notebooks/practicals/day08.ipynb`](/notebooks/practicals/day08.ipynb)
""",
        "day": 8,
    },
    {
        "file": "day09-practical.md",
        "title": "Day 9 Exercise — Train a Tiny GPT",
        "caption": "Decoder-only language model",
        "date": "27-08-2026",
        "desc": "Character-level decoder-only transformer; cross-entropy training loop.",
        "body": """
## Learning objectives

- Implement causal self-attention and a transformer block
- Train on a tiny corpus (~2M parameters budget)
- Track loss curves and generate samples

## Key derivations

**Causal LM loss.**

$$\\mathcal{L} = -\\sum_{i=1}^{L} \\log p_\\theta(x_i \\mid x_{<i}).$$

**Pre-norm block.**

$$h' = h + \\mathrm{MHA}(\\mathrm{LN}(h)), \\quad h'' = h' + \\mathrm{MLP}(\\mathrm{LN}(h')).$$

**RoPE** encodes relative position via rotations in Q/K pairs (see lecture notes).

## Exercises

1. Tokenize character-level corpus
2. Train small GPT — plot loss
3. Sample 200 characters from trained model
4. **Reflection:** estimate FLOPs per token (order of magnitude)

## Notebook

[`notebooks/practicals/day09.ipynb`](/notebooks/practicals/day09.ipynb)
""",
        "day": 9,
    },
    {
        "file": "day10-practical.md",
        "title": "Day 10 Exercise — KV Cache & Inference",
        "caption": "Efficient autoregressive decoding",
        "date": "28-08-2026",
        "desc": "KV cache memory formula, timed generation with/without cache.",
        "body": """
## Learning objectives

- Derive KV cache memory scaling
- Implement cached incremental decoding
- Compare temperature and top-$$p$$ sampling

## Key derivations

**KV cache size (per layer, batch $$B$$, seq $$T$$, head dim $$d_h$$, $$n_{\\mathrm{kv}}$$ heads):**

$$\\mathrm{Memory}_{\\mathrm{KV}} \\approx 2 \\cdot L \\cdot B \\cdot T \\cdot n_{\\mathrm{kv}} \\cdot d_h \\cdot \\texttt{bytes}.$$

**Autoregressive factorization.**

$$p(x_{1:T}) = \\prod_{t=1}^T p(x_t \\mid x_{<t}).$$

**Top-$$p$$ (nucleus):** keep smallest set of tokens with cumulative prob $$\\geq p$$.

## Exercises

1. Compute KV bytes for given $$(L, B, T, d_{\\mathrm{model}})$$
2. Time 500 tokens with vs without cache
3. Compare greedy vs temperature vs top-$$p$$ outputs
4. **Reflection:** throughput vs memory tradeoff at deployment

## Notebook

[`notebooks/practicals/day10.ipynb`](/notebooks/practicals/day10.ipynb)
""",
        "day": 10,
    },
]


def render_page(meta: dict) -> str:
    day = meta["day"]
    images = figs(day, 4)
    fig_md = ""
    if images:
        fig_md = "\n## Figures from lecture materials\n\n"
        for i, p in enumerate(images):
            cap = f"Source illustration — Day {day} (extracted from course PDFs/PPTX)"
            fig_md += img_block(p, cap) + "\n"

    return f"""---
layout: project
title: {meta['title']}
caption: {meta['caption']}
description: >
  {meta['desc']}
date: '{meta['date']}'
sitemap: false
links:
  - title: Exercise notebook
    url: /notebooks/practicals/day{day:02d}.ipynb
  - title: Lecture slides (PDF)
    url: /assets/slides/day{day:02d}.pdf
  - title: Lecture notes
    url: {LECTURE_URLS[day]}
---

# {meta['title']}

{fig_md}
{meta['body'].strip()}
"""


FINAL_ASSESSMENT = """---
layout: project
title: Final Assessment
caption: Math + coding (~3 hours)
description: >
  Closed-book math questions plus PyTorch coding: MLP classifier and diffusion sampler or KV-cache forward pass.
date: '28-08-2026'
sitemap: false
links:
  - title: Assessment notebook
    url: /notebooks/assessment/final_assessment.ipynb
  - title: Dataset
    url: /notebooks/data/assessment_synthetic.csv
---

# Final Assessment

**Duration:** ~3 hours · **Total:** 100 points (50 math + 50 coding) · **Submission:** PDF export of notebook + honor code statement

Work through [`notebooks/assessment/final_assessment.ipynb`](/notebooks/assessment/final_assessment.ipynb); write math answers in its markdown cells and code in its code cells. Before submitting, run **Kernel → Restart & Run All**.

## Part A — Mathematics (50 points)

1. **A1 — Chain rule and gradients (10 pts).** Differentiate $$f(x,y) = \\sin(xy) + x^2$$, evaluate at a point, then apply the chain rule with $$x = t^2,\\ y = 3t$$.
2. **A2 — Backpropagation sketch (10 pts).** For a one-hidden-unit network with squared loss, draw the computation graph and derive $$\\partial \\mathcal{L}/\\partial w_1$$.
3. **A3 — KL divergence and ELBO (10 pts).** Write the closed-form $$\\mathrm{KL}(q \\,\\|\\, p)$$ for two univariate Gaussians, then explain why maximizing the ELBO $$\\mathbb{E}_{q_\\phi(z\\mid x)}[\\log p_\\theta(x\\mid z)] - \\mathrm{KL}(q_\\phi(z\\mid x) \\,\\|\\, p(z))$$ trades off reconstruction against a latent regularizer.
4. **A4 — Forward diffusion and score matching (15 pts).** Show $$x_t = \\sqrt{\\bar{\\alpha}_t}\\,x_0 + \\sqrt{1 - \\bar{\\alpha}_t}\\,\\epsilon$$, relate the optimal noise predictor to the score $$\\nabla_{x_t}\\log q(x_t)$$, and explain why the signal-to-noise ratio falls with $$t$$.
5. **A5 — Scaled dot-product attention (15 pts).** Give the shapes in $$\\mathrm{softmax}(QK^\\top/\\sqrt{d_k})V$$, justify the $$1/\\sqrt{d_k}$$ scaling, and describe the causal mask.

## Part B — Coding (50 points)

Use [`notebooks/assessment/final_assessment.ipynb`](/notebooks/assessment/final_assessment.ipynb) and `notebooks/data/assessment_synthetic.csv`.

| Task | Points |
|------|--------|
| **B1 —** Train an MLP classifier (≥2 hidden layers) on the tabular data; report validation accuracy and plot the training loss | 25 |
| **B2a —** 1-D reverse diffusion sampler for a known Gaussian-mixture score | 25 |
| **B2b —** KV-cache greedy decoding for a tiny causal Transformer; report the tokens/sec speedup | 25 |

For **B2**, complete **one** track (B2a *or* B2b), not both.

## Grading rubric

- Correct math reasoning and notation
- Code runs end-to-end; plots where requested
- Brief written interpretation of results (2–3 sentences per part)

## Figure reference

![The deep generative modeling landscape: transform a simple reference distribution into the data distribution by different model families.](/assets/figures/day06/pdm_dgm_zoo.png)
*The generative-modeling map from Day 6 — useful background for Part A.*
"""


# ---------------------------------------------------------------------------
# Exercises hub (single page with a table; download + Open-in-Colab per day).
# Replaces the previous one-page-per-exercise approach.
# ---------------------------------------------------------------------------
GITHUB = "kierandidi/ml-dl-course"
BRANCH = "main"

TOPICS = [
    (1, "Math foundations"),
    (2, "Statistical learning"),
    (3, "Deep neural networks"),
    (4, "CNNs & vision"),
    (5, "RNNs → Transformers"),
    (6, "Generative modeling & DDPM"),
    (7, "Score, SDEs & flow matching"),
    (8, "Guidance, solvers & fast sampling"),
    (9, "Autoregressive LLMs"),
    (10, "LLM inference & KV cache"),
]


def colab_url(day: int) -> str:
    return (
        f"https://colab.research.google.com/github/{GITHUB}/blob/{BRANCH}"
        f"/notebooks/practicals/day{day:02d}.ipynb"
    )


def build_projects_hub() -> str:
    rows = []
    for day, topic in TOPICS:
        nb = f"/notebooks/practicals/day{day:02d}.ipynb"
        download = f'<a href="{nb}" download>Download</a>'
        colab = f'[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_url(day)})'
        rows.append(f"| {day} | {topic} | {download} | {colab} |")
    table = "\n".join(rows)
    return f"""---
layout: page
title: Exercises
description: >
  Daily exercise notebooks for the ML & Deep Learning course (Aug 2026) —
  download to run locally or open directly in Google Colab.
---

Each exercise is a self-contained Jupyter notebook (~45–60 min) aligned with that
day's lecture notes and slides. **Download** it to run locally, or **Open in Colab**
to run it in your browser (free GPU, no setup).

| Day | Topic | Notebook | Run |
|-----|-------|----------|-----|
{table}

> Colab opens each notebook straight from the [course repository](https://github.com/{GITHUB}). For local use, clone the repo and open the files in [`notebooks/practicals/`](/notebooks/practicals/).

## Final assessment

The course concludes with a [final assessment](/projects/final-assessment/) —
written math plus PyTorch coding (~3 hours).
"""


def main():
    PROJECTS.mkdir(parents=True, exist_ok=True)
    # Single Exercises hub page (replaces per-day project pages).
    (ROOT / "projects.md").write_text(build_projects_hub(), encoding="utf-8")
    print("Wrote projects.md")
    # Final assessment stays as its own project page.
    (PROJECTS / "final-assessment.md").write_text(FINAL_ASSESSMENT, encoding="utf-8")
    print("Wrote _projects/final-assessment.md")
    # Remove the deprecated one-page-per-exercise files.
    for day in range(1, 11):
        stale = PROJECTS / f"day{day:02d}-practical.md"
        if stale.exists():
            stale.unlink()
            print(f"Removed {stale.name}")


if __name__ == "__main__":
    main()
