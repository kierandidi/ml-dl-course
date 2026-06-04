#!/usr/bin/env python3
"""Generate Week 1 Jekyll lecture posts for the ML & DL course."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POSTS_DIR = ROOT / "lectures" / "_posts"
START_DATE = date(2026, 8, 17)

LECTURES = [
    {
        "day": 1,
        "slug": "math-foundations",
        "title": "Math Foundations",
        "description": "Gradients, the chain rule, probability, and maximum likelihood estimation.",
        "reading": [
            "[Bishop — Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/), Ch. 1–2",
            "[Boyd & Vandenberghe — Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/), §2.1–2.3",
            "[3Blue1Brown — Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3tZaY)",
        ],
        "intro": (
            "Before we train neural networks we need a shared mathematical language. "
            "Today we review multivariate calculus, the chain rule, basic probability, "
            "and maximum likelihood — the workhorse principle behind most learning algorithms."
        ),
        "sections": [
            {
                "title": "Multivariate Calculus and Gradients",
                "subsections": [
                    {
                        "heading": "Partial derivatives and the gradient",
                        "definition": (
                            "For a scalar function $$f: \\mathbb{R}^d \\to \\mathbb{R}$$, the "
                            "**gradient** is the vector of partial derivatives "
                            "$$\\nabla f(\\mathbf{x}) = \\left(\\frac{\\partial f}{\\partial x_1}, "
                            "\\ldots, \\frac{\\partial f}{\\partial x_d}\\right)^\\top.$$ "
                            "It points in the direction of steepest ascent."
                        ),
                        "body": """For a quadratic loss $$L(\\mathbf{w}) = \\|\\mathbf{X}\\mathbf{w} - \\mathbf{y}\\|_2^2$$ with design matrix $$\\mathbf{X} \\in \\mathbb{R}^{n \\times d}$$, expanding gives

$$L(\\mathbf{w}) = \\mathbf{w}^\\top \\mathbf{X}^\\top \\mathbf{X} \\mathbf{w} - 2\\mathbf{y}^\\top \\mathbf{X}\\mathbf{w} + \\mathbf{y}^\\top\\mathbf{y}.$$

Differentiating with respect to $$\\mathbf{w}$$ yields the closed-form gradient

$$\\nabla_{\\mathbf{w}} L = 2\\mathbf{X}^\\top(\\mathbf{X}\\mathbf{w} - \\mathbf{y}).$$

Setting $$\\nabla_{\\mathbf{w}} L = \\mathbf{0}$$ recovers the ordinary least-squares solution $$\\hat{\\mathbf{w}} = (\\mathbf{X}^\\top\\mathbf{X})^{-1}\\mathbf{X}^\\top\\mathbf{y}$$ whenever $$\\mathbf{X}^\\top\\mathbf{X}$$ is invertible.

![Gradient descent on a quadratic bowl](/assets/figures/day01/pdf0_page005.png)

The **directional derivative** along unit vector $$\\mathbf{u}$$ is $$D_{\\mathbf{u}} f = \\nabla f \\cdot \\mathbf{u}$$. Steepest descent uses $$\\mathbf{u} = -\\nabla f / \\|\\nabla f\\|$$.""",
                    },
                    {
                        "heading": "The Jacobian and Hessian",
                        "definition": (
                            "For $$\\mathbf{f}: \\mathbb{R}^n \\to \\mathbb{R}^m$$, the **Jacobian** "
                            "$$\\mathbf{J}_{\\mathbf{f}} \\in \\mathbb{R}^{m \\times n}$$ has entries "
                            "$$[\\mathbf{J}_{\\mathbf{f}}]_{ij} = \\partial f_i / \\partial x_j$$. "
                            "For scalar $$f$$, the **Hessian** $$\\mathbf{H} = \\nabla^2 f$$ "
                            "describes local curvature."
                        ),
                        "body": """For a linear layer $$\\mathbf{z} = \\mathbf{W}\\mathbf{x} + \\mathbf{b}$$ with $$\\mathbf{W} \\in \\mathbb{R}^{m \\times n}$$, the Jacobian with respect to $$\\mathbf{x}$$ is simply $$\\mathbf{W}$$. This fact is what makes backpropagation through affine layers trivial.

Near a critical point $$\\mathbf{x}^\\*$$, Taylor expansion gives

$$f(\\mathbf{x}) \\approx f(\\mathbf{x}^\\*) + \\frac{1}{2}(\\mathbf{x}-\\mathbf{x}^\\*)^\\top \\mathbf{H}(\\mathbf{x}^\\*)(\\mathbf{x}-\\mathbf{x}^\\*).$$

Positive definiteness of $$\\mathbf{H}$$ at $$\\mathbf{x}^\\*$$ implies a local minimum — a condition optimizers exploit via second-order methods (Newton, L-BFGS).""",
                    },
                ],
            },
            {
                "title": "The Chain Rule and Computational Graphs",
                "subsections": [
                    {
                        "heading": "Scalar chain rule",
                        "definition": (
                            "If $$y = f(u)$$ and $$u = g(x)$$, then $$\\frac{dy}{dx} = "
                            "\\frac{dy}{du}\\frac{du}{dx}$$. In multivariate form, "
                            "gradients propagate backward through composed functions."
                        ),
                        "body": """Consider a two-layer network with scalar output:

$$\\hat{y} = w_2\\,\\sigma(w_1 x + b_1) + b_2.$$

Define $$z = w_1 x + b_1$$ and $$a = \\sigma(z)$$. Then

$$\\frac{\\partial \\hat{y}}{\\partial w_1} = \\frac{\\partial \\hat{y}}{\\partial a}\\frac{\\partial a}{\\partial z}\\frac{\\partial z}{\\partial w_1} = w_2\\,\\sigma'(z)\\,x.$$

![Computational graph for a tiny network](/assets/figures/day01/pdf0_page010.png)

Each edge in the graph stores a local Jacobian; backpropagation is reverse-mode automatic differentiation that multiplies these Jacobians along paths from the loss to each parameter.""",
                    },
                    {
                        "heading": "Multivariate chain rule",
                        "definition": (
                            "If $$\\mathbf{y} = \\mathbf{f}(\\mathbf{u})$$ and $$\\mathbf{u} = "
                            "\\mathbf{g}(\\mathbf{x})$$, then $$\\frac{\\partial \\mathbf{y}}{\\partial "
                            "\\mathbf{x}} = \\mathbf{J}_{\\mathbf{f}}(\\mathbf{u})\\,"
                            "\\mathbf{J}_{\\mathbf{g}}(\\mathbf{x})$$ by matrix multiplication."
                        ),
                        "body": """For vector-valued intermediates, the chain rule is a product of Jacobians. Given loss $$L$$ and hidden activation $$\\mathbf{h}$$,

$$\\frac{\\partial L}{\\partial \\mathbf{h}} = \\left(\\frac{\\partial L}{\\partial \\mathbf{z}}\\right)^\\top \\mathbf{J}_{\\mathbf{h}}.$$

In practice frameworks never materialize full Jacobians for large layers; they use **vector-Jacobian products** (VJPs) that cost one forward/backward pass per output dimension batch.

The **reverse-mode** trick: one backward pass from a scalar loss costs $$O(\\text{ops})$$ regardless of parameter count, whereas forward-mode AD scales with the number of inputs.""",
                    },
                ],
            },
            {
                "title": "Probability Essentials",
                "subsections": [
                    {
                        "heading": "Random variables and expectations",
                        "definition": (
                            "A **random variable** $$X$$ maps outcomes $$\\omega \\in \\Omega$$ to "
                            "real values. For continuous $$X$$ with density $$p(x)$$, "
                            "$$\\mathbb{E}[X] = \\int x\\,p(x)\\,dx$$ and "
                            "$$\\mathrm{Var}(X) = \\mathbb{E}[(X-\\mathbb{E}[X])^2].$$"
                        ),
                        "body": """Key identities used throughout ML:

$$\\mathrm{Var}(X) = \\mathbb{E}[X^2] - (\\mathbb{E}[X])^2, \\qquad \\mathrm{Cov}(X,Y) = \\mathbb{E}[XY] - \\mathbb{E}[X]\\mathbb{E}[Y].$$

For multivariate Gaussian $$\\mathbf{x} \\sim \\mathcal{N}(\\boldsymbol{\\mu}, \\boldsymbol{\\Sigma})$$,

$$p(\\mathbf{x}) = \\frac{1}{(2\\pi)^{d/2}|\\boldsymbol{\\Sigma}|^{1/2}} \\exp\\!\\left(-\\tfrac{1}{2}(\\mathbf{x}-\\boldsymbol{\\mu})^\\top \\boldsymbol{\\Sigma}^{-1}(\\mathbf{x}-\\boldsymbol{\\mu})\\right).$$

![Gaussian contours in 2D](/assets/figures/day01/pdf0_page015.png)

**Conditional** distributions arise constantly: $$p(\\mathbf{x}|y) \\propto p(y|\\mathbf{x})p(\\mathbf{x})$$. Independence means $$p(\\mathbf{x},\\mathbf{y}) = p(\\mathbf{x})p(\\mathbf{y}).""",
                    },
                    {
                        "heading": "Information and KL divergence",
                        "definition": (
                            "The **Kullback–Leibler divergence** from $$q$$ to $$p$$ is "
                            "$$D_{\\mathrm{KL}}(q\\|p) = \\mathbb{E}_{\\mathbf{x}\\sim q}\\!"
                            "\\left[\\log\\frac{q(\\mathbf{x})}{p(\\mathbf{x})}\\right] \\geq 0$$, "
                            "with equality iff $$q = p$$ almost everywhere."
                        ),
                        "body": """KL divergence is not symmetric but measures how many extra nats are needed to encode samples from $$q$$ using a code optimized for $$p$$.

For Gaussians $$\\mathcal{N}(\\boldsymbol{\\mu}_1, \\boldsymbol{\\Sigma}_1)$$ and $$\\mathcal{N}(\\boldsymbol{\\mu}_2, \\boldsymbol{\\Sigma}_2)$$ in $$d$$ dimensions, a closed form exists and appears in VAEs and diffusion training.

**Entropy** $$H(p) = -\\mathbb{E}_{p}[\\log p]$$ quantifies uncertainty. Cross-entropy $$H(p,q) = -\\mathbb{E}_{p}[\\log q]$$ is the classification loss when $$p$$ is the data distribution and $$q$$ the model.""",
                    },
                ],
            },
            {
                "title": "Maximum Likelihood Estimation",
                "subsections": [
                    {
                        "heading": "Likelihood and log-likelihood",
                        "definition": (
                            "Given i.i.d. data $$\\mathcal{D} = \\{\\mathbf{x}^{(i)}\\}_{i=1}^n$$ "
                            "and parametric model $$p_\\theta(\\mathbf{x})$$, the **likelihood** is "
                            "$$\\mathcal{L}(\\theta) = \\prod_i p_\\theta(\\mathbf{x}^{(i)})$$. "
                            "**MLE** chooses $$\\hat{\\theta} = \\arg\\max_\\theta \\mathcal{L}(\\theta)$$."
                        ),
                        "body": """We almost always maximize the log-likelihood (monotone transform):

$$\\ell(\\theta) = \\sum_{i=1}^n \\log p_\\theta(\\mathbf{x}^{(i)}).$$

For Gaussian noise regression $$y^{(i)} = \\mathbf{w}^\\top\\mathbf{x}^{(i)} + \\epsilon^{(i)}$$ with $$\\epsilon^{(i)} \\sim \\mathcal{N}(0, \\sigma^2)$$, MLE of $$\\mathbf{w}$$ coincides with minimizing squared error — linking probabilistic modeling to empirical risk minimization.

![Likelihood surface for a Bernoulli parameter](/assets/figures/day01/pdf0_page020.png)""",
                    },
                    {
                        "heading": "Properties and regularization as priors",
                        "definition": (
                            "**MAP estimation** maximizes $$p(\\theta|\\mathcal{D}) \\propto "
                            "p(\\mathcal{D}|\\theta)p(\\theta)$$. An $$\\ell_2$$ prior on $$\\mathbf{w}$$ "
                            "yields ridge regression; a Laplace prior yields Lasso."
                        ),
                        "body": """Under regularity conditions, MLE is **consistent** ($$\\hat{\\theta} \\to \\theta^\\*$$ as $$n \\to \\infty$$) and asymptotically normal:

$$\\sqrt{n}(\\hat{\\theta} - \\theta^\\*) \\xrightarrow{d} \\mathcal{N}(\\mathbf{0}, \\mathcal{I}^{-1}(\\theta^\\*)),$$

where $$\\mathcal{I}$$ is the Fisher information matrix.

Taking gradients of $$\\ell(\\theta)$$ and setting to zero often has no closed form — we use gradient ascent:

$$\\theta_{t+1} = \\theta_t + \\eta \\nabla_\\theta \\ell(\\theta_t).$$

This connects MLE directly to the optimization algorithms we will implement in the practical.""",
                    },
                ],
            },
        ],
        "checkpoint": [
            "The gradient $$\\nabla f$$ points uphill; descent steps go opposite to it.",
            "Backpropagation is the multivariate chain rule applied to a computational graph.",
            "MLE turns learning into maximizing $$\\sum_i \\log p_\\theta(x^{(i)})$$.",
            "MAP = MLE + log-prior; common priors recover familiar regularizers.",
        ],
    },
    {
        "day": 2,
        "slug": "statistical-learning",
        "title": "Statistical Learning",
        "description": "Regression, classification, bias–variance, and regularization.",
        "reading": [
            "[Hastie, Tibshirani & Friedman — ESL](https://hastie.su.domains/ElemStatLearn/), Ch. 3–4",
            "[Murphy — Probabilistic Machine Learning: An Introduction](https://probml.github.io/pml-book/book1.html), Ch. 11",
            "[scikit-learn User Guide — Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)",
        ],
        "intro": (
            "Statistical learning formalizes prediction as estimating a function from finite data. "
            "We study linear and logistic models, loss functions, the bias–variance trade-off, "
            "and regularization as the primary tools for controlling generalization."
        ),
        "sections": [
            {
                "title": "Supervised Learning Framework",
                "subsections": [
                    {
                        "heading": "Empirical risk minimization",
                        "definition": (
                            "Given training pairs $$(\\mathbf{x}^{(i)}, y^{(i)})$$ and loss "
                            "$$\\ell$$, **empirical risk minimization (ERM)** seeks "
                            "$$\\hat{f} = \\arg\\min_{f \\in \\mathcal{F}} \\frac{1}{n}\\sum_i "
                            "\\ell(f(\\mathbf{x}^{(i)}), y^{(i)}).$$"
                        ),
                        "body": """The **hypothesis class** $$\\mathcal{F}$$ encodes inductive bias. Linear models use $$\\mathcal{F} = \\{f(\\mathbf{x}) = \\mathbf{w}^\\top\\mathbf{x} + b\\}$$; neural nets use compositional nonlinear functions.

The **true risk** (population loss) is $$R(f) = \\mathbb{E}_{(\\mathbf{x},y)}[\\ell(f(\\mathbf{x}), y)]$$. We only observe the empirical proxy $$\\hat{R}(f)$$. Generalization is the gap $$R(\\hat{f}) - \\hat{R}(\\hat{f})$$.

![Train vs test error as model complexity grows](/assets/figures/day02/pdf0_page005.png)""",
                    },
                    {
                        "heading": "Bias, variance, and noise",
                        "definition": (
                            "For squared loss, expected test error decomposes as "
                            "**bias² + variance + irreducible noise**. High bias underfits; "
                            "high variance overfits."
                        ),
                        "body": """For an estimator $$\\hat{f}$$ at input $$\\mathbf{x}$$,

$$\\mathbb{E}[(y - \\hat{f}(\\mathbf{x}))^2] = \\underbrace{(\\mathbb{E}[\\hat{f}(\\mathbf{x})] - f^\\*(\\mathbf{x}))^2}_{\\text{bias}^2} + \\underbrace{\\mathrm{Var}(\\hat{f}(\\mathbf{x}))}_{\\text{variance}} + \\underbrace{\\sigma^2}_{\\text{noise}}.$$

Complex models reduce bias but increase variance. **Model selection** (cross-validation, validation sets) estimates out-of-sample performance without peeking at the test set.""",
                    },
                ],
            },
            {
                "title": "Linear and Logistic Regression",
                "subsections": [
                    {
                        "heading": "Linear regression",
                        "definition": (
                            "**Linear regression** models $$\\mathbb{E}[y|\\mathbf{x}] = "
                            "\\mathbf{w}^\\top\\mathbf{x} + b$$. Under Gaussian noise, "
                            "OLS minimizes mean squared error."
                        ),
                        "body": """Vectorized training loss:

$$L(\\mathbf{w}) = \\frac{1}{2n}\\|\\mathbf{X}\\mathbf{w} - \\mathbf{y}\\|_2^2.$$

Gradient descent update with learning rate $$\\eta$$:

$$\\mathbf{w} \\leftarrow \\mathbf{w} - \\frac{\\eta}{n}\\mathbf{X}^\\top(\\mathbf{X}\\mathbf{w} - \\mathbf{y}).$$

![Linear fit with residual bands](/assets/figures/day02/pdf0_page010.png)

**Polynomial regression** remains linear in parameters: features $$\\phi(\\mathbf{x}) = (1, x, x^2, \\ldots)$$ and $$\\hat{y} = \\mathbf{w}^\\top\\phi(\\mathbf{x})$$.""",
                    },
                    {
                        "heading": "Logistic regression",
                        "definition": (
                            "For binary labels $$y \\in \\{0,1\\}$$, **logistic regression** "
                            "models $$p(y=1|\\mathbf{x}) = \\sigma(\\mathbf{w}^\\top\\mathbf{x} + b)$$ "
                            "where $$\\sigma(z) = 1/(1+e^{-z})$$."
                        ),
                        "body": """Cross-entropy loss for one example:

$$\\ell = -\\big[y\\log \\hat{p} + (1-y)\\log(1-\\hat{p})\\big], \\quad \\hat{p} = \\sigma(\\mathbf{w}^\\top\\mathbf{x}).$$

The gradient has the elegant form $$\\nabla_{\\mathbf{w}} \\ell = (\\hat{p} - y)\\mathbf{x}$$ — structurally identical to linear regression but with a nonlinear link.

Multiclass **softmax regression** uses

$$p(y=k|\\mathbf{x}) = \\frac{e^{\\mathbf{w}_k^\\top \\mathbf{x}}}{\\sum_j e^{\\mathbf{w}_j^\\top \\mathbf{x}}}.$$

This is the output layer of most classifiers.""",
                    },
                ],
            },
            {
                "title": "Classification Metrics and Decision Theory",
                "subsections": [
                    {
                        "heading": "Confusion matrix and thresholding",
                        "definition": (
                            "A **confusion matrix** tabulates TP, FP, TN, FN. "
                            "**Precision** = TP/(TP+FP); **recall** = TP/(TP+FN). "
                            "The **ROC curve** plots TPR vs FPR as the threshold varies."
                        ),
                        "body": """**F1 score** harmonically combines precision and recall:

$$F_1 = 2 \\cdot \\frac{\\text{precision} \\cdot \\text{recall}}{\\text{precision} + \\text{recall}}.$$

For imbalanced data, accuracy is misleading; prefer PR-AUC or balanced metrics.

![ROC and precision-recall curves](/assets/figures/day02/pdf0_page015.png)

**Bayes optimal classifier** chooses $$\\hat{y} = \\arg\\max_k p(y=k|\\mathbf{x})$$, minimizing 0–1 loss. Logistic regression approximates this when the model is well-specified.""",
                    },
                    {
                        "heading": "Loss functions and surrogates",
                        "definition": (
                            "0–1 loss is non-differentiable; we optimize **surrogate losses** "
                            "(log loss, hinge loss) that upper-bound or approximate classification error."
                        ),
                        "body": """**Hinge loss** for SVMs:

$$\\ell_{\\mathrm{hinge}} = \\max(0, 1 - y\\, f(\\mathbf{x})), \\quad y \\in \\{-1,+1\\}.$$

**Calibration**: predicted probabilities should match empirical frequencies. Platt scaling and isotonic regression post-process scores.

Expected cost under asymmetric misclassification costs $$C_{ij}$$:

$$R = \\sum_{i,j} C_{ij}\\, p(y=j|\\mathbf{x})\\, \\mathbb{1}[\\hat{y}=i].$$""",
                    },
                ],
            },
            {
                "title": "Regularization and Model Selection",
                "subsections": [
                    {
                        "heading": "Ridge, Lasso, and elastic net",
                        "definition": (
                            "**Ridge** ($$\\ell_2$$) penalizes $$\\lambda\\|\\mathbf{w}\\|_2^2$; "
                            "**Lasso** ($$\\ell_1$$) penalizes $$\\lambda\\|\\mathbf{w}\\|_1$ and "
                            "induces sparsity. **Elastic net** combines both."
                        ),
                        "body": """Ridge closed form (when invertible):

$$\\hat{\\mathbf{w}}_{\\mathrm{ridge}} = (\\mathbf{X}^\\top\\mathbf{X} + \\lambda \\mathbf{I})^{-1}\\mathbf{X}^\\top\\mathbf{y}.$$

The penalty shrinks weights toward zero, reducing variance. Geometrically, Lasso's diamond constraint promotes exact zeros at optimality.

![Regularization paths as $$\\lambda$$ varies](/assets/figures/day02/pdf0_page020.png)

**Early stopping** is implicit regularization: halt gradient descent before training loss reaches zero.""",
                    },
                    {
                        "heading": "Cross-validation and hyperparameters",
                        "definition": (
                            "**k-fold cross-validation** splits data into $$k$$ folds, "
                            "training on $$k-1$$ and validating on the held-out fold. "
                            "Hyperparameters ($$\\lambda$$, degree, etc.) are tuned on validation data only."
                        ),
                        "body": """Average validation score:

$$\\mathrm{CV}(\\lambda) = \\frac{1}{k}\\sum_{j=1}^k \\hat{R}_{\\mathrm{val}}^{(j)}(\\lambda).$$

Choose $$\\lambda^\\* = \\arg\\min \\mathrm{CV}(\\lambda)$$, then optionally refit on all training data.

**Nested CV** separates hyperparameter tuning from performance estimation to avoid optimistic bias.

Information criteria (AIC, BIC) penalize model complexity: $$\\mathrm{BIC} = -2\\ell + p\\log n$$ where $$p$$ is parameter count.""",
                    },
                ],
            },
        ],
        "checkpoint": [
            "ERM minimizes average training loss; generalization depends on bias–variance balance.",
            "Linear regression = Gaussian MLE; logistic regression = Bernoulli MLE with sigmoid link.",
            "Use appropriate metrics (F1, AUC) when classes are imbalanced.",
            "Regularization and validation control overfitting without changing the hypothesis class.",
        ],
    },
    {
        "day": 3,
        "slug": "deep-neural-networks",
        "title": "Deep Neural Networks",
        "description": "Backpropagation, activation functions, and stochastic optimization with SGD and Adam.",
        "reading": [
            "[Goodfellow, Bengio & Courville — Deep Learning](https://www.deeplearningbook.org/), Ch. 6–8",
            "[UCLxDeepMind DL2020 — Lecture 2](https://www.youtube.com/playlist?list=PL4LiN2XjWqV8Z9qZqZqZqZqZqZqZqZqZq)",
            "[Karpathy — micrograd](https://github.com/karpathy/micrograd)",
        ],
        "intro": (
            "Deep networks stack nonlinear transformations to learn hierarchical representations. "
            "We derive backpropagation, compare activation functions, and study modern optimizers "
            "that make training large models practical."
        ),
        "sections": [
            {
                "title": "Feedforward Architecture",
                "subsections": [
                    {
                        "heading": "Layer composition",
                        "definition": (
                            "An $$L$$-layer **MLP** computes $$\\mathbf{h}^{(\\ell)} = "
                            "g^{(\\ell)}(\\mathbf{W}^{(\\ell)}\\mathbf{h}^{(\\ell-1)} + "
                            "\\mathbf{b}^{(\\ell)})$$ with $$\\mathbf{h}^{(0)} = \\mathbf{x}$$ and "
                            "output $$\\hat{\\mathbf{y}} = \\mathbf{h}^{(L)}$$."
                        ),
                        "body": """Each layer applies an affine map followed by a nonlinear **activation** $$g$$. Without nonlinearity, the stack collapses to a single linear map.

Universal approximation: a one-hidden-layer network with enough units can approximate continuous functions on compact sets arbitrarily well — depth often improves parameter efficiency.

![Deep vs wide network schematic](/assets/figures/day03/pdf0_page005.png)

Parameter count for layers of sizes $$(d_0, d_1, \\ldots, d_L)$$:

$$|\\theta| = \\sum_{\\ell=1}^L d_{\\ell-1} d_\\ell + d_\\ell \\quad \\text{(weights + biases)}.$$""",
                    },
                    {
                        "heading": "Forward pass notation",
                        "definition": (
                            "Pre-activations $$\\mathbf{z}^{(\\ell)} = \\mathbf{W}^{(\\ell)}"
                            "\\mathbf{h}^{(\\ell-1)} + \\mathbf{b}^{(\\ell)}$$; "
                            "activations $$\\mathbf{h}^{(\\ell)} = g(\\mathbf{z}^{(\\ell)})$$."
                        ),
                        "body": """Batch forward pass for $$\\mathbf{X} \\in \\mathbb{R}^{n \\times d_0}$$:

$$\\mathbf{Z}^{(\\ell)} = \\mathbf{H}^{(\\ell-1)}\\mathbf{W}^{(\\ell)\\top} + \\mathbf{1}\\mathbf{b}^{(\\ell)\\top}, \\quad \\mathbf{H}^{(\\ell)} = g(\\mathbf{Z}^{(\\ell)}).$$

Matrix dimensions must align: $$\\mathbf{W}^{(\\ell)} \\in \\mathbb{R}^{d_\\ell \\times d_{\\ell-1}}$$.

Caching $$\\mathbf{z}^{(\\ell)}$$ and $$\\mathbf{h}^{(\\ell)}$$ during forward pass is required for efficient backward pass.""",
                    },
                ],
            },
            {
                "title": "Backpropagation",
                "subsections": [
                    {
                        "heading": "Output layer gradients",
                        "definition": (
                            "Given loss $$L$$, define error signal $$\\boldsymbol{\\delta}^{(L)} = "
                            "\\nabla_{\\mathbf{z}^{(L)}} L$$. For softmax + cross-entropy with one-hot "
                            "$$\\mathbf{y}$$, $$\\boldsymbol{\\delta}^{(L)} = \\hat{\\mathbf{y}} - \\mathbf{y}$$."
                        ),
                        "body": """Weight gradient for layer $$\\ell$$:

$$\\frac{\\partial L}{\\partial \\mathbf{W}^{(\\ell)}} = \\boldsymbol{\\delta}^{(\\ell)} \\mathbf{h}^{(\\ell-1)\\top}.$$

This is an outer product of output error and input activation — the foundation of efficient GPU kernels.

![Backprop error flow through layers](/assets/figures/day03/pdf0_page010.png)""",
                    },
                    {
                        "heading": "Hidden layer backprop",
                        "definition": (
                            "Errors propagate backward: $$\\boldsymbol{\\delta}^{(\\ell)} = "
                            "(\\mathbf{W}^{(\\ell+1)\\top}\\boldsymbol{\\delta}^{(\\ell+1)}) \\odot "
                            "g'(\\mathbf{z}^{(\\ell)})$$ where $$\\odot$$ is element-wise product."
                        ),
                        "body": """The chain rule multiplies upstream gradient by local derivative. For ReLU, $$g'(z) = \\mathbb{1}[z > 0]$$ — dead neurons have zero gradient.

**Vanishing gradients**: if $$|g'(z)| < 1$$ across many layers, $$\\boldsymbol{\\delta}^{(1)}$$ shrinks exponentially. **Exploding gradients** cause unstable training; **gradient clipping** mitigates:

$$\\mathbf{g} \\leftarrow \\mathbf{g} \\cdot \\min\\!\\left(1, \\frac{\\tau}{\\|\\mathbf{g}\\|}\\right).$$

Modern architectures (residual connections, LayerNorm) were designed to keep gradient magnitudes healthy.""",
                    },
                ],
            },
            {
                "title": "Activation Functions",
                "subsections": [
                    {
                        "heading": "ReLU family",
                        "definition": (
                            "**ReLU**: $$g(z) = \\max(0, z)$$. **Leaky ReLU**: "
                            "$$g(z) = z$$ if $$z > 0$$ else $$\\alpha z$$. **GELU**: "
                            "$$g(z) = z\\,\\Phi(z)$$ where $$\\Phi$$ is the standard normal CDF."
                        ),
                        "body": """ReLU is cheap and avoids saturation on the positive side but can cause **dead neurons**. Leaky ReLU and **Parametric ReLU** (learnable slope) address this.

**Swish / SiLU**: $$g(z) = z\\,\\sigma(z)$$ — smooth, non-monotonic, used in many modern architectures.

![Activation shapes compared](/assets/figures/day03/pdf0_page015.png)

Compare saturation regions: sigmoid/tanh squash large inputs, killing gradients in deep nets pre-2012.""",
                    },
                    {
                        "heading": "Sigmoid, tanh, and softmax",
                        "definition": (
                            "**Sigmoid** $$\\sigma(z) = 1/(1+e^{-z})$$; **tanh** $$= 2\\sigma(2z)-1$$ "
                            "zero-centered. **Softmax** normalizes logits to a probability vector."
                        ),
                        "body": """Sigmoid derivative: $$\\sigma'(z) = \\sigma(z)(1-\\sigma(z)) \\leq 1/4$$.

Tanh derivative: $$1 - \\tanh^2(z)$$, zero-centered so gradients are better conditioned than sigmoid.

Softmax Jacobian is rank-deficient; combined with cross-entropy the gradient simplifies to $$\\hat{\\mathbf{y}} - \\mathbf{y}$$ — always use the fused implementation.""",
                    },
                ],
            },
            {
                "title": "Optimization: SGD and Adam",
                "subsections": [
                    {
                        "heading": "Stochastic gradient descent",
                        "definition": (
                            "**SGD** updates $$\\theta_{t+1} = \\theta_t - \\eta_t "
                            "\\nabla_\\theta \\hat{L}_B$$ where $$\\hat{L}_B$$ is loss on a "
                            "mini-batch $$B$$ of size $$|B| \\ll n$$."
                        ),
                        "body": """Mini-batch gradient is an unbiased estimator of full gradient:

$$\\mathbb{E}[\\nabla \\hat{L}_B] = \\nabla L, \\quad \\mathrm{Var}(\\nabla \\hat{L}_B) \\propto \\frac{1}{|B|}.$$

**Momentum** accumulates velocity:

$$\\mathbf{v}_{t+1} = \\beta \\mathbf{v}_t + \\nabla L, \\quad \\theta_{t+1} = \\theta_t - \\eta \\mathbf{v}_{t+1}.$$

![SGD trajectory vs full-batch GD](/assets/figures/day03/pdf0_page020.png)

Learning rate schedules: step decay, cosine annealing, warm-up — critical for transformers and large-batch training.""",
                    },
                    {
                        "heading": "Adam and adaptive methods",
                        "definition": (
                            "**Adam** maintains per-parameter first moment $$\\mathbf{m}_t$$ and "
                            "second moment $$\\mathbf{v}_t$$ (exponential moving averages), "
                            "with bias correction and update "
                            "$$\\theta_{t+1} = \\theta_t - \\eta \\hat{\\mathbf{m}}_t / "
                            "(\\sqrt{\\hat{\\mathbf{v}}_t} + \\epsilon)$$."
                        ),
                        "body": """Adam update (element-wise):

$$\\begin{aligned}
\\mathbf{m}_t &= \\beta_1 \\mathbf{m}_{t-1} + (1-\\beta_1)\\mathbf{g}_t \\\\
\\mathbf{v}_t &= \\beta_2 \\mathbf{v}_{t-1} + (1-\\beta_2)\\mathbf{g}_t^2 \\\\
\\hat{\\mathbf{m}}_t &= \\mathbf{m}_t / (1-\\beta_1^t), \\quad \\hat{\\mathbf{v}}_t = \\mathbf{v}_t / (1-\\beta_2^t)
\\end{aligned}$$

Default $$\\beta_1=0.9$$, $$\\beta_2=0.999$$, $$\\epsilon=10^{-8}$$. **AdamW** decouples weight decay from the adaptive step — preferred for transformers.

**Weight initialization** (Xavier, He) sets variance of activations stable across layers: He init uses $$\\mathcal{N}(0, 2/n_{\\mathrm{in}})$$ for ReLU nets.""",
                    },
                ],
            },
        ],
        "checkpoint": [
            "MLPs compose affine maps and nonlinear activations; depth enables hierarchical features.",
            "Backprop = chain rule: cache forward values, propagate $$\\delta$$ errors backward.",
            "ReLU and GELU dominate modern nets; watch for dead neurons and gradient health.",
            "SGD + momentum + Adam/AdamW are the default training recipe; tune $$\\eta$$ and batch size.",
        ],
    },
    {
        "day": 4,
        "slug": "convolutional-networks",
        "title": "Convolutional Networks",
        "description": "Convolution, pooling, and canonical CNN architectures for vision.",
        "reading": [
            "[Goodfellow et al. — Deep Learning, Ch. 9](https://www.deeplearningbook.org/contents/convnets.html)",
            "[CS231n — Convolutional Neural Networks](https://cs231n.github.io/convolutional-networks/)",
            "[He et al. — Deep Residual Learning (ResNet)](https://arxiv.org/abs/1512.03385)",
        ],
        "intro": (
            "Convolutional neural networks exploit translation equivariance and local connectivity "
            "to process images efficiently. We build up from the 2D convolution operation through "
            "pooling layers to ResNet-style architectures used in modern vision systems."
        ),
        "sections": [
            {
                "title": "The Convolution Operation",
                "subsections": [
                    {
                        "heading": "Discrete 2D convolution",
                        "definition": (
                            "The **2D convolution** of input $$\\mathbf{X}$$ and kernel $$\\mathbf{K}$$ "
                            "produces output $$\\mathbf{Y}$$ with entries "
                            "$$Y_{ij} = \\sum_{u,v} K_{uv}\\, X_{i+u,\\,j+v}$$ "
                            "(cross-correlation is used in most DL frameworks)."
                        ),
                        "body": """For input $$\\mathbf{X} \\in \\mathbb{R}^{H \\times W \\times C_{\\mathrm{in}}}$$ and $$C_{\\mathrm{out}}$$ filters of size $$k \\times k$$,

$$\\mathbf{Y}_{h,w,c'} = \\sum_{c=1}^{C_{\\mathrm{in}}} \\sum_{i=0}^{k-1}\\sum_{j=0}^{k-1} W_{i,j,c,c'}\\, X_{h+i,\\,w+j,\\,c} + b_{c'}.$$

**Parameter sharing**: one filter slides across the entire spatial grid — far fewer weights than a fully connected layer on flattened pixels.

![Convolution sliding window](/assets/figures/day04/pdf0_page005.png)

Output spatial size with padding $$p$$ and stride $$s$$:

$$H_{\\mathrm{out}} = \\left\\lfloor \\frac{H + 2p - k}{s} \\right\\rfloor + 1.$$""",
                    },
                    {
                        "heading": "Translation equivariance",
                        "definition": (
                            "A layer is **translation equivariant** if shifting the input shifts "
                            "the output: $$f(\\mathrm{shift}(\\mathbf{x})) = \\mathrm{shift}(f(\\mathbf{x}))$$. "
                            "Conv layers satisfy this; fully connected layers do not."
                        ),
                        "body": """Equivariance is ideal for detection and segmentation: a cat in the corner activates the same filter as a cat in the center.

**1×1 convolutions** mix channels without changing spatial size — used in Inception and pointwise expansions in MobileNet.

Depthwise separable conv factorizes into depthwise (spatial) + pointwise (channel) steps, reducing FLOPs:

$$\\text{standard} \\; O(k^2 C_{\\mathrm{in}} C_{\\mathrm{out}}), \\quad \\text{separable} \\; O(k^2 C_{\\mathrm{in}} + C_{\\mathrm{in}} C_{\\mathrm{out}}).$$""",
                    },
                ],
            },
            {
                "title": "Pooling and Normalization",
                "subsections": [
                    {
                        "heading": "Max and average pooling",
                        "definition": (
                            "**Max pooling** takes the maximum in each $$p \\times p$$ window; "
                            "**average pooling** takes the mean. Both reduce spatial resolution "
                            "and provide local translation invariance."
                        ),
                        "body": """Max pool with stride $$s = p$$ halves spatial dimensions (typical $$p=2$$):

$$Y_{i,j} = \\max_{0 \\leq u,v < p} X_{s i + u,\\, s j + v}.$$

Pooling has no learnable parameters but downsamples activations, expanding receptive field in deeper layers.

![Max pool 2×2 stride 2](/assets/figures/day04/pdf0_page010.png)

**Global average pooling** (GAP) averages each feature map to a scalar — common before classification head in ResNet, replacing large fully connected layers.""",
                    },
                    {
                        "heading": "Batch normalization",
                        "definition": (
                            "**BatchNorm** normalizes pre-activations across the mini-batch: "
                            "$$\\hat{z} = (z - \\mu_B)/\\sqrt{\\sigma_B^2 + \\epsilon}$$, "
                            "then applies learnable scale $$\\gamma$$ and shift $$\\beta$$."
                        ),
                        "body": """During training, batch statistics $$\\mu_B, \\sigma_B^2$$; at inference, use running averages $$\\mu_{\\mathrm{EMA}}, \\sigma_{\\mathrm{EMA}}^2$$.

BatchNorm reduces internal covariate shift, allows higher learning rates, and acts as mild regularization.

**LayerNorm** normalizes across features per example — preferred in transformers and RNNs where batch statistics are unreliable.""",
                    },
                ],
            },
            {
                "title": "Canonical Architectures",
                "subsections": [
                    {
                        "heading": "LeNet, AlexNet, and VGG",
                        "definition": (
                            "**LeNet-5** (1998): conv → pool → conv → pool → FC. "
                            "**AlexNet** (2012): deeper, ReLU, dropout, GPU training. "
                            "**VGG**: stacks of 3×3 convs — simplicity over large filters."
                        ),
                        "body": """AlexNet showed that depth + ReLU + data augmentation could dominate ImageNet:

$$\\text{top-5 error: } 15.3\\% \\; (2012) \\; \\text{vs } 26\\% \\; \\text{(previous best)}.$$

VGG-16 uses only 3×3 convs: two 3×3 layers have receptive field 5×5 but more nonlinearities and fewer parameters than one 5×5 layer.

![Architecture comparison timeline](/assets/figures/day04/pdf0_page015.png)""",
                    },
                    {
                        "heading": "ResNet and skip connections",
                        "definition": (
                            "A **residual block** learns $$\\mathcal{F}(\\mathbf{x})$$ with "
                            "$$\\mathbf{y} = \\mathbf{x} + \\mathcal{F}(\\mathbf{x})$$. "
                            "Gradients flow directly through the skip, enabling 100+ layer nets."
                        ),
                        "body": """If optimal mapping is close to identity, residual form lets $$\\mathcal{F} \\approx 0$$ rather than learning identity explicitly.

Bottleneck block (1×1 → 3×3 → 1×1) reduces compute in deep models:

$$\\mathbf{y} = \\mathbf{x} + \\mathcal{F}_{1\\times1}(\\mathcal{F}_{3\\times3}(\\mathcal{F}_{1\\times1}(\\mathbf{x}))).$$

**Receptive field** grows with depth: stacking $$L$$ layers of 3×3 conv gives RF $$\\approx 1 + 2L$$ (without dilation).""",
                    },
                ],
            },
            {
                "title": "Training CNNs for Vision",
                "subsections": [
                    {
                        "heading": "Data augmentation",
                        "definition": (
                            "**Data augmentation** applies label-preserving transforms "
                            "(random crop, flip, color jitter) to expand effective training set "
                            "size and improve robustness."
                        ),
                        "body": """Standard ImageNet pipeline: random resized crop to 224×224, horizontal flip, normalize with dataset mean/std.

**Mixup** blends pairs of examples:

$$\\tilde{\\mathbf{x}} = \\lambda \\mathbf{x}_i + (1-\\lambda)\\mathbf{x}_j, \\quad \\tilde{y} = \\lambda y_i + (1-\\lambda) y_j.$$

**Cutout / Random Erasing** drops random patches, forcing context reasoning.

![Augmentation examples](/assets/figures/day04/pdf0_page020.png)""",
                    },
                    {
                        "heading": "Transfer learning",
                        "definition": (
                            "**Transfer learning** initializes from pretrained weights "
                            "(ImageNet) and fine-tunes on a target task — essential when "
                            "target data is limited."
                        ),
                        "body": """Typical recipe: replace final FC layer, train head with frozen backbone, then unfreeze top layers with small $$\\eta$$.

Feature maps from early layers capture edges/textures (general); late layers capture class-specific semantics.

**Linear probe** vs **full fine-tune**: probe trains only the head (fast baseline); full fine-tune adapts all weights (better with enough data).

Compute FLOPs for conv layer: $$\\approx 2\\, H_{\\mathrm{out}} W_{\\mathrm{out}} C_{\\mathrm{out}} k^2 C_{\\mathrm{in}}$$ multiply-adds.""",
                    },
                ],
            },
        ],
        "checkpoint": [
            "Conv layers share weights spatially and are translation equivariant.",
            "Pooling downsamples; BatchNorm stabilizes training; GAP feeds compact classifiers.",
            "ResNet skip connections solve degradation and enable very deep nets.",
            "Augmentation + transfer learning are standard for practical vision pipelines.",
        ],
    },
    {
        "day": 5,
        "slug": "rnns-and-transformers",
        "title": "RNNs and Transformers",
        "description": "Recurrent models, LSTM, self-attention, and causal masking for sequences.",
        "reading": [
            "[Attention Is All You Need (Vaswani et al.)](https://arxiv.org/abs/1706.03762)",
            "[Karpathy — The Unreasonable Effectiveness of RNNs](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)",
            "[Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)",
        ],
        "intro": (
            "Sequential data requires models that carry context across time steps. We begin with "
            "RNNs and LSTMs, then pivot to the Transformer architecture where self-attention "
            "replaces recurrence and causal masking enables autoregressive language modeling."
        ),
        "sections": [
            {
                "title": "Recurrent Neural Networks",
                "subsections": [
                    {
                        "heading": "Vanilla RNN",
                        "definition": (
                            "An **RNN** maintains hidden state $$\\mathbf{h}_t$$ updated as "
                            "$$\\mathbf{h}_t = g(\\mathbf{W}_{hh}\\mathbf{h}_{t-1} + "
                            "\\mathbf{W}_{xh}\\mathbf{x}_t + \\mathbf{b})$$ and emits "
                            "$$\\mathbf{y}_t = \\mathbf{W}_{hy}\\mathbf{h}_t$$."
                        ),
                        "body": """The same weights are applied at every time step — parameter sharing across sequence length $$T$$.

Unrolled through time, backpropagation through time (BPTT) applies the chain rule across $$T$$ steps:

$$\\frac{\\partial L}{\\partial \\mathbf{h}_t} = \\frac{\\partial L}{\\partial \\mathbf{h}_{t+1}} \\frac{\\partial \\mathbf{h}_{t+1}}{\\partial \\mathbf{h}_t} + \\ldots$$

![RNN unrolled through time](/assets/figures/day05/pdf0_page005.png)

Vanishing/exploding gradients limit vanilla RNNs on long sequences — motivates LSTM and GRU.""",
                    },
                    {
                        "heading": "Bidirectional and many-to-many models",
                        "definition": (
                            "**Bidirectional RNNs** combine forward $$\\overrightarrow{\\mathbf{h}}_t$$ "
                            "and backward $$\\overleftarrow{\\mathbf{h}}_t$$ states. "
                            "Not usable for autoregressive generation (future leakage)."
                        ),
                        "body": """Sequence labeling (NER, POS tagging) uses bidirectional context:

$$\\mathbf{h}_t = [\\overrightarrow{\\mathbf{h}}_t ; \\overleftarrow{\\mathbf{h}}_t].$$

Architecture patterns:
- **Many-to-one**: sentiment (last $$\\mathbf{h}_T$$)
- **One-to-many**: image captioning
- **Seq2seq**: encoder RNN → context vector → decoder RNN

Teacher forcing feeds ground-truth $$\\mathbf{y}_t$$ to decoder during training; exposure bias at inference.""",
                    },
                ],
            },
            {
                "title": "LSTM and Gated Recurrence",
                "subsections": [
                    {
                        "heading": "LSTM cell equations",
                        "definition": (
                            "An **LSTM** uses gates to control information flow: "
                            "**forget** $$\\mathbf{f}_t = \\sigma(\\cdot)$$, "
                            "**input** $$\\mathbf{i}_t = \\sigma(\\cdot)$$, "
                            "**output** $$\\mathbf{o}_t = \\sigma(\\cdot)$$, "
                            "and cell state $$\\mathbf{c}_t$$."
                        ),
                        "body": """Standard LSTM (element-wise operations):

$$\\begin{aligned}
\\mathbf{f}_t &= \\sigma(\\mathbf{W}_f [\\mathbf{h}_{t-1}, \\mathbf{x}_t] + \\mathbf{b}_f) \\\\
\\mathbf{i}_t &= \\sigma(\\mathbf{W}_i [\\mathbf{h}_{t-1}, \\mathbf{x}_t] + \\mathbf{b}_i) \\\\
\\tilde{\\mathbf{c}}_t &= \\tanh(\\mathbf{W}_c [\\mathbf{h}_{t-1}, \\mathbf{x}_t] + \\mathbf{b}_c) \\\\
\\mathbf{c}_t &= \\mathbf{f}_t \\odot \\mathbf{c}_{t-1} + \\mathbf{i}_t \\odot \\tilde{\\mathbf{c}}_t \\\\
\\mathbf{h}_t &= \\mathbf{o}_t \\odot \\tanh(\\mathbf{c}_t)
\\end{aligned}$$

![LSTM gate diagram](/assets/figures/day05/pdf0_page010.png)

The additive cell update $$\\mathbf{c}_t = \\mathbf{f}_t \\odot \\mathbf{c}_{t-1} + \\ldots$$ provides a gradient highway — mitigating vanishing gradients.""",
                    },
                    {
                        "heading": "GRU and when to use recurrence",
                        "definition": (
                            "**GRU** merges forget/input gates into **update gate** $$\\mathbf{z}_t$$ "
                            "and **reset gate** $$\\mathbf{r}_t$$ — fewer parameters than LSTM, "
                            "often comparable performance."
                        ),
                        "body": """GRU update:

$$\\mathbf{h}_t = (1 - \\mathbf{z}_t) \\odot \\mathbf{h}_{t-1} + \\mathbf{z}_t \\odot \\tilde{\\mathbf{h}}_t.$$

RNNs excel on small sequences and streaming data but parallelize poorly ($$O(T)$$ sequential steps). Transformers replaced RNNs in most NLP and are expanding into vision and genomics.

Truncated BPTT: backprop only over last $$K$$ steps to save memory on long sequences.""",
                    },
                ],
            },
            {
                "title": "Self-Attention Mechanism",
                "subsections": [
                    {
                        "heading": "Scaled dot-product attention",
                        "definition": (
                            "Given queries $$\\mathbf{Q}$$, keys $$\\mathbf{K}$$, values $$\\mathbf{V}$$, "
                            "**attention** computes "
                            "$$\\mathrm{Attention}(\\mathbf{Q},\\mathbf{K},\\mathbf{V}) = "
                            "\\mathrm{softmax}(\\mathbf{Q}\\mathbf{K}^\\top / \\sqrt{d_k})\\mathbf{V}.$$"
                        ),
                        "body": """For sequence length $$n$$ and dimension $$d_k$$, attention maps are $$n \\times n$$ — each token attends to all others.

Scaling by $$\\sqrt{d_k}$$ prevents softmax saturation when dot products grow large.

![Attention weight heatmap](/assets/figures/day05/pdf0_page015.png)

**Self-attention**: $$\\mathbf{Q}, \\mathbf{K}, \\mathbf{V}$$ all derived from the same input sequence via learned projections $$\\mathbf{W}_Q, \\mathbf{W}_K, \\mathbf{W}_V$$.""",
                    },
                    {
                        "heading": "Multi-head attention",
                        "definition": (
                            "**Multi-head attention** runs $$h$$ parallel attention heads with "
                            "different projections, concatenates, and projects again: "
                            "$$\\mathrm{MultiHead} = \\mathrm{Concat}(\\mathrm{head}_1,\\ldots,"
                            "\\mathrm{head}_h)\\mathbf{W}^O$$."
                        ),
                        "body": """Each head learns different relational patterns (syntax, coreference, locality).

Transformer block:

$$\\mathbf{x}' = \\mathbf{x} + \\mathrm{MultiHead}(\\mathrm{LN}(\\mathbf{x}))$$
$$\\mathbf{x}'' = \\mathbf{x}' + \\mathrm{FFN}(\\mathrm{LN}(\\mathbf{x}'))$$

FFN is typically two linear layers with GELU: $$\\mathrm{FFN}(\\mathbf{x}) = \\mathbf{W}_2\\,\\mathrm{GELU}(\\mathbf{W}_1 \\mathbf{x} + \\mathbf{b}_1) + \\mathbf{b}_2$$.

Complexity: $$O(n^2 d)$$ per layer — quadratic in sequence length motivates sparse and linear attention variants.""",
                    },
                ],
            },
            {
                "title": "Causal Masking and Autoregressive Models",
                "subsections": [
                    {
                        "heading": "Causal (look-ahead) mask",
                        "definition": (
                            "A **causal mask** sets attention logits to $$-\\infty$$ for "
                            "positions $$j > i$$ so token $$i$$ cannot attend to future tokens — "
                            "required for autoregressive language modeling."
                        ),
                        "body": """Masked attention matrix (lower triangular after softmax):

$$A_{ij} = \\begin{cases} \\mathrm{softmax}(\\mathbf{q}_i^\\top \\mathbf{k}_j / \\sqrt{d_k}) & j \\leq i \\\\ 0 & j > i \\end{cases}$$

Implemented by adding a mask $$M_{ij} = 0$$ if $$j \\leq i$$ else $$-\\infty$$ before softmax.

![Causal mask pattern](/assets/figures/day05/pdf0_page020.png)

**Encoder** (BERT): bidirectional, no causal mask. **Decoder** (GPT): causal mask only.""",
                    },
                    {
                        "heading": "Autoregressive training objective",
                        "definition": (
                            "An **autoregressive LM** factorizes $$p(x_1,\\ldots,x_T) = "
                            "\\prod_{t=1}^T p(x_t | x_{<t})$$ and trains by minimizing "
                            "cross-entropy on next-token prediction."
                        ),
                        "body": """Loss for token sequence:

$$L = -\\sum_{t=1}^T \\log p_\\theta(x_t \\mid x_1, \\ldots, x_{t-1}).$$

At inference, **greedy decoding** picks $$\\hat{x}_t = \\arg\\max p(x_t | x_{<t})$$; **sampling** with temperature $$\\tau$$:

$$p(x_t) \\propto \\exp(z_t / \\tau).$$

**KV cache** stores past key/value projections so each new token costs $$O(n)$$ not $$O(n^2)$$ per step — essential for efficient LLM inference.

Positional information: sinusoidal encodings (original Transformer) or **rotary embeddings (RoPE)** in modern LLMs.""",
                    },
                ],
            },
        ],
        "checkpoint": [
            "RNNs share weights over time; BPTT backprops through unrolled graph; LSTM gates help long-range deps.",
            "Self-attention weighs all pairs of tokens; multi-head captures diverse relations.",
            "Causal masking enforces $$p(x_t | x_{<t})$$ — no peeking at the future.",
            "Autoregressive LMs train with next-token CE; KV cache speeds up generation.",
        ],
    },
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
---"""


def render_lecture(lecture: dict) -> str:
    day = lecture["day"]
    day_str = f"{day:02d}"
    title = lecture["title"]
    lines = [
        front_matter(day, title, lecture["description"]),
        "",
        f"# Day {day} - {title}",
        "",
        "### Optional reading for this lesson",
    ]
    for link in lecture["reading"]:
        lines.append(f"- {link}")
    lines.extend(
        [
            "",
            f"### [Slides](/assets/slides/day{day_str}.pdf)",
            "",
            f"### [Practical](/projects/day{day_str}-practical/)",
            "",
            lecture["intro"],
            "",
            "* toc",
            "{:toc}",
            "",
        ]
    )

    for sec_idx, section in enumerate(lecture["sections"], start=1):
        lines.append(f"## {sec_idx}. {section['title']}")
        lines.append("")
        for sub in section["subsections"]:
            lines.append(f"### {sec_idx}.{section['subsections'].index(sub) + 1} {sub['heading']}")
            lines.append("")
            lines.append(f"> {sub['definition']}")
            lines.append("{:.lead}")
            lines.append("")
            lines.extend(sub["body"].splitlines())
            lines.append("")

    lines.extend(
        [
            "## Checkpoint summary",
            "",
            "Before moving to the practical, confirm you can:",
            "",
        ]
    )
    for item in lecture["checkpoint"]:
        lines.append(f"- {item}")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    POSTS_DIR.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    for i, lecture in enumerate(LECTURES):
        post_date = START_DATE + timedelta(days=i)
        filename = f"{post_date.isoformat()}-day{lecture['day']:02d}-{lecture['slug']}.md"
        path = POSTS_DIR / filename
        content = render_lecture(lecture)
        path.write_text(content, encoding="utf-8")
        line_count = content.count("\n") + 1
        created.append(path)
        print(f"Wrote {path.name} ({line_count} lines)")

    print(f"\nCreated {len(created)} lecture posts in {POSTS_DIR}")


if __name__ == "__main__":
    main()
