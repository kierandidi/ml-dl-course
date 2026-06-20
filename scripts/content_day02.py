"""Day 2 — Statistical learning (MML Part II: Ch. 8–12)."""

FIGURES = [
    # 1. When Models Meet Data (Ch. 8)
    "/assets/figures/day02/ml_taxonomy.png",
    None,
    "/assets/figures/day02/mml_toy_regression.png",
    None,
    "/assets/figures/day02/mml_cross_validation.png",
    None,
    # 2. Linear Regression (Ch. 9)
    "/assets/figures/day02/mml_toy_regression.png",
    None,
    "/assets/figures/day02/mml_linear_regression.png",
    None,
    None,
    "/assets/figures/day02/mml_poly_overfit.png",
    # 3. Dimensionality Reduction / PCA (Ch. 10)
    None,
    "/assets/figures/day02/mml_pca_lowdim.png",
    "/assets/figures/day02/mml_pca_projection.png",
    "/assets/figures/day02/mml_pca_projection.png",
    # 4. Density Estimation / GMM (Ch. 11)
    None,
    "/assets/figures/day02/mml_gmm_1d.png",
    None,
    # 5. Classification / SVM (Ch. 12)
    "/assets/figures/day02/mml_svm_2d.png",
    "/assets/figures/day02/mml_svm_hyperplane.png",
    "/assets/figures/day02/mml_svm_margin.png",
    "/assets/figures/day02/mml_soft_margin.png",
]

SLIDES = (
    "Statistical Learning",
    "Regression, PCA, density estimation, classification",
    [
        (
            "When Models Meet Data",
            [
                (
                    "Machine Learning: A Taxonomy",
                    [
                        "Three paradigms: *supervised*, *unsupervised*, *reinforcement* learning",
                        "Supervised: regression (continuous $y$) and classification (discrete labels)",
                        "Unsupervised: clustering, density estimation, dimensionality reduction",
                        "Reinforcement: sequential decisions from reward signals",
                        "Today covers each of the four pillars in turn",
                    ],
                ),
                (
                    "Four Pillars of Statistical Learning",
                    [
                        "*Linear regression*: predict continuous targets from features",
                        "*Dimensionality reduction* (PCA): find compact representations",
                        "*Density estimation* (GMM): model $p(x)$ and discover structure",
                        "*Classification* (SVM): separate classes with margins",
                        "A shared vocabulary underlies all four: data, models, learning, evaluation",
                    ],
                ),
                (
                    "Data, Models, and Learning",
                    [
                        "Dataset $cal(D) = {(x_n, y_n)}_(n=1)^N$; features $x_n in RR^D$ as *vectors*",
                        "A *model* is a parameterized family $p(y|x, theta)$ or $f_theta(x)$",
                        "*Learning* = choose $theta$ from data so predictions generalize to new $x$",
                        "Similarity of examples (geometry from Day 1) drives regression & classification",
                        "Goal: low error on *unseen* test points, not just training fit",
                    ],
                ),
                (
                    "Empirical Risk Minimization",
                    [
                        "True risk: $R(f) = EE[ell(f(x), y)]$ over the data-generating distribution",
                        'Empirical risk: $R_"emp"(f) = (1/N) sum_n ell(f(x_n), y_n)$ — what we can compute',
                        'ERM: $hat(theta) = arg min_(theta) R_"emp"(theta)$ within hypothesis class $cal(H)$',
                        "Loss examples: squared error (regression), cross-entropy (classification)",
                        "Finite data $arrow.r$ ERM can overfit; need validation and model selection",
                    ],
                ),
                (
                    "Train, Validation, Test & Cross-Validation",
                    [
                        "Hold out a *validation* set to tune hyperparameters ($lambda$, degree, $K$)",
                        "Keep a *test* set untouched until final evaluation",
                        "$K$-fold CV: rotate which chunk is validation; average validation error",
                        "$EE_V[R(f,V)] approx (1/K) sum_k R(f^((k)), V^((k)))$",
                        "Reduces variance of performance estimates at cost of $K$ retrains",
                    ],
                ),
                (
                    "Model Selection & Occam's Razor",
                    [
                        "More complex $cal(H)$ lowers training error but may raise test error",
                        "Bias–variance trade-off: underfit (high bias) vs overfit (high variance)",
                        "Occam: prefer the simpler model among those with similar validation error",
                        '$"BIC" = -2 ell + p log N$',
                        "Regularization (next chapters) encodes simplicity directly in the objective",
                    ],
                ),
            ],
        ),
        (
            "Linear Regression",
            [
                (
                    "Problem Setup",
                    [
                        "Given $(x_n, y_n)$, find $f: RR^D to RR$ with $y_n approx f(x_n) + epsilon_n$",
                        "Noise $epsilon_n tilde cal(N)(0, sigma^2)$ (i.i.d. Gaussian)",
                        "Training fits $f$; we care about prediction at *new* inputs (generalization)",
                        "Vector data lets us use linear algebra from Day 1",
                    ],
                ),
                (
                    "Linear in the Parameters",
                    [
                        "$y = phi(x)^T theta$ — linear in $theta$, possibly nonlinear in $x$",
                        "Plain line: $phi(x) = (1, x)^T$; polynomial: $phi(x) = (1, x, x^2, dots)^T$",
                        "Design matrix $Phi in RR^(N times M)$ stacks $phi(x_n)^T$ as rows",
                        "Batch prediction: $hat(y) = Phi theta$",
                    ],
                ),
                (
                    "Maximum Likelihood = Least Squares",
                    [
                        "Gaussian likelihood: $p(y|x, theta) = cal(N)(phi(x)^T theta, sigma^2)$",
                        "Log-likelihood $prop - (1/(2 sigma^2)) sum_n (y_n - phi(x_n)^T theta)^2$",
                        "MLE $equiv$ minimize MSE: $L(theta) = (1/(2N))||Phi theta - y||^2$",
                        "Closed form (when invertible): $theta^* = (Phi^T Phi)^(-1) Phi^T y$",
                        "Geometric view: orthogonal projection of $y$ onto column space of $Phi$",
                    ],
                ),
                (
                    "Ridge Regression",
                    [
                        "When $Phi^T Phi$ is ill-conditioned, OLS has high variance",
                        "Ridge: $L(theta) = (1/(2N))||Phi theta - y||^2 + (lambda/(2N))||theta||^2$",
                        "Closed form: $theta = (Phi^T Phi + lambda I)^(-1) Phi^T y$",
                        "Bayesian view: Gaussian prior on $theta$ $arrow.r$ MAP estimate",
                        "$lambda$ controls bias–variance; tune with cross-validation",
                    ],
                ),
                (
                    "Polynomial Regression & Overfitting",
                    [
                        "High-degree polynomials fit training data arbitrarily well",
                        "Training error keeps falling; *test* error often rises (overfitting)",
                        "Model selection: pick degree via validation or penalize complexity (ridge)",
                        "Figure 9.6: train vs test error as polynomial degree grows",
                        "Lesson: capacity must match data size and noise level",
                    ],
                ),
            ],
        ),
        (
            "Dimensionality Reduction / PCA",
            [
                (
                    "Why Reduce Dimensionality?",
                    [
                        "Curse of dimensionality: data sparse in high $D$; distance metrics degrade",
                        "Goals: compression, denoising, visualization, faster downstream learning",
                        "Unsupervised: only inputs ${x_n}$, no labels",
                        "Linear method PCA; nonlinear extensions (autoencoders) come in Week 2",
                    ],
                ),
                (
                    "PCA: Maximum Variance Perspective",
                    [
                        'Find direction $b_1$ (unit vector) maximizing $"Var"(b_1^T X)$',
                        "Second component $b_2$ orthogonal to $b_1$, max remaining variance, etc.",
                        "Principal components = eigenvectors of sample covariance $S$",
                        "Eigenvalues $lambda_i$ = variance explained along each axis",
                    ],
                ),
                (
                    "PCA: Projection Perspective",
                    [
                        "Best rank-$M$ approximation: project $x$ onto $M$-dim subspace spanned by top PCs",
                        "$tilde(x) = B B^T x$ with $B = [b_1, dots, b_M]$",
                        "Minimizes reconstruction error $sum_n ||x_n - tilde(x)_n||^2$",
                        "Same solution as maximum-variance view (SVD of centered $X$)",
                    ],
                ),
                (
                    "PCA Algorithm & Applications",
                    [
                        "1. Center data: $tilde(x)_n = x_n - bar(x)$",
                        "2. Compute $S = (1/N) sum_n tilde(x)_n tilde(x)_n^T$ (or SVD of $tilde(X)$)",
                        "3. Take top-$M$ eigenvectors; project $z_n = B^T tilde(x)_n$",
                        "Uses: visualization (2D/3D), compression, whitening, feature preprocessing",
                        "Explained variance ratio guides choice of $M$",
                    ],
                ),
            ],
        ),
        (
            "Density Estimation / GMM",
            [
                (
                    "Learning $p(x)$",
                    [
                        "Generative view: model the *input* distribution, not just $p(y|x)$",
                        "Parametric: choose family $p(x|theta)$ (e.g. Gaussian, GMM)",
                        "Nonparametric: histograms, KDE — flexible but need many samples",
                        "Clustering emerges when we fit multi-modal densities (GMM)",
                    ],
                ),
                (
                    "Gaussian Mixture Models",
                    [
                        "$p(x) = sum_(k=1)^K pi_k cal(N)(x | mu_k, Sigma_k)$, $sum_k pi_k = 1$",
                        "Each component = a cluster; soft assignments via responsibilities",
                        "Latent variable $z_n in {1, dots, K}$: which component generated $x_n$?",
                        "Richer than single Gaussian; still tractable with EM",
                    ],
                ),
                (
                    "EM Algorithm for GMM",
                    [
                        "E-step: $r_(n k) = pi_k cal(N)(x_n|mu_k, Sigma_k) / sum_j pi_j cal(N)(x_n|mu_j, Sigma_j)$",
                        "M-step: update $pi_k, mu_k, Sigma_k$ from weighted sufficient statistics",
                        "Monotonically increases data log-likelihood (local optimum)",
                        "Initialize carefully (k-means++) — symmetries cause multiple local maxima",
                        "Choose $K$ via BIC or held-out log-likelihood",
                    ],
                ),
            ],
        ),
        (
            "Classification / SVM",
            [
                (
                    "Classification Setting",
                    [
                        "Labels $y_n in {-1, +1}$ (or ${0,1}$); learn decision boundary in feature space",
                        'Linear classifier: $f(x) = w^T x + b$; predict $"sign"(f(x))$',
                        "Many separating hyperplanes exist when data are linearly separable",
                        "Which one generalizes best? *Maximum margin* principle (SVM)",
                    ],
                ),
                (
                    "Separating Hyperplanes",
                    [
                        "Separating hyperplane: $w^T x + b = 0$ with $y_n(w^T x_n + b) > 0$ for all $n$",
                        "Margin = distance from hyperplane to nearest point",
                        "Scale $(w, b)$ so nearest points satisfy $|w^T x_n + b| = 1$ (support vectors)",
                        "Hard-margin SVM: maximize margin subject to correct classification",
                    ],
                ),
                (
                    "Maximum-Margin SVM (Primal)",
                    [
                        "Minimize $||w||^2$ subject to $y_n(w^T x_n + b) >= 1$",
                        "Solution depends only on *support vectors* (points on the margin)",
                        "Dual formulation: $max_alpha sum_n alpha_n - (1/2) sum_(n,m) alpha_n alpha_m y_n y_m x_n^T x_m$",
                        "Kernel trick: replace $x_n^T x_m$ with $k(x_n, x_m)$ for nonlinear boundaries",
                    ],
                ),
                (
                    "Soft Margin & Kernels",
                    [
                        "Soft margin: allow slack $xi_n >= 0$; penalize $C sum_n xi_n$ in objective",
                        "Trade-off: large $C$ $arrow.r$ fewer errors, narrower margin (may overfit)",
                        "RBF kernel $k(x, x') = exp(-gamma ||x - x'||^2)$ — implicit feature map",
                        "SVMs vs logistic regression: hinge loss vs log loss; sparse support vectors",
                        "Multiclass: one-vs-rest or structured extensions",
                    ],
                ),
            ],
        ),
    ],
)

LECTURE = {
    "day": 2,
    "slug": "statistical-learning",
    "title": "Statistical Learning",
    "description": (
        "The supervised/unsupervised framework, linear regression, "
        "PCA, Gaussian mixture models, and support vector machines."
    ),
    "reading": [
        "[Deisenroth, Faisal & Ong — *Mathematics for Machine Learning*](https://mml-book.com), Ch. 8–12",
        "[Mathematics for Machine Learning (free PDF)](https://mml-book.github.io/book/mml-book.pdf)",
        "[scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)",
    ],
    "intro": (
        "Day 1 built the mathematical language; today we use it to design learning algorithms. "
        "We start with the **framework** for how data, models, and learning "
        "fit together, then work through the four central pillars: **linear regression**, "
        "**PCA** for dimensionality reduction, **Gaussian mixture models** for "
        "density estimation, and **support vector machines** for classification. "
        "Each section connects back to the linear-algebra and probability tools from Day 1."
    ),
    "sections": [
        {
            "title": "When Models Meet Data",
            "subsections": [
                {
                    "heading": "A taxonomy of machine learning",
                    "definition": (
                        "**Machine learning** learns patterns from data. **Supervised** learning uses "
                        "labeled pairs $$(\\mathbf{x}, y)$$ (regression or classification); "
                        "**unsupervised** learning finds structure in inputs alone (clustering, "
                        "density estimation, dimensionality reduction); **reinforcement** learning "
                        "optimizes sequential decisions from rewards."
                    ),
                    "body": """![Taxonomy of machine learning: supervised (regression & classification), unsupervised (clustering & dimensionality reduction), and reinforcement learning with representative applications.](/assets/figures/day02/ml_taxonomy.png)

This map is the roadmap for the rest of the course. **Regression** (continuous targets) and **classification** (discrete labels) dominate supervised learning and reappear inside deep networks (Days 3–5). **Dimensionality reduction** and **density estimation** are unsupervised tools we use for visualization, preprocessing, and generative modeling. **Reinforcement learning** sits outside today's scope but shares the same ERM spirit: optimize an expected objective from finite samples.

We organize the core methods into four pillars — regression, PCA, GMM, SVM — all introduced through this common lens of data, models, and learning.""",
                },
                {
                    "heading": "Data as vectors, models as hypotheses",
                    "definition": (
                        "A **dataset** is $$\\mathcal{D} = \\{(\\mathbf{x}_n, y_n)\\}_{n=1}^N$$ with "
                        "feature vectors $$\\mathbf{x}_n \\in \\mathbb{R}^D$$. A **model** is a "
                        "parameterized family — e.g. $$p(y\\mid\\mathbf{x}, \\boldsymbol{\\theta})$$ or "
                        "$$f_{\\boldsymbol{\\theta}}(\\mathbf{x})$$ — and **learning** selects "
                        "$$\\boldsymbol{\\theta}$$ from $$\\mathcal{D}$$."
                    ),
                    "body": """![Toy regression data: salary vs age with a query point at age 60.](/assets/figures/day02/mml_toy_regression.png)

Representing examples as vectors lets us reuse linear algebra (Day 1): design matrices, projections, eigen-decompositions. **Similarity** between examples — via inner products or distances — is the geometric backbone of both regression and classification: nearby points should receive similar predictions.

The figure shows a concrete regression task: predict salary from age using $$N$$ training pairs, then query at a new age (60) not in the training set. Success is measured on that *held-out* input, not on memorizing training salaries.""",
                },
                {
                    "heading": "Empirical risk minimization",
                    "definition": (
                        "**Empirical risk minimization (ERM)** chooses "
                        "$$\\hat{\\boldsymbol{\\theta}} = \\arg\\min_{\\boldsymbol{\\theta}} "
                        "\\frac{1}{N}\\sum_{n=1}^N \\ell\\big(f_{\\boldsymbol{\\theta}}(\\mathbf{x}_n), "
                        "y_n\\big)$$ as a proxy for the unknowable **population risk** "
                        "$$R(f) = \\mathbb{E}_{(\\mathbf{x},y)}[\\ell(f(\\mathbf{x}), y)]$$."
                    ),
                    "body": """We never observe the true data distribution — only a finite sample — so we minimize the **empirical** average loss instead. Common choices:

- **Squared error** $$\\ell = (y - \\hat{y})^2$$ for regression (leads to least squares under Gaussian noise).
- **Cross-entropy** for classification (leads to logistic regression; the SVM later uses the hinge loss instead).

ERM is the workhorse of machine learning, but minimizing training loss alone is dangerous: a large enough model can drive empirical risk to zero while **test** error explodes. The sections below show both the power of ERM and the tools to control it (regularization, validation, margins).""",
                },
                {
                    "heading": "Validation, cross-validation, and model selection",
                    "definition": (
                        "Split data into **training**, **validation**, and **test** sets. "
                        "**K-fold cross-validation** rotates the validation fold, averaging "
                        "$$\\frac{1}{K}\\sum_k R(f^{(k)}, \\mathcal{V}^{(k)})$$ as an estimate of "
                        "generalization error."
                    ),
                    "body": """![K-fold cross-validation: the dataset is partitioned into $$K$$ chunks; each chunk serves once as validation while the rest train the model.](/assets/figures/day02/mml_cross_validation.png)

**Hyperparameters** — ridge penalty $$\\lambda$$, polynomial degree, number of GMM components $$K$$, SVM slack $$C$$ — are tuned on validation data only. The **test** set is touched once, at the end, for an unbiased performance estimate.

**Model selection** balances fit and complexity. A useful guiding principle is **Occam's razor**: among models with similar validation error, prefer the simpler one. Information criteria make this explicit:

$$\\mathrm{BIC} = -2\\,\\ell + p\\log N,$$

where $$p$$ is the number of free parameters and $$\\ell$$ is the maximized log-likelihood. Lower BIC favors parsimony.

![Training and validation error vs model capacity: training error decreases monotonically while validation error has a sweet spot.](/assets/figures/day02/mml_poly_overfit.png)

The capacity curve is the picture to remember: more flexible $$\\mathcal{H}$$ reduces **bias** but increases **variance**; validation error is the practical guide to the sweet spot.""",
                },
            ],
        },
        {
            "title": "Linear Regression",
            "subsections": [
                {
                    "heading": "Problem setup and the linear model",
                    "definition": (
                        "In **linear regression**, we model "
                        "$$y = \\boldsymbol{\\phi}(\\mathbf{x})^{\\top}\\boldsymbol{\\theta} + \\epsilon, "
                        "\\quad \\epsilon \\sim \\mathcal{N}(0, \\sigma^2),$$ "
                        "where $$\\boldsymbol{\\phi}$$ maps inputs to features and the model is "
                        "**linear in** $$\\boldsymbol{\\theta}$$."
                    ),
                    "body": """![Linear regression example: candidate lines, training data, and MLE fit.](/assets/figures/day02/mml_linear_regression.png)

The key phrase is *linear in the parameters*. A polynomial $$y = \\theta_0 + \\theta_1 x + \\theta_2 x^2$$ is still linear regression because $$\\boldsymbol{\\phi}(x) = (1, x, x^2)^{\\top}$$. Stacking $$N$$ examples gives the **design matrix** $$\\Phi \\in \\mathbb{R}^{N \\times M}$$ and predictions $$\\hat{\\mathbf{y}} = \\Phi\\boldsymbol{\\theta}$$.

We assume i.i.d. Gaussian noise — a probabilistic model, not just curve fitting — which connects least squares to maximum likelihood.""",
                },
                {
                    "heading": "Derivation: maximum likelihood equals least squares",
                    "definition": (
                        "Under $$p(y\\mid\\mathbf{x}, \\boldsymbol{\\theta}) = "
                        "\\mathcal{N}(\\boldsymbol{\\phi}(\\mathbf{x})^{\\top}\\boldsymbol{\\theta}, "
                        "\\sigma^2)$$, the **MLE** of $$\\boldsymbol{\\theta}$$ minimizes "
                        "$$\\|\\Phi\\boldsymbol{\\theta} - \\mathbf{y}\\|_2^2$$."
                    ),
                    "body": """Least squares is usually presented as "minimize the sum of squared errors", but that choice is not arbitrary: it is the **maximum-likelihood estimator under Gaussian noise**. Making that link explicit is the cleanest way to understand *where losses come from* — and why a different noise assumption gives a different loss.

**From the noise model to the loss.** Assume the targets are generated as $$y_n = \\boldsymbol{\\phi}(\\mathbf{x}_n)^{\\top}\\boldsymbol{\\theta} + \\epsilon_n$$ with i.i.d. $$\\epsilon_n\\sim\\mathcal{N}(0,\\sigma^2)$$. Then each $$y_n$$ is Gaussian about the model prediction, and because the samples are independent the likelihood factorizes. Take its log:

$$\\begin{aligned}
\\log p(\\mathbf{y}\\mid\\Phi,\\boldsymbol{\\theta})
&= \\log \\prod_{n=1}^N \\frac{1}{\\sqrt{2\\pi\\sigma^2}}\\exp\\!\\Big(\\textcolor{teal}{-\\tfrac{1}{2\\sigma^2}\\big(y_n-\\boldsymbol{\\phi}(\\mathbf{x}_n)^{\\top}\\boldsymbol{\\theta}\\big)^2}\\Big) \\\\
&= \\sum_{n=1}^N \\Big[\\textcolor{teal}{-\\tfrac{1}{2\\sigma^2}\\big(y_n-\\boldsymbol{\\phi}(\\mathbf{x}_n)^{\\top}\\boldsymbol{\\theta}\\big)^2} \\;\\underbrace{-\\tfrac12\\log(2\\pi\\sigma^2)}_{\\text{const in }\\boldsymbol{\\theta}}\\Big] \\\\
&= -\\frac{1}{2\\sigma^2}\\,\\big\\|\\Phi\\boldsymbol{\\theta}-\\mathbf{y}\\big\\|_2^2 + \\text{const}.
\\end{aligned}$$

Maximizing the log-likelihood is therefore **identical** to minimizing the sum of squared residuals $$\\|\\Phi\\boldsymbol{\\theta}-\\mathbf{y}\\|_2^2$$ — the noise variance $$\\sigma^2$$ only rescales the objective and drops out of the argmin. So "least squares" is not a modeling choice we make for convenience; it is *implied* by the Gaussian assumption.

**Solving it.** Using the gradient identity $$\\nabla_{\\boldsymbol{\\theta}}\\|\\Phi\\boldsymbol{\\theta}-\\mathbf{y}\\|_2^2 = 2\\Phi^{\\top}(\\Phi\\boldsymbol{\\theta}-\\mathbf{y})$$ (Day 1) and setting it to zero gives the **normal equations** and, when $$\\Phi^{\\top}\\Phi$$ is invertible, the closed form:

$$\\Phi^{\\top}\\Phi\\,\\boldsymbol{\\theta} = \\Phi^{\\top}\\mathbf{y} \\qquad\\Longrightarrow\\qquad \\boldsymbol{\\theta}^{\\star} = (\\Phi^{\\top}\\Phi)^{-1}\\Phi^{\\top}\\mathbf{y}.$$

Geometrically, $$\\Phi\\boldsymbol{\\theta}^{\\star} = \\Phi(\\Phi^{\\top}\\Phi)^{-1}\\Phi^{\\top}\\mathbf{y}$$ is the **orthogonal projection** of $$\\mathbf{y}$$ onto the column space of $$\\Phi$$ — the residual is perpendicular to every feature, so no direction in the model's span can reduce the error further.

**Different noise model $$\\Rightarrow$$ different loss.** The general principle is that the loss is the **negative log-likelihood** of the assumed noise, $$\\ell(y,\\hat y) = -\\log p(y\\mid\\hat y)$$. Changing the noise changes the loss:

| Noise / likelihood $$p(y\\mid\\hat y)$$ | Loss $$-\\log p$$ | Optimal point estimate |
|---|---|---|
| Gaussian $$\\mathcal{N}(\\hat y,\\sigma^2)$$ | squared error $$(y-\\hat y)^2$$ | conditional **mean** |
| Laplace $$\\propto e^{-\\lvert y-\\hat y\\rvert/b}$$ | absolute error $$\\lvert y-\\hat y\\rvert$$ | conditional **median** |
| Student-$$t$$ / heavy-tailed | robust loss (≈ Huber) | outlier-resistant |
| Bernoulli $$\\hat p^{\\,y}(1-\\hat p)^{1-y}$$ | cross-entropy (next section) | class probability |

**What this means in practice.** Squared error implicitly assumes symmetric, light-tailed Gaussian noise, so a single gross **outlier** — which is wildly unlikely under a Gaussian — can dominate the fit and drag the line toward it. If your noise is heavy-tailed, the matched negative-log-likelihood is the $$\\ell_1$$ (absolute) loss or the **Huber** loss (quadratic near zero, linear in the tails), which down-weight outliers and recover the median rather than the mean. Choosing a loss *is* choosing a noise model, whether or not you say so out loud.""",
                },
                {
                    "heading": "Ridge regression and the bias–variance trade-off",
                    "definition": (
                        "**Ridge regression** adds an $$\\ell_2$$ penalty: "
                        "$$\\hat{\\boldsymbol{\\theta}} = "
                        "(\\Phi^{\\top}\\Phi + \\lambda \\mathbf{I})^{-1}\\Phi^{\\top}\\mathbf{y}.$$ "
                        "It stabilizes inversion when features are collinear or $$N \\ll M$$."
                    ),
                    "body": """When $$\\Phi^{\\top}\\Phi$$ is near-singular — collinear features, or more parameters than data ($$M>N$$) — OLS has enormous variance: a tiny perturbation of $$\\mathbf{y}$$ swings $$\\boldsymbol{\\theta}^{\\star}$$ wildly, because we are dividing by near-zero directions of $$\\Phi^{\\top}\\Phi$$. Ridge regression fixes this by adding an $$\\ell_2$$ penalty to the objective.

**Closed form.** Minimize the penalized objective and set the gradient to zero:

$$\\begin{aligned}
L(\\boldsymbol{\\theta}) &= \\tfrac12\\|\\Phi\\boldsymbol{\\theta}-\\mathbf{y}\\|_2^2 + \\tfrac{\\lambda}{2}\\|\\boldsymbol{\\theta}\\|_2^2, \\\\
\\nabla_{\\boldsymbol{\\theta}} L &= \\Phi^{\\top}(\\Phi\\boldsymbol{\\theta}-\\mathbf{y}) + \\lambda\\boldsymbol{\\theta} = \\big(\\textcolor{teal}{\\Phi^{\\top}\\Phi + \\lambda\\mathbf{I}}\\big)\\boldsymbol{\\theta} - \\Phi^{\\top}\\mathbf{y} \\overset{!}{=}\\mathbf{0}, \\\\
\\Rightarrow\\quad \\hat{\\boldsymbol{\\theta}}_{\\text{ridge}} &= \\big(\\Phi^{\\top}\\Phi + \\lambda\\mathbf{I}\\big)^{-1}\\Phi^{\\top}\\mathbf{y}.
\\end{aligned}$$

The term $$\\lambda\\mathbf{I}$$ lifts every eigenvalue of $$\\Phi^{\\top}\\Phi$$ by $$\\lambda>0$$, so the matrix is **always invertible** — the inversion problem disappears.

**Bayesian view (why it is a MAP estimate).** Put a Gaussian prior $$\\boldsymbol{\\theta}\\sim\\mathcal{N}(\\mathbf{0},\\tau^2\\mathbf{I})$$ on the weights and keep the Gaussian likelihood. The log-posterior is

$$\\begin{aligned}
\\log p(\\boldsymbol{\\theta}\\mid\\mathbf{y})
&= \\underbrace{-\\tfrac{1}{2\\sigma^2}\\|\\Phi\\boldsymbol{\\theta}-\\mathbf{y}\\|_2^2}_{\\text{log-likelihood}} \\;\\underbrace{-\\tfrac{1}{2\\tau^2}\\|\\boldsymbol{\\theta}\\|_2^2}_{\\text{log-prior}} + \\text{const}.
\\end{aligned}$$

Maximizing it is exactly ridge with $$\\textcolor{purple}{\\lambda = \\sigma^2/\\tau^2}$$: the penalty strength is the ratio of noise variance to prior variance. A tighter prior (small $$\\tau^2$$) or noisier data (large $$\\sigma^2$$) means more shrinkage. The $$\\ell_2$$ penalty is the log of a Gaussian prior, just as squared error was the log of a Gaussian likelihood.

**What shrinkage does (SVD view).** Write $$\\Phi = U\\Sigma V^{\\top}$$ with singular values $$d_i$$. Then OLS and ridge differ only by a per-direction **shrinkage factor**:

$$\\hat{\\mathbf{y}}_{\\text{ridge}} = \\sum_i \\mathbf{u}_i\\,\\underbrace{\\frac{d_i^2}{d_i^2+\\lambda}}_{\\in(0,1)}\\,\\mathbf{u}_i^{\\top}\\mathbf{y}, \\qquad \\hat{\\mathbf{y}}_{\\text{OLS}} = \\sum_i \\mathbf{u}_i\\,\\mathbf{u}_i^{\\top}\\mathbf{y}.$$

High-variance directions ($$d_i^2\\gg\\lambda$$) pass through almost untouched; low-variance directions ($$d_i^2\\ll\\lambda$$), the unstable ones, are damped toward zero. Ridge trades a little **bias** for a large drop in **variance** — the bias–variance trade-off made mechanical.

![Training and test error vs polynomial degree: training error falls while test error rises after the true degree — a classic overfitting picture.](/assets/figures/day02/mml_poly_overfit.png)

Polynomial regression shows why this matters: a high-degree fit passes through every training point but oscillates wildly between them (huge coefficients). The penalty $$\\lambda$$ — tuned by cross-validation — tames those coefficients and restores generalization. (Swapping the $$\\ell_2$$ penalty for $$\\ell_1$$ gives the **lasso**, whose diamond-shaped constraint drives coefficients exactly to zero, i.e. feature selection.)""",
                },
            ],
        },
        {
            "title": "Dimensionality Reduction with PCA",
            "subsections": [
                {
                    "heading": "Motivation and problem setting",
                    "definition": (
                        "**Principal component analysis (PCA)** finds an orthogonal basis "
                        "$$\\{\\mathbf{b}_1, \\ldots, \\mathbf{b}_D\\}$$ such that the first "
                        "$$M$$ components capture most of the variance in centered data "
                        "$$\\tilde{\\mathbf{x}}_n = \\mathbf{x}_n - \\bar{\\mathbf{x}}$$."
                    ),
                    "body": """High-dimensional data suffer the **curse of dimensionality**: distances become less meaningful, and we need exponentially more samples to fill the space. PCA offers a *linear* compression: represent each point by $$M \\ll D$$ coordinates while minimizing reconstruction error.

Applications include visualization (project to 2D/3D), denoising, whitening features before downstream classifiers, and compression.""",
                },
                {
                    "heading": "Maximum-variance and minimum-reconstruction views",
                    "definition": (
                        "The first principal component solves "
                        "$$\\mathbf{b}_1 = \\arg\\max_{\\|\\mathbf{b}\\|=1} "
                        "\\mathrm{Var}(\\mathbf{b}^{\\top}\\tilde{\\mathbf{X}}).$$ "
                        "Equivalently, PCA minimizes "
                        "$$\\sum_n \\|\\tilde{\\mathbf{x}}_n - \\mathbf{B}\\mathbf{B}^{\\top}\\tilde{\\mathbf{x}}_n\\|^2$$ "
                        "for rank-$$M$$ projection matrix $$\\mathbf{B}$$."
                    ),
                    "body": """![PCA finds a lower-dimensional subspace that preserves variance when data are projected.](/assets/figures/day02/mml_pca_lowdim.png)

![Orthogonal projection onto the principal subspace.](/assets/figures/day02/mml_pca_projection.png)

Both formulations lead to the same solution: **eigenvectors of the sample covariance**

$$\\mathbf{S} = \\frac{1}{N}\\sum_{n=1}^N \\tilde{\\mathbf{x}}_n \\tilde{\\mathbf{x}}_n^{\\top}.$$

Sort eigenvalues $$\\lambda_1 \\geq \\lambda_2 \\geq \\cdots$$; keep the top $$M$$ eigenvectors as columns of $$\\mathbf{B}$$. The **explained variance ratio** $$\\lambda_i / \\sum_j \\lambda_j$$ tells you how many components to retain.

**Algorithm (via SVD).** Center $$\\tilde{\\mathbf{X}}$$, compute $$\\tilde{\\mathbf{X}} = \\mathbf{U}\\boldsymbol{\\Sigma}\\mathbf{V}^{\\top}$$; principal directions are columns of $$\\mathbf{V}$$; coordinates $$\\mathbf{z}_n = \\mathbf{B}^{\\top}\\tilde{\\mathbf{x}}_n$$.""",
                },
            ],
        },
        {
            "title": "Density Estimation with Gaussian Mixture Models",
            "subsections": [
                {
                    "heading": "Parametric density estimation",
                    "definition": (
                        "A **Gaussian mixture model (GMM)** writes "
                        "$$p(\\mathbf{x}) = \\sum_{k=1}^{K} \\pi_k\\,"
                        "\\mathcal{N}(\\mathbf{x}\\mid\\boldsymbol{\\mu}_k, \\boldsymbol{\\Sigma}_k), "
                        "\\quad \\sum_k \\pi_k = 1,\\; \\pi_k \\geq 0.$$"
                    ),
                    "body": """Unlike regression (model $$p(y\\mid\\mathbf{x})$$), density estimation models $$p(\\mathbf{x})$$ itself — useful for outlier detection, sampling, and discovering **clusters** as mixture components.

A single Gaussian is unimodal; a mixture can approximate multimodal data. Each component has mean $$\\boldsymbol{\\mu}_k$$, covariance $$\\boldsymbol{\\Sigma}_k$$, and mixing weight $$\\pi_k$$.

![One-dimensional GMM: sum of Gaussians approximates a complex density.](/assets/figures/day02/mml_gmm_1d.png)""",
                },
                {
                    "heading": "EM algorithm: derivation sketch",
                    "definition": (
                        "The **expectation–maximization (EM)** algorithm alternates: "
                        "**E-step** — compute responsibilities "
                        "$$r_{nk} = \\frac{\\pi_k\\,\\mathcal{N}(\\mathbf{x}_n\\mid\\boldsymbol{\\mu}_k, "
                        "\\boldsymbol{\\Sigma}_k)}{\\sum_j \\pi_j\\,\\mathcal{N}(\\mathbf{x}_n\\mid\\boldsymbol{\\mu}_j, "
                        "\\boldsymbol{\\Sigma}_j)};$$ "
                        "**M-step** — update $$\\pi_k, \\boldsymbol{\\mu}_k, \\boldsymbol{\\Sigma}_k$$ from "
                        "weighted sufficient statistics."
                    ),
                    "body": """Why can't we just maximize the GMM log-likelihood directly? Because the density is a **sum** inside the log, $$\\log\\sum_k \\pi_k\\mathcal{N}(\\mathbf{x}_n\\mid\\boldsymbol{\\mu}_k,\\boldsymbol{\\Sigma}_k)$$, which does not separate into per-component terms — setting the gradient to zero gives coupled equations with no closed form. EM is the standard way around this, and it is far more general than GMMs.

**The general derivation (any latent-variable model).** Let $$\\mathbf{z}$$ be the latent variable and introduce *any* distribution $$q(\\mathbf{z})$$ over it. For every $$q$$, the marginal log-likelihood splits exactly:

$$\\begin{aligned}
\\log p_{\\boldsymbol{\\theta}}(\\mathbf{x})
&= \\underbrace{\\mathbb{E}_{q}\\!\\Big[\\log\\tfrac{p_{\\boldsymbol{\\theta}}(\\mathbf{x},\\mathbf{z})}{q(\\mathbf{z})}\\Big]}_{\\textcolor{teal}{\\mathcal{F}(q,\\,\\boldsymbol{\\theta})\\ =\\ \\text{ELBO}}} \\;+\\; \\underbrace{D_{\\mathrm{KL}}\\!\\big(q(\\mathbf{z})\\,\\|\\,p_{\\boldsymbol{\\theta}}(\\mathbf{z}\\mid\\mathbf{x})\\big)}_{\\textcolor{purple}{\\ \\geq\\ 0}} .
\\end{aligned}$$

(To verify: substitute $$p_{\\boldsymbol{\\theta}}(\\mathbf{x},\\mathbf{z}) = p_{\\boldsymbol{\\theta}}(\\mathbf{z}\\mid\\mathbf{x})p_{\\boldsymbol{\\theta}}(\\mathbf{x})$$ inside the ELBO and the $$\\log p_{\\boldsymbol{\\theta}}(\\mathbf{x})$$ term, being constant in $$\\mathbf{z}$$, pulls out.) Because the KL term is non-negative, the ELBO $$\\mathcal{F}$$ is a **lower bound** on the log-likelihood, tight exactly when $$q = p_{\\boldsymbol{\\theta}}(\\mathbf{z}\\mid\\mathbf{x})$$. EM is **coordinate ascent** on $$\\mathcal{F}$$:

- **E-step** (maximize $$\\mathcal{F}$$ over $$q$$, holding $$\\boldsymbol{\\theta}$$): set $$q(\\mathbf{z}) = p_{\\boldsymbol{\\theta}^{\\text{old}}}(\\mathbf{z}\\mid\\mathbf{x})$$, driving the KL gap to zero so the bound **touches** the curve at $$\\boldsymbol{\\theta}^{\\text{old}}$$.
- **M-step** (maximize $$\\mathcal{F}$$ over $$\\boldsymbol{\\theta}$$, holding $$q$$): since $$q$$ is fixed, this is $$\\boldsymbol{\\theta}^{\\text{new}} = \\arg\\max_{\\boldsymbol{\\theta}} \\mathbb{E}_{q}[\\log p_{\\boldsymbol{\\theta}}(\\mathbf{x},\\mathbf{z})]$$ — the **expected complete-data log-likelihood**, which (with a sum *outside* the log) usually has a closed form.

**Why it monotonically improves.** Chaining the two steps,

$$\\log p_{\\boldsymbol{\\theta}^{\\text{new}}}(\\mathbf{x}) \\;\\geq\\; \\mathcal{F}(q,\\boldsymbol{\\theta}^{\\text{new}}) \\;\\geq\\; \\mathcal{F}(q,\\boldsymbol{\\theta}^{\\text{old}}) \\;=\\; \\log p_{\\boldsymbol{\\theta}^{\\text{old}}}(\\mathbf{x}),$$

where the first inequality is the bound, the second is the M-step's improvement, and the equality is the E-step's tightness. The likelihood **never decreases** — EM climbs to a local optimum.

**Specializing to the GMM.** Here $$z_n\\in\\{1,\\dots,K\\}$$ says which component generated $$\\mathbf{x}_n$$. The E-step posterior is the **responsibility**

$$r_{nk} = p(z_n=k\\mid\\mathbf{x}_n) = \\frac{\\pi_k\\,\\mathcal{N}(\\mathbf{x}_n\\mid\\boldsymbol{\\mu}_k,\\boldsymbol{\\Sigma}_k)}{\\sum_j \\pi_j\\,\\mathcal{N}(\\mathbf{x}_n\\mid\\boldsymbol{\\mu}_j,\\boldsymbol{\\Sigma}_j)},$$

and maximizing the expected complete-data log-likelihood gives **responsibility-weighted** MLE updates:

$$\\pi_k = \\frac{1}{N}\\sum_n r_{nk}, \\quad
\\boldsymbol{\\mu}_k = \\frac{\\sum_n r_{nk}\\,\\mathbf{x}_n}{\\sum_n r_{nk}}, \\quad
\\boldsymbol{\\Sigma}_k = \\frac{\\sum_n r_{nk}(\\mathbf{x}_n - \\boldsymbol{\\mu}_k)(\\mathbf{x}_n - \\boldsymbol{\\mu}_k)^{\\top}}{\\sum_n r_{nk}}.$$

These are just the usual Gaussian MLEs with each point counted *fractionally* by how much it belongs to component $$k$$ — a **soft** version of k-means (which is the $$\\boldsymbol{\\Sigma}_k\\to 0$$, hard-assignment limit). EM finds **local** optima, so initialization matters (k-means++ is standard), and $$K$$ is chosen by BIC or held-out log-likelihood. The same ELBO machinery returns on Day 6 as the training objective for **VAEs**, where $$q$$ is an *amortized* neural encoder instead of the exact posterior.""",
                },
            ],
        },
        {
            "title": "Classification: Logistic Regression and Support Vector Machines",
            "subsections": [
                {
                    "heading": "Logistic regression: cross-entropy from a Bernoulli model",
                    "definition": (
                        "**Logistic regression** models the class probability as a sigmoid of a "
                        "linear score, $$p(y=1\\mid\\mathbf{x}) = \\sigma(\\mathbf{w}^{\\top}\\mathbf{x}+b)$$ "
                        "with $$\\sigma(z) = 1/(1+e^{-z})$$. Maximum likelihood under this **Bernoulli** "
                        "model is exactly minimization of the **cross-entropy** (log) loss."
                    ),
                    "body": """Regression assumed Gaussian noise and got squared error. Classification assumes a **Bernoulli** label and gets cross-entropy — the same recipe (loss $$= -\\log$$ likelihood), a different distribution. This is the second half of "different assumptions $$\\Rightarrow$$ different losses".

**The model.** A linear score $$z=\\mathbf{w}^{\\top}\\mathbf{x}+b$$ is squashed through the **sigmoid** into a probability $$\\hat p = \\sigma(z)\\in(0,1)$$. A single label is Bernoulli, $$p(y\\mid\\mathbf{x}) = \\hat p^{\\,y}(1-\\hat p)^{1-y}$$ for $$y\\in\\{0,1\\}$$.

**From likelihood to cross-entropy.** Take the negative log-likelihood over the dataset:

$$\\begin{aligned}
\\mathcal{L}(\\mathbf{w},b)
&= -\\sum_{n=1}^N \\log\\Big[\\hat p_n^{\\,y_n}(1-\\hat p_n)^{1-y_n}\\Big] \\\\
&= -\\sum_{n=1}^N \\Big[\\textcolor{teal}{y_n\\log\\hat p_n} + \\textcolor{purple}{(1-y_n)\\log(1-\\hat p_n)}\\Big].
\\end{aligned}$$

This is the **binary cross-entropy**: it is precisely $$\\sum_n H(\\text{one-hot } y_n,\\ \\hat p_n)$$, the cross-entropy between the true label and the predicted distribution. Unlike least squares there is **no closed form**, but the loss is **convex** in $$(\\mathbf{w},b)$$, so gradient descent finds the global optimum.

**The gradient is beautifully simple.** Using $$\\sigma'(z)=\\sigma(z)(1-\\sigma(z))$$, the per-example gradient telescopes to the *same* form as linear regression — prediction minus target, times the input:

$$\\nabla_{\\mathbf{w}}\\mathcal{L} = \\sum_{n=1}^N \\big(\\textcolor{orange}{\\hat p_n - y_n}\\big)\\,\\mathbf{x}_n = \\Phi^{\\top}(\\hat{\\mathbf{p}}-\\mathbf{y}).$$

That shared "(prediction − target) · input" structure is no coincidence: both squared error and cross-entropy belong to the **exponential family / generalized linear model** framework, where the matched loss always yields this gradient. (Newton's method on this objective is **iteratively reweighted least squares**.)

**Multiclass.** For $$K$$ classes, replace the sigmoid with the **softmax** $$\\hat p_k = e^{z_k}/\\sum_j e^{z_j}$$ and the Bernoulli with a **categorical** likelihood; the negative log-likelihood becomes the multiclass cross-entropy $$-\\sum_n \\log \\hat p_{n,y_n}$$. This is the exact loss used to train every classifier and language model later in the course.

**What this means in practice.** Cross-entropy outputs **calibrated probabilities** and punishes confident-and-wrong predictions heavily (the $$-\\log$$ blows up as $$\\hat p\\to 0$$ for the true class), which is why it trains faster than squared error on classification: squared error on top of a sigmoid is non-convex and has vanishing gradients when saturated. Logistic regression uses the **log loss**; the SVM below uses the **hinge loss** — two different surrogates for the same goal of separating classes, leading to probabilistic vs. margin-based classifiers.""",
                },
                {
                    "heading": "Separating hyperplanes and margins",
                    "definition": (
                        "A **linear classifier** uses $$f(\\mathbf{x}) = \\mathbf{w}^{\\top}\\mathbf{x} + b$$ "
                        "and predicts $$\\mathrm{sign}(f(\\mathbf{x}))$$. A **separating hyperplane** "
                        "satisfies $$y_n(\\mathbf{w}^{\\top}\\mathbf{x}_n + b) > 0$$ for all training points."
                    ),
                    "body": """![Two-dimensional classification data with two class labels.](/assets/figures/day02/mml_svm_2d.png)

When classes are linearly separable, infinitely many hyperplanes work, and logistic regression's log loss does not single one out. **Support vector machines** pick the hyperplane with **maximum margin** — the largest distance to the nearest training points (the **support vectors**) — which is the choice with the best worst-case robustness.

**Distance from a point to a hyperplane.** The vector $$\\mathbf{w}$$ is normal to the hyperplane $$\\mathbf{w}^{\\top}\\mathbf{x}+b=0$$. Decompose any point as $$\\mathbf{x} = \\mathbf{x}_\\perp + r\\,\\tfrac{\\mathbf{w}}{\\|\\mathbf{w}\\|}$$ with $$\\mathbf{x}_\\perp$$ on the hyperplane. Plugging in and using $$\\mathbf{w}^{\\top}\\mathbf{x}_\\perp+b=0$$,

$$\\mathbf{w}^{\\top}\\mathbf{x}+b = r\\,\\frac{\\mathbf{w}^{\\top}\\mathbf{w}}{\\|\\mathbf{w}\\|} = r\\,\\|\\mathbf{w}\\| \\quad\\Longrightarrow\\quad r = \\frac{\\mathbf{w}^{\\top}\\mathbf{x}+b}{\\|\\mathbf{w}\\|},$$

so the **signed distance** is $$(\\mathbf{w}^{\\top}\\mathbf{x}+b)/\\|\\mathbf{w}\\|$$. We are free to rescale $$(\\mathbf{w},b)$$, so adopt the **canonical** normalization $$\\min_n |\\mathbf{w}^{\\top}\\mathbf{x}_n+b| = 1$$ at the closest points. The distance from the hyperplane to the nearest point is then $$1/\\|\\mathbf{w}\\|$$, and the full **margin** between the two classes is $$2/\\|\\mathbf{w}\\|$$.

![Separating-hyperplane geometry: the normal vector $$\\mathbf{w}$$, offset $$b$$, and the signed distance of a point.](/assets/figures/day02/mml_svm_hyperplane.png)""",
                },
                {
                    "heading": "Hard-margin and soft-margin SVMs",
                    "definition": (
                        "**Hard-margin SVM:** "
                        "$$\\min_{\\mathbf{w}, b} \\tfrac{1}{2}\\|\\mathbf{w}\\|^2 \\quad "
                        "\\text{s.t.}\\quad y_n(\\mathbf{w}^{\\top}\\mathbf{x}_n + b) \\geq 1.$$ "
                        "**Soft margin** adds slack $$\\xi_n \\geq 0$$ and penalty "
                        "$$C\\sum_n \\xi_n$$ for non-separable data."
                    ),
                    "body": """**Maximizing the margin = minimizing $$\\|\\mathbf{w}\\|$$.** Maximizing $$2/\\|\\mathbf{w}\\|$$ is the same as minimizing $$\\tfrac12\\|\\mathbf{w}\\|^2$$ (monotone, and quadratic hence convex). With the canonical scaling, "every point on the correct side of its margin" becomes $$y_n(\\mathbf{w}^{\\top}\\mathbf{x}_n+b)\\geq 1$$. That gives the **hard-margin primal** — a convex quadratic program:

$$\\min_{\\mathbf{w},b}\\ \\tfrac12\\|\\mathbf{w}\\|^2 \\qquad \\text{s.t.}\\quad y_n(\\mathbf{w}^{\\top}\\mathbf{x}_n+b)\\geq 1,\\ \\forall n.$$

![Maximum-margin classifier: the support vectors lie exactly on the margin boundaries.](/assets/figures/day02/mml_svm_margin.png)

**Soft margin for non-separable data.** Real data overlap, so no $$\\mathbf{w}$$ satisfies every constraint. Introduce **slack** $$\\xi_n\\geq 0$$ measuring each point's margin violation and pay for it linearly:

$$\\min_{\\mathbf{w},b,\\boldsymbol{\\xi}}\\ \\tfrac12\\|\\mathbf{w}\\|^2 + C\\sum_n \\xi_n \\qquad \\text{s.t.}\\quad y_n(\\mathbf{w}^{\\top}\\mathbf{x}_n+b)\\geq 1-\\xi_n,\\ \\xi_n\\geq 0.$$

**This is hinge-loss minimization in disguise.** At the optimum each slack takes its smallest feasible value, $$\\xi_n = \\max(0,\\,1 - y_n(\\mathbf{w}^{\\top}\\mathbf{x}_n+b))$$. Substituting turns the constrained QP into an unconstrained regularized loss:

$$\\min_{\\mathbf{w},b}\\ \\underbrace{\\sum_{n}\\max\\big(0,\\,1-y_n(\\mathbf{w}^{\\top}\\mathbf{x}_n+b)\\big)}_{\\textcolor{teal}{\\text{hinge loss}}} + \\underbrace{\\tfrac{1}{2C}\\|\\mathbf{w}\\|^2}_{\\textcolor{purple}{\\ell_2\\text{ regularizer}}}.$$

So the SVM is "hinge loss + ridge penalty", the exact analogue of logistic regression's "log loss + ridge penalty". The **hinge loss** is a convex upper bound on the 0/1 error that is *flat* once a point is correctly classified beyond the margin — which is why only nearby points matter.

![The hinge loss (kinked at margin 1) is a convex upper bound on the 0/1 loss — the surrogate the soft-margin SVM minimizes.](/assets/figures/day02/mml_soft_margin.png)

The hyperparameter $$C$$ sets the trade-off: large $$C$$ punishes slack heavily (narrow margin, few training errors, risk of overfitting); small $$C$$ tolerates violations (wider, simpler margin). Tune $$C$$ by cross-validation.""",
                },
                {
                    "heading": "Dual formulation and the kernel trick",
                    "definition": (
                        "The **dual SVM** depends on inner products "
                        "$$\\mathbf{x}_n^{\\top}\\mathbf{x}_m$$ only. Replace them with a "
                        "**kernel** $$k(\\mathbf{x}_n, \\mathbf{x}_m)$$ to learn nonlinear "
                        "boundaries in an implicit feature space."
                    ),
                    "body": """**Deriving the dual.** Attach multipliers $$\\alpha_n\\geq 0$$ to the hard-margin constraints and form the Lagrangian:

$$\\mathcal{L}(\\mathbf{w},b,\\boldsymbol{\\alpha}) = \\tfrac12\\|\\mathbf{w}\\|^2 - \\sum_n \\alpha_n\\big[y_n(\\mathbf{w}^{\\top}\\mathbf{x}_n+b)-1\\big].$$

Stationarity in the primal variables gives the key identities

$$\\frac{\\partial\\mathcal{L}}{\\partial\\mathbf{w}} = \\mathbf{0}\\ \\Rightarrow\\ \\textcolor{teal}{\\mathbf{w} = \\sum_n \\alpha_n y_n \\mathbf{x}_n}, \\qquad \\frac{\\partial\\mathcal{L}}{\\partial b} = 0\\ \\Rightarrow\\ \\sum_n \\alpha_n y_n = 0.$$

The first says the solution is a **linear combination of the training points**. Substituting both back into $$\\mathcal{L}$$ eliminates $$\\mathbf{w}$$ and $$b$$ and leaves a problem in $$\\boldsymbol{\\alpha}$$ alone — the **dual**:

$$\\max_{\\boldsymbol{\\alpha}}\\ \\sum_{n} \\alpha_n - \\tfrac{1}{2}\\sum_{n,m}\\alpha_n \\alpha_m y_n y_m\\, \\textcolor{purple}{\\mathbf{x}_n^{\\top}\\mathbf{x}_m} \\qquad \\text{s.t.}\\quad 0\\leq\\alpha_n\\leq C,\\ \\ \\sum_n \\alpha_n y_n = 0.$$

(The upper bound $$\\alpha_n\\le C$$ is what the soft-margin slack contributes.)

**Support vectors fall out of the KKT conditions.** Complementary slackness, $$\\alpha_n[y_n(\\mathbf{w}^{\\top}\\mathbf{x}_n+b)-1]=0$$, forces $$\\alpha_n=0$$ for every point strictly beyond its margin. Only points **on** the margin (or violating it) have $$\\alpha_n>0$$ — these are the **support vectors**, and they alone determine $$\\mathbf{w}$$. The classifier is sparse:

$$f(\\mathbf{x}) = \\sum_{n\\in\\mathrm{SV}} \\alpha_n y_n\\, k(\\mathbf{x}_n,\\mathbf{x}) + b.$$

**The kernel trick.** The dual and the predictor touch the data *only* through inner products $$\\mathbf{x}_n^{\\top}\\mathbf{x}_m$$. Replace each with a **kernel** $$k(\\mathbf{x}_n,\\mathbf{x}_m) = \\langle\\boldsymbol{\\phi}(\\mathbf{x}_n),\\boldsymbol{\\phi}(\\mathbf{x}_m)\\rangle$$ and we learn a linear separator in a high-dimensional feature space $$\\boldsymbol{\\phi}$$ **without ever computing $$\\boldsymbol{\\phi}$$**. Common choices: **polynomial** $$k=(\\mathbf{x}^{\\top}\\mathbf{x}'+c)^d$$ and **RBF** $$k=\\exp(-\\gamma\\|\\mathbf{x}-\\mathbf{x}'\\|^2)$$, the latter corresponding to an *infinite*-dimensional feature map.

SVMs (hinge loss, sparse support vectors, margins) and logistic regression (log loss, dense, calibrated probabilities) are the two canonical linear classifiers. Deep networks (Day 3+) take the complementary route: instead of fixing a kernel, they **learn** the feature map $$\\boldsymbol{\\phi}$$ end to end.""",
                },
            ],
        },
    ],
    "checkpoint": [
        "Draw the ML taxonomy and place regression, PCA, GMM, and SVM in it.",
        "State ERM and explain why validation/cross-validation is necessary.",
        "Derive the OLS closed form from Gaussian MLE and interpret it as a projection.",
        "Describe ridge regression and how it trades bias for variance.",
        "Explain PCA from both maximum-variance and minimum-reconstruction views.",
        "Write the GMM density and one EM update step (E and M).",
        "State the hard-margin SVM objective and the role of support vectors.",
        "Explain the kernel trick and how $$C$$ controls the soft-margin trade-off.",
    ],
}
