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
    "MML Part II — regression, PCA, density estimation, classification",
    [
        (
            "When Models Meet Data (MML Ch. 8)",
            [
                (
                    "Machine Learning: A Taxonomy",
                    [
                        "Three paradigms: *supervised*, *unsupervised*, *reinforcement* learning",
                        "Supervised: regression (continuous $y$) and classification (discrete labels)",
                        "Unsupervised: clustering, density estimation, dimensionality reduction",
                        "Reinforcement: sequential decisions from reward signals",
                        "Today maps each pillar to one MML chapter (9–12)",
                    ],
                ),
                (
                    "Four Pillars of Part II",
                    [
                        "Ch. 9 — *Linear regression*: predict continuous targets from features",
                        "Ch. 10 — *Dimensionality reduction* (PCA): find compact representations",
                        "Ch. 11 — *Density estimation* (GMM): model $p(x)$ and discover structure",
                        "Ch. 12 — *Classification* (SVM): separate classes with margins",
                        "Ch. 8 provides the shared vocabulary: data, models, learning, evaluation",
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
            "Linear Regression (MML Ch. 9)",
            [
                (
                    "Problem Setup",
                    [
                        "Given $(x_n, y_n)$, find $f: RR^D to RR$ with $y_n approx f(x_n) + epsilon_n$",
                        "Noise $epsilon_n tilde cal(N)(0, sigma^2)$ (i.i.d. Gaussian in MML)",
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
            "Dimensionality Reduction / PCA (MML Ch. 10)",
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
            "Density Estimation / GMM (MML Ch. 11)",
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
            "Classification / SVM (MML Ch. 12)",
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
        "MML Part II: the supervised/unsupervised framework, linear regression, "
        "PCA, Gaussian mixture models, and support vector machines."
    ),
    "reading": [
        "[Deisenroth, Faisal & Ong — *Mathematics for Machine Learning*](https://mml-book.com), Ch. 8–12",
        "[MML PDF (local)](/material/mml-book.pdf)",
        "[scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)",
    ],
    "intro": (
        "Day 1 built the mathematical language; today we use it to design learning algorithms. "
        "Following MML Part II, we start with the **framework** for how data, models, and learning "
        "fit together (Chapter 8), then work through the four central pillars: **linear regression** "
        "(Ch. 9), **PCA** for dimensionality reduction (Ch. 10), **Gaussian mixture models** for "
        "density estimation (Ch. 11), and **support vector machines** for classification (Ch. 12). "
        "Each section connects back to the linear-algebra and probability tools from Day 1."
    ),
    "sections": [
        {
            "title": "When Models Meet Data (Chapter 8)",
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

MML Part II organizes the core methods into four pillars — regression, PCA, GMM, SVM — all introduced through the common lens of Chapter 8.""",
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
                    "body": """![Toy regression data: salary vs age with a query point at age 60 (MML Fig. 8.1).](/assets/figures/day02/mml_toy_regression.png)

Representing examples as vectors lets us reuse linear algebra (Day 1): design matrices, projections, eigen-decompositions. **Similarity** between examples — via inner products or distances — is the geometric backbone of both regression (Ch. 9) and classification (Ch. 12): nearby points should receive similar predictions.

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
- **Cross-entropy** for classification (leads to logistic regression; see also SVM hinge loss in Ch. 12).

ERM is the workhorse of machine learning, but minimizing training loss alone is dangerous: a large enough model can drive empirical risk to zero while **test** error explodes. Chapters 9–12 show both the power of ERM and the tools to control it (regularization, validation, margins).""",
                },
                {
                    "heading": "Validation, cross-validation, and model selection",
                    "definition": (
                        "Split data into **training**, **validation**, and **test** sets. "
                        "**K-fold cross-validation** rotates the validation fold, averaging "
                        "$$\\frac{1}{K}\\sum_k R(f^{(k)}, \\mathcal{V}^{(k)})$$ as an estimate of "
                        "generalization error."
                    ),
                    "body": """![K-fold cross-validation: the dataset is partitioned into $$K$$ chunks; each chunk serves once as validation while the rest train the model (MML Fig. 8.4).](/assets/figures/day02/mml_cross_validation.png)

**Hyperparameters** — ridge penalty $$\\lambda$$, polynomial degree, number of GMM components $$K$$, SVM slack $$C$$ — are tuned on validation data only. The **test** set is touched once, at the end, for an unbiased performance estimate.

**Model selection** balances fit and complexity. MML emphasizes **Occam's razor**: among models with similar validation error, prefer the simpler one. Information criteria make this explicit:

$$\\mathrm{BIC} = -2\\,\\ell + p\\log N,$$

where $$p$$ is the number of free parameters and $$\\ell$$ is the maximized log-likelihood. Lower BIC favors parsimony.

![Training and validation error vs model capacity: training error decreases monotonically while validation error has a sweet spot (MML Fig. 8.5 — see textbook).](/assets/figures/day02/mml_poly_overfit.png)

The capacity curve is the picture to remember: more flexible $$\\mathcal{H}$$ reduces **bias** but increases **variance**; validation error is the practical guide to the sweet spot.""",
                },
            ],
        },
        {
            "title": "Linear Regression (Chapter 9)",
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
                    "body": """![Linear regression example: candidate lines, training data, and MLE fit (MML Fig. 9.2).](/assets/figures/day02/mml_linear_regression.png)

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
                    "body": """The log-likelihood for one observation is

$$\\log p(y_n\\mid\\mathbf{x}_n, \\boldsymbol{\\theta}) = -\\tfrac{1}{2\\sigma^2}(y_n - \\boldsymbol{\\phi}(\\mathbf{x}_n)^{\\top}\\boldsymbol{\\theta})^2 + \\text{const}.$$

Summing over $$n$$ and dropping constants, MLE maximizes

$$-\\tfrac{1}{2\\sigma^2}\\|\\Phi\\boldsymbol{\\theta} - \\mathbf{y}\\|_2^2,$$

equivalent to minimizing the **mean squared error**. Setting the gradient to zero yields the **normal equations**

$$\\Phi^{\\top}\\Phi\\,\\boldsymbol{\\theta} = \\Phi^{\\top}\\mathbf{y}.$$

When $$\\Phi^{\\top}\\Phi$$ is invertible,

$$\\boldsymbol{\\theta}^{\\star} = (\\Phi^{\\top}\\Phi)^{-1}\\Phi^{\\top}\\mathbf{y}.$$

Geometrically, $$\\Phi\\boldsymbol{\\theta}^{\\star}$$ is the **orthogonal projection** of $$\\mathbf{y}$$ onto the column space of $$\\Phi$$ — the closest point in the model's span.""",
                },
                {
                    "heading": "Ridge regression and the bias–variance trade-off",
                    "definition": (
                        "**Ridge regression** adds an $$\\ell_2$$ penalty: "
                        "$$\\hat{\\boldsymbol{\\theta}} = "
                        "(\\Phi^{\\top}\\Phi + \\lambda \\mathbf{I})^{-1}\\Phi^{\\top}\\mathbf{y}.$$ "
                        "It stabilizes inversion when features are collinear or $$N \\ll M$$."
                    ),
                    "body": """When $$\\Phi^{\\top}\\Phi$$ is near-singular, OLS has enormous variance: small perturbations of $$\\mathbf{y}$$ swing $$\\boldsymbol{\\theta}^{\\star}$$ wildly. Adding $$\\lambda \\mathbf{I}$$ shrinks coefficients toward zero — trading a little **bias** for much lower **variance**.

Equivalently, ridge is the **MAP estimate** under a Gaussian prior $$\\boldsymbol{\\theta} \\sim \\mathcal{N}(\\mathbf{0}, \\tau^2 \\mathbf{I})$$. The penalty $$\\lambda$$ is tuned by cross-validation (Ch. 8).

![Training and test error vs polynomial degree: training error falls while test error rises after the true degree — a classic overfitting picture (MML Fig. 9.6).](/assets/figures/day02/mml_poly_overfit.png)

Polynomial regression illustrates the capacity curve in action: high degree fits every training point but wiggles wildly between them. Ridge or early stopping on degree (via validation) restores generalization.""",
                },
            ],
        },
        {
            "title": "Dimensionality Reduction with PCA (Chapter 10)",
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
                    "body": """![PCA finds a lower-dimensional subspace that preserves variance when data are projected (MML Fig. 10.4).](/assets/figures/day02/mml_pca_lowdim.png)

![Orthogonal projection onto the principal subspace (MML Fig. 10.6).](/assets/figures/day02/mml_pca_projection.png)

Both formulations lead to the same solution: **eigenvectors of the sample covariance**

$$\\mathbf{S} = \\frac{1}{N}\\sum_{n=1}^N \\tilde{\\mathbf{x}}_n \\tilde{\\mathbf{x}}_n^{\\top}.$$

Sort eigenvalues $$\\lambda_1 \\geq \\lambda_2 \\geq \\cdots$$; keep the top $$M$$ eigenvectors as columns of $$\\mathbf{B}$$. The **explained variance ratio** $$\\lambda_i / \\sum_j \\lambda_j$$ tells you how many components to retain.

**Algorithm (via SVD).** Center $$\\tilde{\\mathbf{X}}$$, compute $$\\tilde{\\mathbf{X}} = \\mathbf{U}\\boldsymbol{\\Sigma}\\mathbf{V}^{\\top}$$; principal directions are columns of $$\\mathbf{V}$$; coordinates $$\\mathbf{z}_n = \\mathbf{B}^{\\top}\\tilde{\\mathbf{x}}_n$$.""",
                },
            ],
        },
        {
            "title": "Density Estimation with Gaussian Mixture Models (Chapter 11)",
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

![One-dimensional GMM: sum of Gaussians approximates a complex density (MML Fig. 11.3).](/assets/figures/day02/mml_gmm_1d.png)""",
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
                    "body": """Introduce latent variables $$z_n \\in \\{1,\\ldots,K\\}$$ indicating which component generated $$\\mathbf{x}_n$$. The complete-data log-likelihood is easy to optimize; EM iterates:

1. **E-step:** $$\\mathbb{E}[z_{nk}\\mid\\mathbf{x}_n] = r_{nk}$$ (posterior component membership).
2. **M-step:** weighted MLE updates, e.g.

$$\\boldsymbol{\\mu}_k = \\frac{\\sum_n r_{nk}\\,\\mathbf{x}_n}{\\sum_n r_{nk}}, \\qquad
\\boldsymbol{\\Sigma}_k = \\frac{\\sum_n r_{nk}(\\mathbf{x}_n - \\boldsymbol{\\mu}_k)(\\mathbf{x}_n - \\boldsymbol{\\mu}_k)^{\\top}}{\\sum_n r_{nk}}.$$

Each iteration increases the data log-likelihood (never decreases). EM finds **local** optima — initialization matters (k-means++ is standard).

Choose $$K$$ via BIC or cross-validated held-out log-likelihood. GMMs connect forward to **latent-variable models** and VAEs on Day 6.""",
                },
            ],
        },
        {
            "title": "Classification with Support Vector Machines (Chapter 12)",
            "subsections": [
                {
                    "heading": "Separating hyperplanes and margins",
                    "definition": (
                        "A **linear classifier** uses $$f(\\mathbf{x}) = \\mathbf{w}^{\\top}\\mathbf{x} + b$$ "
                        "and predicts $$\\mathrm{sign}(f(\\mathbf{x}))$$. A **separating hyperplane** "
                        "satisfies $$y_n(\\mathbf{w}^{\\top}\\mathbf{x}_n + b) > 0$$ for all training points."
                    ),
                    "body": """![Two-dimensional classification data with class labels (MML Fig. 12.1).](/assets/figures/day02/mml_svm_2d.png)

When classes are linearly separable, infinitely many hyperplanes work. **Support vector machines** pick the one with **maximum margin** — the distance to the nearest training points (**support vectors**).

![Separating hyperplane geometry: normal vector $$\\mathbf{w}$$ and offset $$b$$ (MML Fig. 12.3).](/assets/figures/day02/mml_svm_hyperplane.png)""",
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
                    "body": """The objective $$\\tfrac{1}{2}\\|\\mathbf{w}\\|^2$$ maximizes the margin $$2/\\|\\mathbf{w}\\|$$ while constraints keep points on the correct side. Only **support vectors** (points with $$y_n(\\mathbf{w}^{\\top}\\mathbf{x}_n + b) = 1$$ or, in the soft case, $$\\xi_n > 0$$) determine the solution — a sparse representation.

![Maximum-margin classifier: support vectors lie on the margin boundaries (MML Fig. 12.7).](/assets/figures/day02/mml_svm_margin.png)

![Hinge loss is a convex upper bound on zero-one loss — the surrogate optimized by soft-margin SVMs (MML Fig. 12.8).](/assets/figures/day02/mml_soft_margin.png)

The hyperparameter $$C$$ trades margin width against training errors: large $$C$$ punishes slack heavily (narrow margin, fewer errors); small $$C$$ tolerates misclassification (wider margin, simpler boundary). Tune $$C$$ by cross-validation.""",
                },
                {
                    "heading": "Dual formulation and the kernel trick",
                    "definition": (
                        "The **dual SVM** depends on inner products "
                        "$$\\mathbf{x}_n^{\\top}\\mathbf{x}_m$$ only. Replace them with a "
                        "**kernel** $$k(\\mathbf{x}_n, \\mathbf{x}_m)$$ to learn nonlinear "
                        "boundaries in an implicit feature space."
                    ),
                    "body": """The Lagrangian dual is

$$\\max_{\\boldsymbol{\\alpha}} \\sum_{n=1}^{N} \\alpha_n - \\tfrac{1}{2}\\sum_{n,m}\\alpha_n \\alpha_m y_n y_m\\, k(\\mathbf{x}_n, \\mathbf{x}_m)$$

subject to $$0 \\leq \\alpha_n \\leq C$$ and $$\\sum_n \\alpha_n y_n = 0$$. Predictions use support vectors:

$$f(\\mathbf{x}) = \\sum_{n \\in \\mathrm{SV}} \\alpha_n y_n\\, k(\\mathbf{x}_n, \\mathbf{x}) + b.$$

Common kernels: **polynomial** $$k(\\mathbf{x}, \\mathbf{x}') = (\\mathbf{x}^{\\top}\\mathbf{x}' + c)^d$$; **RBF** $$k(\\mathbf{x}, \\mathbf{x}') = \\exp(-\\gamma\\|\\mathbf{x} - \\mathbf{x}'\\|^2)$$. The RBF kernel corresponds to an infinite-dimensional feature map — the same "lift to a richer space" idea as polynomial features in regression, but without explicit $$\\boldsymbol{\\phi}(\\mathbf{x})$$.

SVMs use **hinge loss** (via the slack formulation) rather than logistic **log loss**; both are large-margin linear classifiers in feature space. Deep networks (Day 3+) learn $$\\boldsymbol{\\phi}$$ instead of hand-designing or kernelizing it.""",
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
