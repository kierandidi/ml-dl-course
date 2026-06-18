"""Day 1 — Math foundations content (MML book + slopes-expectations + ODE crash course)."""

# Curated, cropped figures aligned to each content slide (in order).
# `None` means the slide has no source figure (no figure slide is emitted).
FIGURES = [
    # The Mathematical Foundation
    "/assets/figures/day01/mml_vectors_types.png",       # Four Pillars of ML
    None,                                                # Roadmap for Today
    # Linear Algebra
    "/assets/figures/day01/mml_linear_system.png",       # From Linear Equations to Matrices
    "/assets/figures/day01/mml_linear_mapping.png",      # What Is a Matrix?
    "/assets/figures/day01/mml_subspace.png",            # Vector Spaces
    "/assets/figures/day01/mml_kernel_nullspace.png",    # Solving & Interpreting Ax=b
    # Analytic Geometry
    "/assets/figures/day01/mml_geometry_mindmap.png",    # Why Similarity Matters
    "/assets/figures/day01/mml_triangle_ineq.png",       # Norms
    "/assets/figures/day01/mml_angle.png",               # Inner Product & Angles
    "/assets/figures/day01/mml_projection.png",          # Projections
    # Vector Calculus & Decompositions
    "/assets/figures/day01/mml_gradient.png",            # Gradients & Jacobians
    "/assets/figures/day01/mml_taylor.png",              # Useful Matrix Derivatives (Taylor)
    "/assets/figures/day01/mml_eigen.png",               # Eigendecomposition
    "/assets/figures/day01/mml_svd.png",                 # SVD & PCA Preview
    # Integration & Differentiation
    "/assets/figures/day01/slopes_map.jpg",              # There and Back Again
    "/assets/figures/day01/integ_unscented.png",         # Integration — Numerical & MC
    "/assets/figures/day01/mml_forward_pass.png",        # Differentiation — Autodiff & Backprop
    "/assets/figures/day01/integ_samples.png",           # Implicit Function & Adjoints
    # Probability & Distributions
    "/assets/figures/day01/mml_distributions.png",       # Basics
    "/assets/figures/day01/mml_gaussian.png",            # The Gaussian
    "/assets/figures/day01/mml_conjugate.png",           # Conjugacy & Change of Variables
    None,                                                # Jensen & KL Divergence (no apt MML figure)
    # Differential Equations
    "/assets/figures/day01/ode_vectorfield.png",         # ODE Intuition
    None,                                                # Solving ODEs
    None,                                                # Numerical ODE Solvers
    None,                                                # SDEs & Numerical Solvers
]
# Each part: (section title, [(slide_title, [bullets]), ...])
SLIDES = (
    "Math Foundations",
    "Linear algebra through differential equations — MML Part I",
    [
        (
            "The Mathematical Foundation",
            [
                (
                    "Four Pillars of ML",
                    [
                        "Data as vectors; tables of data as matrices (linear algebra)",
                        "Similarity via norms and inner products (analytic geometry)",
                        "Structure via matrix decompositions; optimization via gradients (Ch. 4–5)",
                        "Uncertainty via probability; dynamics via differential equations",
                        "Source: Deisenroth, Faisal & Ong — *Mathematics for Machine Learning*",
                    ],
                ),
                (
                    "Roadmap for Today",
                    [
                        "1. Linear algebra — equations, matrices, vector spaces",
                        "2. Analytic geometry — norms, inner products, projections",
                        "3. Vector calculus & matrix decompositions",
                        "4. Integration & differentiation (*There and Back Again*)",
                        "5. Probability & distributions",
                        "6. ODE/SDE crash course (diffusion book appendix)",
                    ],
                ),
            ],
        ),
        (
            "Linear Algebra",
            [
                (
                    "From Linear Equations to Matrices",
                    [
                        "System $A x = b$: each row is one linear constraint on unknowns $x in RR^n$",
                        "Coefficient matrix $A in RR^(m times n)$ encodes all equations at once",
                        "Solution sets are affine subspaces (when consistent)",
                        "Matrix-vector product: each output component is an inner product of a row with $x$",
                    ],
                ),
                (
                    "What Is a Matrix?",
                    [
                        "A matrix is a linear map $f: RR^n to RR^m$ written as $y = A x$",
                        "Columns of $A$ show where basis vectors land — geometric action of the map",
                        "Composition of maps = matrix multiplication: $(A B) x = A (B x)$",
                        "Transpose swaps rows/columns; inverse (when it exists) undoes the map",
                    ],
                ),
                (
                    "Vector Spaces",
                    [
                        "Vector space: closed under addition and scalar multiplication",
                        "Linear combination: $sum_i alpha_i v_i$; span = all reachable combinations",
                        "Linear independence: no vector is a combination of the others",
                        "Basis: minimal spanning set; dimension = number of basis vectors",
                        "Rank of $A$ = dimension of column space = dim of row space",
                    ],
                ),
                (
                    "Solving & Interpreting $A x = b$",
                    [
                        "Gaussian elimination / LU factorization (algorithmic)",
                        "Column picture: $b$ must lie in span of columns of $A$",
                        "Row picture: intersection of hyperplanes",
                        "Null space: all $x$ with $A x = 0$ — directions the map collapses",
                    ],
                ),
            ],
        ),
        (
            "Analytic Geometry",
            [
                (
                    "Why Similarity Matters",
                    [
                        "ML predictors should assign similar outputs to similar inputs",
                        "Vectors encode objects; we need a numeric notion of 'close' or 'aligned'",
                        "Norms measure length/magnitude; inner products measure alignment",
                        "Builds directly on linear-algebra vector operations",
                    ],
                ),
                (
                    "Norms",
                    [
                        "$L_2$ (Euclidean): $||x||_2 = sqrt(sum_i x_i^2)$ — rotation-invariant length",
                        "$L_1$: $||x||_1 = sum_i |x_i|$ — sparsity-inducing geometry",
                        "$L_infinity$: $||x||_infinity = max_i |x_i|$",
                        "Unit ball shapes differ — choice of norm changes regularization (Day 2)",
                    ],
                ),
                (
                    "Inner Product & Angles",
                    [
                        "Inner product $langle x, y rangle = x^T y$ (standard dot product)",
                        "Cauchy–Schwarz: $|langle x,y rangle| <= ||x||_2 ||y||_2$",
                        "Angle: $cos theta = langle x,y rangle / (||x||_2 ||y||_2)$",
                        "Orthogonal vectors: $langle x,y rangle = 0$",
                    ],
                ),
                (
                    "Projections",
                    [
                        "Orthogonal projection of $y$ onto line spanned by $x$: $hat(y) = (x^T y / x^T x) x$",
                        "Projection matrix onto column space of $A$: $P = A (A^T A)^(-1) A^T$",
                        "Least squares: minimize $||A w - y||_2^2$ — project $y$ onto span(columns of $A)$",
                        "Normal equations: $A^T A w = A^T y$",
                    ],
                ),
            ],
        ),
        (
            "Vector Calculus & Decompositions",
            [
                (
                    "Gradients & Jacobians",
                    [
                        "Scalar $f(x)$: gradient $nabla f(x) in RR^n$ — direction of steepest ascent",
                        "Vector $f(x) in RR^m$: Jacobian $J in RR^(m times n)$, $J_(i j) = partial f_i / partial x_j$",
                        "Matrix-valued functions: treat as stacked scalar functions",
                        "Chain rule: $nabla_x (g circ f) = J_f^T nabla_g$ (reverse-mode building block)",
                    ],
                ),
                (
                    "Useful Matrix Derivatives",
                    [
                        "$nabla_x (a^T x) = a$; $nabla_x (x^T A x) = (A + A^T) x$",
                        "$nabla_A ||A x - y||_2^2 = 2 (A x - y) x^T$",
                        "Hessian $H = nabla^2 f$ — second-order curvature for Newton methods",
                        "These identities appear constantly in backprop and closed-form solutions",
                    ],
                ),
                (
                    "Eigendecomposition",
                    [
                        "$A v = lambda v$ — eigenvector direction preserved, scaled by $lambda$",
                        "Symmetric $A$: real eigenvalues, orthogonal eigenvectors",
                        "Spectral theorem: $A = Q Lambda Q^T$ — principal axes of quadratic forms",
                        "Condition number from eigenvalue spread — numerical stability",
                    ],
                ),
                (
                    "SVD & PCA Preview",
                    [
                        "SVD: $A = U Sigma V^T$ — always exists; singular values on diagonal of $Sigma$",
                        "Low-rank approximation: keep top-$k$ singular values (Eckart–Young)",
                        "PCA: eigenvectors of covariance $C = (1/n) X^T X$",
                        "Connects linear algebra geometry to data compression and denoising",
                    ],
                ),
            ],
        ),
        (
            "Integration & Differentiation",
            [
                (
                    "There and Back Again",
                    [
                        "NeurIPS 2020 tutorial: integration *and* differentiation for ML",
                        "Integration: expectations, marginalization, Bayesian evidence",
                        "Differentiation: gradients for optimization, sensitivity analysis",
                        "Map metaphor: numerical integration swamps, Monte Carlo heights, Backprop Bay…",
                    ],
                ),
                (
                    "Integration — Numerical & MC",
                    [
                        "Deterministic quadrature: trapezoid, Simpson, Gauss–Hermite",
                        "Curse of dimensionality motivates Monte Carlo: $EE[f(X)] approx (1/N) sum f(x_i)$",
                        "MC error $O(1/sqrt(N))$ regardless of dimension (with i.i.d. samples)",
                        "Importance sampling & variational methods when $f$ is hard to sample",
                    ],
                ),
                (
                    "Differentiation — Autodiff & Backprop",
                    [
                        "Forward mode: propagate directional derivatives ($O(n)$ for $n$ inputs)",
                        "Reverse mode (backprop): one pass from scalar loss ($O(ops)$ for all params)",
                        "Computational graph stores local Jacobians; chain rule multiplies along paths",
                        "Frameworks implement vector–Jacobian products (VJPs), not full Jacobians",
                    ],
                ),
                (
                    "Implicit Function & Adjoints",
                    [
                        "Implicit Function Theorem: differentiate through $F(x, theta) = 0$ without explicit solve",
                        "Method of adjoints: efficient gradients for ODE/PDE-constrained objectives",
                        "Lagrange multipliers: constrained optimization via augmented Lagrangian",
                        "Stochastic gradient estimators: REINFORCE, reparameterization trick (Day 6–7)",
                    ],
                ),
            ],
        ),
        (
            "Probability & Distributions",
            [
                (
                    "Basics",
                    [
                        "PMF (discrete) / PDF (continuous); CDF $F(x) = P(X <= x)$",
                        'Expectation $EE[X]$, variance $"Var"(X) = EE[(X-EE[X])^2]$',
                        "Joint, marginal, conditional; independence $p(x,y) = p(x)p(y)$",
                        "Bayes: $p(theta|D) prop p(D|theta) p(theta)$",
                    ],
                ),
                (
                    "The Gaussian",
                    [
                        "$N(x; mu, Sigma) = (2 pi)^(-d/2) |Sigma|^(-1/2) exp(-1/2 (x-mu)^T Sigma^(-1)(x-mu))$",
                        "Affine transform: if $x sim N(mu, Sigma)$ then $A x + b sim N(A mu + b, A Sigma A^T)$",
                        "Marginals and conditionals of Gaussians remain Gaussian",
                        "Central role: CLT, linear-Gaussian models, diffusion noise",
                    ],
                ),
                (
                    "Conjugacy & Change of Variables",
                    [
                        "Conjugate prior: posterior same family as prior (e.g. Beta–Bernoulli, Normal–Normal)",
                        "Change of variables: $p_Y(y) = p_X(g^(-1)(y)) |det J_(g^(-1))(y)|$",
                        'Inverse transform sampling: $x = F^(-1)(u)$ for $u sim "Uniform"(0,1)$',
                        "Normalizing flows stack invertible maps with tractable Jacobians",
                    ],
                ),
                (
                    "Jensen & KL Divergence",
                    [
                        "Jensen: $f(EE[X]) <= EE[f(X)]$ for convex $f$ (evidence lower bounds)",
                        'KL divergence: $D_"KL"(q||p) = EE_q[log q/p] >= 0$; not symmetric',
                        "Cross-entropy $H(p,q) = -EE_p[log q]$ — classification loss",
                        "Gaussian KL has closed form — used in VAEs and diffusion training",
                    ],
                ),
            ],
        ),
        (
            "Differential Equations",
            [
                (
                    "ODE Intuition",
                    [
                        "ODE: $dot(x)(t) = v(x(t), t)$ — rate of change depends on state",
                        "Trajectory view: a curve $x(t)$ through state space",
                        "Vector field view: $v(x,t)$ assigns a velocity at each point",
                        "Same equation, two perspectives — flows vs integral curves",
                    ],
                ),
                (
                    "Solving ODEs",
                    [
                        "Linear ODE: exponential integrator / integrating factor",
                        "Example: $dot(x) = a x$ gives $x(t) = e^(a t) x(0)$",
                        "Existence & uniqueness when $v$ is Lipschitz",
                        "Autonomous vs time-varying; stability from eigenvalues of $partial v / partial x$",
                    ],
                ),
                (
                    "Numerical ODE Solvers",
                    [
                        "Euler: $x_(n+1) = x_n + h v(x_n, t_n)$ — first order, $O(h)$ error",
                        "Heun (RK2): predictor–corrector — $O(h^2)$ local error",
                        "Runge–Kutta (RK4): weighted average of slopes — workhorse $O(h^4)$",
                        "Adaptive step size balances accuracy vs cost (used in neural ODEs, diffusion samplers)",
                    ],
                ),
                (
                    "SDEs & Numerical Solvers",
                    [
                        "SDE: $dif X_t = f(X_t,t) dif t + g(X_t,t) dif W_t$ — adds Brownian noise",
                        "Itô calculus: $dif W_t^2 = dif t$ (quadratic variation)",
                        "Euler–Maruyama: $X_(n+1) = X_n + h f + sqrt(h) g xi_n$, $xi_n sim N(0,1)$",
                        "Foundation for diffusion models (Week 2): forward noising = SDE, reverse = learned drift",
                    ],
                ),
            ],
        ),
    ],
)

# Lecture dict for generate_lectures.py (matches SLIDES sections)
LECTURE = {
    "day": 1,
    "slug": "math-foundations",
    "title": "Math Foundations",
    "description": (
        "Linear algebra, analytic geometry, vector calculus, integration & differentiation, "
        "probability, and differential equations — MML Part I foundations."
    ),
    "reading": [
        "[Deisenroth, Faisal & Ong — *Mathematics for Machine Learning*](https://mml-book.com), Ch. 2–6",
        "[Deisenroth & Ong — *There and Back Again: A Tale of Slopes and Expectations*](https://mml-book.github.io/slopes-expectations.html) (NeurIPS 2020 tutorial)",
        "[Modern Integration Methods in ML](https://mml-book.github.io/book/additional_chapters/integration-methods.pdf) (MML supplementary chapter)",
        "[Diffusion Book — Appendix A: Crash Course on Differential Equations](https://arxiv.org/abs/2510.21890) (pp. 399 ff.)",
    ],
    "intro": (
        "Machine learning rests on four mathematical pillars laid out in Part I of "
        "*Mathematics for Machine Learning* (MML): we represent data as vectors and matrices "
        "(linear algebra); measure similarity with norms and inner products (analytic geometry); "
        "exploit matrix structure and gradients (decompositions and vector calculus); and quantify "
        "uncertainty with probability. Today we also cover integration and differentiation "
        "as complementary tools for expectations and optimization, and finish with an ODE/SDE "
        "crash course that foreshadows diffusion models in Week 2."
    ),
    "sections": [
        {
            "title": "Linear Algebra",
            "subsections": [
                {
                    "heading": "From linear equations to matrices",
                    "definition": (
                        "A system of $$m$$ linear equations in $$n$$ unknowns can be written as "
                        "$$\\mathbf{A}\\mathbf{x} = \\mathbf{b}$$ with "
                        "$$\\mathbf{A} \\in \\mathbb{R}^{m \\times n}$$, "
                        "$$\\mathbf{x} \\in \\mathbb{R}^n$$, "
                        "$$\\mathbf{b} \\in \\mathbb{R}^m$$. Each row of $$\\mathbf{A}$$ "
                        "defines one hyperplane; solutions lie at their intersection."
                    ),
                    "body": """Consider two equations in two unknowns:

$$\\begin{aligned} a_{11} x_1 + a_{12} x_2 &= b_1 \\\\ a_{21} x_1 + a_{22} x_2 &= b_2 \\end{aligned}$$

Stacking coefficients gives $$\\mathbf{A} = \\begin{pmatrix} a_{11} & a_{12} \\\\ a_{21} & a_{22} \\end{pmatrix}$$, so one matrix–vector multiply encodes the entire system. This is the bridge from high-school algebra to the language of machine learning, where a dataset of $$n$$ examples each with $$d$$ features is stored as $$\\mathbf{X} \\in \\mathbb{R}^{n \\times d}$$.

![Linear systems and matrix view (MML Fig 2.3)](/assets/figures/day01/mml_linear_system.png)

A matrix is not merely a table of numbers — it is a **linear map** $$f(\\mathbf{x}) = \\mathbf{A}\\mathbf{x}$$. The *column picture* writes $$\\mathbf{b} = x_1 \\mathbf{a}_1 + x_2 \\mathbf{a}_2 + \\cdots$$: $$\\mathbf{b}$$ must lie in the span of the columns. The *row picture* views each equation as a hyperplane.""",
                },
                {
                    "heading": "Vector spaces, basis, and rank",
                    "definition": (
                        "A **vector space** $$V$$ is closed under addition and scalar multiplication. "
                        "A set $$\\{\\mathbf{v}_1, \\ldots, \\mathbf{v}_k\\}$$ is a **basis** if it is "
                        "linearly independent and spans $$V$$. The **rank** of $$\\mathbf{A}$$ is the "
                        "dimension of its column space."
                    ),
                    "body": """Key subspaces associated with $$\\mathbf{A} \\in \\mathbb{R}^{m \\times n}$$:

- **Column space** $$\\mathcal{C}(\\mathbf{A}) \\subseteq \\mathbb{R}^m$$: all reachable outputs $$\\mathbf{A}\\mathbf{x}$$.
- **Null space** $$\\mathcal{N}(\\mathbf{A})$$: all $$\\mathbf{x}$$ with $$\\mathbf{A}\\mathbf{x} = \\mathbf{0}$$.
- **Row space** and **left null space** (orthogonal complements in the appropriate spaces).

The rank–nullity theorem: $$\\mathrm{rank}(\\mathbf{A}) + \\dim \\mathcal{N}(\\mathbf{A}) = n$$.

![Vector subspace (MML Fig 2.6)](/assets/figures/day01/mml_subspace.png)

In ML, features live in a high-dimensional space; learning often finds a low-dimensional subspace (PCA, autoencoders) or selects a sparse subset of coordinates (Lasso).""",
                },
            ],
        },
        {
            "title": "Analytic Geometry",
            "subsections": [
                {
                    "heading": "Norms and inner products",
                    "definition": (
                        "An **inner product** $$\\langle \\mathbf{x}, \\mathbf{y} \\rangle$$ satisfies "
                        "symmetry, linearity, and positive definiteness. The induced **norm** is "
                        "$$\\|\\mathbf{x}\\| = \\sqrt{\\langle \\mathbf{x}, \\mathbf{x} \\rangle}$$. "
                        "For the standard dot product, $$\\|\\mathbf{x}\\|_2 = \\sqrt{\\sum_i x_i^2}$$."
                    ),
                    "body": """Common norms in machine learning:

$$\\|\\mathbf{x}\\|_1 = \\sum_i |x_i|, \\qquad \\|\\mathbf{x}\\|_2 = \\sqrt{\\sum_i x_i^2}, \\qquad \\|\\mathbf{x}\\|_\\infty = \\max_i |x_i|.$$

The **Cauchy–Schwarz inequality** $$|\\langle \\mathbf{x}, \\mathbf{y} \\rangle| \\leq \\|\\mathbf{x}\\|_2 \\|\\mathbf{y}\\|_2$$ lets us define angles between vectors. Two vectors are **orthogonal** when $$\\langle \\mathbf{x}, \\mathbf{y} \\rangle = 0$$.

![Angle between vectors (MML Fig 3.6)](/assets/figures/day01/mml_angle.png)

Why this matters: if two input vectors are close under a chosen norm, we want our predictor to produce similar outputs — a geometric inductive bias.""",
                },
                {
                    "heading": "Projections and least squares",
                    "definition": (
                        "The **orthogonal projection** of $$\\mathbf{y}$$ onto the line spanned by "
                        "$$\\mathbf{x}$$ is $$\\hat{\\mathbf{y}} = \\frac{\\mathbf{x}^\\top \\mathbf{y}}{\\mathbf{x}^\\top \\mathbf{x}} \\mathbf{x}$$. "
                        "More generally, projecting onto $$\\mathcal{C}(\\mathbf{A})$$ uses "
                        "$$\\mathbf{P} = \\mathbf{A}(\\mathbf{A}^\\top \\mathbf{A})^{-1}\\mathbf{A}^\\top$$."
                    ),
                    "body": """**Ordinary least squares** minimizes $$\\|\\mathbf{A}\\mathbf{w} - \\mathbf{y}\\|_2^2$$. Geometrically, $$\\hat{\\mathbf{y}} = \\mathbf{A}\\hat{\\mathbf{w}}$$ is the projection of $$\\mathbf{y}$$ onto the column space of $$\\mathbf{A}$$. Setting the gradient to zero yields the **normal equations**:

$$\\mathbf{A}^\\top \\mathbf{A} \\hat{\\mathbf{w}} = \\mathbf{A}^\\top \\mathbf{y}.$$

![Projection onto a subspace (MML Fig 3.11)](/assets/figures/day01/mml_projection.png)

This connects linear algebra (Day 1) directly to regression (Day 2): fitting a linear model is projecting labels onto the span of features.""",
                },
            ],
        },
        {
            "title": "Vector Calculus and Matrix Decompositions",
            "subsections": [
                {
                    "heading": "Gradients, Jacobians, and the chain rule",
                    "definition": (
                        "For scalar $$f: \\mathbb{R}^n \\to \\mathbb{R}$$, the **gradient** "
                        "$$\\nabla f(\\mathbf{x}) \\in \\mathbb{R}^n$$ points in the direction of "
                        "steepest ascent. For $$\\mathbf{f}: \\mathbb{R}^n \\to \\mathbb{R}^m$$, the "
                        "**Jacobian** $$\\mathbf{J} \\in \\mathbb{R}^{m \\times n}$$ has entries "
                        "$$J_{ij} = \\partial f_i / \\partial x_j$$."
                    ),
                    "body": """Useful identities (memorize these):

$$\\nabla_{\\mathbf{x}} (\\mathbf{a}^\\top \\mathbf{x}) = \\mathbf{a}, \\qquad \\nabla_{\\mathbf{x}} (\\mathbf{x}^\\top \\mathbf{A} \\mathbf{x}) = (\\mathbf{A} + \\mathbf{A}^\\top)\\mathbf{x}.$$

For a quadratic loss $$L(\\mathbf{w}) = \\|\\mathbf{X}\\mathbf{w} - \\mathbf{y}\\|_2^2$$,

$$\\nabla_{\\mathbf{w}} L = 2\\mathbf{X}^\\top(\\mathbf{X}\\mathbf{w} - \\mathbf{y}).$$

![Gradient as the slope of a secant (MML Fig 5.3)](/assets/figures/day01/mml_gradient.png)

The multivariate **chain rule** propagates sensitivities through composed functions — the mathematical content of backpropagation (Section 4).""",
                },
                {
                    "heading": "Eigendecomposition and SVD",
                    "definition": (
                        "If $$\\mathbf{A}\\mathbf{v} = \\lambda \\mathbf{v}$$, then $$\\mathbf{v}$$ is an "
                        "**eigenvector** with **eigenvalue** $$\\lambda$$. For symmetric $$\\mathbf{A}$$, "
                        "$$\\mathbf{A} = \\mathbf{Q}\\boldsymbol{\\Lambda}\\mathbf{Q}^\\top$$. "
                        "The **SVD** is $$\\mathbf{A} = \\mathbf{U}\\boldsymbol{\\Sigma}\\mathbf{V}^\\top$$ "
                        "— always exists."
                    ),
                    "body": """The SVD reveals the action of $$\\mathbf{A}$$ as rotate–scale–rotate. Truncating to the top-$$k$$ singular values gives the best rank-$$k$$ approximation (Eckart–Young).

**PCA** finds orthogonal directions of maximum variance: eigenvectors of the covariance matrix $$\\mathbf{C} = \\frac{1}{n}\\mathbf{X}^\\top \\mathbf{X}$$ (after centering).

![SVD geometry (MML Fig 4.9)](/assets/figures/day01/mml_svd.png)

Eigenvalues of the Hessian at a critical point classify it as minimum, maximum, or saddle — relevant for understanding neural network loss landscapes.""",
                },
            ],
        },
        {
            "title": "Integration and Differentiation",
            "subsections": [
                {
                    "heading": "Integration: expectations and numerical methods",
                    "definition": (
                        "Many ML quantities are **expectations** $$\\mathbb{E}[f(X)] = \\int f(x) p(x)\\,dx$$. "
                        "When the integral is intractable we use **numerical quadrature** (low dimension) "
                        "or **Monte Carlo** (high dimension): "
                        "$$\\mathbb{E}[f(X)] \\approx \\frac{1}{N}\\sum_{i=1}^N f(\\mathbf{x}^{(i)}).$$"
                    ),
                    "body": """The NeurIPS 2020 tutorial [*There and Back Again: A Tale of Slopes and Expectations*](https://mml-book.github.io/slopes-expectations.html) treats integration and differentiation as two directions on the same map — expectations require integration; learning requires differentiation.

![Slopes and expectations map](/assets/figures/day01/slopes_map.jpg)

**Deterministic methods** (trapezoidal rule, Simpson's rule, Gauss–Hermite quadrature) excel in low dimensions. **Monte Carlo** error scales as $$O(1/\\sqrt{N})$$ independently of dimension — crucial for Bayesian marginalization and variational objectives.

![Unscented transform / sigma points (Modern Integration Methods, Fig 8)](/assets/figures/day01/integ_unscented.png)

See also the MML supplementary chapter [*Modern Integration Methods in ML*](https://mml-book.github.io/book/additional_chapters/integration-methods.pdf).""",
                },
                {
                    "heading": "Differentiation: autodiff, adjoints, and gradient estimators",
                    "definition": (
                        "**Reverse-mode automatic differentiation** computes "
                        "$$\\nabla_{\\boldsymbol{\\theta}} L$$ for a scalar loss $$L$$ in one backward pass "
                        "through the computational graph — cost $$O(\\text{ops})$$, not $$O(|\\boldsymbol{\\theta}|)$$ "
                        "times forward cost."
                    ),
                    "body": """**Forward mode** propagates directional derivatives; **reverse mode** (backprop) is preferred when there is one scalar output and many parameters.

When outputs are defined implicitly by $$F(\\mathbf{x}, \\boldsymbol{\\theta}) = \\mathbf{0}$$, the **implicit function theorem** gives

$$\\frac{\\partial \\mathbf{x}}{\\partial \\boldsymbol{\\theta}} = -\\left(\\frac{\\partial F}{\\partial \\mathbf{x}}\\right)^{-1} \\frac{\\partial F}{\\partial \\boldsymbol{\\theta}}.$$

The **method of adjoints** and **Lagrange multipliers** extend this to ODE-constrained and constrained optimization problems.

**Stochastic gradient estimators** (REINFORCE score-function estimator, reparameterization $$\\nabla \\mathbb{E}[f(\\mathbf{z})]$$ via $$\\mathbf{z} = g(\\boldsymbol{\\epsilon}, \\boldsymbol{\\theta})$$) let us differentiate through expectations — central to VAEs and policy gradients.

![Monte Carlo samples across a sequence of distributions (Modern Integration Methods, Fig 6)](/assets/figures/day01/integ_samples.png)""",
                },
            ],
        },
        {
            "title": "Probability and Distributions",
            "subsections": [
                {
                    "heading": "Basics and the Gaussian",
                    "definition": (
                        "A continuous random vector $$\\mathbf{x}$$ with density $$p$$ satisfies "
                        "$$\\mathbb{E}[\\mathbf{x}] = \\int \\mathbf{x}\\, p(\\mathbf{x})\\,d\\mathbf{x}$$. "
                        "The multivariate Gaussian "
                        "$$\\mathcal{N}(\\boldsymbol{\\mu}, \\boldsymbol{\\Sigma})$$ has density "
                        "$$p(\\mathbf{x}) \\propto \\exp\\big(-\\tfrac{1}{2}(\\mathbf{x}-\\boldsymbol{\\mu})^\\top "
                        "\\boldsymbol{\\Sigma}^{-1}(\\mathbf{x}-\\boldsymbol{\\mu})\\big)$$."
                    ),
                    "body": """Key Gaussian closure properties:

- Affine transform: $$\\mathbf{A}\\mathbf{x} + \\mathbf{b} \\sim \\mathcal{N}(\\mathbf{A}\\boldsymbol{\\mu} + \\mathbf{b}, \\mathbf{A}\\boldsymbol{\\Sigma}\\mathbf{A}^\\top)$$.
- Marginals and conditionals of joint Gaussians are Gaussian.
- Sum of independent Gaussians is Gaussian.

![Gaussian distribution (MML Fig 6.7)](/assets/figures/day01/mml_gaussian.png)

The Gaussian is the maximum-entropy distribution with fixed mean and covariance — and the limiting distribution of sums (CLT).""",
                },
                {
                    "heading": "Conjugacy, change of variables, Jensen, and KL",
                    "definition": (
                        "A prior is **conjugate** to a likelihood if the posterior belongs to the same "
                        "parametric family. Under a smooth bijection $$\\mathbf{y} = g(\\mathbf{x})$$, "
                        "$$p_Y(\\mathbf{y}) = p_X(g^{-1}(\\mathbf{y}))\\,|\\det \\mathbf{J}_{g^{-1}}(\\mathbf{y})|$$. "
                        "**Jensen's inequality**: $$f(\\mathbb{E}[X]) \\leq \\mathbb{E}[f(X)]$$ for convex $$f$$."
                    ),
                    "body": """**Conjugate pairs** (Beta–Bernoulli, Normal–Normal, Dirichlet–Multinomial) give closed-form Bayesian updates — useful for pedagogy and some models.

**Inverse transform sampling**: if $$U \\sim \\mathrm{Uniform}(0,1)$$, then $$X = F^{-1}(U)$$ has CDF $$F$$. Normalizing flows compose invertible maps with tractable Jacobians.

**KL divergence** $$D_{\\mathrm{KL}}(q\\|p) = \\mathbb{E}_{\\mathbf{x}\\sim q}[\\log q(\\mathbf{x}) - \\log p(\\mathbf{x})] \\geq 0$$ measures how many extra nats are needed to encode samples from $$q$$ using a code optimized for $$p$$. It is not symmetric.

Jensen's inequality underlies the **evidence lower bound (ELBO)** in variational inference:

$$\\log p(\\mathbf{x}) \\geq \\mathbb{E}_{q(\\mathbf{z})}[\\log p(\\mathbf{x}|\\mathbf{z})] - D_{\\mathrm{KL}}(q(\\mathbf{z})\\|p(\\mathbf{z})).$$

![Conjugate prior example (MML Fig 6.11)](/assets/figures/day01/mml_conjugate.png)""",
                },
            ],
        },
        {
            "title": "Differential Equations (ODE & SDE Crash Course)",
            "subsections": [
                {
                    "heading": "ODEs: vector fields, trajectories, and solvers",
                    "definition": (
                        "An **ordinary differential equation** specifies how a state evolves: "
                        "$$\\dot{\\mathbf{x}}(t) = \\mathbf{v}(\\mathbf{x}(t), t).$$ "
                        "The **vector field** $$\\mathbf{v}$$ assigns a velocity at each point; "
                        "**trajectories** are integral curves following that field."
                    ),
                    "body": """Two equivalent views:

1. **Trajectory**: a curve $$\\mathbf{x}(t)$$ through state space.
2. **Vector field**: an arrow $$\\mathbf{v}(\\mathbf{x}, t)$$ at every point.

For linear $$\\dot{x} = ax$$, the solution is $$x(t) = e^{at} x(0)$$ — the **exponential integrator** idea generalizes to matrix systems $$\\dot{\\mathbf{x}} = \\mathbf{A}\\mathbf{x}$$.

**Numerical solvers** discretize time with step $$h$$:

| Method | Update | Local error |
|--------|--------|-------------|
| Euler | $$\\mathbf{x}_{n+1} = \\mathbf{x}_n + h\\,\\mathbf{v}(\\mathbf{x}_n, t_n)$$ | $$O(h)$$ |
| Heun (RK2) | predictor–corrector average | $$O(h^2)$$ |
| RK4 | four weighted slope evaluations | $$O(h^4)$$ |

![ODE Figure A.1 — left: step-by-step solver updates; right: exact trajectories flowing along the velocity field](/assets/figures/day01/ode_vectorfield.png)

Source: *Diffusion Book* Appendix A (crash course on differential equations), from p. 399.""",
                },
                {
                    "heading": "SDEs and numerical simulation",
                    "definition": (
                        "A **stochastic differential equation** adds noise: "
                        "$$d\\mathbf{X}_t = \\mathbf{f}(\\mathbf{X}_t, t)\\,dt + \\mathbf{g}(\\mathbf{X}_t, t)\\,d\\mathbf{W}_t,$$ "
                        "where $$\\mathbf{W}_t$$ is Brownian motion. "
                        "**Itô's lemma** governs calculus with $$dW_t^2 = dt$$."
                    ),
                    "body": """SDEs model systems with intrinsic randomness — and the forward process in diffusion models.

**Euler–Maruyama** discretization:

$$\\mathbf{X}_{n+1} = \\mathbf{X}_n + h\\,\\mathbf{f}(\\mathbf{X}_n, t_n) + \\sqrt{h}\\,\\mathbf{g}(\\mathbf{X}_n, t_n)\\,\\boldsymbol{\\xi}_n, \\qquad \\boldsymbol{\\xi}_n \\sim \\mathcal{N}(\\mathbf{0}, \\mathbf{I}).$$

Higher-order schemes (Milstein) improve strong convergence when diffusion matters.

In Week 2 we will connect these solvers directly to sampling from diffusion and flow models.""",
                },
            ],
        },
    ],
    "checkpoint": [
        "Translate linear systems into matrix form and interpret column/null spaces.",
        "Compute projections and connect least squares to the normal equations.",
        "Apply gradient identities and interpret eigendecomposition / SVD geometrically.",
        "Contrast numerical quadrature with Monte Carlo for expectations.",
        "Explain reverse-mode AD and when the implicit function theorem applies.",
        "State Gaussian closure properties, change-of-variables formula, and Jensen's inequality.",
        "Describe ODE vector-field vs trajectory views and compare Euler, Heun, and RK4.",
        "Write the Euler–Maruyama update for an SDE.",
    ],
}
