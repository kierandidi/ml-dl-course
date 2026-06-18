#import "../lib.typ": *

#show: course-theme.with(title: [Math Foundations], subtitle: [Day 1 | Aug 2026])

= Day 1: Math Foundations

== Welcome

- *Math Foundations* — Linear algebra through differential equations — MML Part I
- 3 hours lecture + practical
- Slides, notes, and code on the course site

== Outline

- The Mathematical Foundation
- Linear Algebra
- Analytic Geometry
- Vector Calculus & Decompositions
- Integration & Differentiation
- Probability & Distributions
- Differential Equations

= The Mathematical Foundation

== Four Pillars of ML

- Data as vectors; tables of data as matrices (linear algebra)
- Similarity via norms and inner products (analytic geometry)
- Structure via matrix decompositions; optimization via gradients (Ch. 4–5)
- Uncertainty via probability; dynamics via differential equations
- Source: Deisenroth, Faisal & Ong — *Mathematics for Machine Learning*

== Four Pillars of ML — illustration

#align(center)[#image("/assets/figures/day01/mml_vectors_types.png", width: 80%)]

#text(size: 14pt, fill: gray)[The Mathematical Foundation — Four Pillars of ML (source: course materials)]

== Roadmap for Today

- 1. Linear algebra — equations, matrices, vector spaces
- 2. Analytic geometry — norms, inner products, projections
- 3. Vector calculus & matrix decompositions
- 4. Integration & differentiation (*There and Back Again*)
- 5. Probability & distributions
- 6. ODE/SDE crash course (diffusion book appendix)

= Linear Algebra

== From Linear Equations to Matrices

- System $A x = b$: each row is one linear constraint on unknowns $x in RR^n$
- Coefficient matrix $A in RR^(m times n)$ encodes all equations at once
- Solution sets are affine subspaces (when consistent)
- Matrix-vector product: each output component is an inner product of a row with $x$

== From Linear Equations to Matrices — illustration

#align(center)[#image("/assets/figures/day01/mml_linear_system.png", width: 80%)]

#text(size: 14pt, fill: gray)[Linear Algebra — From Linear Equations to Matrices (source: course materials)]

== What Is a Matrix?

- A matrix is a linear map $f: RR^n arrow.r RR^m$ written as $y = A x$
- Columns of $A$ show where basis vectors land — geometric action of the map
- Composition of maps = matrix multiplication: $(A B) x = A (B x)$
- Transpose swaps rows/columns; inverse (when it exists) undoes the map

== What Is a Matrix? — illustration

#align(center)[#image("/assets/figures/day01/mml_linear_mapping.png", width: 80%)]

#text(size: 14pt, fill: gray)[Linear Algebra — What Is a Matrix? (source: course materials)]

== Vector Spaces

- Vector space: closed under addition and scalar multiplication
- Linear combination: $sum_i alpha_i v_i$; span = all reachable combinations
- Linear independence: no vector is a combination of the others
- Basis: minimal spanning set; dimension = number of basis vectors
- Rank of $A$ = dimension of column space = dim of row space

== Vector Spaces — illustration

#align(center)[#image("/assets/figures/day01/mml_subspace.png", width: 80%)]

#text(size: 14pt, fill: gray)[Linear Algebra — Vector Spaces (source: course materials)]

== Solving & Interpreting $A x = b$

- Gaussian elimination / LU factorization (algorithmic)
- Column picture: $b$ must lie in span of columns of $A$
- Row picture: intersection of hyperplanes
- Null space: all $x$ with $A x = 0$ — directions the map collapses

== Solving & Interpreting $A x = b$ — illustration

#align(center)[#image("/assets/figures/day01/mml_kernel_nullspace.png", width: 80%)]

#text(size: 14pt, fill: gray)[Linear Algebra — Solving & Interpreting $A x = b$ (source: course materials)]

= Analytic Geometry

== Why Similarity Matters

- ML predictors should assign similar outputs to similar inputs
- Vectors encode objects; we need a numeric notion of 'close' or 'aligned'
- Norms measure length/magnitude; inner products measure alignment
- Builds directly on linear-algebra vector operations

== Why Similarity Matters — illustration

#align(center)[#image("/assets/figures/day01/mml_geometry_mindmap.png", width: 80%)]

#text(size: 14pt, fill: gray)[Analytic Geometry — Why Similarity Matters (source: course materials)]

== Norms

- $L_2$ (Euclidean): $||x||_2 = sqrt(sum_i x_i^2)$ — rotation-invariant length
- $L_1$: $||x||_1 = sum_i |x_i|$ — sparsity-inducing geometry
- $L_infinity$: $||x||_infinity = max_i |x_i|$
- Unit ball shapes differ — choice of norm changes regularization (Day 2)

== Norms — illustration

#align(center)[#image("/assets/figures/day01/mml_triangle_ineq.png", width: 80%)]

#text(size: 14pt, fill: gray)[Analytic Geometry — Norms (source: course materials)]

== Inner Product & Angles

- Inner product $chevron.l x, y chevron.r = x^T y$ (standard dot product)
- Cauchy–Schwarz: $|chevron.l x,y chevron.r| <= ||x||_2 ||y||_2$
- Angle: $cos theta = chevron.l x,y chevron.r / (||x||_2 ||y||_2)$
- Orthogonal vectors: $chevron.l x,y chevron.r = 0$

== Inner Product & Angles — illustration

#align(center)[#image("/assets/figures/day01/mml_angle.png", width: 80%)]

#text(size: 14pt, fill: gray)[Analytic Geometry — Inner Product & Angles (source: course materials)]

== Projections

- Orthogonal projection of $y$ onto line spanned by $x$: $hat(y) = (x^T y / x^T x) x$
- Projection matrix onto column space of $A$: $P = A (A^T A)^(-1) A^T$
- Least squares: minimize $||A w - y||_2^2$ — project $y$ onto span(columns of $A)$
- Normal equations: $A^T A w = A^T y$

== Projections — illustration

#align(center)[#image("/assets/figures/day01/mml_projection.png", width: 80%)]

#text(size: 14pt, fill: gray)[Analytic Geometry — Projections (source: course materials)]

= Vector Calculus & Decompositions

== Gradients & Jacobians

- Scalar $f(x)$: gradient $nabla f(x) in RR^n$ — direction of steepest ascent
- Vector $f(x) in RR^m$: Jacobian $J in RR^(m times n)$, $J_(i j) = partial f_i / partial x_j$
- Matrix-valued functions: treat as stacked scalar functions
- Chain rule: $nabla_x (g compose f) = J_f^T nabla_g$ (reverse-mode building block)

== Gradients & Jacobians — illustration

#align(center)[#image("/assets/figures/day01/mml_gradient.png", width: 80%)]

#text(size: 14pt, fill: gray)[Vector Calculus & Decompositions — Gradients & Jacobians (source: course materials)]

== Useful Matrix Derivatives

- $nabla_x (a^T x) = a$; $nabla_x (x^T A x) = (A + A^T) x$
- $nabla_A ||A x - y||_2^2 = 2 (A x - y) x^T$
- Hessian $H = nabla^2 f$ — second-order curvature for Newton methods
- These identities appear constantly in backprop and closed-form solutions

== Useful Matrix Derivatives — illustration

#align(center)[#image("/assets/figures/day01/mml_taylor.png", width: 80%)]

#text(size: 14pt, fill: gray)[Vector Calculus & Decompositions — Useful Matrix Derivatives (source: course materials)]

== Eigendecomposition

- $A v = lambda v$ — eigenvector direction preserved, scaled by $lambda$
- Symmetric $A$: real eigenvalues, orthogonal eigenvectors
- Spectral theorem: $A = Q Lambda Q^T$ — principal axes of quadratic forms
- Condition number from eigenvalue spread — numerical stability

== Eigendecomposition — illustration

#align(center)[#image("/assets/figures/day01/mml_eigen.png", width: 80%)]

#text(size: 14pt, fill: gray)[Vector Calculus & Decompositions — Eigendecomposition (source: course materials)]

== SVD & PCA Preview

- SVD: $A = U Sigma V^T$ — always exists; singular values on diagonal of $Sigma$
- Low-rank approximation: keep top-$k$ singular values (Eckart–Young)
- PCA: eigenvectors of covariance $C = (1/n) X^T X$
- Connects linear algebra geometry to data compression and denoising

== SVD & PCA Preview — illustration

#align(center)[#image("/assets/figures/day01/mml_svd.png", width: 80%)]

#text(size: 14pt, fill: gray)[Vector Calculus & Decompositions — SVD & PCA Preview (source: course materials)]

= Integration & Differentiation

== There and Back Again

- NeurIPS 2020 tutorial: integration *and* differentiation for ML
- Integration: expectations, marginalization, Bayesian evidence
- Differentiation: gradients for optimization, sensitivity analysis
- Map metaphor: numerical integration swamps, Monte Carlo heights, Backprop Bay…

== There and Back Again — illustration

#align(center)[#image("/assets/figures/day01/slopes_map.jpg", width: 80%)]

#text(size: 14pt, fill: gray)[Integration & Differentiation — There and Back Again (source: course materials)]

== Integration — Numerical & MC

- Deterministic quadrature: trapezoid, Simpson, Gauss–Hermite
- Curse of dimensionality motivates Monte Carlo: $EE[f(X)] approx (1/N) sum f(x_i)$
- MC error $O(1/sqrt(N))$ regardless of dimension (with i.i.d. samples)
- Importance sampling & variational methods when $f$ is hard to sample

== Integration — Numerical & MC — illustration

#align(center)[#image("/assets/figures/day01/integ_unscented.png", width: 80%)]

#text(size: 14pt, fill: gray)[Integration & Differentiation — Integration — Numerical & MC (source: course materials)]

== Differentiation — Autodiff & Backprop

- Forward mode: propagate directional derivatives ($O(n)$ for $n$ inputs)
- Reverse mode (backprop): one pass from scalar loss ($O("ops")$ for all params)
- Computational graph stores local Jacobians; chain rule multiplies along paths
- Frameworks implement vector–Jacobian products (VJPs), not full Jacobians

== Differentiation — Autodiff & Backprop — illustration

#align(center)[#image("/assets/figures/day01/mml_forward_pass.png", width: 80%)]

#text(size: 14pt, fill: gray)[Integration & Differentiation — Differentiation — Autodiff & Backprop (source: course materials)]

== Implicit Function & Adjoints

- Implicit Function Theorem: differentiate through $F(x, theta) = 0$ without explicit solve
- Method of adjoints: efficient gradients for ODE/PDE-constrained objectives
- Lagrange multipliers: constrained optimization via augmented Lagrangian
- Stochastic gradient estimators: REINFORCE, reparameterization trick (Day 6–7)

== Implicit Function & Adjoints — illustration

#align(center)[#image("/assets/figures/day01/integ_samples.png", width: 80%)]

#text(size: 14pt, fill: gray)[Integration & Differentiation — Implicit Function & Adjoints (source: course materials)]

= Probability & Distributions

== Basics

- PMF (discrete) / PDF (continuous); CDF $F(x) = P(X <= x)$
- Expectation $EE[X]$, variance $"Var"(X) = EE[(X-EE[X])^2]$
- Joint, marginal, conditional; independence $p(x,y) = p(x)p(y)$
- Bayes: $p(theta|D) prop p(D|theta) p(theta)$

== Basics — illustration

#align(center)[#image("/assets/figures/day01/mml_distributions.png", width: 80%)]

#text(size: 14pt, fill: gray)[Probability & Distributions — Basics (source: course materials)]

== The Gaussian

- $N(x; mu, Sigma) = (2 pi)^(-d/2) |Sigma|^(-1/2) exp(-1/2 (x-mu)^T Sigma^(-1)(x-mu))$
- Affine transform: if $x tilde N(mu, Sigma)$ then $A x + b tilde N(A mu + b, A Sigma A^T)$
- Marginals and conditionals of Gaussians remain Gaussian
- Central role: CLT, linear-Gaussian models, diffusion noise

== The Gaussian — illustration

#align(center)[#image("/assets/figures/day01/mml_gaussian.png", width: 80%)]

#text(size: 14pt, fill: gray)[Probability & Distributions — The Gaussian (source: course materials)]

== Conjugacy & Change of Variables

- Conjugate prior: posterior same family as prior (e.g. Beta–Bernoulli, Normal–Normal)
- Change of variables: $p_Y(y) = p_X(g^(-1)(y)) |det J_(g^(-1))(y)|$
- Inverse transform sampling: $x = F^(-1)(u)$ for $u tilde "Uniform"(0,1)$
- Normalizing flows stack invertible maps with tractable Jacobians

== Conjugacy & Change of Variables — illustration

#align(center)[#image("/assets/figures/day01/mml_conjugate.png", width: 80%)]

#text(size: 14pt, fill: gray)[Probability & Distributions — Conjugacy & Change of Variables (source: course materials)]

== Jensen & KL Divergence

- Jensen: $f(EE[X]) <= EE[f(X)]$ for convex $f$ (evidence lower bounds)
- KL divergence: $D_"KL"(q||p) = EE_q[log q/p] >= 0$; not symmetric
- Cross-entropy $H(p,q) = -EE_p[log q]$ — classification loss
- Gaussian KL has closed form — used in VAEs and diffusion training

= Differential Equations

== ODE Intuition

- ODE: $dot(x)(t) = v(x(t), t)$ — rate of change depends on state
- Trajectory view: a curve $x(t)$ through state space
- Vector field view: $v(x,t)$ assigns a velocity at each point
- Same equation, two perspectives — flows vs integral curves

== ODE Intuition — illustration

#align(center)[#image("/assets/figures/day01/ode_vectorfield.png", width: 80%)]

#text(size: 14pt, fill: gray)[Differential Equations — ODE Intuition (source: course materials)]

== Solving ODEs

- Linear ODE: exponential integrator / integrating factor
- Example: $dot(x) = a x$ gives $x(t) = e^(a t) x(0)$
- Existence & uniqueness when $v$ is Lipschitz
- Autonomous vs time-varying; stability from eigenvalues of $partial v / partial x$

== Numerical ODE Solvers

- Euler: $x_(n+1) = x_n + h v(x_n, t_n)$ — first order, $O(h)$ error
- Heun (RK2): predictor–corrector — $O(h^2)$ local error
- Runge–Kutta (RK4): weighted average of slopes — workhorse $O(h^4)$
- Adaptive step size balances accuracy vs cost (used in neural ODEs, diffusion samplers)

== SDEs & Numerical Solvers

- SDE: $dif X_t = f(X_t,t) dif t + g(X_t,t) dif W_t$ — adds Brownian noise
- Itô calculus: $dif W_t^2 = dif t$ (quadratic variation)
- Euler–Maruyama: $X_(n+1) = X_n + h f + sqrt(h) g xi_n$, $xi_n tilde N(0,1)$
- Foundation for diffusion models (Week 2): forward noising = SDE, reverse = learned drift

== Summary

- Day 1: *Math Foundations*
- Linear algebra through differential equations — MML Part I
- Questions welcome — practical follows

== Questions?

#align(center + horizon)[
  #text(size: 44pt, weight: "bold", fill: course-primary)[Questions?]

  #v(1em)
  Practical session follows — see course site.
]
