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

= 1 · The Mathematical Foundation

== 1.1  Four Pillars of ML

- Data as vectors; tables of data as matrices (linear algebra)
- Similarity via norms and inner products (analytic geometry)
- Structure via matrix decompositions; optimization via gradients (Ch. 4–5)
- Uncertainty via probability; dynamics via differential equations
- Source: Deisenroth, Faisal & Ong — *Mathematics for Machine Learning*

== 1.1  Four Pillars of ML

#align(center + horizon)[#image("/assets/figures/day01/mml_vectors_types.png", width: 92%, height: 82%, fit: "contain")]

== 1.2  Roadmap for Today

- 1. Linear algebra — equations, matrices, vector spaces
- 2. Analytic geometry — norms, inner products, projections
- 3. Vector calculus & matrix decompositions
- 4. Integration & differentiation (*There and Back Again*)
- 5. Probability & distributions
- 6. ODE/SDE crash course (diffusion book appendix)

= 2 · Linear Algebra

== 2.1  From Linear Equations to Matrices

- System $A x = b$: each row is one linear constraint on unknowns $x in RR^n$
- Coefficient matrix $A in RR^(m times n)$ encodes all equations at once
- Solution sets are affine subspaces (when consistent)
- Matrix-vector product: each output component is an inner product of a row with $x$

== 2.1  From Linear Equations to Matrices

#align(center + horizon)[#image("/assets/figures/day01/mml_linear_system.png", width: 92%, height: 82%, fit: "contain")]

== 2.2  What Is a Matrix?

- A matrix is a linear map $f: RR^n arrow.r RR^m$ written as $y = A x$
- Columns of $A$ show where basis vectors land — geometric action of the map
- Row picture: each equation a hyperplane; column picture: $b$ in span of columns
- Composition of maps = matrix multiplication: $(A B) x = A (B x)$
- Transpose swaps rows/columns; inverse (when it exists) undoes the map
- Data lives here too: $n$ examples × $d$ features form $X in RR^(n times d)$

== 2.2  What Is a Matrix?

#align(center + horizon)[#image("/assets/figures/day01/mml_linear_mapping.png", width: 92%, height: 82%, fit: "contain")]

== 2.3  Vector Spaces

- Vector space: closed under addition and scalar multiplication
- Linear combination: $sum_i alpha_i v_i$; span = all reachable combinations
- Linear independence: no vector is a combination of the others
- Basis: minimal spanning set; dimension = number of basis vectors
- Rank of $A$ = dimension of column space = dim of row space

== 2.3  Vector Spaces

#align(center + horizon)[#image("/assets/figures/day01/mml_subspace.png", width: 92%, height: 82%, fit: "contain")]

== 2.4  Solving & Interpreting $A x = b$

- Gaussian elimination / LU factorization (algorithmic)
- Column picture: $b$ must lie in span of columns of $A$
- Row picture: intersection of hyperplanes
- Null space: all $x$ with $A x = 0$ — directions the map collapses

== 2.4  Solving & Interpreting $A x = b$

#align(center + horizon)[#image("/assets/figures/day01/mml_kernel_nullspace.png", width: 92%, height: 82%, fit: "contain")]

= 3 · Analytic Geometry

== 3.1  Why Similarity Matters

- ML predictors should assign similar outputs to similar inputs
- Vectors encode objects; we need a numeric notion of 'close' or 'aligned'
- Norms measure length/magnitude; inner products measure alignment
- Builds directly on linear-algebra vector operations

== 3.1  Why Similarity Matters

#align(center + horizon)[#image("/assets/figures/day01/mml_geometry_mindmap.png", width: 92%, height: 82%, fit: "contain")]

== 3.2  Norms

- $L_2$ (Euclidean): $||x||_2 = sqrt(sum_i x_i^2)$ — rotation-invariant length
- $L_1$: $||x||_1 = sum_i |x_i|$ — sparsity-inducing geometry
- $L_infinity$: $||x||_infinity = max_i |x_i|$
- Unit ball shapes differ — choice of norm changes regularization (Day 2)

== 3.2  Norms

#align(center + horizon)[#image("/assets/figures/day01/mml_triangle_ineq.png", width: 92%, height: 82%, fit: "contain")]

== 3.3  Inner Product & Angles

- Inner product $chevron.l x, y chevron.r = x^T y$ (standard dot product)
- Cauchy–Schwarz: $|chevron.l x,y chevron.r| <= ||x||_2 ||y||_2$ keeps $cos theta in [-1,1]$
- Angle: $cos theta = chevron.l x,y chevron.r / (||x||_2 ||y||_2)$ — this is cosine similarity
- Example: $(1,0)$ and $(1,1)$ give $cos theta = 1/sqrt(2)$, so $theta = 45 degree$
- Orthogonal vectors: $chevron.l x,y chevron.r = 0$ (no shared direction)

== 3.3  Inner Product & Angles

#align(center + horizon)[#image("/assets/figures/day01/mml_angle.png", width: 92%, height: 82%, fit: "contain")]

== 3.4  Projections

- Idea: keep the part of $y$ along $x$, discard the orthogonal remainder
- Derivation: residual $y - hat(y) perp x$ forces $hat(y) = (x^T y / x^T x) x$
- Example: project $(2,3)$ onto $(1,0)$ gives $(2,0)$; residual $(0,3) perp (1,0)$
- Projection matrix onto column space of $A$: $P = A (A^T A)^(-1) A^T$
- Least squares: minimize $||A w - y||_2^2$ — project $y$ onto span(columns of $A$)
- Normal equations: $A^T A w = A^T y$ (residual $perp$ every column)

== 3.4  Projections

#align(center + horizon)[#image("/assets/figures/day01/mml_projection.png", width: 92%, height: 82%, fit: "contain")]

= 4 · Vector Calculus & Decompositions

== 4.1  Gradients & Jacobians

- Scalar $f(x)$: gradient $nabla f(x) in RR^n$ — direction of steepest ascent
- Vector $f(x) in RR^m$: Jacobian $J in RR^(m times n)$, $J_(i j) = partial f_i / partial x_j$
- Matrix-valued functions: treat as stacked scalar functions
- Chain rule: $nabla_x (g compose f) = J_f^T nabla_g$ (reverse-mode building block)

== 4.1  Gradients & Jacobians

#align(center + horizon)[#image("/assets/figures/day01/mml_gradient.png", width: 92%, height: 82%, fit: "contain")]

== 4.2  Useful Matrix Derivatives

- $nabla_x (a^T x) = a$; $nabla_x (x^T A x) = (A + A^T) x$
- $nabla_A ||A x - y||_2^2 = 2 (A x - y) x^T$
- Hessian $H = nabla^2 f$ — second-order curvature for Newton methods
- These identities appear constantly in backprop and closed-form solutions

== 4.2  Useful Matrix Derivatives

#align(center + horizon)[#image("/assets/figures/day01/mml_taylor.png", width: 92%, height: 82%, fit: "contain")]

== 4.3  Eigendecomposition

- $A v = lambda v$ — eigenvector direction preserved, scaled by $lambda$
- Symmetric $A$: real eigenvalues, orthogonal eigenvectors
- Spectral theorem: $A = Q Lambda Q^T$ — principal axes of quadratic forms
- Condition number from eigenvalue spread — numerical stability

== 4.3  Eigendecomposition

#align(center + horizon)[#image("/assets/figures/day01/mml_eigen.png", width: 92%, height: 82%, fit: "contain")]

== 4.4  SVD & PCA Preview

- SVD: $A = U Sigma V^T$ — always exists; singular values on diagonal of $Sigma$
- Low-rank approximation: keep top-$k$ singular values (Eckart–Young)
- PCA: eigenvectors of covariance $C = (1/n) X^T X$
- Connects linear algebra geometry to data compression and denoising

== 4.4  SVD & PCA Preview

#align(center + horizon)[#image("/assets/figures/day01/mml_svd.png", width: 92%, height: 82%, fit: "contain")]

= 5 · Integration & Differentiation

== 5.1  There and Back Again

- NeurIPS 2020 tutorial: integration *and* differentiation for ML
- Integration: expectations, marginalization, Bayesian evidence
- Differentiation: gradients for optimization, sensitivity analysis
- Map metaphor: numerical integration swamps, Monte Carlo heights, Backprop Bay…

== 5.1  There and Back Again

#align(center + horizon)[#image("/assets/figures/day01/slopes_map.jpg", width: 92%, height: 82%, fit: "contain")]

== 5.2  Integration — Numerical & MC

- Deterministic quadrature: trapezoid, Simpson, Gauss–Hermite
- Curse of dimensionality motivates Monte Carlo: $EE[f(X)] approx (1/N) sum f(x_i)$
- MC error $O(1/sqrt(N))$ regardless of dimension (with i.i.d. samples)
- Importance sampling & variational methods when $f$ is hard to sample

== 5.2  Integration — Numerical & MC

#align(center + horizon)[#image("/assets/figures/day01/integ_unscented.png", width: 92%, height: 82%, fit: "contain")]

== 5.3  Differentiation — Autodiff & Backprop

- Forward mode: propagate directional derivatives ($O(n)$ for $n$ inputs)
- Reverse mode (backprop): one pass from scalar loss ($O("ops")$ for all params)
- Computational graph stores local Jacobians; chain rule multiplies along paths
- Frameworks implement vector–Jacobian products (VJPs), not full Jacobians

== 5.3  Differentiation — Autodiff & Backprop

#align(center + horizon)[#image("/assets/figures/day01/mml_forward_pass.png", width: 92%, height: 82%, fit: "contain")]

== 5.4  Implicit Function & Adjoints

- Implicit Function Theorem: differentiate through $F(x, theta) = 0$ without explicit solve
- Method of adjoints: efficient gradients for ODE/PDE-constrained objectives
- Lagrange multipliers: constrained optimization via augmented Lagrangian
- Stochastic gradient estimators: REINFORCE, reparameterization trick (Day 6–7)

== 5.4  Implicit Function & Adjoints

#align(center + horizon)[#image("/assets/figures/day01/integ_samples.png", width: 92%, height: 82%, fit: "contain")]

= 6 · Probability & Distributions

== 6.1  Basics

- PMF (discrete) / PDF (continuous); CDF $F(x) = P(X <= x)$
- Expectation $EE[X]$, variance $"Var"(X) = EE[(X-EE[X])^2]$
- Joint, marginal, conditional; independence $p(x,y) = p(x)p(y)$
- Bayes: $p(theta|D) prop p(D|theta) p(theta)$

== 6.1  Basics

#align(center + horizon)[#image("/assets/figures/day01/mml_distributions.png", width: 92%, height: 82%, fit: "contain")]

== 6.2  The Gaussian

- $N(x; mu, Sigma) = (2 pi)^(-d/2) |Sigma|^(-1/2) exp(-1/2 (x-mu)^T Sigma^(-1)(x-mu))$
- Affine transform: if $x tilde N(mu, Sigma)$ then $A x + b tilde N(A mu + b, A Sigma A^T)$
- Marginals and conditionals of Gaussians remain Gaussian
- Central role: CLT, linear-Gaussian models, diffusion noise

== 6.2  The Gaussian

#align(center + horizon)[#image("/assets/figures/day01/mml_gaussian.png", width: 92%, height: 82%, fit: "contain")]

== 6.3  Conjugacy & Change of Variables

- Conjugate prior: posterior same family as prior (e.g. Beta–Bernoulli, Normal–Normal)
- Change of variables: $p_Y(y) = p_X(g^(-1)(y)) |det J_(g^(-1))(y)|$
- Inverse transform sampling: $x = F^(-1)(u)$ for $u tilde "Uniform"(0,1)$
- Normalizing flows stack invertible maps with tractable Jacobians

== 6.3  Conjugacy & Change of Variables

#align(center + horizon)[#image("/assets/figures/day01/mml_conjugate.png", width: 92%, height: 82%, fit: "contain")]

== 6.4  Jensen & KL Divergence

- Jensen: $f(EE[X]) <= EE[f(X)]$ for convex $f$ (evidence lower bounds)
- KL divergence: $D_"KL"(q||p) = EE_q[log q/p] >= 0$; not symmetric
- Cross-entropy $H(p,q) = -EE_p[log q]$ — classification loss
- Gaussian KL has closed form — used in VAEs and diffusion training

= 7 · Differential Equations

== 7.1  ODE Intuition

- ODE: $dot(x)(t) = v(x(t), t)$ — rate of change depends on state
- Trajectory view: a curve $x(t)$ through state space
- Vector field view: $v(x,t)$ assigns a velocity at each point
- Same equation, two perspectives — flows vs integral curves

== 7.1  ODE Intuition

#align(center + horizon)[#image("/assets/figures/day01/ode_vectorfield.png", width: 92%, height: 82%, fit: "contain")]

== 7.2  Solving ODEs

- Linear ODE: exponential integrator / integrating factor
- Example: $dot(x) = a x$ gives $x(t) = e^(a t) x(0)$
- Existence & uniqueness when $v$ is Lipschitz
- Autonomous vs time-varying; stability from eigenvalues of $partial v / partial x$

== 7.3  Numerical ODE Solvers

- Euler: $x_(n+1) = x_n + h v(x_n, t_n)$ — first order, $O(h)$ error
- Heun (RK2): predictor–corrector — $O(h^2)$ local error
- Runge–Kutta (RK4): weighted average of slopes — workhorse $O(h^4)$
- Adaptive step size balances accuracy vs cost (used in neural ODEs, diffusion samplers)

== 7.4  SDEs & Numerical Solvers

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
