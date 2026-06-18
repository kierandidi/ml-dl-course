---
layout: post
title: Day 1 - Math Foundations
image: /assets/img/sampling_space.png
accent_image: 
  background: url('/assets/img/sampling_space.png') center/cover
  overlay: false
accent_color: '#ccc'
theme_color: '#ccc'
description: >
  Linear algebra, analytic geometry, vector calculus, integration & differentiation, probability, and differential equations — MML Part I foundations.
invert_sidebar: true
---

# Day 1 - Math Foundations

### Optional reading for this lesson
- [Deisenroth, Faisal & Ong — *Mathematics for Machine Learning*](https://mml-book.com), Ch. 2–6
- [Deisenroth & Ong — *There and Back Again: A Tale of Slopes and Expectations*](https://mml-book.github.io/slopes-expectations.html) (NeurIPS 2020 tutorial)
- [Modern Integration Methods in ML](https://mml-book.github.io/book/additional_chapters/integration-methods.pdf) (MML supplementary chapter)
- [Diffusion Book — Appendix A: Crash Course on Differential Equations](https://arxiv.org/abs/2510.21890) (pp. 399 ff.)

### [Slides](/assets/slides/day01.pdf)

### [Practical](/projects/day01-practical/)

Machine learning rests on four mathematical pillars laid out in Part I of *Mathematics for Machine Learning* (MML): we represent data as vectors and matrices (linear algebra); measure similarity with norms and inner products (analytic geometry); exploit matrix structure and gradients (decompositions and vector calculus); and quantify uncertainty with probability. Today we also cover integration and differentiation as complementary tools for expectations and optimization, and finish with an ODE/SDE crash course that foreshadows diffusion models in Week 2.

* toc
{:toc}

## 1. Linear Algebra

### 1.1 From linear equations to matrices

> A system of $$m$$ linear equations in $$n$$ unknowns can be written as $$\mathbf{A}\mathbf{x} = \mathbf{b}$$ with $$\mathbf{A} \in \mathbb{R}^{m \times n}$$, $$\mathbf{x} \in \mathbb{R}^n$$, $$\mathbf{b} \in \mathbb{R}^m$$. Each row of $$\mathbf{A}$$ defines one hyperplane; solutions lie at their intersection.
{:.lead}

Consider two equations in two unknowns:

$$\begin{aligned} a_{11} x_1 + a_{12} x_2 &= b_1 \\ a_{21} x_1 + a_{22} x_2 &= b_2 \end{aligned}$$

Stacking coefficients gives $$\mathbf{A} = \begin{pmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{pmatrix}$$, so one matrix–vector multiply encodes the entire system. This is the bridge from high-school algebra to the language of machine learning, where a dataset of $$n$$ examples each with $$d$$ features is stored as $$\mathbf{X} \in \mathbb{R}^{n \times d}$$.

![Linear systems and matrix view (MML Fig 2.3)](/assets/figures/day01/mml_linear_system.png)

A matrix is not merely a table of numbers — it is a **linear map** $$f(\mathbf{x}) = \mathbf{A}\mathbf{x}$$. The *column picture* writes $$\mathbf{b} = x_1 \mathbf{a}_1 + x_2 \mathbf{a}_2 + \cdots$$: $$\mathbf{b}$$ must lie in the span of the columns. The *row picture* views each equation as a hyperplane.

### 1.2 Vector spaces, basis, and rank

> A **vector space** $$V$$ is closed under addition and scalar multiplication. A set $$\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$$ is a **basis** if it is linearly independent and spans $$V$$. The **rank** of $$\mathbf{A}$$ is the dimension of its column space.
{:.lead}

Key subspaces associated with $$\mathbf{A} \in \mathbb{R}^{m \times n}$$:

- **Column space** $$\mathcal{C}(\mathbf{A}) \subseteq \mathbb{R}^m$$: all reachable outputs $$\mathbf{A}\mathbf{x}$$.
- **Null space** $$\mathcal{N}(\mathbf{A})$$: all $$\mathbf{x}$$ with $$\mathbf{A}\mathbf{x} = \mathbf{0}$$.
- **Row space** and **left null space** (orthogonal complements in the appropriate spaces).

The rank–nullity theorem: $$\mathrm{rank}(\mathbf{A}) + \dim \mathcal{N}(\mathbf{A}) = n$$.

![Vector subspace (MML Fig 2.6)](/assets/figures/day01/mml_subspace.png)

In ML, features live in a high-dimensional space; learning often finds a low-dimensional subspace (PCA, autoencoders) or selects a sparse subset of coordinates (Lasso).

## 2. Analytic Geometry

### 2.1 Norms and inner products

> An **inner product** $$\langle \mathbf{x}, \mathbf{y} \rangle$$ satisfies symmetry, linearity, and positive definiteness. The induced **norm** is $$\|\mathbf{x}\| = \sqrt{\langle \mathbf{x}, \mathbf{x} \rangle}$$. For the standard dot product, $$\|\mathbf{x}\|_2 = \sqrt{\sum_i x_i^2}$$.
{:.lead}

Common norms in machine learning:

$$\|\mathbf{x}\|_1 = \sum_i |x_i|, \qquad \|\mathbf{x}\|_2 = \sqrt{\sum_i x_i^2}, \qquad \|\mathbf{x}\|_\infty = \max_i |x_i|.$$

The **Cauchy–Schwarz inequality** $$|\langle \mathbf{x}, \mathbf{y} \rangle| \leq \|\mathbf{x}\|_2 \|\mathbf{y}\|_2$$ lets us define angles between vectors. Two vectors are **orthogonal** when $$\langle \mathbf{x}, \mathbf{y} \rangle = 0$$.

![Angle between vectors (MML Fig 3.6)](/assets/figures/day01/mml_angle.png)

Why this matters: if two input vectors are close under a chosen norm, we want our predictor to produce similar outputs — a geometric inductive bias.

### 2.2 Projections and least squares

> The **orthogonal projection** of $$\mathbf{y}$$ onto the line spanned by $$\mathbf{x}$$ is $$\hat{\mathbf{y}} = \frac{\mathbf{x}^\top \mathbf{y}}{\mathbf{x}^\top \mathbf{x}} \mathbf{x}$$. More generally, projecting onto $$\mathcal{C}(\mathbf{A})$$ uses $$\mathbf{P} = \mathbf{A}(\mathbf{A}^\top \mathbf{A})^{-1}\mathbf{A}^\top$$.
{:.lead}

**Ordinary least squares** minimizes $$\|\mathbf{A}\mathbf{w} - \mathbf{y}\|_2^2$$. Geometrically, $$\hat{\mathbf{y}} = \mathbf{A}\hat{\mathbf{w}}$$ is the projection of $$\mathbf{y}$$ onto the column space of $$\mathbf{A}$$. Setting the gradient to zero yields the **normal equations**:

$$\mathbf{A}^\top \mathbf{A} \hat{\mathbf{w}} = \mathbf{A}^\top \mathbf{y}.$$

![Projection onto a subspace (MML Fig 3.11)](/assets/figures/day01/mml_projection.png)

This connects linear algebra (Day 1) directly to regression (Day 2): fitting a linear model is projecting labels onto the span of features.

## 3. Vector Calculus and Matrix Decompositions

### 3.1 Gradients, Jacobians, and the chain rule

> For scalar $$f: \mathbb{R}^n \to \mathbb{R}$$, the **gradient** $$\nabla f(\mathbf{x}) \in \mathbb{R}^n$$ points in the direction of steepest ascent. For $$\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$$, the **Jacobian** $$\mathbf{J} \in \mathbb{R}^{m \times n}$$ has entries $$J_{ij} = \partial f_i / \partial x_j$$.
{:.lead}

Useful identities (memorize these):

$$\nabla_{\mathbf{x}} (\mathbf{a}^\top \mathbf{x}) = \mathbf{a}, \qquad \nabla_{\mathbf{x}} (\mathbf{x}^\top \mathbf{A} \mathbf{x}) = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}.$$

For a quadratic loss $$L(\mathbf{w}) = \|\mathbf{X}\mathbf{w} - \mathbf{y}\|_2^2$$,

$$\nabla_{\mathbf{w}} L = 2\mathbf{X}^\top(\mathbf{X}\mathbf{w} - \mathbf{y}).$$

![Gradient as the slope of a secant (MML Fig 5.3)](/assets/figures/day01/mml_gradient.png)

The multivariate **chain rule** propagates sensitivities through composed functions — the mathematical content of backpropagation (Section 4).

### 3.2 Eigendecomposition and SVD

> If $$\mathbf{A}\mathbf{v} = \lambda \mathbf{v}$$, then $$\mathbf{v}$$ is an **eigenvector** with **eigenvalue** $$\lambda$$. For symmetric $$\mathbf{A}$$, $$\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top$$. The **SVD** is $$\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$$ — always exists.
{:.lead}

The SVD reveals the action of $$\mathbf{A}$$ as rotate–scale–rotate. Truncating to the top-$$k$$ singular values gives the best rank-$$k$$ approximation (Eckart–Young).

**PCA** finds orthogonal directions of maximum variance: eigenvectors of the covariance matrix $$\mathbf{C} = \frac{1}{n}\mathbf{X}^\top \mathbf{X}$$ (after centering).

![SVD geometry (MML Fig 4.9)](/assets/figures/day01/mml_svd.png)

Eigenvalues of the Hessian at a critical point classify it as minimum, maximum, or saddle — relevant for understanding neural network loss landscapes.

## 4. Integration and Differentiation

### 4.1 Integration: expectations and numerical methods

> Many ML quantities are **expectations** $$\mathbb{E}[f(X)] = \int f(x) p(x)\,dx$$. When the integral is intractable we use **numerical quadrature** (low dimension) or **Monte Carlo** (high dimension): $$\mathbb{E}[f(X)] \approx \frac{1}{N}\sum_{i=1}^N f(\mathbf{x}^{(i)}).$$
{:.lead}

The NeurIPS 2020 tutorial [*There and Back Again: A Tale of Slopes and Expectations*](https://mml-book.github.io/slopes-expectations.html) treats integration and differentiation as two directions on the same map — expectations require integration; learning requires differentiation.

![Slopes and expectations map](/assets/figures/day01/slopes_map.jpg)

**Deterministic methods** (trapezoidal rule, Simpson's rule, Gauss–Hermite quadrature) excel in low dimensions. **Monte Carlo** error scales as $$O(1/\sqrt{N})$$ independently of dimension — crucial for Bayesian marginalization and variational objectives.

![Unscented transform / sigma points (Modern Integration Methods, Fig 8)](/assets/figures/day01/integ_unscented.png)

See also the MML supplementary chapter [*Modern Integration Methods in ML*](https://mml-book.github.io/book/additional_chapters/integration-methods.pdf).

### 4.2 Differentiation: autodiff, adjoints, and gradient estimators

> **Reverse-mode automatic differentiation** computes $$\nabla_{\boldsymbol{\theta}} L$$ for a scalar loss $$L$$ in one backward pass through the computational graph — cost $$O(\text{ops})$$, not $$O(|\boldsymbol{\theta}|)$$ times forward cost.
{:.lead}

**Forward mode** propagates directional derivatives; **reverse mode** (backprop) is preferred when there is one scalar output and many parameters.

When outputs are defined implicitly by $$F(\mathbf{x}, \boldsymbol{\theta}) = \mathbf{0}$$, the **implicit function theorem** gives

$$\frac{\partial \mathbf{x}}{\partial \boldsymbol{\theta}} = -\left(\frac{\partial F}{\partial \mathbf{x}}\right)^{-1} \frac{\partial F}{\partial \boldsymbol{\theta}}.$$

The **method of adjoints** and **Lagrange multipliers** extend this to ODE-constrained and constrained optimization problems.

**Stochastic gradient estimators** (REINFORCE score-function estimator, reparameterization $$\nabla \mathbb{E}[f(\mathbf{z})]$$ via $$\mathbf{z} = g(\boldsymbol{\epsilon}, \boldsymbol{\theta})$$) let us differentiate through expectations — central to VAEs and policy gradients.

![Monte Carlo samples across a sequence of distributions (Modern Integration Methods, Fig 6)](/assets/figures/day01/integ_samples.png)

## 5. Probability and Distributions

### 5.1 Basics and the Gaussian

> A continuous random vector $$\mathbf{x}$$ with density $$p$$ satisfies $$\mathbb{E}[\mathbf{x}] = \int \mathbf{x}\, p(\mathbf{x})\,d\mathbf{x}$$. The multivariate Gaussian $$\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$$ has density $$p(\mathbf{x}) \propto \exp\big(-\tfrac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\big)$$.
{:.lead}

Key Gaussian closure properties:

- Affine transform: $$\mathbf{A}\mathbf{x} + \mathbf{b} \sim \mathcal{N}(\mathbf{A}\boldsymbol{\mu} + \mathbf{b}, \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^\top)$$.
- Marginals and conditionals of joint Gaussians are Gaussian.
- Sum of independent Gaussians is Gaussian.

![Gaussian distribution (MML Fig 6.7)](/assets/figures/day01/mml_gaussian.png)

The Gaussian is the maximum-entropy distribution with fixed mean and covariance — and the limiting distribution of sums (CLT).

### 5.2 Conjugacy, change of variables, Jensen, and KL

> A prior is **conjugate** to a likelihood if the posterior belongs to the same parametric family. Under a smooth bijection $$\mathbf{y} = g(\mathbf{x})$$, $$p_Y(\mathbf{y}) = p_X(g^{-1}(\mathbf{y}))\,|\det \mathbf{J}_{g^{-1}}(\mathbf{y})|$$. **Jensen's inequality**: $$f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]$$ for convex $$f$$.
{:.lead}

**Conjugate pairs** (Beta–Bernoulli, Normal–Normal, Dirichlet–Multinomial) give closed-form Bayesian updates — useful for pedagogy and some models.

**Inverse transform sampling**: if $$U \sim \mathrm{Uniform}(0,1)$$, then $$X = F^{-1}(U)$$ has CDF $$F$$. Normalizing flows compose invertible maps with tractable Jacobians.

**KL divergence** $$D_{\mathrm{KL}}(q\|p) = \mathbb{E}_{\mathbf{x}\sim q}[\log q(\mathbf{x}) - \log p(\mathbf{x})] \geq 0$$ measures how many extra nats are needed to encode samples from $$q$$ using a code optimized for $$p$$. It is not symmetric.

Jensen's inequality underlies the **evidence lower bound (ELBO)** in variational inference:

$$\log p(\mathbf{x}) \geq \mathbb{E}_{q(\mathbf{z})}[\log p(\mathbf{x}|\mathbf{z})] - D_{\mathrm{KL}}(q(\mathbf{z})\|p(\mathbf{z})).$$

![Conjugate prior example (MML Fig 6.11)](/assets/figures/day01/mml_conjugate.png)

## 6. Differential Equations (ODE & SDE Crash Course)

### 6.1 ODEs: vector fields, trajectories, and solvers

> An **ordinary differential equation** specifies how a state evolves: $$\dot{\mathbf{x}}(t) = \mathbf{v}(\mathbf{x}(t), t).$$ The **vector field** $$\mathbf{v}$$ assigns a velocity at each point; **trajectories** are integral curves following that field.
{:.lead}

Two equivalent views:

1. **Trajectory**: a curve $$\mathbf{x}(t)$$ through state space.
2. **Vector field**: an arrow $$\mathbf{v}(\mathbf{x}, t)$$ at every point.

For linear $$\dot{x} = ax$$, the solution is $$x(t) = e^{at} x(0)$$ — the **exponential integrator** idea generalizes to matrix systems $$\dot{\mathbf{x}} = \mathbf{A}\mathbf{x}$$.

**Numerical solvers** discretize time with step $$h$$:

| Method | Update | Local error |
|--------|--------|-------------|
| Euler | $$\mathbf{x}_{n+1} = \mathbf{x}_n + h\,\mathbf{v}(\mathbf{x}_n, t_n)$$ | $$O(h)$$ |
| Heun (RK2) | predictor–corrector average | $$O(h^2)$$ |
| RK4 | four weighted slope evaluations | $$O(h^4)$$ |

![ODE Figure A.1 — left: step-by-step solver updates; right: exact trajectories flowing along the velocity field](/assets/figures/day01/ode_vectorfield.png)

Source: *Diffusion Book* Appendix A (crash course on differential equations), from p. 399.

### 6.2 SDEs and numerical simulation

> A **stochastic differential equation** adds noise: $$d\mathbf{X}_t = \mathbf{f}(\mathbf{X}_t, t)\,dt + \mathbf{g}(\mathbf{X}_t, t)\,d\mathbf{W}_t,$$ where $$\mathbf{W}_t$$ is Brownian motion. **Itô's lemma** governs calculus with $$dW_t^2 = dt$$.
{:.lead}

SDEs model systems with intrinsic randomness — and the forward process in diffusion models.

**Euler–Maruyama** discretization:

$$\mathbf{X}_{n+1} = \mathbf{X}_n + h\,\mathbf{f}(\mathbf{X}_n, t_n) + \sqrt{h}\,\mathbf{g}(\mathbf{X}_n, t_n)\,\boldsymbol{\xi}_n, \qquad \boldsymbol{\xi}_n \sim \mathcal{N}(\mathbf{0}, \mathbf{I}).$$

Higher-order schemes (Milstein) improve strong convergence when diffusion matters.

In Week 2 we will connect these solvers directly to sampling from diffusion and flow models.

## Checkpoint summary

Before moving to the practical, confirm you can:

- Translate linear systems into matrix form and interpret column/null spaces.
- Compute projections and connect least squares to the normal equations.
- Apply gradient identities and interpret eigendecomposition / SVD geometrically.
- Contrast numerical quadrature with Monte Carlo for expectations.
- Explain reverse-mode AD and when the implicit function theorem applies.
- State Gaussian closure properties, change-of-variables formula, and Jensen's inequality.
- Describe ODE vector-field vs trajectory views and compare Euler, Heun, and RK4.
- Write the Euler–Maruyama update for an SDE.
