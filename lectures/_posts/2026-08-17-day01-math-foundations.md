---
layout: post
title: Day 1 - Math Foundations
image: /assets/img/lessons/day01.png
description: >
  Linear algebra, analytic geometry, vector calculus, integration & differentiation, probability, and differential equations — the mathematical foundations.
invert_sidebar: true
---

# Day 1 - Math Foundations

### Optional reading for this lesson
- [Deisenroth, Faisal & Ong — *Mathematics for Machine Learning*](https://mml-book.com), Ch. 2–6
- [Deisenroth & Ong — *There and Back Again: A Tale of Slopes and Expectations*](https://mml-book.github.io/slopes-expectations.html) (NeurIPS 2020 tutorial)
- [Modern Integration Methods in ML](https://mml-book.github.io/book/additional_chapters/integration-methods.pdf) (MML supplementary chapter)
- [Diffusion Book — Appendix A: Crash Course on Differential Equations](https://arxiv.org/abs/2510.21890) (pp. 399 ff.)

### [Slides](/assets/slides/day01.pdf)

### Exercise

[Download the notebook](/notebooks/practicals/day01.ipynb) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kierandidi/ml-dl-course/blob/main/notebooks/practicals/day01.ipynb)

Machine learning rests on four mathematical pillars: we represent data as vectors and matrices (linear algebra); measure similarity with norms and inner products (analytic geometry); exploit matrix structure and gradients (decompositions and vector calculus); and quantify uncertainty with probability. Today we also cover integration and differentiation as complementary tools for expectations and optimization, and finish with an ODE/SDE crash course that foreshadows diffusion models in Week 2.

* toc
{:toc}

## 1. Linear Algebra

### 1.1 From linear equations to matrices

> A system of $$m$$ linear equations in $$n$$ unknowns can be written as $$\mathbf{A}\mathbf{x} = \mathbf{b}$$ with $$\mathbf{A} \in \mathbb{R}^{m \times n}$$, $$\mathbf{x} \in \mathbb{R}^n$$, $$\mathbf{b} \in \mathbb{R}^m$$. Each row of $$\mathbf{A}$$ defines one hyperplane; solutions lie at their intersection.
{:.lead}

**Why this matters.** Almost everything in machine learning starts here: a dataset of $$n$$ examples with $$d$$ features each is just a matrix $$\mathbf{X} \in \mathbb{R}^{n \times d}$$, a linear model is a matrix–vector product, and "fitting" the model means solving (or approximately solving) a linear system. Getting fluent with the two pictures below pays off for the rest of the course.

Consider two equations in two unknowns:

$$\begin{aligned} a_{11} x_1 + a_{12} x_2 &= b_1 \\ a_{21} x_1 + a_{22} x_2 &= b_2 \end{aligned}$$

Stacking the coefficients into a matrix lets one matrix–vector multiply encode the *entire* system:

$$\underbrace{\begin{pmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{pmatrix}}_{\mathbf{A}} \underbrace{\begin{pmatrix} x_1 \\ x_2 \end{pmatrix}}_{\mathbf{x}} = \underbrace{\begin{pmatrix} b_1 \\ b_2 \end{pmatrix}}_{\mathbf{b}}.$$

![Linear systems and matrix view](/assets/figures/day01/mml_linear_system.png)

A matrix is not merely a table of numbers — it is a **linear map** $$f(\mathbf{x}) = \mathbf{A}\mathbf{x}$$. There are two complementary ways to read $$\mathbf{A}\mathbf{x} = \mathbf{b}$$:

- **Row picture.** Each row $$\mathbf{a}_i^\top \mathbf{x} = b_i$$ is one hyperplane; the solution is where all hyperplanes intersect.
- **Column picture.** Write $$\mathbf{b} = x_1 \mathbf{a}_1 + x_2 \mathbf{a}_2 + \cdots$$ — we are asking *which combination of the columns reproduces* $$\mathbf{b}$$. A solution exists iff $$\mathbf{b}$$ lies in the span of the columns.

**Worked example.** Solve
$$\begin{aligned} 2x_1 + x_2 &= 5 \\ x_1 - x_2 &= 1 \end{aligned}$$
Adding the equations eliminates $$x_2$$: $$3x_1 = 6 \Rightarrow x_1 = 2$$, then $$x_2 = 5 - 2x_1 = 1$$. In the column picture, $$\mathbf{b} = (5,1)^\top = 2\,(2,1)^\top + 1\,(1,-1)^\top$$ — exactly the columns of $$\mathbf{A}$$ weighted by the solution. The same elimination, applied systematically, is **Gaussian elimination** / **LU factorization**, the algorithm libraries use under the hood.

### 1.2 Vector spaces, basis, and rank

> A **vector space** $$V$$ is closed under addition and scalar multiplication. A set $$\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$$ is a **basis** if it is linearly independent and spans $$V$$. The **rank** of $$\mathbf{A}$$ is the dimension of its column space.
{:.lead}

**Why this matters.** "Dimension", "rank" and "null space" tell you *how much independent information* a matrix carries. A rank-deficient feature matrix means redundant features (collinearity), an unstable least-squares fit, and directions the model simply cannot see. PCA, autoencoders and low-rank adapters (LoRA) all exploit the fact that real data lives near a low-dimensional subspace.

A set $$\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$$ is **linearly independent** if $$\sum_i \alpha_i \mathbf{v}_i = \mathbf{0}$$ forces all $$\alpha_i = 0$$. A **basis** is a linearly independent set that spans the space; its size is the **dimension**. The key subspaces of $$\mathbf{A} \in \mathbb{R}^{m \times n}$$ are:

- **Column space** $$\mathcal{C}(\mathbf{A}) \subseteq \mathbb{R}^m$$: all reachable outputs $$\mathbf{A}\mathbf{x}$$. Its dimension is the **rank**.
- **Null space** $$\mathcal{N}(\mathbf{A}) \subseteq \mathbb{R}^n$$: all $$\mathbf{x}$$ with $$\mathbf{A}\mathbf{x} = \mathbf{0}$$ — directions the map collapses.

These are tied together by the **rank–nullity theorem**:

$$\mathrm{rank}(\mathbf{A}) + \dim \mathcal{N}(\mathbf{A}) = n.$$

![Vector subspace](/assets/figures/day01/mml_subspace.png)

**Worked example.** Take $$\mathbf{A} = \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}$$. The second row is twice the first, so there is only **one** independent row: $$\mathrm{rank}(\mathbf{A}) = 1$$. By rank–nullity the null space has dimension $$2 - 1 = 1$$; indeed $$\mathbf{A}(2,-1)^\top = \mathbf{0}$$, so $$\mathcal{N}(\mathbf{A}) = \mathrm{span}\{(2,-1)^\top\}$$. Because $$\mathbf{A}$$ is rank-deficient it is **not invertible** — a warning sign that, as a feature matrix, its columns are collinear.

## 2. Analytic Geometry

### 2.1 Norms and inner products

> An **inner product** $$\langle \mathbf{x}, \mathbf{y} \rangle$$ satisfies symmetry, linearity, and positive definiteness. The induced **norm** is $$\Vert \mathbf{x}\Vert  = \sqrt{\langle \mathbf{x}, \mathbf{x} \rangle}$$. For the standard dot product, $$\Vert \mathbf{x}\Vert _2 = \sqrt{\sum_i x_i^2}$$.
{:.lead}

**Why this matters.** A norm is how we measure "how big" an error or weight vector is, and an inner product is how we measure "how aligned" two vectors are. The loss we minimize, the regularizer we add, and the cosine similarity behind retrieval and embeddings are all built from these two objects. Choosing a different norm literally changes the geometry of learning.

Common norms in machine learning:

$$\Vert \mathbf{x}\Vert _1 = \sum_i \vert x_i\vert , \qquad \Vert \mathbf{x}\Vert _2 = \sqrt{\sum_i x_i^2}, \qquad \Vert \mathbf{x}\Vert _\infty = \max_i \vert x_i\vert .$$

The $$\ell_2$$ norm is rotation-invariant (Euclidean length); the $$\ell_1$$ norm has a "diamond" unit ball whose corners sit on the axes, which is *why* $$\ell_1$$ regularization produces sparse solutions (Day 2). The inner product induces the $$\ell_2$$ norm via $$\Vert \mathbf{x}\Vert _2 = \sqrt{\langle \mathbf{x},\mathbf{x}\rangle}$$.

The **Cauchy–Schwarz inequality** $$\vert \langle \mathbf{x}, \mathbf{y} \rangle\vert  \leq \Vert \mathbf{x}\Vert _2 \Vert \mathbf{y}\Vert _2$$ guarantees the ratio below lands in $$[-1,1]$$, so it can define an **angle**:

$$\cos\theta = \frac{\langle \mathbf{x}, \mathbf{y}\rangle}{\Vert \mathbf{x}\Vert _2\,\Vert \mathbf{y}\Vert _2}, \qquad \text{orthogonal} \iff \langle \mathbf{x}, \mathbf{y}\rangle = 0.$$

![Angle between vectors](/assets/figures/day01/mml_angle.png)

**Worked example.** For $$\mathbf{x} = (1,0)^\top$$ and $$\mathbf{y} = (1,1)^\top$$: $$\langle \mathbf{x},\mathbf{y}\rangle = 1$$, $$\Vert \mathbf{x}\Vert _2 = 1$$, $$\Vert \mathbf{y}\Vert _2 = \sqrt{2}$$, so $$\cos\theta = 1/\sqrt{2}$$ and $$\theta = 45^\circ$$. This is exactly the "cosine similarity" used to compare text/image embeddings — it ignores magnitude and measures direction.

### 2.2 Projections and least squares

> The **orthogonal projection** of $$\mathbf{y}$$ onto the line spanned by $$\mathbf{x}$$ is $$\hat{\mathbf{y}} = \frac{\mathbf{x}^\top \mathbf{y}}{\mathbf{x}^\top \mathbf{x}} \mathbf{x}$$. More generally, projecting onto $$\mathcal{C}(\mathbf{A})$$ uses $$\mathbf{P} = \mathbf{A}(\mathbf{A}^\top \mathbf{A})^{-1}\mathbf{A}^\top$$.
{:.lead}

**Why this matters.** Projection is the geometric heart of regression, PCA, and the "subtract the part you can already explain" trick that appears everywhere (Gram–Schmidt, residuals, Krylov solvers). If you understand one worked projection, ordinary least squares becomes obvious.

**Deriving the projection onto a line.** We want the multiple $$\alpha\mathbf{x}$$ of $$\mathbf{x}$$ closest to $$\mathbf{y}$$. The residual $$\mathbf{y} - \alpha\mathbf{x}$$ must be **orthogonal** to $$\mathbf{x}$$:

$$\langle \mathbf{y} - \alpha\mathbf{x}, \mathbf{x}\rangle = 0 \;\Longrightarrow\; \alpha = \frac{\mathbf{x}^\top \mathbf{y}}{\mathbf{x}^\top \mathbf{x}}, \qquad \hat{\mathbf{y}} = \frac{\mathbf{x}^\top \mathbf{y}}{\mathbf{x}^\top \mathbf{x}}\,\mathbf{x}.$$

**Worked example (compute a projection).** Project $$\mathbf{y} = (2,3)^\top$$ onto $$\mathbf{x} = (1,0)^\top$$:

$$\alpha = \frac{\mathbf{x}^\top\mathbf{y}}{\mathbf{x}^\top\mathbf{x}} = \frac{2}{1} = 2, \qquad \hat{\mathbf{y}} = 2\,(1,0)^\top = (2,0)^\top.$$

The residual $$\mathbf{y} - \hat{\mathbf{y}} = (0,3)^\top$$ is indeed orthogonal to $$\mathbf{x}$$ — we kept the component of $$\mathbf{y}$$ along $$\mathbf{x}$$ and discarded the rest.

![Projection onto a subspace](/assets/figures/day01/mml_projection.png)

**From a line to a subspace.** Projecting onto the column space of $$\mathbf{A}$$ uses $$\mathbf{P} = \mathbf{A}(\mathbf{A}^\top \mathbf{A})^{-1}\mathbf{A}^\top$$. **Ordinary least squares** minimizes $$\Vert \mathbf{A}\mathbf{w} - \mathbf{y}\Vert _2^2$$; the same orthogonality condition ("residual $$\perp$$ every column") gives the **normal equations**

$$\mathbf{A}^\top \mathbf{A} \hat{\mathbf{w}} = \mathbf{A}^\top \mathbf{y}, \qquad \hat{\mathbf{y}} = \mathbf{A}\hat{\mathbf{w}} = \mathbf{P}\mathbf{y}.$$

So fitting a linear model *is* projecting the labels onto the span of the features — the bridge from linear algebra (Day 1) to regression (Day 2).

## 3. Vector Calculus and Matrix Decompositions

### 3.1 Gradients, Jacobians, and the chain rule

> For scalar $$f: \mathbb{R}^n \to \mathbb{R}$$, the **gradient** $$\nabla f(\mathbf{x}) \in \mathbb{R}^n$$ points in the direction of steepest ascent. For $$\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$$, the **Jacobian** $$\mathbf{J} \in \mathbb{R}^{m \times n}$$ has entries $$J_{ij} = \partial f_i / \partial x_j$$.
{:.lead}

**Why this matters.** Training is optimization, and optimization runs on gradients. Every parameter update — SGD, Adam, the backward pass of a transformer — is the multivariate chain rule applied at scale. The handful of identities below let you derive closed-form solutions and sanity-check autodiff.

Useful identities (worth memorizing):

$$\nabla_{\mathbf{x}} (\mathbf{a}^\top \mathbf{x}) = \mathbf{a}, \qquad \nabla_{\mathbf{x}} (\mathbf{x}^\top \mathbf{A} \mathbf{x}) = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}.$$

**Derivation for least squares.** Expand $$L(\mathbf{w}) = \Vert \mathbf{X}\mathbf{w} - \mathbf{y}\Vert _2^2 = \mathbf{w}^\top\mathbf{X}^\top\mathbf{X}\mathbf{w} - 2\mathbf{y}^\top\mathbf{X}\mathbf{w} + \mathbf{y}^\top\mathbf{y}$$. Applying the two identities term by term,

$$\nabla_{\mathbf{w}} L = \underbrace{2\mathbf{X}^\top\mathbf{X}\mathbf{w}}_{\text{from } \mathbf{w}^\top\mathbf{X}^\top\mathbf{X}\mathbf{w}} - \underbrace{2\mathbf{X}^\top\mathbf{y}}_{\text{from } -2\mathbf{y}^\top\mathbf{X}\mathbf{w}} = 2\mathbf{X}^\top(\mathbf{X}\mathbf{w} - \mathbf{y}).$$

Setting this to $$\mathbf{0}$$ recovers the normal equations from the previous section — calculus and geometry agree.

![Gradient as the slope of a secant](/assets/figures/day01/mml_gradient.png)

**Chain rule = backprop.** For $$\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$$ the **Jacobian** $$\mathbf{J}$$ has $$J_{ij} = \partial f_i/\partial x_j$$, and for a composition $$g \circ \mathbf{f}$$,
$$\nabla_{\mathbf{x}} (g\circ \mathbf{f}) = \mathbf{J}_{\mathbf{f}}^\top\, \nabla g.$$
Reverse-mode autodiff just multiplies these Jacobian-transposes from the loss backward — the content of Section 4 and Day 3.

### 3.2 Eigendecomposition and SVD

> If $$\mathbf{A}\mathbf{v} = \lambda \mathbf{v}$$, then $$\mathbf{v}$$ is an **eigenvector** with **eigenvalue** $$\lambda$$. For symmetric $$\mathbf{A}$$, $$\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top$$. The **SVD** is $$\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$$ — always exists.
{:.lead}

**Why this matters.** Decompositions turn an opaque matrix into interpretable pieces: principal axes (PCA), compression (low-rank approximation), conditioning (numerical stability), and the curvature of loss landscapes (Hessian eigenvalues). They are the workhorse behind dimensionality reduction and many "why did training blow up?" diagnoses.

**Eigen-intuition.** $$\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$$ says $$\mathbf{v}$$ is a direction the map only *stretches* (by $$\lambda$$), not rotates. For symmetric $$\mathbf{A}$$ the spectral theorem gives orthonormal eigenvectors, $$\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top$$ — the principal axes of the quadratic form $$\mathbf{x}^\top\mathbf{A}\mathbf{x}$$.

**Worked example.** For $$\mathbf{A} = \begin{pmatrix} 2 & 0 \\ 0 & 3 \end{pmatrix}$$ the eigenpairs are $$(\lambda_1=2,\,\mathbf{e}_1)$$ and $$(\lambda_2=3,\,\mathbf{e}_2)$$: the unit circle maps to an ellipse with semi-axes $$2$$ and $$3$$. The **condition number** $$\kappa = \lambda_{\max}/\lambda_{\min} = 3/2$$ controls how much errors are amplified when solving $$\mathbf{A}\mathbf{x}=\mathbf{b}$$.

The **SVD** $$\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$$ always exists and reads as *rotate ($$\mathbf{V}^\top$$) → scale ($$\boldsymbol{\Sigma}$$) → rotate ($$\mathbf{U}$$)*. Truncating to the top-$$k$$ singular values gives the best rank-$$k$$ approximation (**Eckart–Young**) — the math behind image/embedding compression.

**Why the SVD always exists (constructive proof).** Eigendecomposition needs a square (ideally symmetric) matrix, but the SVD works for *any* $$\mathbf{A}\in\mathbb{R}^{m\times n}$$. The trick is to apply the spectral theorem to the symmetric matrix $$\mathbf{A}^\top\mathbf{A}$$. It is symmetric and **positive semidefinite**, since $$\mathbf{x}^\top(\mathbf{A}^\top\mathbf{A})\mathbf{x} = \Vert \mathbf{A}\mathbf{x}\Vert _2^2 \geq 0$$, so it has orthonormal eigenvectors $$\mathbf{v}_i$$ with **non-negative** eigenvalues $$\lambda_i \geq 0$$:

$$\mathbf{A}^\top\mathbf{A}\,\mathbf{v}_i = \lambda_i\mathbf{v}_i, \qquad \lambda_1 \geq \cdots \geq \lambda_r > 0 = \lambda_{r+1} = \cdots.$$

Define the **singular values** $$\sigma_i = \sqrt{\lambda_i}$$ and, for each nonzero one, the left vector $$\mathbf{u}_i = \tfrac{1}{\sigma_i}\mathbf{A}\mathbf{v}_i$$. These $$\mathbf{u}_i$$ are automatically orthonormal:

$$\mathbf{u}_i^\top\mathbf{u}_j = \frac{1}{\sigma_i\sigma_j}\mathbf{v}_i^\top\underbrace{\mathbf{A}^\top\mathbf{A}}_{\lambda_j\mathbf{v}_j}\mathbf{v}_j = \frac{\lambda_j}{\sigma_i\sigma_j}\,\mathbf{v}_i^\top\mathbf{v}_j = \delta_{ij}.$$

By construction $$\mathbf{A}\mathbf{v}_i = \sigma_i\mathbf{u}_i$$, which stacked over all $$i$$ reads $$\mathbf{A}\mathbf{V} = \mathbf{U}\boldsymbol{\Sigma}$$; right-multiplying by $$\mathbf{V}^\top$$ (orthogonal, so $$\mathbf{V}\mathbf{V}^\top=\mathbf{I}$$) gives $$\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$$. The number of nonzero singular values is the **rank** $$r$$, and the squared singular values are the eigenvalues of $$\mathbf{A}^\top\mathbf{A}$$ — the exact link to PCA below.

![Geometry of the SVD: a unit sphere is rotated, axis-scaled by the singular values, then rotated again into an ellipse.](/assets/figures/day01/mml_svd.png)

**PCA** is SVD applied to centered data: the principal directions are eigenvectors of the covariance $$\mathbf{C} = \frac{1}{n}\mathbf{X}^\top \mathbf{X}$$, and the eigenvalues are the variance captured along each axis. At a critical point of a loss, the sign pattern of the Hessian's eigenvalues classifies it as a minimum, maximum, or **saddle** — the dominant feature of high-dimensional neural loss landscapes.

## 4. Integration and Differentiation

### 4.1 Integration: expectations and numerical methods

> Many ML quantities are **expectations** $$\mathbb{E}[f(X)] = \int f(x) p(x)\,dx$$. When the integral is intractable we use **numerical quadrature** (low dimension) or **Monte Carlo** (high dimension): $$\mathbb{E}[f(X)] \approx \frac{1}{N}\sum_{i=1}^N f(\mathbf{x}^{(i)}).$$
{:.lead}

**Why this matters.** A huge fraction of ML quantities are integrals in disguise: a marginal likelihood $$p(\mathbf{x}) = \int p(\mathbf{x}\mid\mathbf{z})p(\mathbf{z})\,d\mathbf{z}$$, an expected reward, the normalizing constant of a posterior. It helps to see integration and differentiation as two directions on the same map — expectations require integration; learning requires differentiation.

![Slopes and expectations map](/assets/figures/day01/slopes_map.jpg)

**Deterministic quadrature** approximates $$\int_a^b f(x)\,dx$$ by a weighted sum of evaluations. The trapezoidal rule on $$N$$ points has error $$O(N^{-2})$$ in **1D**, but a tensor grid needs $$N^d$$ points in $$d$$ dimensions — the **curse of dimensionality**.

**Monte Carlo** sidesteps this. With $$\mathbf{x}^{(i)} \sim p$$,
$$\mathbb{E}_p[f] \approx \hat\mu_N = \frac{1}{N}\sum_{i=1}^N f(\mathbf{x}^{(i)}), \qquad \mathrm{Var}(\hat\mu_N) = \frac{\mathrm{Var}(f)}{N}.$$
So the standard error is $$O(1/\sqrt{N})$$ **regardless of dimension** — the reason MC dominates high-dimensional Bayesian and variational computation. **Importance sampling** reweights samples from an easier proposal $$q$$: $$\mathbb{E}_p[f] = \mathbb{E}_q[f\,p/q]$$.

![Unscented transform / sigma points](/assets/figures/day01/integ_unscented.png)

### 4.2 Differentiation: autodiff, adjoints, and gradient estimators

> **Reverse-mode automatic differentiation** computes $$\nabla_{\boldsymbol{\theta}} L$$ for a scalar loss $$L$$ in one backward pass through the computational graph — cost $$O(\text{ops})$$, not $$O(\vert \boldsymbol{\theta}\vert )$$ times forward cost.
{:.lead}

**Why this matters.** Autodiff is what makes deep learning practical: you write the forward computation, and the framework returns exact gradients for millions of parameters at the cost of roughly one extra forward pass. Knowing *which* mode to use, and how to differentiate through solvers and samplers, is what separates "it trains" from "it trains efficiently".

**Two modes.** For $$\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$$ built from elementary ops with local Jacobians:

- **Forward mode** pushes a direction forward, costing one pass per *input* — good when $$n \ll m$$.
- **Reverse mode (backprop)** pulls the gradient back from the output, costing one pass per *output* — ideal for ML where $$m=1$$ (a scalar loss) and $$n = \vert \boldsymbol{\theta}\vert $$ is huge.

**Tiny worked example.** Let $$y = \sin(w x)$$ with the graph $$u = wx \to y=\sin u$$. Reverse mode seeds $$\bar y = 1$$, then $$\bar u = \cos u$$, then $$\bar w = \bar u\, x = x\cos(wx)$$ — exactly $$\partial y/\partial w$$, obtained by multiplying local derivatives along the path. Frameworks implement these as **vector–Jacobian products**, never forming full Jacobians.

**Differentiating through a solver.** When the output is defined *implicitly* by $$F(\mathbf{x}, \boldsymbol{\theta}) = \mathbf{0}$$, the **implicit function theorem** avoids unrolling:

$$\frac{\partial \mathbf{x}}{\partial \boldsymbol{\theta}} = -\left(\frac{\partial F}{\partial \mathbf{x}}\right)^{-1} \frac{\partial F}{\partial \boldsymbol{\theta}}.$$

The **method of adjoints** (and Lagrange multipliers) applies the same idea to ODE-/constraint-defined objectives — the backbone of neural ODEs.

**Differentiating through randomness.** To get $$\nabla_{\boldsymbol\theta}\mathbb{E}[f(\mathbf{z})]$$ we use either the **score-function/REINFORCE** estimator $$\mathbb{E}[f(\mathbf{z})\,\nabla_{\boldsymbol\theta}\log p_{\boldsymbol\theta}(\mathbf{z})]$$, or the lower-variance **reparameterization** $$\mathbf{z} = g(\boldsymbol{\epsilon}, \boldsymbol{\theta})$$ with $$\boldsymbol\epsilon$$ noise — central to VAEs (Day 6) and policy gradients.

![Monte Carlo samples across a sequence of distributions](/assets/figures/day01/integ_samples.png)

## 5. Probability and Distributions

### 5.1 Basics and the Gaussian

> A continuous random vector $$\mathbf{x}$$ with density $$p$$ satisfies $$\mathbb{E}[\mathbf{x}] = \int \mathbf{x}\, p(\mathbf{x})\,d\mathbf{x}$$. The multivariate Gaussian $$\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$$ has density $$p(\mathbf{x}) \propto \exp\big(-\tfrac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\big)$$.
{:.lead}

**Why this matters.** The Gaussian is the single most important distribution in ML: it is the noise model in least squares, the prior/posterior in linear-Gaussian models, the latent prior in VAEs, and the noise injected at every step of a diffusion model. Its closure properties make otherwise-intractable computations exact.

A continuous random vector has mean $$\mathbb{E}[\mathbf{x}] = \int \mathbf{x}\,p(\mathbf{x})\,d\mathbf{x}$$ and covariance $$\boldsymbol\Sigma = \mathbb{E}[(\mathbf{x}-\boldsymbol\mu)(\mathbf{x}-\boldsymbol\mu)^\top]$$. The multivariate Gaussian density is

$$\mathcal{N}(\mathbf{x};\boldsymbol\mu,\boldsymbol\Sigma) = (2\pi)^{-d/2}\,\vert \boldsymbol\Sigma\vert ^{-1/2}\exp\!\Big(-\tfrac{1}{2}(\mathbf{x}-\boldsymbol\mu)^\top\boldsymbol\Sigma^{-1}(\mathbf{x}-\boldsymbol\mu)\Big).$$

Key **closure** properties (each keeps you inside the Gaussian family):

- **Affine maps:** $$\mathbf{A}\mathbf{x} + \mathbf{b} \sim \mathcal{N}(\mathbf{A}\boldsymbol{\mu} + \mathbf{b},\, \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^\top)$$.
- **Marginals and conditionals** of a joint Gaussian are Gaussian.
- **Sums** of independent Gaussians are Gaussian.

**Worked example (the diffusion forward step).** If $$\mathbf{x}_0$$ is data and $$\boldsymbol\epsilon \sim \mathcal{N}(\mathbf{0},\mathbf{I})$$, then $$\mathbf{x}_t = \sqrt{\bar\alpha_t}\,\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon$$ is an affine map of a Gaussian, so $$\mathbf{x}_t \mid \mathbf{x}_0 \sim \mathcal{N}(\sqrt{\bar\alpha_t}\,\mathbf{x}_0,\,(1-\bar\alpha_t)\mathbf{I})$$ in closed form — this is exactly the trick that makes diffusion training cheap (Day 6).

![Gaussian distribution](/assets/figures/day01/mml_gaussian.png)

The Gaussian is also the **maximum-entropy** distribution with fixed mean and covariance, and the limiting distribution of normalized sums (**CLT**) — two reasons it shows up "by default".

### 5.2 Conjugacy, change of variables, Jensen, and KL

> A prior is **conjugate** to a likelihood if the posterior belongs to the same parametric family. Under a smooth bijection $$\mathbf{y} = g(\mathbf{x})$$, $$p_Y(\mathbf{y}) = p_X(g^{-1}(\mathbf{y}))\,\vert \det \mathbf{J}_{g^{-1}}(\mathbf{y})\vert $$. **Jensen's inequality**: $$f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]$$ for convex $$f$$.
{:.lead}

**Why this matters.** These four ideas are the toolkit of probabilistic ML: conjugacy gives closed-form Bayesian updates, change-of-variables powers normalizing flows, Jensen's inequality produces the ELBO, and KL divergence is the loss that trains VAEs and diffusion models.

**Conjugate pairs** (Beta–Bernoulli, Normal–Normal, Dirichlet–Multinomial) keep the posterior in the prior's family. For example, with a $$\mathrm{Beta}(a,b)$$ prior and $$k$$ successes in $$n$$ Bernoulli trials, the posterior is simply $$\mathrm{Beta}(a+k,\,b+n-k)$$ — updating beliefs is just adding counts.

**Change of variables.** Under a smooth bijection $$\mathbf{y} = g(\mathbf{x})$$, densities transform by the Jacobian:
$$p_Y(\mathbf{y}) = p_X(g^{-1}(\mathbf{y}))\,\big\vert \det \mathbf{J}_{g^{-1}}(\mathbf{y})\big\vert .$$
The special 1D case gives **inverse transform sampling**: if $$U \sim \mathrm{Uniform}(0,1)$$ then $$X = F^{-1}(U)$$ has CDF $$F$$. Normalizing flows stack many invertible maps with tractable Jacobians (Day 7).

**KL divergence** measures the cost (in nats) of using $$p$$ to encode samples from $$q$$:
$$D_{\mathrm{KL}}(q\,\Vert \,p) = \mathbb{E}_{\mathbf{x}\sim q}\!\big[\log q(\mathbf{x}) - \log p(\mathbf{x})\big] \geq 0,$$
with equality iff $$q=p$$ (a direct consequence of **Jensen's inequality** applied to the convex $$-\log$$). It is **not symmetric**: forward KL is mode-covering, reverse KL is mode-seeking.

**Worked example (Gaussian KL).** For two 1D Gaussians,
$$D_{\mathrm{KL}}\big(\mathcal{N}(\mu_1,\sigma_1^2)\,\Vert \,\mathcal{N}(\mu_2,\sigma_2^2)\big) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1-\mu_2)^2}{2\sigma_2^2} - \frac12.$$
Against a standard normal ($$\mu_2=0,\sigma_2=1$$) this reduces to $$\tfrac12(\sigma_1^2 + \mu_1^2 - 1 - \log\sigma_1^2)$$ — the exact KL term in the VAE objective.

Finally, Jensen's inequality yields the **evidence lower bound (ELBO)**:
$$\log p(\mathbf{x}) \geq \mathbb{E}_{q(\mathbf{z})}[\log p(\mathbf{x}\mid\mathbf{z})] - D_{\mathrm{KL}}(q(\mathbf{z})\,\Vert \,p(\mathbf{z})).$$

![Conjugate prior example](/assets/figures/day01/mml_conjugate.png)

## 6. Differential Equations (ODE & SDE Crash Course)

### 6.1 ODEs: vector fields, trajectories, and solvers

> An **ordinary differential equation** specifies how a state evolves: $$\dot{\mathbf{x}}(t) = \mathbf{v}(\mathbf{x}(t), t).$$ The **vector field** $$\mathbf{v}$$ assigns a velocity at each point; **trajectories** are integral curves following that field.
{:.lead}

**Why this matters.** Continuous-time dynamics are the modern language of generative modeling: neural ODEs, probability-flow ODEs, and the deterministic samplers of diffusion/flow models are all "integrate a learned vector field". The numerical solvers below are *literally* the samplers you will run in Week 2.

Two equivalent views of $$\dot{\mathbf{x}}(t) = \mathbf{v}(\mathbf{x}(t), t)$$:

1. **Trajectory** — a single curve $$\mathbf{x}(t)$$ threading through state space.
2. **Vector field** — an arrow $$\mathbf{v}(\mathbf{x},t)$$ at *every* point; trajectories are its integral curves.

**Exact solution by integrating factor.** For the scalar linear ODE $$\dot x = a x$$, rewrite as $$\dot x - a x = 0$$ and multiply by $$e^{-at}$$:
$$\tfrac{d}{dt}\big(e^{-at}x\big) = e^{-at}(\dot x - a x) = 0 \;\Rightarrow\; e^{-at}x = \text{const} \;\Rightarrow\; x(t) = e^{at}x(0).$$
The same **matrix exponential** $$\mathbf{x}(t) = e^{\mathbf{A}t}\mathbf{x}(0)$$ solves $$\dot{\mathbf{x}} = \mathbf{A}\mathbf{x}$$; the sign of $$a$$ (or the real parts of the eigenvalues of $$\mathbf{A}$$) decides stability. Existence and uniqueness hold whenever $$\mathbf{v}$$ is **Lipschitz** (Picard–Lindelöf).

**Numerical solvers** discretize time with step $$h$$ — trading accuracy for function evaluations:

| Method | Update | Local error |
|--------|--------|-------------|
| Euler | $$\mathbf{x}_{n+1} = \mathbf{x}_n + h\,\mathbf{v}(\mathbf{x}_n, t_n)$$ | $$O(h^2)$$ |
| Heun (RK2) | predictor $$\tilde{\mathbf{x}} = \mathbf{x}_n + h\mathbf{v}_n$$, then average $$\tfrac{h}{2}(\mathbf{v}_n + \mathbf{v}(\tilde{\mathbf{x}}, t_{n+1}))$$ | $$O(h^3)$$ |
| RK4 | weighted average of four slope evaluations | $$O(h^5)$$ |

(The orders above are *local* per-step errors; the *global* errors are one power of $$h$$ lower, i.e. Euler is first-order overall.)

**Worked Euler step.** For $$\dot x = x$$, $$x(0)=1$$, $$h=0.5$$: one Euler step gives $$x_1 = 1 + 0.5\cdot 1 = 1.5$$, versus the exact $$e^{0.5}\approx 1.649$$. Halving $$h$$ roughly halves the error — the hallmark of a first-order method.

![Left: step-by-step solver updates; right: exact trajectories flowing along the velocity field](/assets/figures/day01/ode_vectorfield.png)

### 6.2 SDEs and numerical simulation

> A **stochastic differential equation** adds noise: $$d\mathbf{X}_t = \mathbf{f}(\mathbf{X}_t, t)\,dt + \mathbf{g}(\mathbf{X}_t, t)\,d\mathbf{W}_t,$$ where $$\mathbf{W}_t$$ is Brownian motion. **Itô's lemma** governs calculus with $$dW_t^2 = dt$$.
{:.lead}

**Why this matters.** The *forward* (noising) process of a diffusion model is an SDE, and the *reverse* (generation) process is another SDE driven by the learned score. Understanding the Euler–Maruyama step here is understanding the diffusion sampler you will implement in Week 2.

An SDE adds a noise term to an ODE: $$d\mathbf{X}_t = \underbrace{\mathbf{f}(\mathbf{X}_t,t)\,dt}_{\text{drift}} + \underbrace{\mathbf{g}(\mathbf{X}_t,t)\,d\mathbf{W}_t}_{\text{diffusion}}$$, where $$\mathbf{W}_t$$ is **Brownian motion**: independent Gaussian increments with $$\mathbf{W}_{t+h}-\mathbf{W}_t \sim \mathcal{N}(\mathbf{0},h\mathbf{I})$$.

**Why the $$\sqrt{h}$$ appears.** Because the increment has variance $$h$$, its *standard deviation* scales like $$\sqrt{h}$$, not $$h$$. This is the defining feature of stochastic calculus and the reason **Itô's lemma** keeps the second-order term: $$dW_t^2 = dt$$.

**Euler–Maruyama** discretization (the SDE analogue of Euler):

$$\mathbf{X}_{n+1} = \mathbf{X}_n + h\,\mathbf{f}(\mathbf{X}_n, t_n) + \sqrt{h}\,\mathbf{g}(\mathbf{X}_n, t_n)\,\boldsymbol{\xi}_n, \qquad \boldsymbol{\xi}_n \sim \mathcal{N}(\mathbf{0}, \mathbf{I}).$$

**Worked example (Ornstein–Uhlenbeck).** For $$dX_t = -X_t\,dt + dW_t$$ with $$X_0=0$$, $$h=0.1$$: one step draws $$\xi_0\sim\mathcal{N}(0,1)$$ and sets $$X_1 = 0 + 0.1\cdot(-0) + \sqrt{0.1}\,\xi_0 = 0.316\,\xi_0$$. Repeating, the process relaxes toward its stationary $$\mathcal{N}(0,\tfrac12)$$ — a mean-reverting noise process, exactly the structure of the variance-preserving diffusion forward SDE.

Higher-order schemes (Milstein) improve strong convergence when the diffusion term varies with state. In Week 2 we connect these solvers directly to sampling from diffusion and flow models (Days 6–8).

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
