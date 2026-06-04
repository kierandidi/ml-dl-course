#import "../lib.typ": *

#show: course-theme.with(title: [Math Foundations], subtitle: [Day 1 | Aug 2026])

= Day 1: Math Foundations

== Welcome

- *Math Foundations* — Linear algebra, calculus, probability, optimization
- 3 hours lecture + practical
- Slides, notes, and code on the course site

== Outline

- Linear Algebra
- Calculus & Optimization
- Probability
- Putting It Together

= Linear Algebra

== Vectors and Matrices

- Scalars, vectors, matrices — shapes and notation
- Matrix multiply: $(A B)_(i j) = sum_k A_(i k) B_(k j)$
- Transpose, inverse, rank, and conditioning
- Geometric view: linear maps rotate, scale, shear

== L1 introduct 00

#align(center)[#image("/assets/figures/day01/L1_introduct_00.png", width: 92%)]

#text(size: 14pt, fill: gray)[Linear Algebra — Vectors and Matrices (source: course materials)]

== Eigenvalues & SVD

- Eigen-decomposition: $A v = lambda v$
- Symmetric PSD matrices and quadratic forms
- SVD: $A = U Sigma V^T$ — low-rank approximations
- Principal components as top eigenvectors of covariance

== L1 introduct 03

#align(center)[#image("/assets/figures/day01/L1_introduct_03.png", width: 92%)]

#text(size: 14pt, fill: gray)[Linear Algebra — Eigenvalues & SVD (source: course materials)]

== Norms and Projections

- $L_2$ norm: $||x||_2 = sqrt(sum_i x_i^2)$
- Orthogonal projection onto a subspace
- Least squares as minimizing $||A w - y||_2^2$
- Normal equations: $A^T A w = A^T y$

== L1 introduct 06

#align(center)[#image("/assets/figures/day01/L1_introduct_06.png", width: 92%)]

#text(size: 14pt, fill: gray)[Linear Algebra — Norms and Projections (source: course materials)]

== Tensors in ML

- Batch dimension, channels, spatial axes
- Einstein summation and broadcasting rules
- Gradients w.r.t. matrix variables

== L2 descripti 01

#align(center)[#image("/assets/figures/day01/L2_descripti_01.png", width: 92%)]

#text(size: 14pt, fill: gray)[Linear Algebra — Tensors in ML (source: course materials)]

= Calculus & Optimization

== Derivatives & Gradients

- Partial derivative: $partial f / partial x_i$
- Gradient $nabla f(x)$ points uphill; $-nabla f$ is descent
- Chain rule for composed functions
- Jacobian $J_(i j) = partial f_i / partial x_j$

== L2 descripti 03

#align(center)[#image("/assets/figures/day01/L2_descripti_03.png", width: 92%)]

#text(size: 14pt, fill: gray)[Calculus & Optimization — Derivatives & Gradients (source: course materials)]

== Taylor Approximations

- First-order: $f(x + Delta) approx f(x) + nabla f(x)^T Delta$
- Second-order curvature via Hessian $H$
- Newton step: $Delta = -H^(-1) nabla f$

== L2 descripti 04

#align(center)[#image("/assets/figures/day01/L2_descripti_04.png", width: 92%)]

#text(size: 14pt, fill: gray)[Calculus & Optimization — Taylor Approximations (source: course materials)]

== Convexity

- Convex set: line segment stays inside
- Convex function: $f(t x + (1-t) y) <= t f(x) + (1-t) f(y)$
- Local minima are global for convex problems
- Examples: linear regression, logistic loss (in w)

== L2 descripti 05

#align(center)[#image("/assets/figures/day01/L2_descripti_05.png", width: 92%)]

#text(size: 14pt, fill: gray)[Calculus & Optimization — Convexity (source: course materials)]

== Gradient Descent

- Update: $w_(t+1) = w_t - eta nabla L(w_t)$
- Learning rate $eta$: too large diverges, too small is slow
- Stochastic GD: noisy but cheap per step
- Momentum and adaptive methods preview (Day 3)

== L2 descripti 06

#align(center)[#image("/assets/figures/day01/L2_descripti_06.png", width: 92%)]

#text(size: 14pt, fill: gray)[Calculus & Optimization — Gradient Descent (source: course materials)]

= Probability

== Random Variables

- PMF / PDF, CDF, expectation $EE[X]$
- Variance: $"Var"(X) = EE[(X - EE[X])^2]$
- Common distributions: Gaussian, Bernoulli, categorical

== L2 descripti 08

#align(center)[#image("/assets/figures/day01/L2_descripti_08.png", width: 92%)]

#text(size: 14pt, fill: gray)[Probability — Random Variables (source: course materials)]

== Joint & Conditional

- Joint $p(x, y)$, marginal $p(x) = sum_y p(x, y)$
- Conditional: $p(y|x) = p(x,y) / p(x)$
- Independence: $p(x,y) = p(x) p(y)$

== L2 descripti 09

#align(center)[#image("/assets/figures/day01/L2_descripti_09.png", width: 92%)]

#text(size: 14pt, fill: gray)[Probability — Joint & Conditional (source: course materials)]

== Bayes' Rule

- $p(theta|D) prop p(D|theta) p(theta)$
- Prior, likelihood, posterior
- MAP vs full Bayesian inference

== L2 descripti 10

#align(center)[#image("/assets/figures/day01/L2_descripti_10.png", width: 92%)]

#text(size: 14pt, fill: gray)[Probability — Bayes' Rule (source: course materials)]

== Information Theory Preview

- Entropy: $H(p) = -sum_x p(x) log p(x)$
- Cross-entropy and KL divergence (Day 6)
- Maximum likelihood = minimize cross-entropy

== pptx0 00

#align(center)[#image("/assets/figures/day01/pptx0_00.png", width: 92%)]

#text(size: 14pt, fill: gray)[Probability — Information Theory Preview (source: course materials)]

= Putting It Together

== Loss as Expected Risk

- Empirical risk: $hat(R)(w) = (1/n) sum_i ell(y_i, f_w(x_i))$
- Population risk: $R(w) = EE[ell(Y, f_w(X))]$
- Bias–variance tradeoff (Day 2)

== Worked Example: Linear Regression

- Model: $hat(y) = w^T x + b$
- MSE loss and closed-form solution
- Connection to projection and Gauss–Markov

== Checklist for the Course

- Comfort with matrix calculus notation
- Read gradients off simple computational graphs
- Interpret probabilities as degrees of belief

== Summary

- Day 1: *Math Foundations*
- Linear algebra, calculus, probability, optimization
- Questions welcome — practical follows

== Questions?

#align(center + horizon)[
  #text(size: 44pt, weight: "bold", fill: course-primary)[Questions?]

  #v(1em)
  Practical session follows — see course site.
]
