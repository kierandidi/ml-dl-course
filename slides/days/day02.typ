#import "../lib.typ": *

#show: course-theme.with(title: [Statistical Learning], subtitle: [Day 2 | Aug 2026])

= Day 2: Statistical Learning

== Welcome

- *Statistical Learning* — MML Part II — regression, PCA, density estimation, classification
- 3 hours lecture + practical
- Slides, notes, and code on the course site

== Outline

- When Models Meet Data (MML Ch. 8)
- Linear Regression (MML Ch. 9)
- Dimensionality Reduction / PCA (MML Ch. 10)
- Density Estimation / GMM (MML Ch. 11)
- Classification / SVM (MML Ch. 12)

= 1 · When Models Meet Data (MML Ch. 8)

== 1.1  Machine Learning: A Taxonomy

- Three paradigms: *supervised*, *unsupervised*, *reinforcement* learning
- Supervised: regression (continuous $y$) and classification (discrete labels)
- Unsupervised: clustering, density estimation, dimensionality reduction
- Reinforcement: sequential decisions from reward signals
- Today maps each pillar to one MML chapter (9–12)

== 1.1  Machine Learning: A Taxonomy

#align(center + horizon)[#image("/assets/figures/day02/ml_taxonomy.png", width: 92%, height: 82%, fit: "contain")]

== 1.2  Four Pillars of Part II

- Ch. 9 — *Linear regression*: predict continuous targets from features
- Ch. 10 — *Dimensionality reduction* (PCA): find compact representations
- Ch. 11 — *Density estimation* (GMM): model $p(x)$ and discover structure
- Ch. 12 — *Classification* (SVM): separate classes with margins
- Ch. 8 provides the shared vocabulary: data, models, learning, evaluation

== 1.3  Data, Models, and Learning

- Dataset $cal(D) = {(x_n, y_n)}_(n=1)^N$; features $x_n in RR^D$ as *vectors*
- A *model* is a parameterized family $p(y|x, theta)$ or $f_theta(x)$
- *Learning* = choose $theta$ from data so predictions generalize to new $x$
- Similarity of examples (geometry from Day 1) drives regression & classification
- Goal: low error on *unseen* test points, not just training fit

== 1.3  Data, Models, and Learning

#align(center + horizon)[#image("/assets/figures/day02/mml_toy_regression.png", width: 92%, height: 82%, fit: "contain")]

== 1.4  Empirical Risk Minimization

- True risk: $R(f) = EE[ell(f(x), y)]$ over the data-generating distribution
- Empirical risk: $R_"emp"(f) = (1/N) sum_n ell(f(x_n), y_n)$ — what we can compute
- ERM: $hat(theta) = "arg min"_(theta) R_"emp"(theta)$ within hypothesis class $cal(H)$
- Loss examples: squared error (regression), cross-entropy (classification)
- Finite data $arrow.r$ ERM can overfit; need validation and model selection

== 1.5  Train, Validation, Test & Cross-Validation

- Hold out a *validation* set to tune hyperparameters ($lambda$, degree, $K$)
- Keep a *test* set untouched until final evaluation
- $K$-fold CV: rotate which chunk is validation; average validation error
- $EE_V[R(f,V)] approx (1/K) sum_k R(f^((k)), V^((k)))$
- Reduces variance of performance estimates at cost of $K$ retrains

== 1.5  Train, Validation, Test & Cross-Validation

#align(center + horizon)[#image("/assets/figures/day02/mml_cross_validation.png", width: 92%, height: 82%, fit: "contain")]

== 1.6  Model Selection & Occam's Razor

- More complex $cal(H)$ lowers training error but may raise test error
- Bias–variance trade-off: underfit (high bias) vs overfit (high variance)
- Occam: prefer the simpler model among those with similar validation error
- $"BIC" = -2 ell + p log N$
- Regularization (next chapters) encodes simplicity directly in the objective

= 2 · Linear Regression (MML Ch. 9)

== 2.1  Problem Setup

- Given $(x_n, y_n)$, find $f: RR^D arrow.r RR$ with $y_n approx f(x_n) + epsilon_n$
- Noise $epsilon_n tilde cal(N)(0, sigma^2)$ (i.i.d. Gaussian in MML)
- Training fits $f$; we care about prediction at *new* inputs (generalization)
- Vector data lets us use linear algebra from Day 1

== 2.1  Problem Setup

#align(center + horizon)[#image("/assets/figures/day02/mml_toy_regression.png", width: 92%, height: 82%, fit: "contain")]

== 2.2  Linear in the Parameters

- $y = phi(x)^T theta$ — linear in $theta$, possibly nonlinear in $x$
- Plain line: $phi(x) = (1, x)^T$; polynomial: $phi(x) = (1, x, x^2, dots)^T$
- Design matrix $Phi in RR^(N times M)$ stacks $phi(x_n)^T$ as rows
- Batch prediction: $hat(y) = Phi theta$

== 2.3  Maximum Likelihood = Least Squares

- Gaussian likelihood: $p(y|x, theta) = cal(N)(phi(x)^T theta, sigma^2)$
- Log-likelihood $prop - (1/(2 sigma^2)) sum_n (y_n - phi(x_n)^T theta)^2$
- MLE $equiv$ minimize MSE: $L(theta) = (1/(2N))||Phi theta - y||^2$
- Closed form (when invertible): $theta^* = (Phi^T Phi)^(-1) Phi^T y$
- Geometric view: orthogonal projection of $y$ onto column space of $Phi$

== 2.3  Maximum Likelihood = Least Squares

#align(center + horizon)[#image("/assets/figures/day02/mml_linear_regression.png", width: 92%, height: 82%, fit: "contain")]

== 2.4  Ridge Regression

- When $Phi^T Phi$ is ill-conditioned, OLS has high variance
- Ridge: $L(theta) = (1/(2N))||Phi theta - y||^2 + (lambda/(2N))||theta||^2$
- Closed form: $theta = (Phi^T Phi + lambda I)^(-1) Phi^T y$
- Bayesian view: Gaussian prior on $theta$ $arrow.r$ MAP estimate
- $lambda$ controls bias–variance; tune with cross-validation

== 2.5  Polynomial Regression & Overfitting

- High-degree polynomials fit training data arbitrarily well
- Training error keeps falling; *test* error often rises (overfitting)
- Model selection: pick degree via validation or penalize complexity (ridge)
- Figure 9.6: train vs test error as polynomial degree grows
- Lesson: capacity must match data size and noise level

= 3 · Dimensionality Reduction / PCA (MML Ch. 10)

== 3.1  Why Reduce Dimensionality?

- Curse of dimensionality: data sparse in high $D$; distance metrics degrade
- Goals: compression, denoising, visualization, faster downstream learning
- Unsupervised: only inputs ${x_n}$, no labels
- Linear method PCA; nonlinear extensions (autoencoders) come in Week 2

== 3.1  Why Reduce Dimensionality?

#align(center + horizon)[#image("/assets/figures/day02/mml_poly_overfit.png", width: 92%, height: 82%, fit: "contain")]

== 3.2  PCA: Maximum Variance Perspective

- Find direction $b_1$ (unit vector) maximizing $"Var"(b_1^T X)$
- Second component $b_2$ orthogonal to $b_1$, max remaining variance, etc.
- Principal components = eigenvectors of sample covariance $S$
- Eigenvalues $lambda_i$ = variance explained along each axis

== 3.3  PCA: Projection Perspective

- Best rank-$M$ approximation: project $x$ onto $M$-dim subspace spanned by top PCs
- $tilde(x) = B B^T x$ with $B = [b_1, dots, b_M]$
- Minimizes reconstruction error $sum_n ||x_n - tilde(x)_n||^2$
- Same solution as maximum-variance view (SVD of centered $X$)

== 3.3  PCA: Projection Perspective

#align(center + horizon)[#image("/assets/figures/day02/mml_pca_lowdim.png", width: 92%, height: 82%, fit: "contain")]

== 3.4  PCA Algorithm & Applications

- 1. Center data: $tilde(x)_n = x_n - bar(x)$
- 2. Compute $S = (1/N) sum_n tilde(x)_n tilde(x)_n^T$ (or SVD of $tilde(X)$)
- 3. Take top-$M$ eigenvectors; project $z_n = B^T tilde(x)_n$
- Uses: visualization (2D/3D), compression, whitening, feature preprocessing
- Explained variance ratio guides choice of $M$

== 3.4  PCA Algorithm & Applications

#align(center + horizon)[#image("/assets/figures/day02/mml_pca_projection.png", width: 92%, height: 82%, fit: "contain")]

= 4 · Density Estimation / GMM (MML Ch. 11)

== 4.1  Learning $p(x)$

- Generative view: model the *input* distribution, not just $p(y|x)$
- Parametric: choose family $p(x|theta)$ (e.g. Gaussian, GMM)
- Nonparametric: histograms, KDE — flexible but need many samples
- Clustering emerges when we fit multi-modal densities (GMM)

== 4.1  Learning $p(x)$

#align(center + horizon)[#image("/assets/figures/day02/mml_pca_projection.png", width: 92%, height: 82%, fit: "contain")]

== 4.2  Gaussian Mixture Models

- $p(x) = sum_(k=1)^K pi_k cal(N)(x | mu_k, Sigma_k)$, $sum_k pi_k = 1$
- Each component = a cluster; soft assignments via responsibilities
- Latent variable $z_n in {1, dots, K}$: which component generated $x_n$?
- Richer than single Gaussian; still tractable with EM

== 4.3  EM Algorithm for GMM

- E-step: $r_(n k) = pi_k cal(N)(x_n|mu_k, Sigma_k) / sum_j pi_j cal(N)(x_n|mu_j, Sigma_j)$
- M-step: update $pi_k, mu_k, Sigma_k$ from weighted sufficient statistics
- Monotonically increases data log-likelihood (local optimum)
- Initialize carefully (k-means++) — symmetries cause multiple local maxima
- Choose $K$ via BIC or held-out log-likelihood

== 4.3  EM Algorithm for GMM

#align(center + horizon)[#image("/assets/figures/day02/mml_gmm_1d.png", width: 92%, height: 82%, fit: "contain")]

= 5 · Classification / SVM (MML Ch. 12)

== 5.1  Classification Setting

- Labels $y_n in {-1, +1}$ (or ${0,1}$); learn decision boundary in feature space
- Linear classifier: $f(x) = w^T x + b$; predict $"sign"(f(x))$
- Many separating hyperplanes exist when data are linearly separable
- Which one generalizes best? *Maximum margin* principle (SVM)

== 5.2  Separating Hyperplanes

- Separating hyperplane: $w^T x + b = 0$ with $y_n(w^T x_n + b) > 0$ for all $n$
- Margin = distance from hyperplane to nearest point
- Scale $(w, b)$ so nearest points satisfy $|w^T x_n + b| = 1$ (support vectors)
- Hard-margin SVM: maximize margin subject to correct classification

== 5.2  Separating Hyperplanes

#align(center + horizon)[#image("/assets/figures/day02/mml_svm_2d.png", width: 92%, height: 82%, fit: "contain")]

== 5.3  Maximum-Margin SVM (Primal)

- Minimize $||w||^2$ subject to $y_n(w^T x_n + b) >= 1$
- Solution depends only on *support vectors* (points on the margin)
- Dual formulation: $max_alpha sum_n alpha_n - (1/2) sum_(n,m) alpha_n alpha_m y_n y_m x_n^T x_m$
- Kernel trick: replace $x_n^T x_m$ with $k(x_n, x_m)$ for nonlinear boundaries

== 5.3  Maximum-Margin SVM (Primal)

#align(center + horizon)[#image("/assets/figures/day02/mml_svm_hyperplane.png", width: 92%, height: 82%, fit: "contain")]

== 5.4  Soft Margin & Kernels

- Soft margin: allow slack $xi_n >= 0$; penalize $C sum_n xi_n$ in objective
- Trade-off: large $C$ $arrow.r$ fewer errors, narrower margin (may overfit)
- RBF kernel $k(x, x') = exp(-gamma ||x - x'||^2)$ — implicit feature map
- SVMs vs logistic regression: hinge loss vs log loss; sparse support vectors
- Multiclass: one-vs-rest or structured extensions

== 5.4  Soft Margin & Kernels

#align(center + horizon)[#image("/assets/figures/day02/mml_svm_margin.png", width: 92%, height: 82%, fit: "contain")]

= Additional Figures

== Supplementary figure 1

#align(center + horizon)[#image("/assets/figures/day02/mml_soft_margin.png", width: 92%, height: 82%, fit: "contain")]

== Summary

- Day 2: *Statistical Learning*
- MML Part II — regression, PCA, density estimation, classification
- Questions welcome — practical follows

== Questions?

#align(center + horizon)[
  #text(size: 44pt, weight: "bold", fill: course-primary)[Questions?]

  #v(1em)
  Practical session follows — see course site.
]
