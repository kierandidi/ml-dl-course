---
layout: post
title: Day 2 - Statistical Learning
image: /assets/img/lessons/day02.png
description: >
  MML Part II: the supervised/unsupervised framework, linear regression, PCA, Gaussian mixture models, and support vector machines.
invert_sidebar: true
---

# Day 2 - Statistical Learning

### Optional reading for this lesson
- [Deisenroth, Faisal & Ong — *Mathematics for Machine Learning*](https://mml-book.com), Ch. 8–12
- [Mathematics for Machine Learning (free PDF)](https://mml-book.github.io/book/mml-book.pdf)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

### [Slides](/assets/slides/day02.pdf)

### Exercise

[Download the notebook](/notebooks/practicals/day02.ipynb) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kierandidi/ml-dl-course/blob/main/notebooks/practicals/day02.ipynb)

Day 1 built the mathematical language; today we use it to design learning algorithms. Following MML Part II, we start with the **framework** for how data, models, and learning fit together (Chapter 8), then work through the four central pillars: **linear regression** (Ch. 9), **PCA** for dimensionality reduction (Ch. 10), **Gaussian mixture models** for density estimation (Ch. 11), and **support vector machines** for classification (Ch. 12). Each section connects back to the linear-algebra and probability tools from Day 1.

* toc
{:toc}

## 1. When Models Meet Data (Chapter 8)

### 1.1 A taxonomy of machine learning

> **Machine learning** learns patterns from data. **Supervised** learning uses labeled pairs $$(\mathbf{x}, y)$$ (regression or classification); **unsupervised** learning finds structure in inputs alone (clustering, density estimation, dimensionality reduction); **reinforcement** learning optimizes sequential decisions from rewards.
{:.lead}

![Taxonomy of machine learning: supervised (regression & classification), unsupervised (clustering & dimensionality reduction), and reinforcement learning with representative applications.](/assets/figures/day02/ml_taxonomy.png)

This map is the roadmap for the rest of the course. **Regression** (continuous targets) and **classification** (discrete labels) dominate supervised learning and reappear inside deep networks (Days 3–5). **Dimensionality reduction** and **density estimation** are unsupervised tools we use for visualization, preprocessing, and generative modeling. **Reinforcement learning** sits outside today's scope but shares the same ERM spirit: optimize an expected objective from finite samples.

MML Part II organizes the core methods into four pillars — regression, PCA, GMM, SVM — all introduced through the common lens of Chapter 8.

### 1.2 Data as vectors, models as hypotheses

> A **dataset** is $$\mathcal{D} = \{(\mathbf{x}_n, y_n)\}_{n=1}^N$$ with feature vectors $$\mathbf{x}_n \in \mathbb{R}^D$$. A **model** is a parameterized family — e.g. $$p(y\mid\mathbf{x}, \boldsymbol{\theta})$$ or $$f_{\boldsymbol{\theta}}(\mathbf{x})$$ — and **learning** selects $$\boldsymbol{\theta}$$ from $$\mathcal{D}$$.
{:.lead}

![Toy regression data: salary vs age with a query point at age 60 (MML Fig. 8.1).](/assets/figures/day02/mml_toy_regression.png)

Representing examples as vectors lets us reuse linear algebra (Day 1): design matrices, projections, eigen-decompositions. **Similarity** between examples — via inner products or distances — is the geometric backbone of both regression (Ch. 9) and classification (Ch. 12): nearby points should receive similar predictions.

The figure shows a concrete regression task: predict salary from age using $$N$$ training pairs, then query at a new age (60) not in the training set. Success is measured on that *held-out* input, not on memorizing training salaries.

### 1.3 Empirical risk minimization

> **Empirical risk minimization (ERM)** chooses $$\hat{\boldsymbol{\theta}} = \arg\min_{\boldsymbol{\theta}} \frac{1}{N}\sum_{n=1}^N \ell\big(f_{\boldsymbol{\theta}}(\mathbf{x}_n), y_n\big)$$ as a proxy for the unknowable **population risk** $$R(f) = \mathbb{E}_{(\mathbf{x},y)}[\ell(f(\mathbf{x}), y)]$$.
{:.lead}

We never observe the true data distribution — only a finite sample — so we minimize the **empirical** average loss instead. Common choices:

- **Squared error** $$\ell = (y - \hat{y})^2$$ for regression (leads to least squares under Gaussian noise).
- **Cross-entropy** for classification (leads to logistic regression; see also SVM hinge loss in Ch. 12).

ERM is the workhorse of machine learning, but minimizing training loss alone is dangerous: a large enough model can drive empirical risk to zero while **test** error explodes. Chapters 9–12 show both the power of ERM and the tools to control it (regularization, validation, margins).

### 1.4 Validation, cross-validation, and model selection

> Split data into **training**, **validation**, and **test** sets. **K-fold cross-validation** rotates the validation fold, averaging $$\frac{1}{K}\sum_k R(f^{(k)}, \mathcal{V}^{(k)})$$ as an estimate of generalization error.
{:.lead}

![K-fold cross-validation: the dataset is partitioned into $$K$$ chunks; each chunk serves once as validation while the rest train the model (MML Fig. 8.4).](/assets/figures/day02/mml_cross_validation.png)

**Hyperparameters** — ridge penalty $$\lambda$$, polynomial degree, number of GMM components $$K$$, SVM slack $$C$$ — are tuned on validation data only. The **test** set is touched once, at the end, for an unbiased performance estimate.

**Model selection** balances fit and complexity. MML emphasizes **Occam's razor**: among models with similar validation error, prefer the simpler one. Information criteria make this explicit:

$$\mathrm{BIC} = -2\,\ell + p\log N,$$

where $$p$$ is the number of free parameters and $$\ell$$ is the maximized log-likelihood. Lower BIC favors parsimony.

![Training and validation error vs model capacity: training error decreases monotonically while validation error has a sweet spot (MML Fig. 8.5 — see textbook).](/assets/figures/day02/mml_poly_overfit.png)

The capacity curve is the picture to remember: more flexible $$\mathcal{H}$$ reduces **bias** but increases **variance**; validation error is the practical guide to the sweet spot.

## 2. Linear Regression (Chapter 9)

### 2.1 Problem setup and the linear model

> In **linear regression**, we model $$y = \boldsymbol{\phi}(\mathbf{x})^{\top}\boldsymbol{\theta} + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2),$$ where $$\boldsymbol{\phi}$$ maps inputs to features and the model is **linear in** $$\boldsymbol{\theta}$$.
{:.lead}

![Linear regression example: candidate lines, training data, and MLE fit (MML Fig. 9.2).](/assets/figures/day02/mml_linear_regression.png)

The key phrase is *linear in the parameters*. A polynomial $$y = \theta_0 + \theta_1 x + \theta_2 x^2$$ is still linear regression because $$\boldsymbol{\phi}(x) = (1, x, x^2)^{\top}$$. Stacking $$N$$ examples gives the **design matrix** $$\Phi \in \mathbb{R}^{N \times M}$$ and predictions $$\hat{\mathbf{y}} = \Phi\boldsymbol{\theta}$$.

We assume i.i.d. Gaussian noise — a probabilistic model, not just curve fitting — which connects least squares to maximum likelihood.

### 2.2 Derivation: maximum likelihood equals least squares

> Under $$p(y\mid\mathbf{x}, \boldsymbol{\theta}) = \mathcal{N}(\boldsymbol{\phi}(\mathbf{x})^{\top}\boldsymbol{\theta}, \sigma^2)$$, the **MLE** of $$\boldsymbol{\theta}$$ minimizes $$\Vert \Phi\boldsymbol{\theta} - \mathbf{y}\Vert _2^2$$.
{:.lead}

The log-likelihood for one observation is

$$\log p(y_n\mid\mathbf{x}_n, \boldsymbol{\theta}) = -\tfrac{1}{2\sigma^2}(y_n - \boldsymbol{\phi}(\mathbf{x}_n)^{\top}\boldsymbol{\theta})^2 + \text{const}.$$

Summing over $$n$$ and dropping constants, MLE maximizes

$$-\tfrac{1}{2\sigma^2}\Vert \Phi\boldsymbol{\theta} - \mathbf{y}\Vert _2^2,$$

equivalent to minimizing the **mean squared error**. Setting the gradient to zero yields the **normal equations**

$$\Phi^{\top}\Phi\,\boldsymbol{\theta} = \Phi^{\top}\mathbf{y}.$$

When $$\Phi^{\top}\Phi$$ is invertible,

$$\boldsymbol{\theta}^{\star} = (\Phi^{\top}\Phi)^{-1}\Phi^{\top}\mathbf{y}.$$

Geometrically, $$\Phi\boldsymbol{\theta}^{\star}$$ is the **orthogonal projection** of $$\mathbf{y}$$ onto the column space of $$\Phi$$ — the closest point in the model's span.

### 2.3 Ridge regression and the bias–variance trade-off

> **Ridge regression** adds an $$\ell_2$$ penalty: $$\hat{\boldsymbol{\theta}} = (\Phi^{\top}\Phi + \lambda \mathbf{I})^{-1}\Phi^{\top}\mathbf{y}.$$ It stabilizes inversion when features are collinear or $$N \ll M$$.
{:.lead}

When $$\Phi^{\top}\Phi$$ is near-singular, OLS has enormous variance: small perturbations of $$\mathbf{y}$$ swing $$\boldsymbol{\theta}^{\star}$$ wildly. Adding $$\lambda \mathbf{I}$$ shrinks coefficients toward zero — trading a little **bias** for much lower **variance**.

Equivalently, ridge is the **MAP estimate** under a Gaussian prior $$\boldsymbol{\theta} \sim \mathcal{N}(\mathbf{0}, \tau^2 \mathbf{I})$$. The penalty $$\lambda$$ is tuned by cross-validation (Ch. 8).

![Training and test error vs polynomial degree: training error falls while test error rises after the true degree — a classic overfitting picture (MML Fig. 9.6).](/assets/figures/day02/mml_poly_overfit.png)

Polynomial regression illustrates the capacity curve in action: high degree fits every training point but wiggles wildly between them. Ridge or early stopping on degree (via validation) restores generalization.

## 3. Dimensionality Reduction with PCA (Chapter 10)

### 3.1 Motivation and problem setting

> **Principal component analysis (PCA)** finds an orthogonal basis $$\{\mathbf{b}_1, \ldots, \mathbf{b}_D\}$$ such that the first $$M$$ components capture most of the variance in centered data $$\tilde{\mathbf{x}}_n = \mathbf{x}_n - \bar{\mathbf{x}}$$.
{:.lead}

High-dimensional data suffer the **curse of dimensionality**: distances become less meaningful, and we need exponentially more samples to fill the space. PCA offers a *linear* compression: represent each point by $$M \ll D$$ coordinates while minimizing reconstruction error.

Applications include visualization (project to 2D/3D), denoising, whitening features before downstream classifiers, and compression.

### 3.2 Maximum-variance and minimum-reconstruction views

> The first principal component solves $$\mathbf{b}_1 = \arg\max_{\Vert \mathbf{b}\Vert =1} \mathrm{Var}(\mathbf{b}^{\top}\tilde{\mathbf{X}}).$$ Equivalently, PCA minimizes $$\sum_n \Vert \tilde{\mathbf{x}}_n - \mathbf{B}\mathbf{B}^{\top}\tilde{\mathbf{x}}_n\Vert ^2$$ for rank-$$M$$ projection matrix $$\mathbf{B}$$.
{:.lead}

![PCA finds a lower-dimensional subspace that preserves variance when data are projected (MML Fig. 10.4).](/assets/figures/day02/mml_pca_lowdim.png)

![Orthogonal projection onto the principal subspace (MML Fig. 10.6).](/assets/figures/day02/mml_pca_projection.png)

Both formulations lead to the same solution: **eigenvectors of the sample covariance**

$$\mathbf{S} = \frac{1}{N}\sum_{n=1}^N \tilde{\mathbf{x}}_n \tilde{\mathbf{x}}_n^{\top}.$$

Sort eigenvalues $$\lambda_1 \geq \lambda_2 \geq \cdots$$; keep the top $$M$$ eigenvectors as columns of $$\mathbf{B}$$. The **explained variance ratio** $$\lambda_i / \sum_j \lambda_j$$ tells you how many components to retain.

**Algorithm (via SVD).** Center $$\tilde{\mathbf{X}}$$, compute $$\tilde{\mathbf{X}} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^{\top}$$; principal directions are columns of $$\mathbf{V}$$; coordinates $$\mathbf{z}_n = \mathbf{B}^{\top}\tilde{\mathbf{x}}_n$$.

## 4. Density Estimation with Gaussian Mixture Models (Chapter 11)

### 4.1 Parametric density estimation

> A **Gaussian mixture model (GMM)** writes $$p(\mathbf{x}) = \sum_{k=1}^{K} \pi_k\,\mathcal{N}(\mathbf{x}\mid\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k), \quad \sum_k \pi_k = 1,\; \pi_k \geq 0.$$
{:.lead}

Unlike regression (model $$p(y\mid\mathbf{x})$$), density estimation models $$p(\mathbf{x})$$ itself — useful for outlier detection, sampling, and discovering **clusters** as mixture components.

A single Gaussian is unimodal; a mixture can approximate multimodal data. Each component has mean $$\boldsymbol{\mu}_k$$, covariance $$\boldsymbol{\Sigma}_k$$, and mixing weight $$\pi_k$$.

![One-dimensional GMM: sum of Gaussians approximates a complex density (MML Fig. 11.3).](/assets/figures/day02/mml_gmm_1d.png)

### 4.2 EM algorithm: derivation sketch

> The **expectation–maximization (EM)** algorithm alternates: **E-step** — compute responsibilities $$r_{nk} = \frac{\pi_k\,\mathcal{N}(\mathbf{x}_n\mid\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_j \pi_j\,\mathcal{N}(\mathbf{x}_n\mid\boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)};$$ **M-step** — update $$\pi_k, \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k$$ from weighted sufficient statistics.
{:.lead}

Introduce latent variables $$z_n \in \{1,\ldots,K\}$$ indicating which component generated $$\mathbf{x}_n$$. The complete-data log-likelihood is easy to optimize; EM iterates:

1. **E-step:** $$\mathbb{E}[z_{nk}\mid\mathbf{x}_n] = r_{nk}$$ (posterior component membership).
2. **M-step:** weighted MLE updates, e.g.

$$\boldsymbol{\mu}_k = \frac{\sum_n r_{nk}\,\mathbf{x}_n}{\sum_n r_{nk}}, \qquad
\boldsymbol{\Sigma}_k = \frac{\sum_n r_{nk}(\mathbf{x}_n - \boldsymbol{\mu}_k)(\mathbf{x}_n - \boldsymbol{\mu}_k)^{\top}}{\sum_n r_{nk}}.$$

Each iteration increases the data log-likelihood (never decreases). EM finds **local** optima — initialization matters (k-means++ is standard).

Choose $$K$$ via BIC or cross-validated held-out log-likelihood. GMMs connect forward to **latent-variable models** and VAEs on Day 6.

## 5. Classification with Support Vector Machines (Chapter 12)

### 5.1 Separating hyperplanes and margins

> A **linear classifier** uses $$f(\mathbf{x}) = \mathbf{w}^{\top}\mathbf{x} + b$$ and predicts $$\mathrm{sign}(f(\mathbf{x}))$$. A **separating hyperplane** satisfies $$y_n(\mathbf{w}^{\top}\mathbf{x}_n + b) > 0$$ for all training points.
{:.lead}

![Two-dimensional classification data with class labels (MML Fig. 12.1).](/assets/figures/day02/mml_svm_2d.png)

When classes are linearly separable, infinitely many hyperplanes work. **Support vector machines** pick the one with **maximum margin** — the distance to the nearest training points (**support vectors**).

![Separating hyperplane geometry: normal vector $$\mathbf{w}$$ and offset $$b$$ (MML Fig. 12.3).](/assets/figures/day02/mml_svm_hyperplane.png)

### 5.2 Hard-margin and soft-margin SVMs

> **Hard-margin SVM:** $$\min_{\mathbf{w}, b} \tfrac{1}{2}\Vert \mathbf{w}\Vert ^2 \quad \text{s.t.}\quad y_n(\mathbf{w}^{\top}\mathbf{x}_n + b) \geq 1.$$ **Soft margin** adds slack $$\xi_n \geq 0$$ and penalty $$C\sum_n \xi_n$$ for non-separable data.
{:.lead}

The objective $$\tfrac{1}{2}\Vert \mathbf{w}\Vert ^2$$ maximizes the margin $$2/\Vert \mathbf{w}\Vert $$ while constraints keep points on the correct side. Only **support vectors** (points with $$y_n(\mathbf{w}^{\top}\mathbf{x}_n + b) = 1$$ or, in the soft case, $$\xi_n > 0$$) determine the solution — a sparse representation.

![Maximum-margin classifier: support vectors lie on the margin boundaries (MML Fig. 12.7).](/assets/figures/day02/mml_svm_margin.png)

![Hinge loss is a convex upper bound on zero-one loss — the surrogate optimized by soft-margin SVMs (MML Fig. 12.8).](/assets/figures/day02/mml_soft_margin.png)

The hyperparameter $$C$$ trades margin width against training errors: large $$C$$ punishes slack heavily (narrow margin, fewer errors); small $$C$$ tolerates misclassification (wider margin, simpler boundary). Tune $$C$$ by cross-validation.

### 5.3 Dual formulation and the kernel trick

> The **dual SVM** depends on inner products $$\mathbf{x}_n^{\top}\mathbf{x}_m$$ only. Replace them with a **kernel** $$k(\mathbf{x}_n, \mathbf{x}_m)$$ to learn nonlinear boundaries in an implicit feature space.
{:.lead}

The Lagrangian dual is

$$\max_{\boldsymbol{\alpha}} \sum_{n=1}^{N} \alpha_n - \tfrac{1}{2}\sum_{n,m}\alpha_n \alpha_m y_n y_m\, k(\mathbf{x}_n, \mathbf{x}_m)$$

subject to $$0 \leq \alpha_n \leq C$$ and $$\sum_n \alpha_n y_n = 0$$. Predictions use support vectors:

$$f(\mathbf{x}) = \sum_{n \in \mathrm{SV}} \alpha_n y_n\, k(\mathbf{x}_n, \mathbf{x}) + b.$$

Common kernels: **polynomial** $$k(\mathbf{x}, \mathbf{x}') = (\mathbf{x}^{\top}\mathbf{x}' + c)^d$$; **RBF** $$k(\mathbf{x}, \mathbf{x}') = \exp(-\gamma\Vert \mathbf{x} - \mathbf{x}'\Vert ^2)$$. The RBF kernel corresponds to an infinite-dimensional feature map — the same "lift to a richer space" idea as polynomial features in regression, but without explicit $$\boldsymbol{\phi}(\mathbf{x})$$.

SVMs use **hinge loss** (via the slack formulation) rather than logistic **log loss**; both are large-margin linear classifiers in feature space. Deep networks (Day 3+) learn $$\boldsymbol{\phi}$$ instead of hand-designing or kernelizing it.

## Checkpoint summary

Before moving to the practical, confirm you can:

- Draw the ML taxonomy and place regression, PCA, GMM, and SVM in it.
- State ERM and explain why validation/cross-validation is necessary.
- Derive the OLS closed form from Gaussian MLE and interpret it as a projection.
- Describe ridge regression and how it trades bias for variance.
- Explain PCA from both maximum-variance and minimum-reconstruction views.
- Write the GMM density and one EM update step (E and M).
- State the hard-margin SVM objective and the role of support vectors.
- Explain the kernel trick and how $$C$$ controls the soft-margin trade-off.
