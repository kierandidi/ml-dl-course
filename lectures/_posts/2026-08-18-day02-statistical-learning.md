---
layout: post
title: Day 2 - Statistical Learning
image: /assets/img/sampling_space.png
accent_image: 
  background: url('/assets/img/sampling_space.png') center/cover
  overlay: false
accent_color: '#ccc'
theme_color: '#ccc'
description: >
  Regression, classification, bias–variance, and regularization.
invert_sidebar: true
---

# Day 2 - Statistical Learning

### Optional reading for this lesson
- [Hastie, Tibshirani & Friedman — ESL](https://hastie.su.domains/ElemStatLearn/), Ch. 3–4
- [Murphy — Probabilistic Machine Learning: An Introduction](https://probml.github.io/pml-book/book1.html), Ch. 11
- [scikit-learn User Guide — Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)

### [Slides](/assets/slides/day02.pdf)

### [Practical](/projects/day02-practical/)

Statistical learning formalizes prediction as estimating a function from finite data. We study linear and logistic models, loss functions, the bias–variance trade-off, and regularization as the primary tools for controlling generalization.

* toc
{:toc}

## 1. Supervised Learning Framework

### 1.1 Empirical risk minimization

> Given training pairs $$(\mathbf{x}^{(i)}, y^{(i)})$$ and loss $$\ell$$, **empirical risk minimization (ERM)** seeks $$\hat{f} = \arg\min_{f \in \mathcal{F}} \frac{1}{n}\sum_i \ell(f(\mathbf{x}^{(i)}), y^{(i)}).$$
{:.lead}

The **hypothesis class** $$\mathcal{F}$$ encodes inductive bias. Linear models use $$\mathcal{F} = \{f(\mathbf{x}) = \mathbf{w}^\top\mathbf{x} + b\}$$; neural nets use compositional nonlinear functions.

The **true risk** (population loss) is $$R(f) = \mathbb{E}_{(\mathbf{x},y)}[\ell(f(\mathbf{x}), y)]$$. We only observe the empirical proxy $$\hat{R}(f)$$. Generalization is the gap $$R(\hat{f}) - \hat{R}(\hat{f})$$.

![Train vs test error as model complexity grows](/assets/figures/day02/pdf0_page005.png)

### 1.2 Bias, variance, and noise

> For squared loss, expected test error decomposes as **bias² + variance + irreducible noise**. High bias underfits; high variance overfits.
{:.lead}

For an estimator $$\hat{f}$$ at input $$\mathbf{x}$$,

$$\mathbb{E}[(y - \hat{f}(\mathbf{x}))^2] = \underbrace{(\mathbb{E}[\hat{f}(\mathbf{x})] - f^\*(\mathbf{x}))^2}_{\text{bias}^2} + \underbrace{\mathrm{Var}(\hat{f}(\mathbf{x}))}_{\text{variance}} + \underbrace{\sigma^2}_{\text{noise}}.$$

Complex models reduce bias but increase variance. **Model selection** (cross-validation, validation sets) estimates out-of-sample performance without peeking at the test set.

## 2. Linear and Logistic Regression

### 2.1 Linear regression

> **Linear regression** models $$\mathbb{E}[y|\mathbf{x}] = \mathbf{w}^\top\mathbf{x} + b$$. Under Gaussian noise, OLS minimizes mean squared error.
{:.lead}

Vectorized training loss:

$$L(\mathbf{w}) = \frac{1}{2n}\|\mathbf{X}\mathbf{w} - \mathbf{y}\|_2^2.$$

Gradient descent update with learning rate $$\eta$$:

$$\mathbf{w} \leftarrow \mathbf{w} - \frac{\eta}{n}\mathbf{X}^\top(\mathbf{X}\mathbf{w} - \mathbf{y}).$$

![Linear fit with residual bands](/assets/figures/day02/pdf0_page010.png)

**Polynomial regression** remains linear in parameters: features $$\phi(\mathbf{x}) = (1, x, x^2, \ldots)$$ and $$\hat{y} = \mathbf{w}^\top\phi(\mathbf{x})$$.

### 2.2 Logistic regression

> For binary labels $$y \in \{0,1\}$$, **logistic regression** models $$p(y=1|\mathbf{x}) = \sigma(\mathbf{w}^\top\mathbf{x} + b)$$ where $$\sigma(z) = 1/(1+e^{-z})$$.
{:.lead}

Cross-entropy loss for one example:

$$\ell = -\big[y\log \hat{p} + (1-y)\log(1-\hat{p})\big], \quad \hat{p} = \sigma(\mathbf{w}^\top\mathbf{x}).$$

The gradient has the elegant form $$\nabla_{\mathbf{w}} \ell = (\hat{p} - y)\mathbf{x}$$ — structurally identical to linear regression but with a nonlinear link.

Multiclass **softmax regression** uses

$$p(y=k|\mathbf{x}) = \frac{e^{\mathbf{w}_k^\top \mathbf{x}}}{\sum_j e^{\mathbf{w}_j^\top \mathbf{x}}}.$$

This is the output layer of most classifiers.

## 3. Classification Metrics and Decision Theory

### 3.1 Confusion matrix and thresholding

> A **confusion matrix** tabulates TP, FP, TN, FN. **Precision** = TP/(TP+FP); **recall** = TP/(TP+FN). The **ROC curve** plots TPR vs FPR as the threshold varies.
{:.lead}

**F1 score** harmonically combines precision and recall:

$$F_1 = 2 \cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}.$$

For imbalanced data, accuracy is misleading; prefer PR-AUC or balanced metrics.

![ROC and precision-recall curves](/assets/figures/day02/pdf0_page015.png)

**Bayes optimal classifier** chooses $$\hat{y} = \arg\max_k p(y=k|\mathbf{x})$$, minimizing 0–1 loss. Logistic regression approximates this when the model is well-specified.

### 3.2 Loss functions and surrogates

> 0–1 loss is non-differentiable; we optimize **surrogate losses** (log loss, hinge loss) that upper-bound or approximate classification error.
{:.lead}

**Hinge loss** for SVMs:

$$\ell_{\mathrm{hinge}} = \max(0, 1 - y\, f(\mathbf{x})), \quad y \in \{-1,+1\}.$$

**Calibration**: predicted probabilities should match empirical frequencies. Platt scaling and isotonic regression post-process scores.

Expected cost under asymmetric misclassification costs $$C_{ij}$$:

$$R = \sum_{i,j} C_{ij}\, p(y=j|\mathbf{x})\, \mathbb{1}[\hat{y}=i].$$

## 4. Regularization and Model Selection

### 4.1 Ridge, Lasso, and elastic net

> **Ridge** ($$\ell_2$$) penalizes $$\lambda\|\mathbf{w}\|_2^2$; **Lasso** ($$\ell_1$$) penalizes $$\lambda\|\mathbf{w}\|_1$ and induces sparsity. **Elastic net** combines both.
{:.lead}

Ridge closed form (when invertible):

$$\hat{\mathbf{w}}_{\mathrm{ridge}} = (\mathbf{X}^\top\mathbf{X} + \lambda \mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}.$$

The penalty shrinks weights toward zero, reducing variance. Geometrically, Lasso's diamond constraint promotes exact zeros at optimality.

![Regularization paths as $$\lambda$$ varies](/assets/figures/day02/pdf0_page020.png)

**Early stopping** is implicit regularization: halt gradient descent before training loss reaches zero.

### 4.2 Cross-validation and hyperparameters

> **k-fold cross-validation** splits data into $$k$$ folds, training on $$k-1$$ and validating on the held-out fold. Hyperparameters ($$\lambda$$, degree, etc.) are tuned on validation data only.
{:.lead}

Average validation score:

$$\mathrm{CV}(\lambda) = \frac{1}{k}\sum_{j=1}^k \hat{R}_{\mathrm{val}}^{(j)}(\lambda).$$

Choose $$\lambda^\* = \arg\min \mathrm{CV}(\lambda)$$, then optionally refit on all training data.

**Nested CV** separates hyperparameter tuning from performance estimation to avoid optimistic bias.

Information criteria (AIC, BIC) penalize model complexity: $$\mathrm{BIC} = -2\ell + p\log n$$ where $$p$$ is parameter count.

## Checkpoint summary

Before moving to the practical, confirm you can:

- ERM minimizes average training loss; generalization depends on bias–variance balance.
- Linear regression = Gaussian MLE; logistic regression = Bernoulli MLE with sigmoid link.
- Use appropriate metrics (F1, AUC) when classes are imbalanced.
- Regularization and validation control overfitting without changing the hypothesis class.
