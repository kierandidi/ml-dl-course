#import "../lib.typ": *

#show: course-theme.with(title: [Statistical Learning], subtitle: [Day 2 | Aug 2026])

= Day 2: Statistical Learning

== Welcome

- *Statistical Learning* — Regression, classification, regularization
- 3 hours lecture + practical
- Slides, notes, and code on the course site

== Outline

- Learning Setup
- Linear & Logistic Models
- Regularization
- Beyond Linearity (Preview)

= 1 · Learning Setup

== 1.1  Supervised Learning

- Dataset $D = {(x_i, y_i)}_(i=1)^n$
- Hypothesis class $cal(H)$ and capacity
- Train / validation / test splits
- Generalization: performance on unseen data

== 1.1  Supervised Learning

#align(center + horizon)[#image("/assets/figures/day02/L5_probabili_01.png", width: 92%, height: 82%, fit: "contain")]

== 1.2  Regression

- Targets $y in RR$ (continuous)
- Linear model: $f_w(x) = w^T phi(x)$
- MSE: $L(w) = (1/n) sum_i (y_i - f_w(x_i))^2$
- Polynomial features and basis expansion

== 1.2  Regression

#align(center + horizon)[#image("/assets/figures/day02/L5_probabili_02.png", width: 92%, height: 82%, fit: "contain")]

== 1.3  Classification

- Binary labels $y in {0, 1}$ or $y in {-1, +1}$
- Linear classifier: $f_w(x) = w^T x$
- Decision boundary where $f_w(x) = 0$

== 1.3  Classification

#align(center + horizon)[#image("/assets/figures/day02/L6_decision__00.png", width: 92%, height: 82%, fit: "contain")]

== 1.4  Evaluation Metrics

- Regression: MSE, MAE, $R^2$
- Classification: accuracy, precision, recall, F1
- Confusion matrix and class imbalance

== 1.4  Evaluation Metrics

#align(center + horizon)[#image("/assets/figures/day02/L6_decision__01.png", width: 92%, height: 82%, fit: "contain")]

= 2 · Linear & Logistic Models

== 2.1  Ordinary Least Squares

- Normal equations: $w^* = (X^T X)^(-1) X^T y$
- Geometric interpretation: orthogonal residual
- When $X^T X$ is ill-conditioned

== 2.1  Ordinary Least Squares

#align(center + horizon)[#image("/assets/figures/day02/L6_decision__02.png", width: 92%, height: 82%, fit: "contain")]

== 2.2  Logistic Regression

- $p(y=1|x) = sigma(w^T x)$ with $sigma(z) = 1/(1+e^(-z))$
- Cross-entropy loss for Bernoulli labels
- Decision boundary remains linear in feature space

== 2.2  Logistic Regression

#align(center + horizon)[#image("/assets/figures/day02/L6_decision__05.png", width: 92%, height: 82%, fit: "contain")]

== 2.3  Softmax Multiclass

- $p(y=k|x) = exp(w_k^T x) / sum_j exp(w_j^T x)$
- One-vs-rest vs multinomial logistic
- Cross-entropy over $K$ classes

== 2.3  Softmax Multiclass

#align(center + horizon)[#image("/assets/figures/day02/L6_decision__06.png", width: 92%, height: 82%, fit: "contain")]

== 2.4  Probabilistic View

- Model specifies $p(y|x, w)$
- MLE chooses $w$ maximizing $product_i p(y_i|x_i, w)$
- Log-likelihood turns products into sums

== 2.4  Probabilistic View

#align(center + horizon)[#image("/assets/figures/day02/L6_decision__07.png", width: 92%, height: 82%, fit: "contain")]

= 3 · Regularization

== 3.1  Overfitting

- Low train error, high test error
- Model too flexible for data size
- Memorization vs learning structure

== 3.1  Overfitting

#align(center + horizon)[#image("/assets/figures/day02/L6_decision__08.png", width: 92%, height: 82%, fit: "contain")]

== 3.2  L2 (Ridge)

- $L(w) = sum_i ell_i + lambda ||w||_2^2$
- Shrinks weights toward zero
- Closed form for ridge regression

== 3.2  L2 (Ridge)

#align(center + horizon)[#image("/assets/figures/day02/L6_decision__09.png", width: 92%, height: 82%, fit: "contain")]

== 3.3  L1 (Lasso)

- $L(w) = sum_i ell_i + lambda ||w||_1$
- Promotes sparsity — feature selection
- Non-smooth at zero; subgradient methods

== 3.3  L1 (Lasso)

#align(center + horizon)[#image("/assets/figures/day02/L6_decision__10.png", width: 92%, height: 82%, fit: "contain")]

== 3.4  Model Selection

- Validation curves vs $lambda$
- Early stopping as implicit regularization
- Bias–variance decomposition intuition

== 3.4  Model Selection

#align(center + horizon)[#image("/assets/figures/day02/L6_decision__11.png", width: 92%, height: 82%, fit: "contain")]

= 4 · Beyond Linearity (Preview)

== 4.1  Kernel Trick

- Implicit feature map $phi(x)$ via $k(x, x')$
- Representer theorem: $w = sum_i alpha_i phi(x_i)$
- RBF kernel for non-linear boundaries

== 4.2  Feature Engineering

- Domain knowledge beats raw pixels (often)
- Normalization and standardization
- Handling missing data and outliers

== 4.3  Bridge to Deep Learning

- Neural nets learn features end-to-end (Day 3)
- Same loss + regularization story, richer $cal(H)$
- Practical today: start simple, add complexity

== Summary

- Day 2: *Statistical Learning*
- Regression, classification, regularization
- Questions welcome — practical follows

== Questions?

#align(center + horizon)[
  #text(size: 44pt, weight: "bold", fill: course-primary)[Questions?]

  #v(1em)
  Practical session follows — see course site.
]
