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

= Learning Setup

== Supervised Learning

- Dataset $D = {(x_i, y_i)}_(i=1)^n$
- Hypothesis class $cal(H)$ and capacity
- Train / validation / test splits
- Generalization: performance on unseen data

== Supervised Learning — illustration

#align(center)[#image("/assets/figures/day02/L5_probabili_01.png", width: 80%)]

#text(size: 14pt, fill: gray)[Learning Setup — Supervised Learning (source: course materials)]

== Regression

- Targets $y in RR$ (continuous)
- Linear model: $f_w(x) = w^T phi(x)$
- MSE: $L(w) = (1/n) sum_i (y_i - f_w(x_i))^2$
- Polynomial features and basis expansion

== Regression — illustration

#align(center)[#image("/assets/figures/day02/L5_probabili_02.png", width: 80%)]

#text(size: 14pt, fill: gray)[Learning Setup — Regression (source: course materials)]

== Classification

- Binary labels $y in {0, 1}$ or $y in {-1, +1}$
- Linear classifier: $f_w(x) = w^T x$
- Decision boundary where $f_w(x) = 0$

== Classification — illustration

#align(center)[#image("/assets/figures/day02/L6_decision__00.png", width: 80%)]

#text(size: 14pt, fill: gray)[Learning Setup — Classification (source: course materials)]

== Evaluation Metrics

- Regression: MSE, MAE, $R^2$
- Classification: accuracy, precision, recall, F1
- Confusion matrix and class imbalance

== Evaluation Metrics — illustration

#align(center)[#image("/assets/figures/day02/L6_decision__01.png", width: 80%)]

#text(size: 14pt, fill: gray)[Learning Setup — Evaluation Metrics (source: course materials)]

= Linear & Logistic Models

== Ordinary Least Squares

- Normal equations: $w^* = (X^T X)^(-1) X^T y$
- Geometric interpretation: orthogonal residual
- When $X^T X$ is ill-conditioned

== Ordinary Least Squares — illustration

#align(center)[#image("/assets/figures/day02/L6_decision__02.png", width: 80%)]

#text(size: 14pt, fill: gray)[Linear & Logistic Models — Ordinary Least Squares (source: course materials)]

== Logistic Regression

- $p(y=1|x) = sigma(w^T x)$ with $sigma(z) = 1/(1+e^(-z))$
- Cross-entropy loss for Bernoulli labels
- Decision boundary remains linear in feature space

== Logistic Regression — illustration

#align(center)[#image("/assets/figures/day02/L6_decision__05.png", width: 80%)]

#text(size: 14pt, fill: gray)[Linear & Logistic Models — Logistic Regression (source: course materials)]

== Softmax Multiclass

- $p(y=k|x) = exp(w_k^T x) / sum_j exp(w_j^T x)$
- One-vs-rest vs multinomial logistic
- Cross-entropy over $K$ classes

== Softmax Multiclass — illustration

#align(center)[#image("/assets/figures/day02/L6_decision__06.png", width: 80%)]

#text(size: 14pt, fill: gray)[Linear & Logistic Models — Softmax Multiclass (source: course materials)]

== Probabilistic View

- Model specifies $p(y|x, w)$
- MLE chooses $w$ maximizing $product_i p(y_i|x_i, w)$
- Log-likelihood turns products into sums

== Probabilistic View — illustration

#align(center)[#image("/assets/figures/day02/L6_decision__07.png", width: 80%)]

#text(size: 14pt, fill: gray)[Linear & Logistic Models — Probabilistic View (source: course materials)]

= Regularization

== Overfitting

- Low train error, high test error
- Model too flexible for data size
- Memorization vs learning structure

== Overfitting — illustration

#align(center)[#image("/assets/figures/day02/L6_decision__08.png", width: 80%)]

#text(size: 14pt, fill: gray)[Regularization — Overfitting (source: course materials)]

== L2 (Ridge)

- $L(w) = sum_i ell_i + lambda ||w||_2^2$
- Shrinks weights toward zero
- Closed form for ridge regression

== L2 (Ridge) — illustration

#align(center)[#image("/assets/figures/day02/L6_decision__09.png", width: 80%)]

#text(size: 14pt, fill: gray)[Regularization — L2 (Ridge) (source: course materials)]

== L1 (Lasso)

- $L(w) = sum_i ell_i + lambda ||w||_1$
- Promotes sparsity — feature selection
- Non-smooth at zero; subgradient methods

== L1 (Lasso) — illustration

#align(center)[#image("/assets/figures/day02/L6_decision__10.png", width: 80%)]

#text(size: 14pt, fill: gray)[Regularization — L1 (Lasso) (source: course materials)]

== Model Selection

- Validation curves vs $lambda$
- Early stopping as implicit regularization
- Bias–variance decomposition intuition

== Model Selection — illustration

#align(center)[#image("/assets/figures/day02/L6_decision__11.png", width: 80%)]

#text(size: 14pt, fill: gray)[Regularization — Model Selection (source: course materials)]

= Beyond Linearity (Preview)

== Kernel Trick

- Implicit feature map $phi(x)$ via $k(x, x')$
- Representer theorem: $w = sum_i alpha_i phi(x_i)$
- RBF kernel for non-linear boundaries

== Feature Engineering

- Domain knowledge beats raw pixels (often)
- Normalization and standardization
- Handling missing data and outliers

== Bridge to Deep Learning

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
