---
layout: post
title: Day 1 - Math Foundations
image: /assets/img/lessons/day01.png
accent_image: 
  background: url('/assets/img/sampling_space.png') center/cover
  overlay: false
accent_color: '#ccc'
theme_color: '#ccc'
description: >
  Gradients, the chain rule, probability, and maximum likelihood estimation.
invert_sidebar: true
---

# Day 1 - Math Foundations

### Optional reading for this lesson
- [Bishop — Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/), Ch. 1–2
- [Boyd & Vandenberghe — Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/), §2.1–2.3
- [3Blue1Brown — Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3tZaY)
- [Complete reading list for Day 1](/publications/#day-1) (all resources for this lecture)


### [Slides](/assets/slides/day01.pdf)

### [Practical](/projects/day01-practical/)

Before we train neural networks we need a shared mathematical language. Today we review multivariate calculus, the chain rule, basic probability, and maximum likelihood — the workhorse principle behind most learning algorithms.

* toc
{:toc}

## 1. Multivariate Calculus and Gradients

### 1.1 Partial derivatives and the gradient

> For a scalar function $$f: \mathbb{R}^d \to \mathbb{R}$$, the **gradient** is the vector of partial derivatives $$\nabla f(\mathbf{x}) = \left(\frac{\partial f}{\partial x_1}, \ldots, \frac{\partial f}{\partial x_d}\right)^\top.$$ It points in the direction of steepest ascent.
{:.lead}

For a quadratic loss $$L(\mathbf{w}) = \|\mathbf{X}\mathbf{w} - \mathbf{y}\|_2^2$$ with design matrix $$\mathbf{X} \in \mathbb{R}^{n \times d}$$, expanding gives

$$L(\mathbf{w}) = \mathbf{w}^\top \mathbf{X}^\top \mathbf{X} \mathbf{w} - 2\mathbf{y}^\top \mathbf{X}\mathbf{w} + \mathbf{y}^\top\mathbf{y}.$$

Differentiating with respect to $$\mathbf{w}$$ yields the closed-form gradient

$$\nabla_{\mathbf{w}} L = 2\mathbf{X}^\top(\mathbf{X}\mathbf{w} - \mathbf{y}).$$

Setting $$\nabla_{\mathbf{w}} L = \mathbf{0}$$ recovers the ordinary least-squares solution $$\hat{\mathbf{w}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$$ whenever $$\mathbf{X}^\top\mathbf{X}$$ is invertible.

![Gradient descent on a quadratic bowl](/assets/figures/day01/pdf0_page005.png)

The **directional derivative** along unit vector $$\mathbf{u}$$ is $$D_{\mathbf{u}} f = \nabla f \cdot \mathbf{u}$$. Steepest descent uses $$\mathbf{u} = -\nabla f / \|\nabla f\|$$.

### 1.2 The Jacobian and Hessian

> For $$\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$$, the **Jacobian** $$\mathbf{J}_{\mathbf{f}} \in \mathbb{R}^{m \times n}$$ has entries $$[\mathbf{J}_{\mathbf{f}}]_{ij} = \partial f_i / \partial x_j$$. For scalar $$f$$, the **Hessian** $$\mathbf{H} = \nabla^2 f$$ describes local curvature.
{:.lead}

For a linear layer $$\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}$$ with $$\mathbf{W} \in \mathbb{R}^{m \times n}$$, the Jacobian with respect to $$\mathbf{x}$$ is simply $$\mathbf{W}$$. This fact is what makes backpropagation through affine layers trivial.

Near a critical point $$\mathbf{x}^\*$$, Taylor expansion gives

$$f(\mathbf{x}) \approx f(\mathbf{x}^\*) + \frac{1}{2}(\mathbf{x}-\mathbf{x}^\*)^\top \mathbf{H}(\mathbf{x}^\*)(\mathbf{x}-\mathbf{x}^\*).$$

Positive definiteness of $$\mathbf{H}$$ at $$\mathbf{x}^\*$$ implies a local minimum — a condition optimizers exploit via second-order methods (Newton, L-BFGS).

## 2. The Chain Rule and Computational Graphs

### 2.1 Scalar chain rule

> If $$y = f(u)$$ and $$u = g(x)$$, then $$\frac{dy}{dx} = \frac{dy}{du}\frac{du}{dx}$$. In multivariate form, gradients propagate backward through composed functions.
{:.lead}

Consider a two-layer network with scalar output:

$$\hat{y} = w_2\,\sigma(w_1 x + b_1) + b_2.$$

Define $$z = w_1 x + b_1$$ and $$a = \sigma(z)$$. Then

$$\frac{\partial \hat{y}}{\partial w_1} = \frac{\partial \hat{y}}{\partial a}\frac{\partial a}{\partial z}\frac{\partial z}{\partial w_1} = w_2\,\sigma'(z)\,x.$$

![Computational graph for a tiny network](/assets/figures/day01/pdf0_page010.png)

Each edge in the graph stores a local Jacobian; backpropagation is reverse-mode automatic differentiation that multiplies these Jacobians along paths from the loss to each parameter.

### 2.2 Multivariate chain rule

> If $$\mathbf{y} = \mathbf{f}(\mathbf{u})$$ and $$\mathbf{u} = \mathbf{g}(\mathbf{x})$$, then $$\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \mathbf{J}_{\mathbf{f}}(\mathbf{u})\,\mathbf{J}_{\mathbf{g}}(\mathbf{x})$$ by matrix multiplication.
{:.lead}

For vector-valued intermediates, the chain rule is a product of Jacobians. Given loss $$L$$ and hidden activation $$\mathbf{h}$$,

$$\frac{\partial L}{\partial \mathbf{h}} = \left(\frac{\partial L}{\partial \mathbf{z}}\right)^\top \mathbf{J}_{\mathbf{h}}.$$

In practice frameworks never materialize full Jacobians for large layers; they use **vector-Jacobian products** (VJPs) that cost one forward/backward pass per output dimension batch.

The **reverse-mode** trick: one backward pass from a scalar loss costs $$O(\text{ops})$$ regardless of parameter count, whereas forward-mode AD scales with the number of inputs.

## 3. Probability Essentials

### 3.1 Random variables and expectations

> A **random variable** $$X$$ maps outcomes $$\omega \in \Omega$$ to real values. For continuous $$X$$ with density $$p(x)$$, $$\mathbb{E}[X] = \int x\,p(x)\,dx$$ and $$\mathrm{Var}(X) = \mathbb{E}[(X-\mathbb{E}[X])^2].$$
{:.lead}

Key identities used throughout ML:

$$\mathrm{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2, \qquad \mathrm{Cov}(X,Y) = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y].$$

For multivariate Gaussian $$\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$$,

$$p(\mathbf{x}) = \frac{1}{(2\pi)^{d/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\!\left(-\tfrac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right).$$

![Gaussian contours in 2D](/assets/figures/day01/pdf0_page015.png)

**Conditional** distributions arise constantly: $$p(\mathbf{x}|y) \propto p(y|\mathbf{x})p(\mathbf{x})$$. Independence means $$p(\mathbf{x},\mathbf{y}) = p(\mathbf{x})p(\mathbf{y}).

### 3.2 Information and KL divergence

> The **Kullback–Leibler divergence** from $$q$$ to $$p$$ is $$D_{\mathrm{KL}}(q\|p) = \mathbb{E}_{\mathbf{x}\sim q}\!\left[\log\frac{q(\mathbf{x})}{p(\mathbf{x})}\right] \geq 0$$, with equality iff $$q = p$$ almost everywhere.
{:.lead}

KL divergence is not symmetric but measures how many extra nats are needed to encode samples from $$q$$ using a code optimized for $$p$$.

For Gaussians $$\mathcal{N}(\boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1)$$ and $$\mathcal{N}(\boldsymbol{\mu}_2, \boldsymbol{\Sigma}_2)$$ in $$d$$ dimensions, a closed form exists and appears in VAEs and diffusion training.

**Entropy** $$H(p) = -\mathbb{E}_{p}[\log p]$$ quantifies uncertainty. Cross-entropy $$H(p,q) = -\mathbb{E}_{p}[\log q]$$ is the classification loss when $$p$$ is the data distribution and $$q$$ the model.

## 4. Maximum Likelihood Estimation

### 4.1 Likelihood and log-likelihood

> Given i.i.d. data $$\mathcal{D} = \{\mathbf{x}^{(i)}\}_{i=1}^n$$ and parametric model $$p_\theta(\mathbf{x})$$, the **likelihood** is $$\mathcal{L}(\theta) = \prod_i p_\theta(\mathbf{x}^{(i)})$$. **MLE** chooses $$\hat{\theta} = \arg\max_\theta \mathcal{L}(\theta)$$.
{:.lead}

We almost always maximize the log-likelihood (monotone transform):

$$\ell(\theta) = \sum_{i=1}^n \log p_\theta(\mathbf{x}^{(i)}).$$

For Gaussian noise regression $$y^{(i)} = \mathbf{w}^\top\mathbf{x}^{(i)} + \epsilon^{(i)}$$ with $$\epsilon^{(i)} \sim \mathcal{N}(0, \sigma^2)$$, MLE of $$\mathbf{w}$$ coincides with minimizing squared error — linking probabilistic modeling to empirical risk minimization.

![Likelihood surface for a Bernoulli parameter](/assets/figures/day01/pdf0_page020.png)

### 4.2 Properties and regularization as priors

> **MAP estimation** maximizes $$p(\theta|\mathcal{D}) \propto p(\mathcal{D}|\theta)p(\theta)$$. An $$\ell_2$$ prior on $$\mathbf{w}$$ yields ridge regression; a Laplace prior yields Lasso.
{:.lead}

Under regularity conditions, MLE is **consistent** ($$\hat{\theta} \to \theta^\*$$ as $$n \to \infty$$) and asymptotically normal:

$$\sqrt{n}(\hat{\theta} - \theta^\*) \xrightarrow{d} \mathcal{N}(\mathbf{0}, \mathcal{I}^{-1}(\theta^\*)),$$

where $$\mathcal{I}$$ is the Fisher information matrix.

Taking gradients of $$\ell(\theta)$$ and setting to zero often has no closed form — we use gradient ascent:

$$\theta_{t+1} = \theta_t + \eta \nabla_\theta \ell(\theta_t).$$

This connects MLE directly to the optimization algorithms we will implement in the practical.

## Checkpoint summary

Before moving to the practical, confirm you can:

- The gradient $$\nabla f$$ points uphill; descent steps go opposite to it.
- Backpropagation is the multivariate chain rule applied to a computational graph.
- MLE turns learning into maximizing $$\sum_i \log p_\theta(x^{(i)})$$.
- MAP = MLE + log-prior; common priors recover familiar regularizers.
