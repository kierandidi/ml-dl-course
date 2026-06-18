---
layout: post
title: Day 3 - Deep Neural Networks
image: /assets/img/sampling_space.png
accent_image: 
  background: url('/assets/img/sampling_space.png') center/cover
  overlay: false
accent_color: '#ccc'
theme_color: '#ccc'
description: >
  Backpropagation, activation functions, and stochastic optimization with SGD and Adam.
invert_sidebar: true
---

# Day 3 - Deep Neural Networks

### Optional reading for this lesson
- [Goodfellow, Bengio & Courville — Deep Learning](https://www.deeplearningbook.org/), Ch. 6–8
- [UCLxDeepMind DL2020 — Lecture 2](https://www.youtube.com/playlist?list=PL4LiN2XjWqV8Z9qZqZqZqZqZqZqZqZqZq)
- [Karpathy — micrograd](https://github.com/karpathy/micrograd)

### [Slides](/assets/slides/day03.pdf)

### [Practical](/projects/day03-practical/)

Deep networks stack nonlinear transformations to learn hierarchical representations. We derive backpropagation, compare activation functions, and study modern optimizers that make training large models practical.

* toc
{:toc}

## 1. Feedforward Architecture

### 1.1 Layer composition

> An $$L$$-layer **MLP** computes $$\mathbf{h}^{(\ell)} = g^{(\ell)}(\mathbf{W}^{(\ell)}\mathbf{h}^{(\ell-1)} + \mathbf{b}^{(\ell)})$$ with $$\mathbf{h}^{(0)} = \mathbf{x}$$ and output $$\hat{\mathbf{y}} = \mathbf{h}^{(L)}$$.
{:.lead}

Each layer applies an affine map followed by a nonlinear **activation** $$g$$. Without nonlinearity, the stack collapses to a single linear map.

Universal approximation: a one-hidden-layer network with enough units can approximate continuous functions on compact sets arbitrarily well — depth often improves parameter efficiency.

![Deep vs wide network schematic](/assets/figures/day03/pdf0_page005.png)

Parameter count for layers of sizes $$(d_0, d_1, \ldots, d_L)$$:

$$|\theta| = \sum_{\ell=1}^L d_{\ell-1} d_\ell + d_\ell \quad \text{(weights + biases)}.$$

### 1.2 Forward pass notation

> Pre-activations $$\mathbf{z}^{(\ell)} = \mathbf{W}^{(\ell)}\mathbf{h}^{(\ell-1)} + \mathbf{b}^{(\ell)}$$; activations $$\mathbf{h}^{(\ell)} = g(\mathbf{z}^{(\ell)})$$.
{:.lead}

Batch forward pass for $$\mathbf{X} \in \mathbb{R}^{n \times d_0}$$:

$$\mathbf{Z}^{(\ell)} = \mathbf{H}^{(\ell-1)}\mathbf{W}^{(\ell)\top} + \mathbf{1}\mathbf{b}^{(\ell)\top}, \quad \mathbf{H}^{(\ell)} = g(\mathbf{Z}^{(\ell)}).$$

Matrix dimensions must align: $$\mathbf{W}^{(\ell)} \in \mathbb{R}^{d_\ell \times d_{\ell-1}}$$.

Caching $$\mathbf{z}^{(\ell)}$$ and $$\mathbf{h}^{(\ell)}$$ during forward pass is required for efficient backward pass.

## 2. Backpropagation

### 2.1 Output layer gradients

> Given loss $$L$$, define error signal $$\boldsymbol{\delta}^{(L)} = \nabla_{\mathbf{z}^{(L)}} L$$. For softmax + cross-entropy with one-hot $$\mathbf{y}$$, $$\boldsymbol{\delta}^{(L)} = \hat{\mathbf{y}} - \mathbf{y}$$.
{:.lead}

Weight gradient for layer $$\ell$$:

$$\frac{\partial L}{\partial \mathbf{W}^{(\ell)}} = \boldsymbol{\delta}^{(\ell)} \mathbf{h}^{(\ell-1)\top}.$$

This is an outer product of output error and input activation — the foundation of efficient GPU kernels.

![Backprop error flow through layers](/assets/figures/day03/pdf0_page010.png)

### 2.2 Hidden layer backprop

> Errors propagate backward: $$\boldsymbol{\delta}^{(\ell)} = (\mathbf{W}^{(\ell+1)\top}\boldsymbol{\delta}^{(\ell+1)}) \odot g'(\mathbf{z}^{(\ell)})$$ where $$\odot$$ is element-wise product.
{:.lead}

The chain rule multiplies upstream gradient by local derivative. For ReLU, $$g'(z) = \mathbb{1}[z > 0]$$ — dead neurons have zero gradient.

**Vanishing gradients**: if $$|g'(z)| < 1$$ across many layers, $$\boldsymbol{\delta}^{(1)}$$ shrinks exponentially. **Exploding gradients** cause unstable training; **gradient clipping** mitigates:

$$\mathbf{g} \leftarrow \mathbf{g} \cdot \min\!\left(1, \frac{\tau}{\|\mathbf{g}\|}\right).$$

Modern architectures (residual connections, LayerNorm) were designed to keep gradient magnitudes healthy.

## 3. Activation Functions

### 3.1 ReLU family

> **ReLU**: $$g(z) = \max(0, z)$$. **Leaky ReLU**: $$g(z) = z$$ if $$z > 0$$ else $$\alpha z$$. **GELU**: $$g(z) = z\,\Phi(z)$$ where $$\Phi$$ is the standard normal CDF.
{:.lead}

ReLU is cheap and avoids saturation on the positive side but can cause **dead neurons**. Leaky ReLU and **Parametric ReLU** (learnable slope) address this.

**Swish / SiLU**: $$g(z) = z\,\sigma(z)$$ — smooth, non-monotonic, used in many modern architectures.

![Activation shapes compared](/assets/figures/day03/pdf0_page015.png)

Compare saturation regions: sigmoid/tanh squash large inputs, killing gradients in deep nets pre-2012.

### 3.2 Sigmoid, tanh, and softmax

> **Sigmoid** $$\sigma(z) = 1/(1+e^{-z})$$; **tanh** $$= 2\sigma(2z)-1$$ zero-centered. **Softmax** normalizes logits to a probability vector.
{:.lead}

Sigmoid derivative: $$\sigma'(z) = \sigma(z)(1-\sigma(z)) \leq 1/4$$.

Tanh derivative: $$1 - \tanh^2(z)$$, zero-centered so gradients are better conditioned than sigmoid.

Softmax Jacobian is rank-deficient; combined with cross-entropy the gradient simplifies to $$\hat{\mathbf{y}} - \mathbf{y}$$ — always use the fused implementation.

## 4. Optimization: SGD and Adam

### 4.1 Stochastic gradient descent

> **SGD** updates $$\theta_{t+1} = \theta_t - \eta_t \nabla_\theta \hat{L}_B$$ where $$\hat{L}_B$$ is loss on a mini-batch $$B$$ of size $$|B| \ll n$$.
{:.lead}

Mini-batch gradient is an unbiased estimator of full gradient:

$$\mathbb{E}[\nabla \hat{L}_B] = \nabla L, \quad \mathrm{Var}(\nabla \hat{L}_B) \propto \frac{1}{|B|}.$$

**Momentum** accumulates velocity:

$$\mathbf{v}_{t+1} = \beta \mathbf{v}_t + \nabla L, \quad \theta_{t+1} = \theta_t - \eta \mathbf{v}_{t+1}.$$

![SGD trajectory vs full-batch GD](/assets/figures/day03/pdf0_page020.png)

Learning rate schedules: step decay, cosine annealing, warm-up — critical for transformers and large-batch training.

### 4.2 Adam and adaptive methods

> **Adam** maintains per-parameter first moment $$\mathbf{m}_t$$ and second moment $$\mathbf{v}_t$$ (exponential moving averages), with bias correction and update $$\theta_{t+1} = \theta_t - \eta \hat{\mathbf{m}}_t / (\sqrt{\hat{\mathbf{v}}_t} + \epsilon)$$.
{:.lead}

Adam update (element-wise):

$$\begin{aligned}
\mathbf{m}_t &= \beta_1 \mathbf{m}_{t-1} + (1-\beta_1)\mathbf{g}_t \\
\mathbf{v}_t &= \beta_2 \mathbf{v}_{t-1} + (1-\beta_2)\mathbf{g}_t^2 \\
\hat{\mathbf{m}}_t &= \mathbf{m}_t / (1-\beta_1^t), \quad \hat{\mathbf{v}}_t = \mathbf{v}_t / (1-\beta_2^t)
\end{aligned}$$

Default $$\beta_1=0.9$$, $$\beta_2=0.999$$, $$\epsilon=10^{-8}$$. **AdamW** decouples weight decay from the adaptive step — preferred for transformers.

**Weight initialization** (Xavier, He) sets variance of activations stable across layers: He init uses $$\mathcal{N}(0, 2/n_{\mathrm{in}})$$ for ReLU nets.

## Checkpoint summary

Before moving to the practical, confirm you can:

- MLPs compose affine maps and nonlinear activations; depth enables hierarchical features.
- Backprop = chain rule: cache forward values, propagate $$\delta$$ errors backward.
- ReLU and GELU dominate modern nets; watch for dead neurons and gradient health.
- SGD + momentum + Adam/AdamW are the default training recipe; tune $$\eta$$ and batch size.
