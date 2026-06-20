---
layout: post
title: Day 3 - Deep Neural Networks
image: /assets/img/lessons/day03.png
accent_image: 
  background: url('/assets/img/lessons/day03.png') center/cover
  overlay: false
accent_color: '#ccc'
theme_color: '#ccc'
description: >
  Multilayer perceptrons, backpropagation, and the optimization and regularization that make them learn.
invert_sidebar: true
---

# Day 3 - Deep Neural Networks

### Optional reading for this lesson
- [UCL x DeepMind DL2020 — L2: Neural Networks Foundations](https://www.youtube.com/watch?v=FBggC-XVF4M)
- [UCL x DeepMind DL2020 — L5: Optimization for Machine Learning](https://www.youtube.com/watch?v=kVU8zTI-Od0)
- [Mathematics for Machine Learning](https://mml-book.github.io/), Ch. 5 (vector calculus)
- [Goodfellow, Bengio & Courville — Deep Learning](https://www.deeplearningbook.org/), Ch. 6–8

### [Slides](/assets/slides/day03.pdf)

### Exercise

[Download the notebook](/notebooks/practicals/day03.ipynb) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kierandidi/ml-dl-course/blob/main/notebooks/practicals/day03.ipynb)

Yesterday we fit linear and logistic models; today we make them deep. A neural network is nothing more than a *composition* of simple parameterized maps, but composition buys an enormous amount: the ability to learn features rather than hand-craft them. We will build the multilayer perceptron from scratch, derive backpropagation as reverse-mode automatic differentiation, understand why deep gradients vanish or explode, and study the optimization and regularization tricks that turn a pile of matrix multiplies into a model that generalizes.

* toc
{:toc}

## 1. From Shallow to Deep

### 1.1 Why linear models are not enough

> A **linear model** $$f(\boldsymbol{x}) = \boldsymbol{w}^\top\boldsymbol{x} + b$$ splits the input space with a single hyperplane. A **neural network** replaces the fixed features by a learned composition $$f(\boldsymbol{x}) = g^{(L)}\!\circ\cdots\circ g^{(1)}(\boldsymbol{x}),$$ where each $$g^{(l)}$$ is an affine map followed by a nonlinearity.
{:.lead}

**Why this matters.** The whole reason deep learning exists is that most interesting decision boundaries are *not* hyperplanes. The textbook example is XOR: four points $$(0,0),(1,1)\to 0$$ and $$(0,1),(1,0)\to 1$$ cannot be separated by any single line. No choice of $$\boldsymbol{w},b$$ in $$f(\boldsymbol{x})=\boldsymbol{w}^\top\boldsymbol{x}+b$$ works.

The classical workaround is a **feature map** $$\boldsymbol{\phi}$$: instead of feeding raw $$\boldsymbol{x}$$, we feed $$\boldsymbol{\phi}(\boldsymbol{x})$$ and learn $$f(\boldsymbol{x})=\boldsymbol{w}^\top\boldsymbol{\phi}(\boldsymbol{x})$$. For XOR, the single extra feature $$\phi_3 = x_1 x_2$$ makes the problem linearly separable. The catch: *someone has to design* $$\boldsymbol{\phi}$$. For images, audio, or text this is hopeless by hand.

**The deep-learning idea** is to *learn* $$\boldsymbol{\phi}$$ from data by stacking simple maps and optimizing all of them jointly. Each layer is an affine transform followed by an elementwise nonlinearity:

$$\boldsymbol{a}^{(l)} = g\big(W^{(l)}\boldsymbol{a}^{(l-1)} + \boldsymbol{b}^{(l)}\big), \qquad \boldsymbol{a}^{(0)} = \boldsymbol{x}.$$

The early layers play the role of $$\boldsymbol{\phi}$$ (representation), the last layer plays the role of $$\boldsymbol{w}$$ (classifier), and gradient descent tunes them together.

### 1.2 Universal approximation and the role of depth

> **Universal approximation theorem.** For any continuous function $$f:K\subset\mathbb{R}^n\to\mathbb{R}$$ on a compact set and any $$\varepsilon>0$$, there is a one-hidden-layer network $$\hat f(\boldsymbol{x}) = \sum_{j=1}^{N} v_j\,\sigma(\boldsymbol{w}_j^\top\boldsymbol{x}+b_j)$$ with $$\sup_{\boldsymbol{x}\in K}\vert f(\boldsymbol{x})-\hat f(\boldsymbol{x})\vert <\varepsilon.$$
{:.lead}

![A single hidden layer with a squashing nonlinearity is dense in the space of continuous functions — but the theorem says nothing about how *wide* the layer must be (UCL L2).](/assets/figures/day03/dnn_universal_approx.png)

This is a reassuring **existence** result: neural networks are *dense* in the space of continuous functions, so there is no function we are fundamentally unable to represent. But read the fine print:

- It says a good approximator **exists**; it does not say gradient descent will **find** it.
- The required width $$N$$ can grow **exponentially** with the input dimension $$n$$.

**Why depth helps.** Consider building a bump: with one hidden sigmoid layer you can make a step, and a difference of two steps makes a bump; summing bumps tiles any 1-D function. In high dimensions this tiling needs exponentially many bumps. Depth changes the accounting. Each additional layer can *compose* and *fold* the regions created by the previous one, so the number of linear regions a network can express grows **multiplicatively** with depth but only **additively** with width. A function that needs width $$2^k$$ at depth 1 may need only width $$O(k)$$ at depth $$k$$.

There is also a representational story that matches intuition: a deep vision network learns edges, then textures, then object parts, then objects — a **hierarchy** of features in which later concepts reuse earlier ones. Depth is what makes that reuse possible.

## 2. Anatomy of a Neural Network

### 2.1 The multilayer perceptron as a computational graph

> A **multilayer perceptron (MLP)** with $$L$$ layers computes, for $$l=1,\dots,L$$, $$\boldsymbol{z}^{(l)} = W^{(l)}\boldsymbol{a}^{(l-1)} + \boldsymbol{b}^{(l)},\qquad \boldsymbol{a}^{(l)} = g\big(\boldsymbol{z}^{(l)}\big),$$ with $$\boldsymbol{a}^{(0)}=\boldsymbol{x}$$ and output $$\hat{\boldsymbol{y}}=\boldsymbol{a}^{(L)}$$.
{:.lead}

![A neural network is a computational graph: nodes are operations, edges carry tensors. Reverse-mode autodiff walks this graph backwards (UCL L2).](/assets/figures/day03/dnn_compgraph.png)

It pays to think of the network as a **computational graph** rather than a formula. Nodes are primitive operations (matmul, add, ReLU, softmax); edges carry tensors. Two facts about graphs drive everything that follows:

1. The **forward pass** is just a topological evaluation of the graph: compute each node once its inputs are ready, caching $$\boldsymbol{z}^{(l)}$$ and $$\boldsymbol{a}^{(l)}$$.
2. The **backward pass** (backprop) walks the same graph in reverse, multiplying local Jacobians. Because each cached value is reused, computing *all* gradients costs about the same as one forward pass.

**Shapes.** If layer $$l$$ has $$n_l$$ units, then $$W^{(l)}\in\mathbb{R}^{n_l\times n_{l-1}}$$ and $$\boldsymbol{b}^{(l)}\in\mathbb{R}^{n_l}$$. The full parameter vector is $$\theta=\{W^{(l)},\boldsymbol{b}^{(l)}\}_{l=1}^L$$, often tens of millions of numbers — yet backprop gives us every partial derivative in one sweep.

**Worked example (a tiny forward pass).** Take a 2–2–1 network with
$$W^{(1)}=\begin{pmatrix}1 & -1\\ 1 & 1\end{pmatrix},\ \boldsymbol{b}^{(1)}=\begin{pmatrix}0\\0\end{pmatrix},\ g=\mathrm{ReLU},\quad W^{(2)}=\begin{pmatrix}1 & 1\end{pmatrix},\ b^{(2)}=0.$$
For $$\boldsymbol{x}=(1,0)^\top$$: $$\boldsymbol{z}^{(1)}=(1,1)^\top$$, $$\boldsymbol{a}^{(1)}=\mathrm{ReLU}(1,1)=(1,1)^\top$$, and $$\hat y = 1\cdot1+1\cdot1=2.$$ For $$\boldsymbol{x}=(0,0)^\top$$ the output is $$0$$. The hidden ReLUs have carved the plane into regions on which the network is locally linear.

### 2.2 Activation functions

> An **activation function** $$g$$ is the elementwise nonlinearity applied after each affine layer. Without it, $$W^{(2)}(W^{(1)}\boldsymbol{x}+\boldsymbol{b}^{(1)})+\boldsymbol{b}^{(2)}$$ is *still affine*, so the whole network would collapse to a single linear map.
{:.lead}

The nonlinearity is what makes depth meaningful. The common choices and their trade-offs:

- **Sigmoid** $$\sigma(z)=\dfrac{1}{1+e^{-z}}$$, with $$\sigma'(z)=\sigma(z)(1-\sigma(z))\le \tfrac14$$. Smooth and probabilistic, but it **saturates**: for large $$\vert z\vert $$ the gradient is ~0, which stalls learning in deep stacks.
- **Tanh** $$\tanh(z)$$ is a rescaled sigmoid; zero-centered (nicer for optimization) but still saturates.
- **ReLU** $$\mathrm{ReLU}(z)=\max(0,z)$$, with derivative $$\mathbf{1}[z>0]$$. Cheap, non-saturating for $$z>0$$, and induces sparse activations. The default. Its weakness is "dead" units that get stuck at $$z<0$$.
- **Leaky ReLU / GELU / SiLU** keep a small gradient for $$z<0$$, fixing dead units. GELU $$z\,\Phi(z)$$ (where $$\Phi$$ is the Gaussian CDF) is standard in Transformers.

**Why saturation hurts.** Backprop multiplies the activation derivative at every layer (next section). If each factor is $$\le\tfrac14$$ (sigmoid), a 10-layer network attenuates the gradient by up to $$4^{-10}\approx 10^{-6}$$. ReLU's derivative is exactly $$1$$ on the active region, which is precisely why it rescued deep training.

### 2.3 The forward pass and output heads

> The **forward pass** evaluates the network left-to-right to produce a prediction $$\hat{\boldsymbol{y}}$$, which is then scored against the target by a **loss** $$\ell(\hat{\boldsymbol{y}},\boldsymbol{y})$$.
{:.lead}

The last layer is shaped by the task:

- **Regression**: linear output, mean-squared-error loss $$\ell=\tfrac12\Vert \hat{\boldsymbol{y}}-\boldsymbol{y}\Vert ^2$$.
- **Classification**: a **softmax** head $$p_k=\dfrac{e^{z_k}}{\sum_j e^{z_j}}$$ turning logits into a probability vector, scored by **cross-entropy** $$\ell=-\sum_k y_k\log p_k$$.

A small miracle makes classification training clean: the gradient of softmax-cross-entropy with respect to the logits is simply

$$\frac{\partial \ell}{\partial \boldsymbol{z}^{(L)}} = \boldsymbol{p} - \boldsymbol{y},$$

the difference between predicted and true distributions. This is the error signal that backprop will propagate. During the forward pass we **cache** every $$\boldsymbol{z}^{(l)}$$ and $$\boldsymbol{a}^{(l)}$$ because the backward pass needs them — trading memory for a huge saving in compute.

## 3. Backpropagation

### 3.1 Learning as loss minimization

> **Training** chooses parameters to minimize the empirical risk $$L(\theta) = \frac{1}{m}\sum_{i=1}^m \ell\big(f(\boldsymbol{x}_i;\theta),\boldsymbol{y}_i\big),$$ by following the negative gradient $$-\nabla_\theta L$$.
{:.lead}

![The standard training objective: an average loss over the data, minimized over the network's parameters (UCL L5).](/assets/figures/day03/dnn_train_objective.png)

Everything reduces to computing $$\nabla_\theta L$$ — possibly hundreds of millions of partial derivatives. The naive options are both unusable:

- **Symbolic differentiation** produces gigantic expressions that explode in size.
- **Numerical differentiation** $$\frac{\partial L}{\partial\theta_i}\approx\frac{L(\theta+h\boldsymbol{e}_i)-L(\theta)}{h}$$ costs one forward pass *per parameter* and is numerically fragile.

**Backpropagation** computes the exact gradient of all parameters in a single backward sweep, at the cost of one extra forward-pass-worth of work. It is reverse-mode automatic differentiation applied to the network's computational graph. (Numerical differentiation is still useful for *checking* an implementation — "gradient checking".)

### 3.2 The chain rule on a computational graph

> Define the **error signal** of layer $$l$$ as $$\boldsymbol{\delta}^{(l)} := \frac{\partial L}{\partial \boldsymbol{z}^{(l)}}.$$ Backprop is the recursion that computes $$\boldsymbol{\delta}^{(l)}$$ from $$\boldsymbol{\delta}^{(l+1)}$$ using the layer's local Jacobian.
{:.lead}

Recall the scalar chain rule $$\frac{\mathrm d}{\mathrm dx}f(g(x)) = f'(g(x))\,g'(x)$$. For vector-valued maps the derivatives become **Jacobians**, and the chain rule says they **multiply right-to-left**. Reverse mode evaluates this product starting from the scalar loss, so at every stage we carry a *vector* (the error signal) rather than a full Jacobian matrix — that is the efficiency win.

Concretely, the loss depends on $$\boldsymbol{z}^{(l)}$$ only through $$\boldsymbol{z}^{(l+1)} = W^{(l+1)}g(\boldsymbol{z}^{(l)}) + \boldsymbol{b}^{(l+1)}$$. Applying the chain rule through this single hop gives the backward recursion derived next. Two sweeps total:

1. **Forward** — compute and cache $$\boldsymbol{z}^{(l)},\boldsymbol{a}^{(l)}$$.
2. **Backward** — compute $$\boldsymbol{\delta}^{(L)},\dots,\boldsymbol{\delta}^{(1)}$$ and read off parameter gradients.

### 3.3 Derivation: the four equations of backprop

> Backpropagation is four equations: an output-layer error, a recursion, and the two parameter gradients. We derive each from the chain rule.
{:.lead}

Write $$g'(\boldsymbol{z}^{(l)})$$ for the elementwise derivative and $$\odot$$ for the elementwise (Hadamard) product.

**(1) Output error.** The loss sees $$\boldsymbol{z}^{(L)}$$ through $$\boldsymbol{a}^{(L)}=g(\boldsymbol{z}^{(L)})$$, so by the chain rule, componentwise $$\delta^{(L)}_j = \frac{\partial L}{\partial a^{(L)}_j}\,g'(z^{(L)}_j)$$, i.e.

$$\boldsymbol{\delta}^{(L)} = \textcolor{teal}{\nabla_{\boldsymbol{a}}\ell}\;\odot\;\textcolor{purple}{g'(\boldsymbol{z}^{(L)})}.$$

(For softmax + cross-entropy this collapses to the clean $$\boldsymbol{\delta}^{(L)}=\boldsymbol{p}-\boldsymbol{y}$$.)

**(2) Backward recursion.** Since $$\boldsymbol{z}^{(l+1)} = W^{(l+1)}\boldsymbol{a}^{(l)}+\boldsymbol{b}^{(l+1)}$$ and $$\boldsymbol{a}^{(l)}=g(\boldsymbol{z}^{(l)})$$, the chain rule through this hop gives

$$\boldsymbol{\delta}^{(l)} = \frac{\partial L}{\partial \boldsymbol{z}^{(l)}} = \underbrace{\Big(\tfrac{\partial \boldsymbol{z}^{(l+1)}}{\partial \boldsymbol{a}^{(l)}}\Big)^{\!\top}}_{(W^{(l+1)})^\top}\,\boldsymbol{\delta}^{(l+1)} \;\odot\; g'(\boldsymbol{z}^{(l)}) = \textcolor{teal}{\big(W^{(l+1)}\big)^{\top}\boldsymbol{\delta}^{(l+1)}}\;\odot\;\textcolor{purple}{g'(\boldsymbol{z}^{(l)})}.$$

**(3) Weight gradient.** Because $$z^{(l)}_j = \sum_k W^{(l)}_{jk}a^{(l-1)}_k + b^{(l)}_j$$, we have $$\frac{\partial z^{(l)}_j}{\partial W^{(l)}_{jk}} = a^{(l-1)}_k$$, hence

$$\frac{\partial L}{\partial W^{(l)}} = \boldsymbol{\delta}^{(l)}\big(\boldsymbol{a}^{(l-1)}\big)^{\top}\quad(\text{an outer product}).$$

**(4) Bias gradient.** Since $$\frac{\partial z^{(l)}_j}{\partial b^{(l)}_j}=1$$,

$$\frac{\partial L}{\partial \boldsymbol{b}^{(l)}} = \boldsymbol{\delta}^{(l)}.$$

These four lines, applied from $$l=L$$ down to $$l=1$$, *are* backpropagation. Note the recurring structure: every gradient is a product of the upstream error and a cached forward value.

### 3.4 Vanishing and exploding gradients

> Because $$\boldsymbol{\delta}^{(l)}$$ is a product of many factors $$(W^{(l+1)})^\top$$ and $$g'(\boldsymbol{z}^{(l)})$$, its magnitude can shrink toward zero (**vanishing**) or blow up (**exploding**) as it propagates back through a deep network.
{:.lead}

Unrolling the recursion, the error at layer $$l$$ contains a product of $$L-l$$ Jacobian-like factors. If a typical factor has norm $$<1$$, the product decays geometrically; if $$>1$$, it grows geometrically.

**Vanishing.** With sigmoid/tanh, $$g'\le\tfrac14$$, so deep gradients are crushed and early layers barely learn. Cures:
- **ReLU activations** ($$g'=1$$ on the active region).
- **Careful initialization** that preserves variance across layers: He init $$\mathrm{Var}(W)=2/n_{l-1}$$ for ReLU, Xavier/Glorot $$2/(n_{l-1}+n_l)$$ for tanh.
- **Residual connections** $$\boldsymbol{a}^{(l)}=\boldsymbol{a}^{(l-1)}+\mathcal{F}(\boldsymbol{a}^{(l-1)})$$, which add an identity path so gradients flow undiminished.
- **Normalization** layers that re-standardize activations.

**Exploding.** Common in recurrent nets (Day 5) where the same matrix is applied many times. The standard fix is **gradient clipping**: rescale $$\nabla L$$ whenever $$\Vert \nabla L\Vert $$ exceeds a threshold. Together, good activations + init + normalization + residuals are what make networks with hundreds of layers trainable.

## 4. Optimization

### 4.1 Gradient descent is steepest descent

> **Gradient descent** updates $$\theta_{k+1} = \theta_k - \eta\,\nabla L(\theta_k)$$. The direction $$-\nabla L$$ is the steepest local decrease, and $$\eta$$ is the **learning rate**.
{:.lead}

![Among all unit directions, $$-\nabla L$$ gives the greatest first-order decrease; how far we can trust it depends on how smooth $$L$$ is (UCL L5).](/assets/figures/day03/opt_steepest.png)

The justification is the first-order Taylor expansion around the current point:

$$L(\theta + \boldsymbol{d}) \approx L(\theta) + \nabla L(\theta)^\top \boldsymbol{d}.$$

To decrease $$L$$ fastest for a fixed step length, we minimize $$\nabla L^\top\boldsymbol{d}$$ over unit $$\boldsymbol{d}$$; Cauchy–Schwarz makes the minimizer $$\boldsymbol{d}\propto-\nabla L$$. The catch is that the linear model is only trustworthy in a small neighborhood — the smoother (less curved) $$L$$ is, the larger a step $$\eta$$ we can take. Too large and we overshoot or diverge; too small and progress is glacial. The learning rate is the single most important hyperparameter in deep learning.

### 4.2 Ill-conditioning: the narrow-valley problem

> When the loss is **ill-conditioned** — much steeper in some directions than others — a single learning rate cannot serve all directions, and gradient descent **zig-zags**. Conditioning is measured by $$\kappa = \lambda_{\max}/\lambda_{\min}$$ of the Hessian.
{:.lead}

![Gradient descent in a narrow valley: a large learning rate bounces across the steep walls, a small one crawls along the flat floor (UCL L5).](/assets/figures/day03/opt_narrow_valley.png)

Picture a long, narrow valley. The gradient points mostly across the valley (the steep direction), so steps bounce from wall to wall while making little progress along the floor (the flat direction we actually want to travel). Quantitatively, for a quadratic with Hessian eigenvalues $$\lambda_{\min}\le\cdots\le\lambda_{\max}$$, the stable learning rate is bounded by $$\eta<2/\lambda_{\max}$$ (set by the steep direction), while progress along the flat direction goes as $$(1-\eta\lambda_{\min})$$. The number of iterations to converge scales with the **condition number** $$\kappa=\lambda_{\max}/\lambda_{\min}$$.

Neural-network losses are badly conditioned, so plain GD is slow. The next methods attack exactly this: momentum averages out the cross-valley oscillation, and adaptive methods give each coordinate its own effective step size.

### 4.3 SGD, momentum, and adaptive methods

> **Stochastic gradient descent (SGD)** replaces the full-data gradient by a minibatch estimate. **Momentum** and **Adam** accelerate it by reusing the history of past gradients.
{:.lead}

![Trajectories in the narrow valley: plain gradient descent zig-zags, momentum smooths the path, and second-order methods cut straight to the minimum (UCL L5).](/assets/figures/day03/opt_methods.png)

**SGD.** Computing $$\nabla L$$ on the full dataset every step is wasteful. SGD uses a random minibatch $$\mathcal{B}$$: $$\nabla L\approx\frac{1}{\vert \mathcal{B}\vert }\sum_{i\in\mathcal{B}}\nabla\ell_i$$. This is an *unbiased* estimate (Day 1's Monte-Carlo idea), so steps are cheap and noisy. The noise is not just tolerable — it helps the model escape sharp minima and acts as implicit regularization.

**Momentum.** Maintain a velocity that is an exponential moving average of gradients:

$$\boldsymbol{v}_{k+1} = \beta\,\boldsymbol{v}_k + \nabla L(\theta_k), \qquad \theta_{k+1} = \theta_k - \eta\,\boldsymbol{v}_{k+1}.$$

**Adam** keeps running estimates of the first moment $$\boldsymbol{m}$$ (mean) and second moment $$\boldsymbol{v}$$ (uncentered variance) of the gradient, then steps with $$\theta\leftarrow\theta-\eta\,\hat{\boldsymbol{m}}/(\sqrt{\hat{\boldsymbol{v}}}+\epsilon)$$. Dividing by $$\sqrt{\hat{\boldsymbol{v}}}$$ gives each parameter its own adaptive step, directly addressing ill-conditioning. In practice: AdamW (Adam + decoupled weight decay) with a warmup-then-cosine-decay schedule is a strong default.

### 4.4 Derivation: momentum as a heavy ball

> Momentum can be read as giving the optimizer **inertia**: the velocity is an exponentially weighted sum of all past gradients, so consistent directions accelerate while oscillations cancel.
{:.lead}

Unroll the velocity recursion $$\boldsymbol{v}_{k+1}=\beta\boldsymbol{v}_k+\nabla L(\theta_k)$$ from $$\boldsymbol{v}_0=\mathbf{0}$$:

$$\boldsymbol{v}_{k+1} = \sum_{j=0}^{k}\beta^{\,k-j}\,\nabla L(\theta_j).$$

So the update is a **geometrically weighted average** of the gradient history, with decay $$\beta$$ (typically $$0.9$$). Two consequences:

- **Steady slope.** If the gradient is roughly constant $$\boldsymbol{g}$$ over many steps, then $$\boldsymbol{v}\to\boldsymbol{g}\sum_{i\ge0}\beta^i = \boldsymbol{g}/(1-\beta)$$, so the effective step is amplified to $$\eta/(1-\beta)$$ — momentum *accelerates* along consistent directions. With $$\beta=0.9$$ that is a $$10\times$$ boost.
- **Oscillation.** In the steep cross-valley direction the gradient flips sign each step; consecutive terms in the sum partially **cancel**, damping the zig-zag.

The name "heavy ball" comes from the physical analogy: plain GD is a massless particle that instantly follows the local slope, while momentum is a ball with mass rolling on the loss surface — its inertia carries it smoothly along the valley floor and through small bumps. Nesterov's variant evaluates the gradient at the *look-ahead* point $$\theta_k-\eta\beta\boldsymbol{v}_k$$ and enjoys a better worst-case convergence rate.

## 5. Generalization and Regularization

### 5.1 Overfitting and the bias–variance view

> **Generalization** is performance on unseen data. As model capacity grows, training error falls monotonically while test error is **U-shaped**: it first falls (less bias) then rises (more variance).
{:.lead}

![Training risk keeps dropping with capacity, but test risk is U-shaped; regularization and more data move the sweet spot to the right (UCL L2).](/assets/figures/day03/dnn_overfitting.png)

The decomposition of expected test error into **bias** (error from too-simple a model) plus **variance** (sensitivity to the particular training sample) plus irreducible noise makes the trade-off precise. A model with too little capacity *underfits* (high bias); one with too much *overfits* (high variance), memorizing noise in the training set.

The practical consequence is that **zero training error is not the goal** — low *test* error is. We hold out a **validation set** to estimate test performance and to choose capacity and hyperparameters, keeping a final **test set** untouched for an honest estimate. (Modern over-parameterized networks complicate the classical picture — the "double descent" phenomenon — but the operational advice, regularize and validate, is unchanged.)

### 5.2 Weight decay, dropout, and data augmentation

> **Regularization** is any modification that reduces test error at the possible cost of training error. The workhorses are weight penalties, dropout, and data augmentation.
{:.lead}

**Weight decay ($$L_2$$).** Add $$\frac{\lambda}{2}\Vert \theta\Vert ^2$$ to the loss. The gradient gains a term $$\lambda\theta$$, so each step shrinks the weights: $$\theta\leftarrow(1-\eta\lambda)\theta-\eta\nabla L$$. From a Bayesian angle this is a **Gaussian prior** on the weights — the MAP estimate of $$\theta$$ under $$\theta\sim\mathcal{N}(0,\tau^2 I)$$ (Day 2's connection between regularization and priors).

**$$L_1$$ penalty** $$\lambda\Vert \theta\Vert _1$$ drives weights exactly to zero, giving sparse, interpretable models (feature selection).

**Dropout.** During training, randomly zero each unit with probability $$p$$. This prevents fragile co-adaptation between units and is equivalent to training an **exponential ensemble** of sub-networks that share weights; at test time we use all units and scale by $$1-p$$.

**Data augmentation.** Apply label-preserving transformations (crops, flips, color jitter for images; token masking for text). It enlarges the effective dataset for free and bakes in the invariances we want the model to have — often the single most effective regularizer.

### 5.3 Normalization and early stopping

> **Normalization layers** keep activations well-scaled across depth, and **early stopping** halts training when validation loss starts to climb.
{:.lead}

**Batch normalization** standardizes each activation over the minibatch, $$\hat z = (z-\mu_{\mathcal{B}})/\sqrt{\sigma_{\mathcal{B}}^2+\epsilon}$$, then applies a learned scale and shift $$\gamma\hat z+\beta$$. It smooths the loss landscape, allows larger learning rates, and has a mild regularizing effect via batch noise. **Layer normalization** does the same statistics *per example* across features instead of across the batch; it is the default in Transformers (Day 5) and in any setting with small or variable batch sizes.

**Early stopping** monitors validation loss and stops at its minimum, keeping the best checkpoint. It is a cheap, implicit regularizer — limiting optimization time limits how far weights can grow, much like an $$L_2$$ penalty.

**A reliable recipe.** ReLU/GELU activations + He/Xavier initialization + a normalization layer + residual connections + AdamW with weight decay + a warmup–cosine learning-rate schedule + early stopping. This combination is what reliably trains the deep architectures we turn to next: convolutional networks (Day 4) and sequence models (Day 5).

## Checkpoint summary

Before moving to the practical, confirm you can:

- Explain why a nonlinearity is essential and what the universal approximation theorem does and does not guarantee.
- Write down the forward pass of an MLP and compute it by hand for a tiny network.
- Derive the four backpropagation equations from the chain rule and explain why a gradient costs about one forward pass.
- Diagnose vanishing/exploding gradients and list concrete fixes (ReLU, init, residuals, normalization, clipping).
- Contrast SGD, momentum, and Adam, and explain how each addresses the narrow-valley (ill-conditioning) problem.
- Choose appropriate regularization (weight decay, dropout, augmentation, early stopping) and justify it via the bias–variance trade-off.
