#import "../lib.typ": *

#show: course-theme.with(title: [Deep Neural Networks], subtitle: [Day 3 | Aug 2026])

= Day 3: Deep Neural Networks

== Welcome

- *Deep Neural Networks* — From the perceptron to backprop and optimization
- 3 hours lecture + practical
- Slides, notes, and code on the course site

== Outline

- From Shallow to Deep
- Anatomy of a Neural Network
- Backpropagation
- Optimization
- Generalization & Regularization

= 1 · From Shallow to Deep

== 1.1  Why Go Beyond Linear Models

- Linear model $f(x) = w^T x + b$ can only carve the input with a *hyperplane*
- XOR, image classes, language — not linearly separable
- Fix-feature trick: $f(x) = w^T phi(x)$ needs hand-designed $phi$
- Deep nets *learn* the features $phi$ jointly with the classifier
- Composition of simple maps $arrow.r$ rich, reusable representations

== 1.2  Universal Approximation

- One hidden layer + a squashing nonlinearity can approximate *any* continuous function on a compact set
- Formally: for any continuous $f$ and $epsilon > 0$, a wide enough net is within $epsilon$ uniformly
- Existence only — says nothing about *learnability* or *width needed*
- Width may grow exponentially in input dimension
- Motivates depth: a more parameter-efficient route to the same functions

== 1.2  Universal Approximation

#align(center + horizon)[#image("/assets/figures/day03/dnn_universal_approx.png", width: 92%, height: 82%, fit: "contain")]

== 1.3  Depth vs Width

- Some functions need exponential width at depth 1, but linear width with more depth
- Each layer composes & folds the space $arrow.r$ piecewise-linear regions multiply
- Depth = hierarchy: edges $arrow.r$ parts $arrow.r$ objects
- Trade-off: depth helps expressivity but complicates optimization

= 2 · Anatomy of a Neural Network

== 2.1  The Multilayer Perceptron

- Layer $l$: $z^((l)) = W^((l)) a^((l-1)) + b^((l))$, then $a^((l)) = g(z^((l)))$
- $a^((0)) = x$ (input), $hat(y) = a^((L))$ (output)
- Parameters $theta = {W^((l)), b^((l))}_(l=1)^L$
- A *computational graph*: nodes = operations, edges = tensors
- Width = neurons per layer; depth = number of layers $L$

== 2.1  The Multilayer Perceptron

#align(center + horizon)[#image("/assets/figures/day03/dnn_compgraph.png", width: 92%, height: 82%, fit: "contain")]

== 2.2  Activation Functions

- Sigmoid $sigma(z) = 1\\/(1 + e^(-z))$ — saturates, gradients vanish
- Tanh — zero-centered but still saturates
- ReLU $max(0, z)$ — cheap, sparse, no saturation for $z>0$ (default)
- Leaky ReLU / GELU / SiLU — keep a gradient for $z<0$
- Without a nonlinearity, stacked layers collapse to one linear map

== 2.3  The Forward Pass

- Propagate $x arrow.r a^((1)) arrow.r dots.h arrow.r a^((L)) = hat(y)$
- Classification head: softmax $p_k = e^(z_k)\\/sum_j e^(z_j)$
- Loss compares $hat(y)$ to target $y$ (cross-entropy, MSE)
- Cache the $z^((l))$, $a^((l))$ — backprop reuses them

= 3 · Backpropagation

== 3.1  Learning = Minimizing a Loss

- Objective $L(theta) = 1/m sum_(i=1)^m ell(f(x_i; theta), y_i)$
- Find $theta^* = "arg min"_theta L(theta)$ by gradient descent
- Need $nabla_theta L$ — millions of partials, computed *efficiently*
- Backprop = reverse-mode autodiff = chain rule + caching

== 3.1  Learning = Minimizing a Loss

#align(center + horizon)[#image("/assets/figures/day03/dnn_train_objective.png", width: 92%, height: 82%, fit: "contain")]

== 3.2  The Chain Rule on a Graph

- Scalar chain rule: $(dif)/(dif x) f(g(x)) = f'(g(x)) g'(x)$
- Vector form: Jacobians multiply right-to-left
- Define the error signal $delta^((l)) = (partial L)/(partial z^((l)))$
- One forward sweep (values) + one backward sweep (gradients)
- Cost of gradient $approx$ cost of forward pass

== 3.3  Derivation: The Four Backprop Equations

- Output layer: $delta^((L)) = nabla_a ell dot.o g'(z^((L)))$
- Recurse: $delta^((l)) = (W^((l+1)))^T delta^((l+1)) dot.o g'(z^((l)))$
- Weight grad: $(partial L)/(partial W^((l))) = delta^((l)) (a^((l-1)))^T$
- Bias grad: $(partial L)/(partial b^((l))) = delta^((l))$
- Each $dot.o$ is elementwise; full derivation in the notes

== 3.4  Vanishing & Exploding Gradients

- $delta^((l))$ is a product of many Jacobians $arrow.r$ can shrink/blow up
- Sigmoid/tanh: $g' <= 1\\/4 arrow.r$ deep gradients vanish
- Fixes: ReLU, careful init (He/Xavier), residual connections
- Normalization (Batch/Layer) keeps activations well-scaled
- Gradient clipping bounds explosions (esp. RNNs, Day 5)

= 4 · Optimization

== 4.1  Gradient Descent is Steepest Descent

- $theta_(k+1) = theta_k - eta nabla L(theta_k)$
- $-nabla L$ = direction of steepest local decrease
- 1st-order Taylor: $L(theta + d) approx L(theta) + nabla L^T d$
- Step size $eta$ (learning rate) is the key knob
- Too big $arrow.r$ diverge; too small $arrow.r$ crawl

== 4.1  Gradient Descent is Steepest Descent

#align(center + horizon)[#image("/assets/figures/day03/opt_steepest.png", width: 92%, height: 82%, fit: "contain")]

== 4.2  The Narrow-Valley Problem

- Ill-conditioned loss: steep in one direction, flat in another
- GD zig-zags across the valley, crawls along it
- Conditioning $kappa = lambda_max\\/lambda_min$ of the Hessian sets the rate
- A single $eta$ cannot suit both directions
- Momentum & adaptive methods damp the zig-zag

== 4.2  The Narrow-Valley Problem

#align(center + horizon)[#image("/assets/figures/day03/opt_narrow_valley.png", width: 92%, height: 82%, fit: "contain")]

== 4.3  SGD, Momentum & Adam

- SGD: estimate $nabla L$ on a *minibatch* $arrow.r$ cheap, noisy steps
- Momentum: $v_(k+1) = beta v_k + nabla L$, $theta_(k+1) = theta_k - eta v_(k+1)$
- Adam: per-parameter step from running 1st/2nd moments
- Noise in SGD acts as implicit regularization
- Schedules: warmup then decay (cosine, step)

== 4.3  SGD, Momentum & Adam

#align(center + horizon)[#image("/assets/figures/day03/opt_methods.png", width: 92%, height: 82%, fit: "contain")]

== 4.4  Derivation: Momentum as a Heavy Ball

- Plain GD: a massless particle on the loss surface
- Momentum adds *inertia*: velocity accumulates past gradients
- $v_k = sum_(j<=k) beta^(k-j) nabla L(theta_j)$ — EMA of gradients
- Consistent directions reinforce; oscillations cancel
- Effective step $approx eta\\/(1-beta)$ along a steady slope

= 5 · Generalization & Regularization

== 5.1  Overfitting & the Bias-Variance View

- Training error falls with capacity; test error is U-shaped
- Underfit = high bias; overfit = high variance
- Goal: lowest *test* risk, not zero training risk
- Validation set picks capacity & hyperparameters
- Deep nets: more data and regularization push the sweet spot right

== 5.1  Overfitting & the Bias-Variance View

#align(center + horizon)[#image("/assets/figures/day03/dnn_overfitting.png", width: 92%, height: 82%, fit: "contain")]

== 5.2  Weight Decay & Dropout

- $L_2$ / weight decay: add $lambda \\|theta\\|^2$ $arrow.r$ smaller weights
- Equivalent to a Gaussian prior on weights (MAP view)
- Dropout: randomly zero units $arrow.r$ ensemble of subnetworks
- $L_1$ encourages sparsity (feature selection)
- Data augmentation = free, task-specific regularization

== 5.3  Normalization & Early Stopping

- BatchNorm: standardize activations per minibatch
- LayerNorm: standardize per example (default in Transformers, Day 5)
- Stabilizes & speeds training; mild regularizing effect
- Early stopping: halt when validation loss climbs
- Recipe: ReLU + good init + normalization + Adam + decay

== Summary

- Day 3: *Deep Neural Networks*
- From the perceptron to backprop and optimization
- Questions welcome — practical follows

== Questions?

#align(center + horizon)[
  #text(size: 44pt, weight: "bold", fill: course-primary)[Questions?]

  #v(1em)
  Practical session follows — see course site.
]
