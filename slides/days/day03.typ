#import "../lib.typ": *

#show: course-theme.with(title: [Deep Neural Networks], subtitle: [Day 3 | Aug 2026])

= Day 3: Deep Neural Networks

== Welcome

- *Deep Neural Networks* — Backpropagation, activations, optimizers
- 3 hours lecture + practical
- Slides, notes, and code on the course site

== Outline

- From Linear to Deep
- Backpropagation
- Training Loop
- Optimizers

= From Linear to Deep

== Limitations of Linear Models

- XOR and non-linearly separable data
- Need composed non-linear transformations
- Universal approximation intuition

== L1 introduct 00

#align(center)[#image("/assets/figures/day03/L1_introduct_00.png", width: 92%)]

#text(size: 14pt, fill: gray)[From Linear to Deep — Limitations of Linear Models (source: course materials)]

== Layer Composition

- $h^(l) = sigma(W^(l) h^(l-1) + b^(l))$
- Depth vs width tradeoffs
- Parameter count scales with layer sizes

== L1 introduct 06

#align(center)[#image("/assets/figures/day03/L1_introduct_06.png", width: 92%)]

#text(size: 14pt, fill: gray)[From Linear to Deep — Layer Composition (source: course materials)]

== Activations

- ReLU: $max(0, z)$ — sparse, fast
- Sigmoid / tanh — saturating, vanishing gradients
- GELU, SiLU in modern transformers

== L1 introduct 08

#align(center)[#image("/assets/figures/day03/L1_introduct_08.png", width: 92%)]

#text(size: 14pt, fill: gray)[From Linear to Deep — Activations (source: course materials)]

== Forward Pass

- Cache intermediates for backward pass
- Batching: tensor shape $(N, d)$
- Numerical stability: log-sum-exp trick

== L5- CNN 00

#align(center)[#image("/assets/figures/day03/L5- CNN_00.png", width: 92%)]

#text(size: 14pt, fill: gray)[From Linear to Deep — Forward Pass (source: course materials)]

= Backpropagation

== Computational Graph

- Nodes = ops; edges = tensors
- Local gradients multiply via chain rule
- Reverse-mode AD = backprop

== L5- CNN 02

#align(center)[#image("/assets/figures/day03/L5- CNN_02.png", width: 92%)]

#text(size: 14pt, fill: gray)[Backpropagation — Computational Graph (source: course materials)]

== Output Layer Gradients

- MSE: $partial L / partial hat(y) = 2(hat(y) - y)$
- Softmax + CE: gradient simplifies to $hat(y) - y$
- Sigmoid + BCE similarly clean

== L5- CNN 03

#align(center)[#image("/assets/figures/day03/L5- CNN_03.png", width: 92%)]

#text(size: 14pt, fill: gray)[Backpropagation — Output Layer Gradients (source: course materials)]

== Hidden Layer Gradients

- $delta^(l) = (W^(l+1))^T delta^(l+1) dot.op sigma'(z^(l))$
- Vanishing / exploding gradients in deep nets
- Skip connections mitigate (ResNet, Day 4+)

== L5- CNN 04

#align(center)[#image("/assets/figures/day03/L5- CNN_04.png", width: 92%)]

#text(size: 14pt, fill: gray)[Backpropagation — Hidden Layer Gradients (source: course materials)]

== Implementation Notes

- Autograd in PyTorch / JAX
- Detach, stop_gradient, custom Function
- Check gradients with finite differences

== L5- CNN 10

#align(center)[#image("/assets/figures/day03/L5- CNN_10.png", width: 92%)]

#text(size: 14pt, fill: gray)[Backpropagation — Implementation Notes (source: course materials)]

= Training Loop

== Mini-batch SGD

- Sample batch $B subset D$ each step
- Loss averaged over batch
- Epoch = one pass over training set

== L5- CNN 11

#align(center)[#image("/assets/figures/day03/L5- CNN_11.png", width: 92%)]

#text(size: 14pt, fill: gray)[Training Loop — Mini-batch SGD (source: course materials)]

== Learning Rate Schedules

- Step decay, cosine annealing, warmup
- $eta_t$ often largest hyperparameter
- Monitor train vs val loss curves

== pdf0 page000

#align(center)[#image("/assets/figures/day03/pdf0_page000.png", width: 92%)]

#text(size: 14pt, fill: gray)[Training Loop — Learning Rate Schedules (source: course materials)]

== Initialization

- Xavier / He scaling for variance preservation
- Bad init → dead ReLUs or blow-up
- LayerNorm reduces sensitivity (Day 5)

== pdf0 page002

#align(center)[#image("/assets/figures/day03/pdf0_page002.png", width: 92%)]

#text(size: 14pt, fill: gray)[Training Loop — Initialization (source: course materials)]

== Batch Normalization

- Normalize activations per mini-batch
- Learnable scale and shift $gamma, beta$
- Regularization side effect

== pdf0 page004

#align(center)[#image("/assets/figures/day03/pdf0_page004.png", width: 92%)]

#text(size: 14pt, fill: gray)[Training Loop — Batch Normalization (source: course materials)]

= Optimizers

== Momentum

- $v_(t+1) = beta v_t + nabla L$; $w_(t+1) = w_t - eta v_(t+1)$
- Accumulates consistent gradient direction
- Nesterov lookahead variant

== Adam

- Adaptive per-parameter learning rates
- First and second moment estimates
- Default choice for many DL experiments

== Weight Decay

- Decoupled WD vs L2 in AdamW
- Regularization interacts with normalization

== Debugging Training

- Loss not decreasing → LR, init, bugs
- Overfit small batch sanity check
- Gradient clipping for RNNs / LLMs

== Summary

- Day 3: *Deep Neural Networks*
- Backpropagation, activations, optimizers
- Questions welcome — practical follows

== Questions?

#align(center + horizon)[
  #text(size: 44pt, weight: "bold", fill: course-primary)[Questions?]

  #v(1em)
  Practical session follows — see course site.
]
