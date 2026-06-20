#import "../lib.typ": *

#show: course-theme.with(title: [Sequence Models & Transformers], subtitle: [Day 5 | Aug 2026])

= Day 5: Sequence Models & Transformers

== Welcome

- *Sequence Models & Transformers* — From recurrence to attention
- 3 hours lecture + practical
- Slides, notes, and code on the course site

== Outline

- Sequence Modeling
- Recurrent Neural Networks
- Sequence to Sequence
- Building Intuition for Attention
- Standard Notation and Why Transformers Won

= 1 · Sequence Modeling

== 1.1  Why Sequences are Different

- Inputs/outputs have *variable length*: text, audio, time series
- Order matters: 'dog bites man' $!=$ 'man bites dog'
- Long-range dependencies span many steps
- Need parameter sharing *across positions in time*
- Two ideas today: recurrence, then attention

== 1.2  Autoregressive Factorization

- Any joint factorizes by the chain rule of probability:
- $p(x) = product_(i=1)^n p(x_i | x_1, dots.h, x_(i-1))$
- Model each conditional with a shared network
- Train by next-token prediction (max likelihood)
- Generate by sampling one token at a time

== 1.2  Autoregressive Factorization

#align(center + horizon)[#image("/assets/figures/day05/rnn_pixelrnn.png", width: 92%, height: 82%, fit: "contain")]

== 1.3  Sharing Across Time

- Reuse the *same* weights at every time step
- Like CNNs share across space, RNNs share across time
- Handles arbitrary length with a fixed parameter count
- A hidden state carries a summary of the past
- Output head: softmax over the vocabulary

= 2 · Recurrent Neural Networks

== 2.1  The RNN Recurrence

- State update: $h_t = tanh(W_h h_(t-1) + W_x x_t + b)$
- Prediction: $y_t = "softmax"(W_y h_t)$
- $h_t$ is a running summary of $x_1, dots.h, x_t$
- Same $W_h, W_x, W_y$ shared over all $t$
- Unroll in time $arrow.r$ a deep feedforward net

== 2.1  The RNN Recurrence

#align(center + horizon)[#image("/assets/figures/day05/rnn_unrolled.png", width: 92%, height: 82%, fit: "contain")]

== 2.2  Backprop Through Time

- Unroll the recurrence, then apply backprop
- Gradient at step $t$ sums contributions from all later steps
- Involves products of Jacobians $product (partial h_(k))/(partial h_(k-1))$
- Truncate BPTT for very long sequences
- Cost grows with sequence length; steps are sequential

== 2.3  Vanishing Gradients & LSTM/GRU

- Repeated Jacobian products $arrow.r$ gradients vanish/explode
- Plain RNNs forget long-range context
- LSTM: a *cell state* with additive updates + gates
- Gates (forget/input/output) control information flow
- GRU: a lighter gated variant; clip to tame explosions

= 3 · Sequence to Sequence

== 3.1  Encoder-Decoder

- Encoder RNN reads the source into a context vector
- Decoder RNN generates the target, token by token
- Enabled end-to-end neural machine translation
- Closed much of the gap to human-quality translation
- One architecture for translation, summarization, dialogue

== 3.1  Encoder-Decoder

#align(center + horizon)[#image("/assets/figures/day05/seq2seq_nmt.png", width: 92%, height: 82%, fit: "contain")]

== 3.2  The Bottleneck Problem

- All source information squeezed into *one* fixed vector
- Long sentences overflow the bottleneck $arrow.r$ quality drops
- Decoder can't 'look back' at specific source words
- Fix: let the decoder attend to *all* encoder states
- This is attention — the bridge to Transformers

= 4 · Building Intuition for Attention

== 4.1  Tokens as a D x N Matrix

- Input: $N$ tokens, each a $D$-dimensional vector
- Collect into $X^(0) in RR^(D times N)$: features *down*, sequence *across*
- Words, image patches, amino-acid embeddings — all tokenise the same way
- Transformer returns $X^(M)$, another $D times N$ representation
- Source: Turner, *An Introduction to Transformers* (arXiv:2304.10557)

== 4.2  Attention = Weighted Linear Combination

- At position $n$, form a new vector by *averaging* all tokens
- $y_n = sum_(j=1)^N x_j A_(j n)$ — a convex combination
- Weights $A_(j n) >= 0$, columns sum to 1: $sum_j A_(j n)=1$
- High $A_(j n)$ = position $j$ is *relevant* for updating $n$
- Nothing exotic yet — just a learned weighted sum

== 4.3  Matrix Form: Y = X A

- Stack all positions: $Y = X A$ ($D times N$)
- Each column of $A$ is a softmax-normalised weight vector
- Stage 1 of the block: mix information *across the sequence*
- Operates row-wise on $X$ (each feature independently)
- Stage 2 will refine features *down* each column

== 4.3  Matrix Form: Y = X A

#align(center + horizon)[#image("/assets/figures/day05/attn_alignment.png", width: 92%, height: 82%, fit: "contain")]

== 4.4  Self-Attention: Where Does A Come From?

- Naive: $A_(n j) prop exp(x_n^T x_j)$ — similarity of raw features
- Problem: similarity is entangled with content
- Fix: compare *projected* features $U x_n$ instead
- Still symmetric — 'caulking iron' $arrow.l.r$ 'tool' but not vice versa
- Need *asymmetric* queries and keys (next slide)

== 4.5  Queries and Keys

- $q_n = U_q x_n$, $k_n = U_k x_n$ — two linear maps of the input
- $A_(n j) = "softmax"_j exp(q_n^T k_j)$
- Only parameters so far: $U_q, U_k$ ($K times D$ each)
- Query = what position $n$ is looking for
- Key = what position $j$ offers; value = content (standard view)

== 4.5  Queries and Keys

#align(center + horizon)[#image("/assets/figures/day05/attn_content.png", width: 92%, height: 82%, fit: "contain")]

== 4.6  Multi-Head Self-Attention

- One $N times N$ attention map can be a bottleneck
- Run $H$ attention heads in parallel (different $U_(q,h), U_(k,h)$)
- Each head learns a different notion of 'relevance'
- Concatenate and project: $Y = sum_h V_h X A_h$
- Analogous to multiple convolution filters (Turner)

== 4.7  Stage 2: MLP Across Features

- Second stage refines each token's feature vector independently
- $x_n = "MLP"(y_n)$ — same MLP weights at every position $n$
- Acts *vertically* down each column of $X$
- After $M$ blocks, token $n$, feature $d$ uses info from $(j, k)$ anywhere
- Horizontal (attention) + vertical (MLP) = full mixing

== 4.8  Residual Connections and LayerNorm

- Parameterise updates as *residuals*: $x^(m) = x^(m-1) + "res"(x^(m-1))$
- Each stage applies a mild correction; depth composes large transforms
- LayerNorm standardises each feature dimension: zero mean, unit variance
- Prevents activations blowing up through deep stacks
- Full block: LN $arrow.r$ MHSA $arrow.r$ residual $arrow.r$ LN $arrow.r$ MLP $arrow.r$ residual

== 4.9  Position Encoding and Causal Masking

- Attention alone is permutation-equivariant — order is lost
- Fix: add (or concatenate) a position embedding to each token
- Autoregressive training: mask $A$ so $j > n$ gets weight 0
- Upper-triangular mask $arrow.r$ train on full sequence in one pass
- Contrast RNNs: all time-points treated identically in attention

= 5 · Standard Notation and Why Transformers Won

== 5.1  Self-Attention in Translation

- Seq2seq attention = special case: decoder queries, encoder keys/values
- Self-attention: $Q, K, V$ all from the same sequence
- Alignment matrix shows which source words each output used
- Interpretable, differentiable, no fixed bottleneck

== 5.1  Self-Attention in Translation

#align(center + horizon)[#image("/assets/figures/day05/attn_implicit.png", width: 92%, height: 82%, fit: "contain")]

== 5.2  Scaled Dot-Product Attention (Batch Form)

- Standard ML notation: $X in RR^(N times d)$ (rows = tokens)
- $"Attention"(Q,K,V) = "softmax"(Q K^T \\/ sqrt(d_k)) V$
- Same math as Turner; $1\\/sqrt(d_k)$ keeps softmax gradients healthy
- Fully parallel over $N$ — unlike sequential RNNs
- Cost: $O(N^2)$ in sequence length

== 5.3  Transformers vs RNNs

- RNN: nearby inputs treated differently from distant ones (recurrence)
- Transformer: every pair interacts in one layer, same treatment
- Constant path length $arrow.r$ long-range deps no harder than short
- Parallel training $arrow.r$ scales to internet-scale data (LLMs)
- Day 9-10: pretraining; proteins/sequences: see structural-bioinformatics course

== Summary

- Day 5: *Sequence Models & Transformers*
- From recurrence to attention
- Questions welcome — practical follows

== Questions?

#align(center + horizon)[
  #text(size: 44pt, weight: "bold", fill: course-primary)[Questions?]

  #v(1em)
  Practical session follows — see course site.
]
