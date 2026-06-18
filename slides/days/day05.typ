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
- Attention
- The Transformer

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

= 4 · Attention

== 4.1  Alignment: a Soft Lookup

- At each output step, learn which inputs to focus on
- Alignment matrix = soft correspondence input$arrow.l.r$output
- Differentiable 'soft' lookup, trained end-to-end
- No fixed bottleneck: direct access to every source state
- Interpretable: the weights show what the model used

== 4.1  Alignment: a Soft Lookup

#align(center + horizon)[#image("/assets/figures/day05/attn_alignment.png", width: 92%, height: 82%, fit: "contain")]

== 4.2  Content-Based Addressing

- A query $q$ is compared to each key $k_j$ by a similarity
- Normalize similarities with a softmax $arrow.r$ weights $a_j$
- Output = weighted sum of values: $sum_j a_j v_j$
- $a_j = "softmax"_j (q dot.op k_j)$
- Query/key/value: the vocabulary of attention

== 4.2  Content-Based Addressing

#align(center + horizon)[#image("/assets/figures/day05/attn_content.png", width: 92%, height: 82%, fit: "contain")]

== 4.3  Derivation: Scaled Dot-Product Attention

- Stack queries, keys, values into $Q, K, V$
- Scores $S = Q K^T$ (all query-key dot products)
- Scale by $sqrt(d_k)$ to keep softmax gradients healthy
- $"Attention"(Q,K,V) = "softmax"(Q K^T \\/ sqrt(d_k)) V$
- Full derivation of the scaling in the notes

= 5 · The Transformer

== 5.1  Self-Attention

- Each token attends to *every* token in the sequence
- $Q, K, V$ are all linear projections of the *same* input
- Captures long-range dependencies in *one* step
- Fully parallel over positions (unlike RNNs)
- Attention maps are interpretable alignments

== 5.1  Self-Attention

#align(center + horizon)[#image("/assets/figures/day05/attn_implicit.png", width: 92%, height: 82%, fit: "contain")]

== 5.2  Multi-Head Attention

- Run $h$ attention heads in parallel on projected subspaces
- Each head learns a different relation (syntax, coref, ...)
- Concatenate heads, then project: $W_O [h_1, dots.h, h_n]$
- More expressive than a single attention at the same cost
- Heads attend to different positions simultaneously

== 5.3  Positional Encoding & the Block

- Attention is permutation-equivariant $arrow.r$ inject position
- Sinusoidal or learned positional encodings added to inputs
- Block: (multi-head attn $arrow.r$ MLP), each with residual + LayerNorm
- Decoder uses *masked* self-attention (no peeking ahead)
- Stack $N$ blocks; same template for encoder & decoder

== 5.4  Why Transformers Won

- Parallel training: no sequential recurrence
- Constant path length between any two tokens
- Scales with data and compute — the basis of LLMs
- One architecture for text, vision (ViT), audio, proteins
- Day 9-10: pretraining and large language models

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
