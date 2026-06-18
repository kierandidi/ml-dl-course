#import "../lib.typ": *

#show: course-theme.with(title: [Sequence Models], subtitle: [Day 5 | Aug 2026])

= Day 5: Sequence Models

== Welcome

- *Sequence Models* — RNNs, attention, and Transformers
- 3 hours lecture + practical
- Slides, notes, and code on the course site

== Outline

- Sequential Data
- Attention
- Transformer Architecture
- Applications & Scaling

= Sequential Data

== Motivation

- Text, speech, time series — ordered inputs
- Variable length sequences
- Markov assumptions and history

== Motivation — illustration

#align(center)[#image("/assets/figures/day05/ucl0_p081.png", width: 80%)]

#text(size: 14pt, fill: gray)[Sequential Data — Motivation (source: course materials)]

== Fixed-window Baselines

- Bag-of-words loses order
- N-gram language models
- Need latent state summarizing past

== Fixed-window Baselines — illustration

#align(center)[#image("/assets/figures/day05/ucl0_p082.png", width: 80%)]

#text(size: 14pt, fill: gray)[Sequential Data — Fixed-window Baselines (source: course materials)]

== RNN Recurrence

- $h_t = sigma(W_h h_(t-1) + W_x x_t + b)$
- Same weights applied at each time step
- Unroll graph for BPTT

== RNN Recurrence — illustration

#align(center)[#image("/assets/figures/day05/ucl0_p083.png", width: 80%)]

#text(size: 14pt, fill: gray)[Sequential Data — RNN Recurrence (source: course materials)]

== Vanishing Gradients

- Long-range dependencies are hard
- LSTM / GRU gating mechanisms
- Truncated BPTT for long sequences

== Vanishing Gradients — illustration

#align(center)[#image("/assets/figures/day05/ucl0_p084.png", width: 80%)]

#text(size: 14pt, fill: gray)[Sequential Data — Vanishing Gradients (source: course materials)]

= Attention

== Seq2seq Bottleneck

- Encoder final state must encode everything
- Attention reads all encoder states

== Seq2seq Bottleneck — illustration

#align(center)[#image("/assets/figures/day05/ucl0_p085.png", width: 80%)]

#text(size: 14pt, fill: gray)[Attention — Seq2seq Bottleneck (source: course materials)]

== Scaled Dot-Product Attention

- $"Attention"(Q,K,V) = "softmax"(Q K^T / sqrt(d_k)) V$
- Query, Key, Value interpretations
- Soft alignment weights over positions

== Scaled Dot-Product Attention — illustration

#align(center)[#image("/assets/figures/day05/ucl0_p114.png", width: 80%)]

#text(size: 14pt, fill: gray)[Attention — Scaled Dot-Product Attention (source: course materials)]

== Multi-Head Attention

- Parallel heads in subspaces
- Concatenate and project
- Expressivity vs compute

== Multi-Head Attention — illustration

#align(center)[#image("/assets/figures/day05/ucl0_p125.png", width: 80%)]

#text(size: 14pt, fill: gray)[Attention — Multi-Head Attention (source: course materials)]

== Self-Attention

- Q, K, V from same sequence
- Direct long-range links $O(n^2)$
- Positional information required

== Self-Attention — illustration

#align(center)[#image("/assets/figures/day05/ucl1_p010.png", width: 80%)]

#text(size: 14pt, fill: gray)[Attention — Self-Attention (source: course materials)]

= Transformer Architecture

== Encoder Block

- MHA → Add&Norm → FFN → Add&Norm
- FFN: two linear layers with non-linearity
- Pre-norm vs post-norm variants

== Encoder Block — illustration

#align(center)[#image("/assets/figures/day05/ucl1_p034.png", width: 80%)]

#text(size: 14pt, fill: gray)[Transformer Architecture — Encoder Block (source: course materials)]

== Positional Encoding

- Sinusoidal or learned position embeddings
- RoPE in modern LLMs (Day 9–10)
- Relative position bias

== Positional Encoding — illustration

#align(center)[#image("/assets/figures/day05/ucl1_p036.png", width: 80%)]

#text(size: 14pt, fill: gray)[Transformer Architecture — Positional Encoding (source: course materials)]

== Decoder & Masking

- Causal mask: token $t$ sees $<= t$ only
- Cross-attention to encoder (MT)
- Decoder-only for language modeling

== Decoder & Masking — illustration

#align(center)[#image("/assets/figures/day05/ucl1_p080.png", width: 80%)]

#text(size: 14pt, fill: gray)[Transformer Architecture — Decoder & Masking (source: course materials)]

== Complexity

- Self-attention: $O(n^2 d)$ time and memory
- Motivates sparse / linear attention research
- KV cache for inference (Day 10)

== Complexity — illustration

#align(center)[#image("/assets/figures/day05/ucl1_p081.png", width: 80%)]

#text(size: 14pt, fill: gray)[Transformer Architecture — Complexity (source: course materials)]

= Applications & Scaling

== Language Modeling

- Next-token prediction objective
- Perplexity metric: $exp(-(1/N) sum log p)$
- BPE / SentencePiece tokenization

== Pre-training + Fine-tuning

- Self-supervised pretrain on large corpus
- Supervised fine-tune on downstream task
- Instruction tuning and RLHF preview

== Scaling Laws

- Loss improves predictably with compute/data/params
- Chinchilla-optimal token budgets
- Emergent capabilities (debated)

== Summary

- Day 5: *Sequence Models*
- RNNs, attention, and Transformers
- Questions welcome — practical follows

== Questions?

#align(center + horizon)[
  #text(size: 44pt, weight: "bold", fill: course-primary)[Questions?]

  #v(1em)
  Practical session follows — see course site.
]
