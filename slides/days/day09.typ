#import "../lib.typ": *

#show: course-theme.with(title: [Autoregressive Language Models], subtitle: [Day 9 | Aug 2026])

= Day 9: Autoregressive Language Models

== Welcome

- *Autoregressive Language Models* — Build & train a decoder-only LM
- 3 hours lecture + practical
- Slides, notes, and code on the course site

== Outline

- Back to Autoregressive (Week 2 Arc)
- From Text to Tokens
- The Decoder-Only Transformer Block
- Training One Model End-to-End

= 1 · Back to Autoregressive (Week 2 Arc)

== 1.1  Where We Are: the Generative Map (Day 6)

- Day 6 goal: transport noise to the data distribution
- Likelihood *exact*: autoregressive, normalizing flows
- Likelihood *bound*: VAEs, diffusion; *implicit*: GANs
- Days 6-8 went deep on diffusion / score / flow
- Today we return to the *autoregressive* family — LLMs

== 1.1  Where We Are: the Generative Map (Day 6)

#align(center + horizon)[#image("/assets/figures/day06/pdm_dgm_zoo.png", width: 100%, height: 90%, fit: "contain")]

== 1.2  Autoregressive vs Diffusion

- AR: $p(x) = product_t p(x_t | x_(<t))$ — exact likelihood
- Diffusion: ELBO / score matching — bound or implicit
- AR training: next-token cross-entropy
- AR sampling: *one token at a time* (Day 10: KV cache)
- Diffusion sampling: *many steps* along a schedule / ODE

== 1.3  The Autoregressive Factorization

- Chain rule: $p(x) = product_(t=1)^T p(x_t | x_(<t))$
- Model every conditional with one shared network
- Train by maximum likelihood = next-token cross-entropy
- Causal mask (Day 5) computes all conditionals in one pass
- Generation = ancestral sampling, token by token

== 1.4  Architecture Families

- Encoder-only (BERT): bidirectional, for understanding
- Encoder-decoder (T5, NMT): map one sequence to another
- Decoder-only (GPT, chat): causal, next-token prediction
- Today: the *decoder-only* stack that powers modern LLMs
- One architecture, scaled with data + compute

== 1.4  Architecture Families

#align(center + horizon)[#image("/assets/figures/day09/llmks_paradigm.png", width: 100%, height: 90%, fit: "contain")]

= 2 · From Text to Tokens

== 2.1  Tokenization (BPE)

- Text is split into *subword* tokens, not words or characters
- Byte-pair encoding: merge frequent byte pairs greedily
- Balances vocabulary size vs sequence length
- Handles any string (rare words, code, emoji) — no OOV
- Vocabulary $V$ ~ 30k-130k for modern LMs

== 2.2  Special Tokens & Document Packing

- Special tokens: BOS / EOS / PAD mark boundaries
- Concatenate documents into one long token stream
- Slice into fixed-length blocks (the context window)
- Packing keeps GPUs busy — no wasted padding
- Attention/loss masks stop attending across document joins

== 2.3  Embeddings

- Each token id indexes a row of an embedding table $E in RR^(V times d)$
- Token id $arrow.r$ vector $x_t in RR^d$
- Often *tie* the embedding with the output `lm_head` weights
- Position is added separately (Day 5) — or via RoPE (later)
- Output of this stage: a $T times d$ matrix of token vectors

= 3 · The Decoder-Only Transformer Block

== 3.1  The Pre-Norm Residual Block

- Modern blocks normalize *before* each sublayer (pre-norm)
- $h' = h + "MHA"("LN"(h))$
- $h'' = h' + "MLP"("LN"(h'))$
- Residual path keeps gradients healthy through depth
- Same identity-path idea as ResNet (Day 4)

== 3.1  The Pre-Norm Residual Block

#align(center + horizon)[#image("/assets/figures/day09/llmks_residual.png", width: 100%, height: 90%, fit: "contain")]

== 3.2  RMSNorm

- Normalize each token vector by its root-mean-square
- $"RMSNorm"(x) = x \\/ "rms"(x) dot.op gamma$
- No mean subtraction, no bias — cheaper than LayerNorm
- Stabilizes activations in deep stacks
- Default norm in LLaMA-style models

== 3.3  Causal Self-Attention

- Project tokens to queries, keys, values: $Q, K, V$
- $A = "softmax"(Q K^T \\/ sqrt(d_k))$ — soft dictionary lookup
- Causal mask: set $A_(i,j) = 0$ for $j > i$ (no peeking ahead)
- Output $= A V$ — a weighted average of past values
- This is the conditional $p(x_t | x_(<t))$ in action

== 3.3  Causal Self-Attention

#align(center + horizon)[#image("/assets/figures/day09/llmks_attention_dict.png", width: 100%, height: 90%, fit: "contain")]

== 3.4  Multi-Head & Grouped-Query Attention

- Run $H$ attention heads in parallel, then concatenate
- Each head learns a different relation (syntax, coref, ...)
- Multi-query / grouped-query: heads *share* K, V
- GQA shrinks the KV cache (Day 10) with little quality loss
- Heads = parallel relevance maps, like CNN filters

== 3.4  Multi-Head & Grouped-Query Attention

#align(center + horizon)[#image("/assets/figures/day09/llmks_mha.png", width: 100%, height: 90%, fit: "contain")]

== 3.5  Rotary Position Embeddings (RoPE)

- Rotate $Q, K$ by an angle that depends on position
- Dot product then depends on *relative* offset $m - n$
- No learned position table; extrapolates to longer context
- YaRN: rescale frequencies to extend context length
- Standard in LLaMA, Qwen, Mistral, ...

== 3.5  Rotary Position Embeddings (RoPE)

#align(center + horizon)[#image("/assets/figures/day09/llmks_rope.png", width: 100%, height: 90%, fit: "contain")]

== 3.6  The MLP: GeGLU / SwiGLU

- Position-wise FFN refines each token independently
- Vanilla: $W_2 "GELU"(W_1 x)$
- Gated (GeGLU/SwiGLU): $W_2 ( "GELU"(W_0 x) dot.op (W_1 x) )$
- Where most parameters and FLOPs live
- Hidden width typically $4 d$ (or ~$8/3 d$ for gated)

== 3.6  The MLP: GeGLU / SwiGLU

#align(center + horizon)[#image("/assets/figures/day09/llmks_ffn.png", width: 100%, height: 90%, fit: "contain")]

== 3.7  Stacking the Block to Logits

- Stack $L$ identical blocks: $h^((l)) = "Block"(h^((l-1)))$
- Final norm, then `lm_head`: $h arrow.r RR^V$ logits
- Softmax over the vocabulary $arrow.r$ next-token distribution
- Same weights at every position (parameter sharing)
- This is a full GPT forward pass

== 3.7  Stacking the Block to Logits

#align(center + horizon)[#image("/assets/figures/day09/llmks_architecture.png", width: 100%, height: 90%, fit: "contain")]

= 4 · Training One Model End-to-End

== 4.1  The Training Objective

- Loss: $L = - sum_i log p_theta (x_i | x_(<i))$
- = next-token cross-entropy over the vocabulary
- Teacher forcing: feed ground-truth context, predict next
- Labels = inputs shifted by one position
- Report *perplexity* $= exp(L)$

== 4.1  The Training Objective

#align(center + horizon)[#image("/assets/figures/day09/llmks_training.png", width: 100%, height: 90%, fit: "contain")]

== 4.2  Optimization Stack

- AdamW with weight decay
- Learning-rate warmup, then cosine decay
- Gradient clipping for stability
- Mixed precision (bf16); gradient accumulation for big batches
- Long runs over billions of tokens

== 4.3  Parameters & FLOPs

- Params per block ~ $12 d^2$ (attention $4d^2$ + MLP $8d^2$)
- Total ~ $12 L d^2$ plus $V d$ embeddings
- Training compute ~ $6 N$ FLOPs per token ($N$ = params)
- Forward ~ $2N$, backward ~ $4N$
- Order-of-magnitude reasoning guides model/data sizing

== 4.4  nanoGPT Code Map

- `model.py`: GPT, Block, CausalSelfAttention, MLP
- `train.py`: data loader, training loop, optimizer
- `config`: n_layer, n_head, n_embd, block_size
- `sample.py`: generation (preview of Day 10)
- ~300 lines: the whole stack you can read in an evening

== 4.4  nanoGPT Code Map

#align(center + horizon)[#image("/assets/figures/day09/llmks_decoder_masked.png", width: 100%, height: 90%, fit: "contain")]

== 4.5  Scaling and What Is Next

- Scaling laws: loss falls predictably with params/data/compute
- Chinchilla: balance model size against training tokens
- Distributed training (data / tensor / pipeline parallel)
- Day 10: how a trained base LM is *served* and *aligned*
- Same 'make generation fast' theme as Day 8 — different bottleneck

== Summary

- Day 9: *Autoregressive Language Models*
- Build & train a decoder-only LM
- Questions welcome — practical follows

== Questions?

#align(center + horizon)[
  #text(size: 44pt, weight: "bold", fill: course-primary)[Questions?]

  #v(1em)
  Practical session follows — see course site.
]
