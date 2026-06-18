---
layout: post
title: Day 9 - Autoregressive Language Models
image: /assets/img/sampling_space.png
accent_image: 
  background: url('/assets/img/sampling_space.png') center/cover
  overlay: false
accent_color: '#ccc'
theme_color: '#ccc'
description: >
  Encoder–decoder vs decoder-only stacks, training with cross-entropy, RoPE, and the transformer block.
invert_sidebar: true
---

# Day 9 - Autoregressive Language Models

### [Slides](/assets/slides/day09.pdf)

### [Practical](/projects/day09-practical/)

### Optional reading for this lesson
- [Vaswani et al. — Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Gordić — Inside the Transformer: The Life of a Token](https://www.aleksagordic.com/blog/transformer)
- [Su et al. — RoFormer / RoPE](https://arxiv.org/abs/2104.09864)

* toc
{:toc}

Large language models are **autoregressive**: they factorize sequences with causal conditioning. We map architectural
families, the standard training loop, and the mathematical core of modern decoder-only transformers.

## 1. Model families: encoder, decoder, and hybrids

> **Autoregressive factorization.**
> $$p(\mathbf{x}) = \prod_{i=1}^{L} p(x_i \mid x_{<i})$$ with causal masking so token $$i$$ never sees the future.
{:.lead}

| Architecture | Attention | Typical use |
|--------------|-----------|-------------|
| **Encoder-only** | Bidirectional | Classification, embeddings (BERT) |
| **Decoder-only** | Causal | GPT-style LMs, chat models |
| **Encoder–decoder** | Cross + causal | Translation, summarization (T5) |

**Encoder–decoder** maps input tokens to memory with a bidirectional encoder; the decoder attends to memory with
cross-attention and to past outputs with a causal mask.

**Decoder-only** treats prompt and completion as one sequence—today's default for general-purpose LLMs.

![Architecture families](/assets/figures/day09/pdf0_page025.png)
*Figure: where information flows in each stack.*

### 1.1 Causal mask

For positions $$i, j$$, attention logits satisfy $$A_{ij} = -\infty$$ when $$j > i$$. After softmax, token $$i$$
only aggregates keys/values from positions $$\le i$$.

## 2. Training loop and cross-entropy

Given tokenized sequence $$\mathbf{x} = (x_1, \ldots, x_L)$$, the model outputs logits $$\mathbf{z}_i \in \mathbb{R}^{|\mathcal{V}|}$$
for each position. **Next-token prediction** maximizes

$$
\mathcal{L}(\theta) = -\sum_{i=1}^{L} \log p_\theta(x_i \mid x_{<i})
= -\sum_{i=1}^{L} \log \mathrm{softmax}(\mathbf{z}_i)_{x_i}.
$$

This is **multiclass cross-entropy** averaged over non-masked positions (padding masked out in the loss).

> **Teacher forcing.** During training, the model always conditions on ground-truth prefixes $$x_{<i}$$,
> not its own previous predictions—stable gradients, train/inference mismatch handled at decode time (day 10).
{:.lead}

![Training batch](/assets/figures/day09/pdf1_page020.png)
*Figure: packed sequences and label shift by one.*

### 2.1 Optimization stack

- AdamW with weight decay on matrices (not biases/LayerNorm).
- Learning-rate warmup + cosine decay.
- Gradient clipping and mixed-precision (bf16/fp16).

## 3. RoPE positional embeddings

**Sinusoidal** absolute positions add to embeddings; **RoPE (rotary position embedding)** encodes relative position
in attention by rotating query/key pairs in 2D subspaces.

For head dimension pairs $$(2k, 2k+1)$$ and position $$m$$,

$$
\mathrm{RoPE}(\mathbf{q}, m) =
\begin{pmatrix}
\cos m\theta_k & -\sin m\theta_k \\
\sin m\theta_k & \cos m\theta_k
\end{pmatrix}
\begin{pmatrix} q_{2k} \\ q_{2k+1} \end{pmatrix},
\qquad \theta_k = 10000^{-2k/d_{\mathrm{head}}}.
$$

Attention score $$\langle \mathrm{RoPE}(\mathbf{q}, m), \mathrm{RoPE}(\mathbf{k}, n)\rangle$$ depends on $$m-n$$,
improving length extrapolation (YaRN scales frequencies for very long contexts).

![RoPE intuition](/assets/figures/day09/pdf1_page030.png)
*Figure: relative position via rotation.*

## 4. Transformer block internals

A **pre-norm** decoder block (schematic):

$$
\mathbf{h}' = \mathbf{h} + \mathrm{MHA}(\mathrm{LN}(\mathbf{h})),\qquad
\mathbf{h}'' = \mathbf{h}' + \mathrm{MLP}(\mathrm{LN}(\mathbf{h}')).
$$

**Multi-head attention (MHA):**

$$
\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_{\mathrm{head}}}}\right) V,
$$

with $$Q = XW_Q$$, $$K = XW_K$$, $$V = XW_V$$, split into heads. **GQA/MQA** share key/value heads to cut memory at inference.

**MLP (SwiGLU / GeGLU):** gated feed-forward expands dimension (e.g. $$4\times$$) then projects back.

### 4.1 Life of a token (forward pass)

1. Embed token IDs → vectors.
2. For each layer: causal self-attention + MLP with residuals.
3. Final linear **lm_head** → logits; softmax for loss or sampling.

Stack depth $$L$$, hidden size $$d_{\mathrm{model}}$$, and head count set capacity; FLOPs scale roughly $$\mathcal{O}(L\, d_{\mathrm{model}}^2)$$ per token.

### 4.2 Parameter and FLOP scaling (sketch)

Per layer, attention matrices contribute $$\approx 4 d_{\mathrm{model}}^2$$ parameters (Q,K,V,O projections) and
MLP another $$\approx 8 d_{\mathrm{model}}^2$$ with expansion ratio 4. Total parameters scale
$$\mathcal{O}(L_{\mathrm{layers}}\, d_{\mathrm{model}}^2)$$—the basis for Chinchilla-style compute–data trade-offs.

### 4.3 Tokenization and packing

Subword tokenizers (BPE, SentencePiece) map bytes/characters to a vocabulary $$\mathcal{V}$$.
**Document packing** concatenates multiple examples with attention masks so padding waste drops in training batches.

![Transformer block](/assets/figures/day09/pdf0_page035.png)
*Figure: residual stream through attention and MLP.*

## Checkpoint summary

- **Decoder-only** LMs dominate general text generation; encoder–decoder remains strong for fixed input→output tasks.
- Training = sum of cross-entropies on next tokens with causal masking.
- **RoPE** bakes relative position into attention via rotations.
- A block = LN + attention + LN + MLP with residuals; depth stacks identical blocks with distinct weights.
