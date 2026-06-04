---
layout: post
title: Day 5 - RNNs and Transformers
image: /assets/img/lessons/day05.png
accent_image: 
  background: url('/assets/img/sampling_space.png') center/cover
  overlay: false
accent_color: '#ccc'
theme_color: '#ccc'
description: >
  Recurrent models, LSTM, self-attention, and causal masking for sequences.
invert_sidebar: true
---

# Day 5 - RNNs and Transformers

### Optional reading for this lesson
- [Attention Is All You Need (Vaswani et al.)](https://arxiv.org/abs/1706.03762)
- [Karpathy — The Unreasonable Effectiveness of RNNs](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)
- [Complete reading list for Day 5](/publications/#day-5) (all resources for this lecture)


### [Slides](/assets/slides/day05.pdf)

### [Practical](/projects/day05-practical/)

Sequential data requires models that carry context across time steps. We begin with RNNs and LSTMs, then pivot to the Transformer architecture where self-attention replaces recurrence and causal masking enables autoregressive language modeling.

* toc
{:toc}

## 1. Recurrent Neural Networks

### 1.1 Vanilla RNN

> An **RNN** maintains hidden state $$\mathbf{h}_t$$ updated as $$\mathbf{h}_t = g(\mathbf{W}_{hh}\mathbf{h}_{t-1} + \mathbf{W}_{xh}\mathbf{x}_t + \mathbf{b})$$ and emits $$\mathbf{y}_t = \mathbf{W}_{hy}\mathbf{h}_t$$.
{:.lead}

The same weights are applied at every time step — parameter sharing across sequence length $$T$$.

Unrolled through time, backpropagation through time (BPTT) applies the chain rule across $$T$$ steps:

$$\frac{\partial L}{\partial \mathbf{h}_t} = \frac{\partial L}{\partial \mathbf{h}_{t+1}} \frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t} + \ldots$$

![RNN unrolled through time](/assets/figures/day05/pdf0_page005.png)

Vanishing/exploding gradients limit vanilla RNNs on long sequences — motivates LSTM and GRU.

### 1.2 Bidirectional and many-to-many models

> **Bidirectional RNNs** combine forward $$\overrightarrow{\mathbf{h}}_t$$ and backward $$\overleftarrow{\mathbf{h}}_t$$ states. Not usable for autoregressive generation (future leakage).
{:.lead}

Sequence labeling (NER, POS tagging) uses bidirectional context:

$$\mathbf{h}_t = [\overrightarrow{\mathbf{h}}_t ; \overleftarrow{\mathbf{h}}_t].$$

Architecture patterns:
- **Many-to-one**: sentiment (last $$\mathbf{h}_T$$)
- **One-to-many**: image captioning
- **Seq2seq**: encoder RNN → context vector → decoder RNN

Teacher forcing feeds ground-truth $$\mathbf{y}_t$$ to decoder during training; exposure bias at inference.

## 2. LSTM and Gated Recurrence

### 2.1 LSTM cell equations

> An **LSTM** uses gates to control information flow: **forget** $$\mathbf{f}_t = \sigma(\cdot)$$, **input** $$\mathbf{i}_t = \sigma(\cdot)$$, **output** $$\mathbf{o}_t = \sigma(\cdot)$$, and cell state $$\mathbf{c}_t$$.
{:.lead}

Standard LSTM (element-wise operations):

$$\begin{aligned}
\mathbf{f}_t &= \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f) \\
\mathbf{i}_t &= \sigma(\mathbf{W}_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i) \\
\tilde{\mathbf{c}}_t &= \tanh(\mathbf{W}_c [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c) \\
\mathbf{c}_t &= \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t \\
\mathbf{h}_t &= \mathbf{o}_t \odot \tanh(\mathbf{c}_t)
\end{aligned}$$

![LSTM gate diagram](/assets/figures/day05/pdf0_page010.png)

The additive cell update $$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \ldots$$ provides a gradient highway — mitigating vanishing gradients.

### 2.2 GRU and when to use recurrence

> **GRU** merges forget/input gates into **update gate** $$\mathbf{z}_t$$ and **reset gate** $$\mathbf{r}_t$$ — fewer parameters than LSTM, often comparable performance.
{:.lead}

GRU update:

$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t.$$

RNNs excel on small sequences and streaming data but parallelize poorly ($$O(T)$$ sequential steps). Transformers replaced RNNs in most NLP and are expanding into vision and genomics.

Truncated BPTT: backprop only over last $$K$$ steps to save memory on long sequences.

## 3. Self-Attention Mechanism

### 3.1 Scaled dot-product attention

> Given queries $$\mathbf{Q}$$, keys $$\mathbf{K}$$, values $$\mathbf{V}$$, **attention** computes $$\mathrm{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \mathrm{softmax}(\mathbf{Q}\mathbf{K}^\top / \sqrt{d_k})\mathbf{V}.$$
{:.lead}

For sequence length $$n$$ and dimension $$d_k$$, attention maps are $$n \times n$$ — each token attends to all others.

Scaling by $$\sqrt{d_k}$$ prevents softmax saturation when dot products grow large.

![Attention weight heatmap](/assets/figures/day05/pdf0_page015.png)

**Self-attention**: $$\mathbf{Q}, \mathbf{K}, \mathbf{V}$$ all derived from the same input sequence via learned projections $$\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$$.

### 3.2 Multi-head attention

> **Multi-head attention** runs $$h$$ parallel attention heads with different projections, concatenates, and projects again: $$\mathrm{MultiHead} = \mathrm{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_h)\mathbf{W}^O$$.
{:.lead}

Each head learns different relational patterns (syntax, coreference, locality).

Transformer block:

$$\mathbf{x}' = \mathbf{x} + \mathrm{MultiHead}(\mathrm{LN}(\mathbf{x}))$$
$$\mathbf{x}'' = \mathbf{x}' + \mathrm{FFN}(\mathrm{LN}(\mathbf{x}'))$$

FFN is typically two linear layers with GELU: $$\mathrm{FFN}(\mathbf{x}) = \mathbf{W}_2\,\mathrm{GELU}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2$$.

Complexity: $$O(n^2 d)$$ per layer — quadratic in sequence length motivates sparse and linear attention variants.

## 4. Causal Masking and Autoregressive Models

### 4.1 Causal (look-ahead) mask

> A **causal mask** sets attention logits to $$-\infty$$ for positions $$j > i$$ so token $$i$$ cannot attend to future tokens — required for autoregressive language modeling.
{:.lead}

Masked attention matrix (lower triangular after softmax):

$$A_{ij} = \begin{cases} \mathrm{softmax}(\mathbf{q}_i^\top \mathbf{k}_j / \sqrt{d_k}) & j \leq i \\ 0 & j > i \end{cases}$$

Implemented by adding a mask $$M_{ij} = 0$$ if $$j \leq i$$ else $$-\infty$$ before softmax.

![Causal mask pattern](/assets/figures/day05/pdf0_page020.png)

**Encoder** (BERT): bidirectional, no causal mask. **Decoder** (GPT): causal mask only.

### 4.2 Autoregressive training objective

> An **autoregressive LM** factorizes $$p(x_1,\ldots,x_T) = \prod_{t=1}^T p(x_t | x_{<t})$$ and trains by minimizing cross-entropy on next-token prediction.
{:.lead}

Loss for token sequence:

$$L = -\sum_{t=1}^T \log p_\theta(x_t \mid x_1, \ldots, x_{t-1}).$$

At inference, **greedy decoding** picks $$\hat{x}_t = \arg\max p(x_t | x_{<t})$$; **sampling** with temperature $$\tau$$:

$$p(x_t) \propto \exp(z_t / \tau).$$

**KV cache** stores past key/value projections so each new token costs $$O(n)$$ not $$O(n^2)$$ per step — essential for efficient LLM inference.

Positional information: sinusoidal encodings (original Transformer) or **rotary embeddings (RoPE)** in modern LLMs.

## Checkpoint summary

Before moving to the practical, confirm you can:

- RNNs share weights over time; BPTT backprops through unrolled graph; LSTM gates help long-range deps.
- Self-attention weighs all pairs of tokens; multi-head captures diverse relations.
- Causal masking enforces $$p(x_t | x_{<t})$$ — no peeking at the future.
- Autoregressive LMs train with next-token CE; KV cache speeds up generation.
