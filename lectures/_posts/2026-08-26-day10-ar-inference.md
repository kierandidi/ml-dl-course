---
layout: post
title: Day 10 - Autoregressive Inference
image: /assets/img/sampling_space.png
accent_image: 
  background: url('/assets/img/sampling_space.png') center/cover
  overlay: false
accent_color: '#ccc'
theme_color: '#ccc'
description: >
  KV caching, temperature, nucleus (top-p) sampling, and batched decoding.
invert_sidebar: true
---

# Day 10 - Autoregressive Inference

### [Slides](/assets/slides/day10.pdf)

### [Practical](/projects/day10-practical/)

### Optional reading for this lesson
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — §3 (complexity)
- [Holtzman et al. — The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)
- [Gordić — KV cache discussion](https://www.aleksagordic.com/blog/transformer)

* toc
{:toc}

Inference generates one token at a time. Efficiency hinges on **KV caching**, while **temperature** and **top-p**
shape output quality. Production systems batch requests under memory and latency constraints.

## 1. Autoregressive decoding

At step $$t$$, we have prefix $$x_{\le t}$$. The model outputs distribution $$p_\theta(x_{t+1}\mid x_{\le t})$$.
We sample or argmax $$x_{t+1}$$, append, and repeat until EOS or max length.

> **Exposure bias.** Training uses teacher forcing; inference feeds the model its own samples—errors compound.
> Mitigations: scheduled sampling, distillation, RL fine-tuning (out of scope here).
{:.lead}

![Decoding loop](/assets/figures/day10/pdf0_page000.png)
*Figure: single-token extension per step.*

### 1.1 Complexity without cache

Recomputing keys/values for all past tokens each step costs $$\mathcal{O}(t)$$ attention per step and
$$\mathcal{O}(L^2)$$ over full length $$L$$—prohibitive for long contexts.

## 2. KV cache

For each layer $$\ell$$ and head, projections produce queries, keys, values. **Keys and values for past tokens are fixed**
under causal attention, so we store them.

> **KV cache memory (per layer, rough).**
> $$2 \times L_{\mathrm{heads}} \times L_{\mathrm{seq}} \times d_{\mathrm{head}} \times \texttt{bytes\_per\_elem}$$
> for keys plus values, times number of layers.
{:.lead}

Let $$C_{\mathrm{KV}}$$ denote cached tensors. At step $$t$$,

$$
K^{(\ell)} = \big[ K^{(\ell)}_{\mathrm{cache}} \;\|\; k^{(\ell)}_t \big], \qquad
V^{(\ell)} = \big[ V^{(\ell)}_{\mathrm{cache}} \;\|\; v^{(\ell)}_t \big],
$$

and only $$q^{(\ell)}_t$$ attends to the concatenated length-$$t$$ sequence. Per-step cost becomes $$\mathcal{O}(t)$$
attention per layer instead of recomputing from scratch.

![KV cache growth](/assets/figures/day10/pdf0_page005.png)
*Figure: cache size grows linearly with context.*

### 2.1 Multi-request batching

Batching $$B$$ sequences pads to a common length (or uses ragged/paged attention). Cache is indexed per sequence;
**PagedAttention** stores KV blocks in non-contiguous pages to reduce fragmentation on GPUs.

## 3. Temperature and top-p sampling

Logits $$\mathbf{z}$$ become probabilities

$$
p_i = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}, \qquad \tau > 0 \;\text{(temperature)}.
$$

- $$\tau \to 0^+$$: distribution sharpens → greedy / near-argmax behavior.
- $$\tau = 1$$: training-scale probabilities.
- $$\tau > 1$$: flatter, more random outputs.

**Top-p (nucleus) sampling:** sort probabilities $$p_{(1)} \ge p_{(2)} \ge \cdots$$ and keep the smallest set
$$V_p \subset \mathcal{V}$$ such that $$\sum_{i \in V_p} p_i \ge p_{\mathrm{cut}}$$ (e.g. $$p_{\mathrm{cut}}=0.9$$).
Renormalize and sample within $$V_p$$. This adapts the support size to model confidence.

![Sampling trade-offs](/assets/figures/day10/pdf1_page010.png)
*Figure: temperature vs diversity.*

### 3.1 Other decoding knobs

- **Top-k:** restrict to $$k$$ largest logits before softmax.
- **Repetition penalty:** down-weight logits of tokens already generated.
- **Stop sequences:** user-defined EOS strings.

## 4. Batching, throughput, and serving

**Static batching** waits until $$B$$ requests fill—simple but increases latency.

**Continuous batching** admits new prompts as others finish generations; improves GPU utilization in serving systems.

Metrics:

- **Time-to-first-token (TTFT):** prefill processes the prompt (large matmul, fills KV cache).
- **Inter-token latency:** decode steps with cache—memory-bandwidth bound.

$$
\text{Throughput} \approx \frac{B}{\text{latency per step}} \quad\text{(tokens/s, rough)}.
$$

Quantization (INT8/INT4 weights, FP8 KV) trades accuracy for larger batch or longer context within the same VRAM.

![Serving stack](/assets/figures/day10/pdf1_page000.png)
*Figure: prefill vs decode phases.*

### 4.1 Practical checklist

1. Enable KV cache for all decode paths.
2. Tune $$\tau$$ and top-p jointly on a validation prompt set.
3. Cap `max_new_tokens` and monitor cache memory $$\propto L \times d_{\mathrm{model}} \times L_{\mathrm{layers}}$$.

### 4.2 Speculative decoding

A **draft model** (small, fast) proposes $$k$$ tokens; the **target model** verifies them in parallel with one forward pass.
Accepted prefixes advance without $$k$$ serial steps—latency drops when draft and target distributions align.

### 4.3 Beam search (brief)

Beam search keeps $$B$$ highest-probability partial hypotheses. Score accumulation uses length normalization

$$
\frac{1}{t^\alpha} \sum_{i=1}^{t} \log p_\theta(x_i \mid x_{<i}),
$$

with $$\alpha \in [0,1]$$ to penalize overly short outputs. Used in translation; less common in open-ended chat where sampling dominates.

![Batched inference](/assets/figures/day10/pdf0_page010.png)
*Figure: prefill batch vs decode batch on GPU.*

## Checkpoint summary

- **Autoregressive inference** extends the sequence one token at a time.
- **KV cache** avoids recomputing past keys/values; memory grows with context length.
- **Temperature** and **top-p** control randomness vs coherence.
- **Batching + paging** maximize hardware utilization in real deployments.
