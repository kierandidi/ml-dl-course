#import "../lib.typ": *

#show: course-theme.with(title: [Autoregressive Inference], subtitle: [Day 10 | Aug 2026])

= Day 10: Autoregressive Inference

== Welcome

- *Autoregressive Inference* — Decoding strategies and KV cache
- 3 hours lecture + practical
- Slides, notes, and code on the course site

== Outline

- Text Generation Loop
- KV Cache
- Efficient Attention
- Serving & Systems

= Text Generation Loop

== Autoregressive Decoding

- Start from prompt tokens
- Repeat: forward pass → logits → sample/argmax → append
- Stop at EOS or max length

== pdf0 page000

#align(center)[#image("/assets/figures/day10/pdf0_page000.png", width: 92%)]

#text(size: 14pt, fill: gray)[Text Generation Loop — Autoregressive Decoding (source: course materials)]

== Greedy & Beam Search

- Greedy: $x_t = "arg max" p(x_t|x_(<t))$
- Beam search keeps top-$k$ partial sequences
- Deterministic but often dull

== pdf0 page002

#align(center)[#image("/assets/figures/day10/pdf0_page002.png", width: 92%)]

#text(size: 14pt, fill: gray)[Text Generation Loop — Greedy & Beam Search (source: course materials)]

== Sampling Methods

- Temperature $tau$: soften $"softmax"("logits"/tau)$
- Top-$k$ and nucleus (top-$p$) filtering
- Repetition penalty

== pdf0 page004

#align(center)[#image("/assets/figures/day10/pdf0_page004.png", width: 92%)]

#text(size: 14pt, fill: gray)[Text Generation Loop — Sampling Methods (source: course materials)]

== Latency Metrics

- Time to first token (TTFT)
- Tokens per second (TPS)
- Prefill vs decode phases

== pdf0 page005

#align(center)[#image("/assets/figures/day10/pdf0_page005.png", width: 92%)]

#text(size: 14pt, fill: gray)[Text Generation Loop — Latency Metrics (source: course materials)]

= KV Cache

== Motivation

- Self-attention recomputes keys/values for all past tokens
- At step $t$, past $K,V$ are unchanged
- Cache avoids $O(t^2)$ redundant work per step

== pdf0 page006

#align(center)[#image("/assets/figures/day10/pdf0_page006.png", width: 92%)]

#text(size: 14pt, fill: gray)[KV Cache — Motivation (source: course materials)]

== Cache Structure

- Store $K,V$ per layer per head
- Memory $O(L dot H dot T dot d_h)$ grows with context
- Batching pads to max length in batch

== pdf0 page008

#align(center)[#image("/assets/figures/day10/pdf0_page008.png", width: 92%)]

#text(size: 14pt, fill: gray)[KV Cache — Cache Structure (source: course materials)]

== Prefill vs Decode

- Prefill: process prompt in parallel (compute-bound)
- Decode: one token at a time (memory-bandwidth)
- Continuous batching in serving systems

== pdf0 page010

#align(center)[#image("/assets/figures/day10/pdf0_page010.png", width: 92%)]

#text(size: 14pt, fill: gray)[KV Cache — Prefill vs Decode (source: course materials)]

== Multi-Query / GQA

- Share K,V heads across query heads
- Reduces cache size with minimal quality loss
- Standard in modern LLM inference

== pdf0 page012

#align(center)[#image("/assets/figures/day10/pdf0_page012.png", width: 92%)]

#text(size: 14pt, fill: gray)[KV Cache — Multi-Query / GQA (source: course materials)]

= Efficient Attention

== FlashAttention

- Tiled softmax without materializing full $n times n$
- IO-aware — faster on GPU memory hierarchy
- Training and prefill benefit most

== pdf0 page014

#align(center)[#image("/assets/figures/day10/pdf0_page014.png", width: 92%)]

#text(size: 14pt, fill: gray)[Efficient Attention — FlashAttention (source: course materials)]

== PagedAttention (vLLM)

- Non-contiguous KV blocks like virtual memory
- Reduces fragmentation in batched serving
- Higher GPU utilization

== pdf0 page016

#align(center)[#image("/assets/figures/day10/pdf0_page016.png", width: 92%)]

#text(size: 14pt, fill: gray)[Efficient Attention — PagedAttention (vLLM) (source: course materials)]

== Speculative Decoding

- Draft model proposes several tokens
- Target model verifies in parallel
- Acceptance rate determines speedup

== pdf0 page018

#align(center)[#image("/assets/figures/day10/pdf0_page018.png", width: 92%)]

#text(size: 14pt, fill: gray)[Efficient Attention — Speculative Decoding (source: course materials)]

== Long Context

- RoPE scaling, YaRN, ALiBi
- Ring attention for very long sequences
- Context window vs true reasoning

== pdf1 page000

#align(center)[#image("/assets/figures/day10/pdf1_page000.png", width: 92%)]

#text(size: 14pt, fill: gray)[Efficient Attention — Long Context (source: course materials)]

= Serving & Systems

== Quantization for Inference

- Weight-only INT4 (GPTQ, AWQ)
- KV cache quantization
- Accuracy vs throughput tradeoffs

== Batching Strategies

- Static vs continuous batching
- Request scheduling and preemption
- Multi-tenant SLA constraints

== Course Recap

- Week 1: ML/DL foundations → Transformers
- Week 2: generative models → production LLM inference
- Final assessment ties math + code together

== Summary

- Day 10: *Autoregressive Inference*
- Decoding strategies and KV cache
- Questions welcome — practical follows

== Questions?

#align(center + horizon)[
  #text(size: 44pt, weight: "bold", fill: course-primary)[Questions?]

  #v(1em)
  Practical session follows — see course site.
]
