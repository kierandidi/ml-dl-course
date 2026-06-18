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

= 1 · Text Generation Loop

== 1.1  Autoregressive Decoding

- Start from prompt tokens
- Repeat: forward pass → logits → sample/argmax → append
- Stop at EOS or max length

== 1.1  Autoregressive Decoding

#align(center + horizon)[#image("/assets/figures/day10/pdf0_page000.png", width: 92%, height: 82%, fit: "contain")]

== 1.2  Greedy & Beam Search

- Greedy: $x_t = "arg max" p(x_t|x_(<t))$
- Beam search keeps top-$k$ partial sequences
- Deterministic but often dull

== 1.2  Greedy & Beam Search

#align(center + horizon)[#image("/assets/figures/day10/pdf0_page002.png", width: 92%, height: 82%, fit: "contain")]

== 1.3  Sampling Methods

- Temperature $tau$: soften $"softmax"("logits"/tau)$
- Top-$k$ and nucleus (top-$p$) filtering
- Repetition penalty

== 1.3  Sampling Methods

#align(center + horizon)[#image("/assets/figures/day10/pdf0_page004.png", width: 92%, height: 82%, fit: "contain")]

== 1.4  Latency Metrics

- Time to first token (TTFT)
- Tokens per second (TPS)
- Prefill vs decode phases

== 1.4  Latency Metrics

#align(center + horizon)[#image("/assets/figures/day10/pdf0_page005.png", width: 92%, height: 82%, fit: "contain")]

= 2 · KV Cache

== 2.1  Motivation

- Self-attention recomputes keys/values for all past tokens
- At step $t$, past $K,V$ are unchanged
- Cache avoids $O(t^2)$ redundant work per step

== 2.1  Motivation

#align(center + horizon)[#image("/assets/figures/day10/pdf0_page006.png", width: 92%, height: 82%, fit: "contain")]

== 2.2  Cache Structure

- Store $K,V$ per layer per head
- Memory $O(L dot H dot T dot d_h)$ grows with context
- Batching pads to max length in batch

== 2.2  Cache Structure

#align(center + horizon)[#image("/assets/figures/day10/pdf0_page008.png", width: 92%, height: 82%, fit: "contain")]

== 2.3  Prefill vs Decode

- Prefill: process prompt in parallel (compute-bound)
- Decode: one token at a time (memory-bandwidth)
- Continuous batching in serving systems

== 2.3  Prefill vs Decode

#align(center + horizon)[#image("/assets/figures/day10/pdf0_page010.png", width: 92%, height: 82%, fit: "contain")]

== 2.4  Multi-Query / GQA

- Share K,V heads across query heads
- Reduces cache size with minimal quality loss
- Standard in modern LLM inference

== 2.4  Multi-Query / GQA

#align(center + horizon)[#image("/assets/figures/day10/pdf0_page012.png", width: 92%, height: 82%, fit: "contain")]

= 3 · Efficient Attention

== 3.1  FlashAttention

- Tiled softmax without materializing full $n times n$
- IO-aware — faster on GPU memory hierarchy
- Training and prefill benefit most

== 3.1  FlashAttention

#align(center + horizon)[#image("/assets/figures/day10/pdf0_page014.png", width: 92%, height: 82%, fit: "contain")]

== 3.2  PagedAttention (vLLM)

- Non-contiguous KV blocks like virtual memory
- Reduces fragmentation in batched serving
- Higher GPU utilization

== 3.2  PagedAttention (vLLM)

#align(center + horizon)[#image("/assets/figures/day10/pdf0_page016.png", width: 92%, height: 82%, fit: "contain")]

== 3.3  Speculative Decoding

- Draft model proposes several tokens
- Target model verifies in parallel
- Acceptance rate determines speedup

== 3.3  Speculative Decoding

#align(center + horizon)[#image("/assets/figures/day10/pdf0_page018.png", width: 92%, height: 82%, fit: "contain")]

== 3.4  Long Context

- RoPE scaling, YaRN, ALiBi
- Ring attention for very long sequences
- Context window vs true reasoning

== 3.4  Long Context

#align(center + horizon)[#image("/assets/figures/day10/pdf1_page000.png", width: 92%, height: 82%, fit: "contain")]

= 4 · Serving & Systems

== 4.1  Quantization for Inference

- Weight-only INT4 (GPTQ, AWQ)
- KV cache quantization
- Accuracy vs throughput tradeoffs

== 4.2  Batching Strategies

- Static vs continuous batching
- Request scheduling and preemption
- Multi-tenant SLA constraints

== 4.3  Course Recap

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
