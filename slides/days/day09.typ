#import "../lib.typ": *

#show: course-theme.with(title: [Autoregressive Models & LLM Training], subtitle: [Day 9 | Aug 2026])

= Day 9: Autoregressive Models & LLM Training

== Welcome

- *Autoregressive Models & LLM Training* — Causal language modeling at scale
- 3 hours lecture + practical
- Slides, notes, and code on the course site

== Outline

- Autoregressive LM
- Training at Scale
- Alignment & Fine-tuning
- From Training to Inference

= Autoregressive LM

== Factorization

- $p(x) = product_(t=1)^T p(x_t | x_(<t); theta)$
- Causal masking enforces ordering
- Teacher forcing during training

== pdf0 page000

#align(center)[#image("/assets/figures/day09/pdf0_page000.png", width: 92%)]

#text(size: 14pt, fill: gray)[Autoregressive LM — Factorization (source: course materials)]

== Tokenization

- BPE merges frequent pairs
- Vocabulary size vs sequence length
- Special tokens: BOS, EOS, PAD

== pdf0 page004

#align(center)[#image("/assets/figures/day09/pdf0_page004.png", width: 92%)]

#text(size: 14pt, fill: gray)[Autoregressive LM — Tokenization (source: course materials)]

== Architecture Choices

- Decoder-only Transformer (GPT)
- RMSNorm, SwiGLU FFN, RoPE positions
- Grouped-query attention (GQA)

== pdf0 page008

#align(center)[#image("/assets/figures/day09/pdf0_page008.png", width: 92%)]

#text(size: 14pt, fill: gray)[Autoregressive LM — Architecture Choices (source: course materials)]

== Loss & Metrics

- Cross-entropy over next token
- Perplexity $PP = exp(H)$
- Bits-per-byte for comparison across vocabs

== pdf0 page012

#align(center)[#image("/assets/figures/day09/pdf0_page012.png", width: 92%)]

#text(size: 14pt, fill: gray)[Autoregressive LM — Loss & Metrics (source: course materials)]

= Training at Scale

== Data Pipeline

- Web crawl filtering and deduplication
- Quality heuristics and safety filters
- Mixture-of-sources ratios matter

== pdf0 page016

#align(center)[#image("/assets/figures/day09/pdf0_page016.png", width: 92%)]

#text(size: 14pt, fill: gray)[Training at Scale — Data Pipeline (source: course materials)]

== Optimization

- AdamW + cosine LR + warmup
- Gradient accumulation for large effective batch
- Loss spikes: skip step, reduce LR

== pdf0 page020

#align(center)[#image("/assets/figures/day09/pdf0_page020.png", width: 92%)]

#text(size: 14pt, fill: gray)[Training at Scale — Optimization (source: course materials)]

== Distributed Training

- Data parallel: replicate model, shard batch
- Tensor / pipeline / sequence parallel
- ZeRO optimizer state sharding

== pdf0 page024

#align(center)[#image("/assets/figures/day09/pdf0_page024.png", width: 92%)]

#text(size: 14pt, fill: gray)[Training at Scale — Distributed Training (source: course materials)]

== Checkpointing

- Save model, optimizer, RNG, step
- Resume long runs after failure
- HF format interoperability

== pdf0 page025

#align(center)[#image("/assets/figures/day09/pdf0_page025.png", width: 92%)]

#text(size: 14pt, fill: gray)[Training at Scale — Checkpointing (source: course materials)]

= Alignment & Fine-tuning

== Supervised Fine-Tuning

- Instruction-response pairs
- Catastrophic forgetting mitigation
- LoRA: low-rank adapter updates

== pdf0 page028

#align(center)[#image("/assets/figures/day09/pdf0_page028.png", width: 92%)]

#text(size: 14pt, fill: gray)[Alignment & Fine-tuning — Supervised Fine-Tuning (source: course materials)]

== RLHF Overview

- Reward model from human preferences
- PPO fine-tune policy against reward
- DPO direct preference optimization

== pdf0 page032

#align(center)[#image("/assets/figures/day09/pdf0_page032.png", width: 92%)]

#text(size: 14pt, fill: gray)[Alignment & Fine-tuning — RLHF Overview (source: course materials)]

== Evaluation

- Benchmark suites: MMLU, GSM8K, etc.
- Human eval for chat quality
- Contamination concerns

== pdf0 page035

#align(center)[#image("/assets/figures/day09/pdf0_page035.png", width: 92%)]

#text(size: 14pt, fill: gray)[Alignment & Fine-tuning — Evaluation (source: course materials)]

== Safety

- Red teaming and refusal behavior
- System prompts and moderation
- Alignment is ongoing, not solved

== pdf0 page036

#align(center)[#image("/assets/figures/day09/pdf0_page036.png", width: 92%)]

#text(size: 14pt, fill: gray)[Alignment & Fine-tuning — Safety (source: course materials)]

= From Training to Inference

== Train vs Inference Workloads

- Training: parallel across sequence (with mask)
- Inference: sequential token generation
- Memory dominated by activations vs KV

== Model Compression Preview

- Quantization INT8/INT4 weights
- Knowledge distillation
- Day 10: serving optimizations

== Open vs Closed Ecosystem

- Open weights: Llama, Mistral, etc.
- API-only: GPT-4 class models
- Responsible release considerations

== Summary

- Day 9: *Autoregressive Models & LLM Training*
- Causal language modeling at scale
- Questions welcome — practical follows

== Questions?

#align(center + horizon)[
  #text(size: 44pt, weight: "bold", fill: course-primary)[Questions?]

  #v(1em)
  Practical session follows — see course site.
]
