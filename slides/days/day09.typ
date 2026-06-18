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

= 1 · Autoregressive LM

== 1.1  Factorization

- $p(x) = product_(t=1)^T p(x_t | x_(<t); theta)$
- Causal masking enforces ordering
- Teacher forcing during training

== 1.1  Factorization

#align(center + horizon)[#image("/assets/figures/day09/pdf0_page000.png", width: 92%, height: 82%, fit: "contain")]

== 1.2  Tokenization

- BPE merges frequent pairs
- Vocabulary size vs sequence length
- Special tokens: BOS, EOS, PAD

== 1.2  Tokenization

#align(center + horizon)[#image("/assets/figures/day09/pdf0_page004.png", width: 92%, height: 82%, fit: "contain")]

== 1.3  Architecture Choices

- Decoder-only Transformer (GPT)
- RMSNorm, SwiGLU FFN, RoPE positions
- Grouped-query attention (GQA)

== 1.3  Architecture Choices

#align(center + horizon)[#image("/assets/figures/day09/pdf0_page008.png", width: 92%, height: 82%, fit: "contain")]

== 1.4  Loss & Metrics

- Cross-entropy over next token
- Perplexity $PP = exp(H)$
- Bits-per-byte for comparison across vocabs

== 1.4  Loss & Metrics

#align(center + horizon)[#image("/assets/figures/day09/pdf0_page012.png", width: 92%, height: 82%, fit: "contain")]

= 2 · Training at Scale

== 2.1  Data Pipeline

- Web crawl filtering and deduplication
- Quality heuristics and safety filters
- Mixture-of-sources ratios matter

== 2.1  Data Pipeline

#align(center + horizon)[#image("/assets/figures/day09/pdf0_page016.png", width: 92%, height: 82%, fit: "contain")]

== 2.2  Optimization

- AdamW + cosine LR + warmup
- Gradient accumulation for large effective batch
- Loss spikes: skip step, reduce LR

== 2.2  Optimization

#align(center + horizon)[#image("/assets/figures/day09/pdf0_page020.png", width: 92%, height: 82%, fit: "contain")]

== 2.3  Distributed Training

- Data parallel: replicate model, shard batch
- Tensor / pipeline / sequence parallel
- ZeRO optimizer state sharding

== 2.3  Distributed Training

#align(center + horizon)[#image("/assets/figures/day09/pdf0_page024.png", width: 92%, height: 82%, fit: "contain")]

== 2.4  Checkpointing

- Save model, optimizer, RNG, step
- Resume long runs after failure
- HF format interoperability

== 2.4  Checkpointing

#align(center + horizon)[#image("/assets/figures/day09/pdf0_page025.png", width: 92%, height: 82%, fit: "contain")]

= 3 · Alignment & Fine-tuning

== 3.1  Supervised Fine-Tuning

- Instruction-response pairs
- Catastrophic forgetting mitigation
- LoRA: low-rank adapter updates

== 3.1  Supervised Fine-Tuning

#align(center + horizon)[#image("/assets/figures/day09/pdf0_page028.png", width: 92%, height: 82%, fit: "contain")]

== 3.2  RLHF Overview

- Reward model from human preferences
- PPO fine-tune policy against reward
- DPO direct preference optimization

== 3.2  RLHF Overview

#align(center + horizon)[#image("/assets/figures/day09/pdf0_page032.png", width: 92%, height: 82%, fit: "contain")]

== 3.3  Evaluation

- Benchmark suites: MMLU, GSM8K, etc.
- Human eval for chat quality
- Contamination concerns

== 3.3  Evaluation

#align(center + horizon)[#image("/assets/figures/day09/pdf0_page035.png", width: 92%, height: 82%, fit: "contain")]

== 3.4  Safety

- Red teaming and refusal behavior
- System prompts and moderation
- Alignment is ongoing, not solved

== 3.4  Safety

#align(center + horizon)[#image("/assets/figures/day09/pdf0_page036.png", width: 92%, height: 82%, fit: "contain")]

= 4 · From Training to Inference

== 4.1  Train vs Inference Workloads

- Training: parallel across sequence (with mask)
- Inference: sequential token generation
- Memory dominated by activations vs KV

== 4.2  Model Compression Preview

- Quantization INT8/INT4 weights
- Knowledge distillation
- Day 10: serving optimizations

== 4.3  Open vs Closed Ecosystem

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
