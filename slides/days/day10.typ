#import "../lib.typ": *

#show: course-theme.with(title: [LLM Inference & Alignment], subtitle: [Day 10 | Aug 2026])

= Day 10: LLM Inference & Alignment

== Welcome

- *LLM Inference & Alignment* — From a base model to a usable assistant
- 3 hours lecture + practical
- Slides, notes, and code on the course site

== Outline

- Autoregressive Decoding
- KV Cache & Memory
- Sampling & Systems
- From GPT to ChatGPT (light)

= 1 · Autoregressive Decoding

== 1.1  The Decode Loop

- Day 9 gave us $p_theta (x_t | x_(<t))$ — now we *generate*
- Sample $x_t$, append it, feed back, repeat
- Inherently *serial*: token $t$ needs tokens $< t$
- Two phases: prefill (the prompt) then decode (one by one)
- Latency = time-to-first-token + per-token time

== 1.1  The Decode Loop

#align(center + horizon)[#image("/assets/figures/day10/llmks_generation.png", width: 92%, height: 82%, fit: "contain")]

== 1.2  Exposure Bias

- Training: teacher forcing on ground-truth context
- Generation: model conditions on its *own* past outputs
- Small errors can compound over a long generation
- Mitigated by scale, good data, and decoding choices
- A reason sampling strategy matters at inference

== 1.3  Greedy, Sampling, and Beam

- Greedy: take the argmax each step — repetitive, brittle
- Sampling: draw from the distribution — diverse
- Beam search: keep top-$k$ partial sequences (translation legacy)
- Beam helps short, high-precision outputs; hurts open-ended text
- Modern chat: temperature + nucleus sampling

== 1.3  Greedy, Sampling, and Beam

#align(center + horizon)[#image("/assets/figures/day10/llmks_beam.png", width: 92%, height: 82%, fit: "contain")]

= 2 · KV Cache & Memory

== 2.1  Why Decode Is Quadratic Without a Cache

- Naive decode recomputes attention over the whole prefix
- Step $t$ costs $O(t)$; a full sequence costs $O(T^2)$
- But keys/values of past tokens never change
- Idea: compute them once, store, and reuse
- This is the single biggest inference optimization

== 2.2  The KV Cache

- Cache $K, V$ for every layer and past position
- Each new token computes its own $q, k, v$
- Append $k, v$ to the cache; attend over the cache
- Per-step cost drops from $O(t)$ recompute to $O(1)$ append
- Prefill fills the cache; decode extends it

== 2.2  The KV Cache

#align(center + horizon)[#image("/assets/figures/day10/llmks_kvcache.png", width: 92%, height: 82%, fit: "contain")]

== 2.3  KV-Cache Memory and GQA

- Cache size ~ $2 dot L dot n_"kv" dot d_"head" dot T$ per sequence
- Grows with context length $T$ and batch size
- Often the real bottleneck — bandwidth and memory bound
- Grouped-query attention (Day 9) shrinks $n_"kv"$
- PagedAttention manages the cache like virtual memory

== 2.3  KV-Cache Memory and GQA

#align(center + horizon)[#image("/assets/figures/day10/llmks_kv_bottleneck.png", width: 92%, height: 82%, fit: "contain")]

= 3 · Sampling & Systems

== 3.1  Temperature

- Rescale logits before softmax: $p_i prop exp(z_i \\/ T)$
- $T < 1$: sharper, more deterministic
- $T > 1$: flatter, more random
- $T arrow.r 0$ recovers greedy decoding
- The first knob for the quality/diversity trade-off

== 3.1  Temperature

#align(center + horizon)[#image("/assets/figures/day10/llmks_sampling.png", width: 92%, height: 82%, fit: "contain")]

== 3.2  Top-k and Top-p (Nucleus)

- Top-$k$: sample only from the $k$ most likely tokens
- Top-$p$: smallest set whose probability mass exceeds $p$
- Truncates the unreliable long tail of the distribution
- Nucleus (top-$p$) adapts the cutoff to each step
- Common default: temperature + top-$p$ together

== 3.2  Top-k and Top-p (Nucleus)

#align(center + horizon)[#image("/assets/figures/day10/llmks_topk_topp.png", width: 92%, height: 82%, fit: "contain")]

== 3.3  Throughput: Prefill vs Decode

- Prefill is compute-bound; decode is memory-bandwidth-bound
- Decode reuses weights for one token at a time
- Batch many requests to amortize weight reads
- Continuous batching: add/remove requests mid-flight
- Metrics: tokens/sec, TTFT, inter-token latency

== 3.4  FlashAttention & PagedAttention

- FlashAttention: IO-aware exact attention, no big $N times N$ matrix
- Tiles the computation in fast on-chip memory
- Speeds up training and prefill, saves memory
- PagedAttention (vLLM): non-contiguous KV cache in pages
- Less fragmentation $arrow.r$ higher batch sizes, more throughput

== 3.5  Speculative Decoding & Quantization

- Speculative: a small draft model proposes several tokens
- The big model *verifies* them in one parallel pass
- Accepted tokens are free — same distribution, fewer big-model steps
- Quantization: INT8 / INT4 weights shrink memory + bandwidth
- Both attack the serial / memory-bound decode cost

= 4 · From GPT to ChatGPT (light)

== 4.1  From Base Model to Assistant

- Base LM (Day 9): predicts internet text, not 'helpful'
- Pipeline: pretrain $arrow.r$ SFT $arrow.r$ preference tuning
- Then: system prompt + safety + tools
- Each stage uses far less data than pretraining
- Turns a next-token predictor into a chat model

== 4.1  From Base Model to Assistant

#align(center + horizon)[#image("/assets/figures/day10/llmks_paradigm_shift.png", width: 92%, height: 82%, fit: "contain")]

== 4.2  Supervised Fine-Tuning (SFT)

- Fine-tune on curated (instruction, response) conversations
- Same next-token loss, masked to the *assistant* tokens
- Teaches format, instruction-following, tone
- LoRA / PEFT: adapt cheaply without full fine-tuning
- Small, high-quality data matters more than volume

== 4.3  Preference Tuning (RLHF / DPO)

- Collect human preferences: response A vs response B
- RLHF: train a reward model, optimize it with PPO
- DPO: optimize the preferences directly, no RL loop
- Aligns the model with human judgments of quality
- Details in the optional notes — concept is the takeaway

== 4.4  Course Recap

- Week 1: ML/DL foundations $arrow.r$ CNNs $arrow.r$ Transformers
- Week 2: generative models, diffusion, and autoregressive LLMs
- Three generative paths: AR, diffusion/score, flows
- Same backbone — Transformers — powers all of it
- Transformers everywhere: text, vision, audio, biology

== 4.4  Course Recap

#align(center + horizon)[#image("/assets/figures/day10/llmks_summary.png", width: 92%, height: 82%, fit: "contain")]

== Summary

- Day 10: *LLM Inference & Alignment*
- From a base model to a usable assistant
- Questions welcome — practical follows

== Questions?

#align(center + horizon)[
  #text(size: 44pt, weight: "bold", fill: course-primary)[Questions?]

  #v(1em)
  Practical session follows — see course site.
]
