---
layout: post
title: Day 10 - LLM Inference and Alignment
image: /assets/img/lessons/day10.png
description: >
  How a trained base language model is served efficiently (decode loop, KV cache, sampling, systems) and lightly post-trained into a chat assistant (SFT and preference tuning).
invert_sidebar: true
---

# Day 10 - LLM Inference and Alignment

### Optional reading for this lesson
- [Aleksa Gordić — Inside the Transformer (KV cache, FLOPs, inference)](https://www.aleksagordic.com/blog/transformer)
- [Holtzman et al. — The Curious Case of Neural Text Degeneration (nucleus sampling, 2019)](https://arxiv.org/abs/1904.09751)
- [Dao et al. — FlashAttention (2022)](https://arxiv.org/abs/2205.14135)
- [Kwon et al. — Efficient Memory Management for LLM Serving with PagedAttention (vLLM, 2023)](https://arxiv.org/abs/2309.06180)
- [Leviathan et al. — Fast Inference via Speculative Decoding (2023)](https://arxiv.org/abs/2211.17192)
- [Ouyang et al. — Training Language Models to Follow Instructions (InstructGPT/RLHF, 2022)](https://arxiv.org/abs/2203.02155)
- [Rafailov et al. — Direct Preference Optimization (DPO, 2023)](https://arxiv.org/abs/2305.18290)
- [Hu et al. — LoRA: Low-Rank Adaptation (2021)](https://arxiv.org/abs/2106.09685)

### [Slides](/assets/slides/day10.pdf)

### Exercise

[Download the notebook](/notebooks/practicals/day10.ipynb) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kierandidi/ml-dl-course/blob/main/notebooks/practicals/day10.ipynb)

Day 9 left us with a **base language model**: a decoder-only Transformer that predicts the next token. Today we make it useful in two senses. First, **inference** — how to actually generate text, and how to do it efficiently. Autoregressive decoding is inherently serial, so we study the decode loop, the **KV cache** that makes it tractable, the **sampling** strategies that shape output quality, and the systems (FlashAttention, PagedAttention, continuous batching, speculative decoding, quantization) that turn a model into a service. Second, **alignment** — a light pass over how a base GPT becomes something like ChatGPT through supervised fine-tuning and preference tuning. We keep the alignment treatment at the level of ideas and diagrams; the RLHF and DPO objectives are spelled out in an optional, collapsible block for the curious. We close Week 2 — and the course — by comparing the three generative families students can now reason about: autoregressive, diffusion/score, and flows.

* toc
{:toc}

## 1. Autoregressive Decoding

### 1.1 The decode loop

> **Autoregressive decoding** generates a sequence by repeatedly sampling the next token from $$p_\theta(x_t\mid x_{<t})$$ and feeding it back as input. It runs in two phases: a parallel **prefill** of the prompt, then serial **decode** of one token at a time.
{:.lead}

![Generation proceeds one token at a time: the model's output is appended to the input and fed back in.](/assets/figures/day10/llmks_generation.png)

A trained LM is a function from a context to a distribution over the next token. To **generate**, we sample a token from that distribution, append it to the context, and run the model again — over and over. This loop has a structure worth naming:

- **Prefill.** The prompt is processed in a single parallel forward pass (just like training), producing the distribution for the first new token and populating the cache (next section).
- **Decode.** Each subsequent token requires its own forward pass that depends on all previous tokens. This phase is **inherently serial**: token $$t$$ cannot be computed until token $$t-1$$ exists.

This serial dependency is the central fact of LLM inference. It is why two latency metrics matter: **time-to-first-token** (dominated by prefill) and **inter-token latency** (dominated by decode). Everything else in the first half of today is about making this loop cheaper.

### 1.2 Exposure bias

> **Exposure bias** is the mismatch between training and generation: the model is trained with **teacher forcing** on ground-truth prefixes but at inference must condition on its **own** (possibly imperfect) past outputs.
{:.lead}

Recall from Day 9 that we train with teacher forcing: at every position the model conditions on the *true* previous tokens. At generation time there is no ground truth — the model conditions on the tokens it *itself* produced. If it makes an early mistake, that mistake becomes part of the context for every later step, and errors can compound. This gap is called **exposure bias**.

In practice, large models trained on large, clean corpora are fairly robust to it, and good **decoding strategies** (the next subsection) keep generation on the data manifold by avoiding low-probability tokens. It is still useful to understand, because it explains why naive greedy decoding can spiral into repetition and why sampling choices have an outsized effect on perceived quality.

### 1.3 Greedy, sampling, and beam search

> **Greedy** decoding takes the argmax at each step; **sampling** draws from the predicted distribution; **beam search** keeps the $$k$$ highest-probability partial sequences. The right choice depends on whether the task is precise or open-ended.
{:.lead}

![Beam search expands several high-probability partial sequences in parallel and keeps the best.](/assets/figures/day10/llmks_beam.png)

Three classic strategies, then the modern default:

- **Greedy** picks the single most likely token each step. It is deterministic but myopic and tends to produce repetitive, degenerate text in open-ended generation.
- **Beam search** maintains the top-$$k$$ partial hypotheses and expands all of them, approximating the most probable *sequence* rather than the most probable next *token*. It is the legacy workhorse of machine translation, where outputs are short and high-precision — but it actively hurts open-ended generation, which is *not* served by maximizing sequence probability.
- **Sampling** draws from the distribution, giving diversity. Raw sampling occasionally picks bad low-probability tokens, which is why we shape the distribution with temperature and truncation (next section).

For chat and creative generation, the practical default is **temperature plus nucleus (top-p) sampling**.

## 2. The KV Cache and Inference Memory

### 2.1 Why naive decoding is quadratic

> Without caching, generating token $$t$$ recomputes attention over the entire prefix of length $$t$$, so producing a length-$$T$$ sequence costs $$\mathcal{O}(T^2)$$ — even though the keys and values of past tokens never change.
{:.lead}

Consider the decode loop literally. To produce token $$t$$, attention needs the keys and values of all positions $$1,\dots,t-1$$. If we recompute everything from scratch each step, step $$t$$ does $$\mathcal{O}(t)$$ work in attention, and the whole generation costs

$$\sum_{t=1}^{T} \mathcal{O}(t) = \mathcal{O}(T^2).$$

The crucial observation: the key and value vectors for a past token are a **fixed function of that token and its position**. Once computed, they never change as generation continues. So recomputing them every step is pure waste — and removing that waste is the single most important inference optimization.

### 2.2 The KV cache

> The **KV cache** stores the key and value vectors of every past token, for every layer. Each new token computes only its own $$q,k,v$$, appends $$k,v$$ to the cache, and attends over the cache — turning per-step attention from recompute into an $$\mathcal{O}(1)$$ append plus a read.
{:.lead}

![With a KV cache, only the new token's query/key/value are computed; past keys and values are read from the cache instead of recomputed.](/assets/figures/day10/llmks_kvcache.png)

The fix follows directly. Maintain, for each layer, a growing buffer of the keys and values seen so far. At each decode step:

1. Compute the new token's query, key, and value: $$q_t, k_t, v_t$$.
2. **Append** $$k_t, v_t$$ to the layer's cache.
3. Attend: $$\mathrm{softmax}\!\big(q_t K_{\le t}^\top/\sqrt{d_k}\big) V_{\le t}$$, reading $$K,V$$ from the cache.

Now each step does a constant amount of *new* attention work (one query against the cached keys), instead of recomputing the whole prefix. **Prefill** is exactly the operation that fills the cache for the prompt in one pass; **decode** extends it one token at a time. The cost of generation drops from quadratic recompute to linear, and the loop becomes practical.

### 2.3 KV-cache memory and grouped-query attention

> The KV cache trades compute for **memory**: its size scales as $$\approx 2\,L\,n_{\text{kv}}\,d_{\text{head}}\,T$$ per sequence. Grouped-query attention reduces $$n_{\text{kv}}$$, and PagedAttention manages the cache without fragmentation.
{:.lead}

![As context and batch grow, the KV cache — not compute — becomes the bottleneck; multi-query and grouped-query attention shrink it.](/assets/figures/day10/llmks_kv_bottleneck.png)

The cache is not free. Storing keys and values for every layer and position costs, per sequence, on the order of

$$2 \times L \times n_{\text{kv}} \times d_{\text{head}} \times T$$

numbers (the leading 2 is keys *and* values; $$L$$ layers, $$n_{\text{kv}}$$ key/value heads, $$d_{\text{head}}$$ per-head width, $$T$$ tokens). This grows with **context length** and **batch size**, and at long context it — not arithmetic — becomes the binding constraint, making decode **memory-bandwidth-bound**.

Two responses connect back to Day 9 and forward to systems. **Grouped-query attention** shrinks $$n_{\text{kv}}$$ by sharing key/value heads across query heads, directly cutting cache size and bandwidth. **PagedAttention** (vLLM) stores the cache in fixed-size *pages* rather than one contiguous block, eliminating fragmentation so that far more sequences fit in memory at once — the key to high-throughput serving.

## 3. Sampling and Serving Systems

### 3.1 Temperature

> **Temperature** $$T$$ rescales the logits before the softmax, $$p_i \propto \exp(z_i / T)$$, controlling how peaked the next-token distribution is: $$T<1$$ sharpens, $$T>1$$ flattens, and $$T\to 0$$ recovers greedy decoding.
{:.lead}

![Temperature reshapes the next-token distribution before sampling, trading determinism for diversity.](/assets/figures/day10/llmks_sampling.png)

The model outputs logits $$z$$; the sampling distribution is $$p_i = \mathrm{softmax}(z/T)_i \propto \exp(z_i/T)$$. The **temperature** $$T$$ is the simplest quality/diversity knob:

- $$T < 1$$ concentrates mass on the top tokens — more focused, more deterministic, less surprising.
- $$T > 1$$ spreads mass toward the tail — more random and creative, but riskier.
- $$T \to 0$$ puts all mass on the argmax: greedy decoding.

Temperature alone still leaves a long tail of low-probability tokens in play, which is where truncation comes in.

### 3.2 Top-k and nucleus (top-p) sampling

> **Top-k** sampling restricts the draw to the $$k$$ most probable tokens; **top-p (nucleus)** sampling restricts it to the smallest set of tokens whose cumulative probability exceeds $$p$$. Both truncate the unreliable tail.
{:.lead}

![Top-k keeps a fixed number of candidates; top-p keeps a dynamic set covering probability mass p.](/assets/figures/day10/llmks_topk_topp.png)

Holtzman et al. showed that the long tail of the distribution is where text degeneration comes from: even a tiny per-step probability of a bad token, accumulated over hundreds of steps, produces incoherent output. Two truncations fix this:

- **Top-k**: keep only the $$k$$ highest-probability tokens, renormalize, and sample. Simple, but a fixed $$k$$ is too aggressive when the model is uncertain and too permissive when it is confident.
- **Top-p (nucleus)**: keep the smallest set of tokens whose probabilities sum to at least $$p$$ (e.g. 0.9), renormalize, and sample. The candidate set **adapts** to the step's uncertainty, which is why nucleus sampling is the common default, usually combined with a temperature.

### 3.3 Throughput: prefill vs decode and batching

> **Prefill** is compute-bound and parallel; **decode** is memory-bandwidth-bound because it reuses the weights for a single token per step. **Batching** many requests amortizes weight reads and is the main lever for throughput.
{:.lead}

The two inference phases have opposite bottlenecks. **Prefill** processes many prompt tokens at once, so it is **compute-bound** — lots of arithmetic per weight read. **Decode** produces one token per step, reading the entire weight matrix to make a single prediction, so it is **memory-bandwidth-bound** — the GPU spends its time moving weights, not computing.

The fix for decode is **batching**: process many sequences together so that one read of the weights serves many tokens, amortizing the bandwidth cost. Naive static batching wastes capacity when sequences finish at different times, so production servers use **continuous (in-flight) batching**, adding and removing requests from the running batch as they arrive and complete. The metrics that matter for a service are **throughput** (tokens/sec across all users), **time-to-first-token**, and **inter-token latency**, and they trade off against one another.

### 3.4 FlashAttention, PagedAttention, speculative decoding, quantization

> Systems techniques attack the two inference costs: **FlashAttention** makes attention IO-efficient; **PagedAttention** manages the KV cache; **speculative decoding** verifies many draft tokens at once; **quantization** shrinks weights to INT8/INT4.
{:.lead}

A short tour of the techniques behind fast serving, each targeting a specific cost:

- **FlashAttention** computes exact attention without ever materializing the full $$N\times N$$ score matrix. It tiles the computation to keep data in fast on-chip SRAM, which both speeds up training and prefill and cuts memory from quadratic to linear in sequence length.
- **PagedAttention** (vLLM) stores the KV cache in non-contiguous fixed-size pages, eliminating the memory fragmentation that otherwise caps batch size — directly raising throughput.
- **Speculative decoding** runs a small, cheap **draft** model to propose several tokens, then has the large model **verify** them all in one parallel pass. Tokens the big model agrees with are accepted for free, so several tokens can be emitted per big-model step **without changing the output distribution**.
- **Quantization** stores weights (and sometimes activations and the KV cache) in INT8 or INT4 instead of 16-bit floats, shrinking memory and bandwidth — the dominant decode cost — usually with minimal quality loss.

All of these chip away at the same problem: autoregressive decode is serial and memory-bound. This is the inference analogue of Day 8's fight against the many-step cost of diffusion sampling.

## 4. From GPT to ChatGPT (Light)

### 4.1 From base model to assistant

> A **base LM** predicts internet text; it is not intrinsically helpful or safe. The path to an assistant is **pretrain → supervised fine-tuning (SFT) → preference tuning**, followed by a system prompt and safety layer — each stage using far less data than pretraining.
{:.lead}

![Transformers enabled a change of paradigm: one scalable, general model adapted to many uses.](/assets/figures/day10/llmks_paradigm_shift.png)

The base model from Day 9 is extraordinarily knowledgeable but only knows how to *continue text*. Ask it a question and it might continue with more questions. Turning it into a helpful, harmless assistant is **post-training**, a pipeline of comparatively small steps on top of pretraining:

1. **Pretraining** (Day 9): next-token prediction on internet-scale text → a base model.
2. **Supervised fine-tuning (SFT):** imitate high-quality example conversations → an instruction-following model.
3. **Preference tuning (RLHF or DPO):** optimize toward human judgments of which response is better → an aligned model.
4. **System prompt, safety, and tools:** runtime conditioning and guardrails.

Each later stage uses orders of magnitude less data than pretraining, yet has an outsized effect on how the model *behaves*. The next two subsections cover stages 2 and 3 at the level of ideas; the underlying RLHF/DPO objectives are in the optional block.

### 4.2 Supervised fine-tuning and parameter-efficient adaptation

> **Supervised fine-tuning (SFT)** continues next-token training on curated (instruction, response) conversations, with the loss masked to the assistant's tokens. **LoRA / PEFT** adapt a model by training small added matrices instead of all weights.
{:.lead}

**SFT** is conceptually just more Day 9 training, with two twists. The data is a curated set of high-quality conversations in a fixed chat format (using the role special tokens from Day 9), and the cross-entropy loss is **masked to the assistant turns** so the model learns to *produce* good responses, not to predict the user's messages. This single step already gives a usable instruction-following model; data *quality* matters far more than quantity.

Fine-tuning all the weights of a large model is expensive, so **parameter-efficient fine-tuning** is common. **LoRA** (low-rank adaptation) freezes the pretrained weights and learns small low-rank update matrices $$\Delta W = BA$$ added to selected layers; only $$A,B$$ are trained and stored. This makes adaptation cheap, swappable, and easy to share, with quality close to full fine-tuning.

<details class="optional-derivation" markdown="1">
<summary><strong>RLHF and DPO objectives (optional deep dive)</strong> (optional — click to expand)</summary>

This block sketches the math behind preference tuning, for students who want it. The lecture itself only requires the *concept*: optimize the model toward responses humans prefer.

**Setup.** Collect preference data: for a prompt $$x$$, annotators compare two responses and label the winner $$y_w$$ over the loser $$y_l$$. We assume a **Bradley–Terry** model in which the probability that $$y_w$$ is preferred depends on a latent reward $$r(x,y)$$:

$$p(y_w \succ y_l \mid x) = \sigma\big(r(x,y_w) - r(x,y_l)\big).$$

**RLHF (InstructGPT).** Two steps. First fit a **reward model** $$r_\phi$$ by maximizing the likelihood of the preference data above. Then optimize the policy $$\pi_\theta$$ (the LM) to maximize expected reward while staying close to the SFT reference $$\pi_{\text{ref}}$$, using a KL penalty to prevent reward hacking:

$$\max_{\theta}\; \mathbb{E}_{x,\,y\sim\pi_\theta}\big[r_\phi(x,y)\big] - \beta\, \mathrm{KL}\big(\pi_\theta(\cdot\mid x)\,\Vert\,\pi_{\text{ref}}(\cdot\mid x)\big).$$

This is optimized with a policy-gradient method (**PPO**), which requires sampling from the model during training — powerful but operationally heavy (a reward model plus an RL loop).

**DPO (Direct Preference Optimization).** The key insight is that the KL-regularized objective above has a closed-form optimal policy, which can be inverted to express the implicit reward in terms of the policy itself: $$r(x,y) = \beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \text{const}$$. Substituting into the Bradley–Terry likelihood turns preference learning into a **simple classification loss on the policy**, with no reward model and no RL:

$$\mathcal{L}_{\text{DPO}} = -\,\mathbb{E}_{(x,y_w,y_l)}\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right)\right].$$

DPO reaches similar alignment quality with a far simpler, more stable training recipe, which is why it is widely used. Either way, the takeaway for this course is the same: a small amount of preference data, used to nudge the model toward human-preferred responses, is what separates a raw next-token predictor from a helpful assistant.

</details>

### 4.3 Preference tuning and the course recap

> **Preference tuning** aligns the SFT model with human judgments using preference comparisons, via RLHF (reward model + PPO) or DPO (a direct classification loss). It is the final stage that makes a model feel like a helpful assistant.
{:.lead}

![Week 2 in one picture: a single Transformer backbone, trained as a generator, serves text, vision, audio, and biology.](/assets/figures/day10/llmks_summary.png)

**Preference tuning** is stage 3 of the pipeline. We gather comparisons — humans (or a model) judging which of two responses is better — and use them to push the model toward preferred behavior. **RLHF** does this with a learned reward model optimized by PPO; **DPO** does it with a single direct loss and no RL loop (objectives in the optional block above). Benchmarks such as MMLU are named here only as further reading; evaluating aligned models is its own large topic.

**Course recap.** Over ten days we went from linear algebra and statistical learning to deep networks, CNNs, and Transformers (Week 1), then to generative modeling: the DDPM/score/SDE/flow family (Days 6-8) and the autoregressive LLM family (Days 9-10). You can now place a problem on the generative map and choose a family — **autoregressive** (exact likelihood, serial decode), **diffusion/score** (iterative denoising, guidance), or **flows** (deterministic ODE transport) — and you have seen that one backbone, the Transformer, underlies text, vision, audio, and biological sequence modeling. That is the throughline to carry forward: a small set of ideas, scaled.

## Checkpoint summary

Before moving to the practical, confirm you can:

- Describe the autoregressive decode loop and distinguish the prefill and decode phases and their latency metrics.
- Explain exposure bias and why decoding strategy affects generation quality.
- Contrast greedy, sampling, and beam search, and say when each is appropriate.
- Show why naive decoding is $$\mathcal{O}(T^2)$$ and how the KV cache makes it linear.
- Write the KV-cache memory scaling and explain how grouped-query attention and PagedAttention reduce or manage it.
- Explain temperature, top-k, and top-p (nucleus) sampling and why truncating the tail prevents degeneration.
- Explain why decode is memory-bandwidth-bound and how batching, FlashAttention, PagedAttention, speculative decoding, and quantization speed up serving.
- Lay out the pretrain → SFT → preference-tuning pipeline and explain, at a high level, how RLHF and DPO differ.
