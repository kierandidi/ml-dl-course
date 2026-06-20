---
layout: post
title: Day 9 - Autoregressive Language Models
image: /assets/img/lessons/day09.png
description: >
  Returning to the autoregressive family from Day 6: build and train a decoder-only Transformer language model, from tokens to logits, following Gordić, a Beyer-style sequence, and nanoGPT.
invert_sidebar: true
---

# Day 9 - Autoregressive Language Models

### Optional reading for this lesson
- [Aleksa Gordić — Inside the Transformer (the life of a token)](https://www.aleksagordic.com/blog/transformer)
- [Andrej Karpathy — nanoGPT (README + model.py)](https://github.com/karpathy/nanoGPT)
- [Vaswani et al. — Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762)
- [Su et al. — RoFormer: Rotary Position Embedding (2021)](https://arxiv.org/abs/2104.09864)
- [Peng et al. — YaRN: Efficient Context Window Extension (2023)](https://arxiv.org/abs/2309.00071)
- [Shazeer — GLU Variants Improve Transformer (2020)](https://arxiv.org/abs/2002.05202)

### [Slides](/assets/slides/day09.pdf)

### Exercise

[Download the notebook](/notebooks/practicals/day09.ipynb) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kierandidi/ml-dl-course/blob/main/notebooks/practicals/day09.ipynb)

On Day 6 we drew the map of deep generative models and placed **autoregressive models** on it as the likelihood-based, *exact* family — then spent Days 6-8 inside the diffusion/score/flow world because it dominates image and audio generation. Today we return to autoregressive modeling, because large language models are where it *is* the main story. We take the Transformer block built on Day 5 and turn it into a production **decoder-only** language model, following the 'life of a token' from raw text to a trained next-token predictor. We tokenize text, look up embeddings, walk through every submodule of a modern pre-norm block (RMSNorm, causal multi-head and grouped-query attention, rotary position embeddings, a gated MLP), stack it into a full GPT forward pass, and train it with next-token cross-entropy. Throughout, [Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) is the concrete codebase we trace, and the practical has you build and train one yourself. Day 10 then covers how this base model is served efficiently and turned into a chat assistant.

* toc
{:toc}

## 1. Back to Autoregressive: the Week 2 Arc

### 1.1 Where we are on the generative map

> An **autoregressive model** factorizes a joint distribution into a product of next-element conditionals, $$p(x) = \prod_{t=1}^{T} p(x_t \mid x_{<t}),$$ and is trained by exact maximum likelihood — the *exact* likelihood-based corner of the Day 6 taxonomy.
{:.lead}

![The Day 6 map of deep generative models: a shared goal (turn a simple reference distribution into $$p_{\text{data}}$$) reached by different families. Autoregressive models sit in the exact-likelihood corner; diffusion in the bounded-likelihood corner (Principles Fig 1.2).](/assets/figures/day06/pdm_dgm_zoo.png)

On **Day 6** we established a single goal shared by every deep generative model: transform a simple reference distribution (noise) into the data distribution $$p_{\text{data}}$$. We then sorted the major families:

- **Likelihood-based, exact:** normalizing flows and **autoregressive models** — they compute $$\log p_\theta(x)$$ directly.
- **Likelihood-based, bounded:** VAEs and **diffusion** — they optimize a bound (ELBO) or a score-matching surrogate.
- **Implicit:** GANs — no explicit density, just a sampler.

**Days 6-8** then went deep into diffusion, score-based models, SDEs, and flow matching, because that family dominates image and audio generation today. We even flagged, back on Day 6, that autoregressive modeling was named but deferred. **Today is the promised return.** Large language models are the place where the autoregressive factorization above *is* the dominant paradigm, so we now build one end to end using the Transformer machinery from Day 5.

### 1.2 Autoregressive vs diffusion, side by side

> Both families pursue the same generative goal but differ in how they **factorize** the distribution, what **loss** they optimize, and how they **sample**: autoregressive models decode one token at a time; diffusion models integrate dynamics over many steps.
{:.lead}

It is worth making the contrast explicit, because the two halves of Week 2 are two routes to the *same* destination:

| | **Autoregressive LM (Days 9-10)** | **Diffusion (Days 6-8)** |
|--|-----------------------------------|---------------------------|
| Factorization | $$p(x)=\prod_t p(x_t\mid x_{<t})$$ | score / flow on the marginals $$p_t$$ |
| Likelihood | Exact $$\log p_\theta(x)$$ via the chain rule | ELBO / score matching (bound or implicit) |
| Training target | Next-token cross-entropy | Denoise / score / velocity regression |
| Sampling | **One token at a time** (serial) | **Many steps** along a schedule or ODE |
| Architecture | Decoder-only Transformer | Same backbone possible; different objective + sampler |
| Where Week 1 fits | Day 5 attention **implements** the conditional | Day 3-4 nets parameterize the denoiser |

A useful through-line for the rest of Week 2: diffusion fought the cost of *many sampling steps* (NFEs) with high-order solvers and flow maps (Day 8); autoregressive models fight the cost of *serial token-by-token decoding* with the KV cache and batching (Day 10). Same "make generation fast" theme, different bottleneck.

### 1.3 The factorization and the architecture families

> A **decoder-only** Transformer is a causal model that, at every position, outputs a distribution over the next token. Stacked and masked, it computes **all** the conditionals $$p(x_t\mid x_{<t})$$ of a sequence in a single parallel forward pass.
{:.lead}

![After Transformers, one architecture family — scaled with data and compute — replaced the per-task zoo of older NLP systems (Beyer-style deck).](/assets/figures/day09/llmks_paradigm.png)

The chain rule (Day 5) is exact and fully general:

$$p(x) = \prod_{t=1}^{T} p(x_t \mid x_1,\dots,x_{t-1}).$$

We model every conditional with **one shared network** and train by maximum likelihood, i.e. minimizing the negative log-likelihood, which for discrete tokens is the **next-token cross-entropy**. Three architecture families build on the Transformer block:

- **Encoder-only** (BERT): bidirectional attention, great for *understanding* tasks (classification, retrieval), but not a generator.
- **Encoder-decoder** (T5, classic NMT): an encoder reads the source, a decoder generates the target — natural for sequence-to-sequence.
- **Decoder-only** (GPT, chat models): a single causal stack that predicts the next token. This is the architecture behind modern LLMs, and the one we build today.

The decoder-only choice is what makes the causal mask from Day 5 so important: it lets us evaluate every next-token prediction in the sequence at once during training, while still forbidding each position from seeing the future.

## 2. From Text to Tokens

### 2.1 Tokenization with byte-pair encoding

> **Tokenization** maps a string to a sequence of integer ids from a fixed **vocabulary**. **Byte-pair encoding (BPE)** builds that vocabulary by greedily merging the most frequent adjacent symbol pairs, yielding *subword* units between characters and whole words.
{:.lead}

A language model does not see text; it sees integers. We need a reversible map from strings to a fixed set of tokens. Two extremes are poor: a **word** vocabulary cannot handle unseen words (out-of-vocabulary) and explodes in size, while a **character** vocabulary makes sequences very long and forces the model to relearn spelling everywhere.

**Byte-pair encoding** strikes the balance. Starting from bytes (or characters), it repeatedly finds the most frequent adjacent pair and merges it into a new symbol, recording the merge. Common words become single tokens; rare words split into a few subwords; any string is representable, so there is **no out-of-vocabulary problem**. Practical vocabularies $$V$$ are roughly 30k-130k tokens. The cost is a subtlety students should remember: token boundaries are not word or syllable boundaries, which is why models sometimes miscount letters or mishandle digits.

### 2.2 Special tokens and document packing

> **Special tokens** (BOS, EOS, PAD, and chat-role markers) carry structure the raw text does not. **Document packing** concatenates many documents into one long token stream that is sliced into fixed-length training blocks.
{:.lead}

Beyond ordinary tokens, the vocabulary includes **special tokens**: a beginning-of-sequence marker, an end-of-sequence marker, padding, and — for chat models — role markers (system / user / assistant). These let one flat token stream encode boundaries and conversational structure.

For efficient pretraining we **pack** documents: concatenate the whole corpus into one long stream and slice it into contiguous blocks of length `block_size` (the context window). Packing avoids wasting compute on padding, keeping the GPU fully utilized. To prevent the model from "attending across" an unrelated document boundary, the attention or loss mask is reset at the join — a small bookkeeping detail with a real effect on quality. The output of preprocessing is a tensor of token ids, ready for embedding lookup.

### 2.3 Embeddings and weight tying

> The **embedding table** $$E\in\mathbb{R}^{V\times d}$$ maps each token id to a $$d$$-dimensional vector by row lookup. **Weight tying** reuses $$E$$ as the output projection (`lm_head`), saving parameters and often improving quality.
{:.lead}

Each token id $$t$$ selects row $$E_t\in\mathbb{R}^{d}$$ of the embedding table — a learned vector that the rest of the network refines. A length-$$T$$ block of ids becomes a $$T\times d$$ matrix $$X^{(0)}$$, exactly the "bag of token vectors" view from Day 5.

Position must be injected too. The classic approach (Day 5) adds a learned or sinusoidal **position embedding** to each row; modern decoder-only models instead rotate the queries and keys inside attention (rotary embeddings, below). Finally, many models **tie weights**: the same matrix $$E$$ that maps ids to vectors is transposed to map the final hidden states back to vocabulary logits in the output head `lm_head`. This couples input and output representations, removes $$V\times d$$ parameters, and tends to help.

## 3. The Decoder-Only Transformer Block

### 3.1 The pre-norm residual block

> A modern Transformer **block** applies normalization *before* each sublayer and adds the result back through a residual connection: $$h' = h + \mathrm{MHA}(\mathrm{LN}(h)),\qquad h'' = h' + \mathrm{MLP}(\mathrm{LN}(h')).$$
{:.lead}

![A residual (skip) connection wraps each sublayer so gradients flow cleanly through a deep stack (Beyer-style deck).](/assets/figures/day09/llmks_residual.png)

Following Gordić, we teach the block from the outside in. The skeleton is two **pre-norm residual** sublayers: first multi-head attention, then a position-wise MLP, each wrapped as $$h \leftarrow h + \mathrm{sublayer}(\mathrm{LN}(h))$$. Two design choices distinguish this from the original (post-norm) Transformer of Day 5:

- **Pre-norm.** Normalizing the *input* to each sublayer (rather than the output) gives a clean residual highway from input to output, which makes very deep stacks trainable without delicate learning-rate warmup tricks. Almost all modern LLMs are pre-norm.
- **Residual highway.** As on Day 4 (ResNet), the identity path means each sublayer only has to learn a *correction*, and gradients reach early layers undiminished.

Everything else in this section — RMSNorm, attention, RoPE, the gated MLP — slots into this skeleton.

### 3.2 RMSNorm

> **RMSNorm** rescales each token vector by its root-mean-square and a learned gain, with no mean subtraction or bias: $$\mathrm{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}}\odot\gamma.$$
{:.lead}

LayerNorm (Day 5) centers and scales each vector: subtract the mean, divide by the standard deviation, then apply a learned gain and bias. **RMSNorm** drops the mean-centering and the bias, normalizing only by the root-mean-square magnitude:

$$\mathrm{RMSNorm}(x) = \frac{x}{\mathrm{rms}(x)}\odot\gamma,\qquad \mathrm{rms}(x)=\sqrt{\tfrac{1}{d}\textstyle\sum_{i=1}^{d} x_i^2 + \epsilon}.$$

Empirically the centering step contributes little, so RMSNorm matches LayerNorm's stability at lower cost and fewer parameters. It is the default normalization in LLaMA-style models and is what you will most often see inside a modern pre-norm block.

### 3.3 Causal self-attention

> **Causal self-attention** computes, for each position $$i$$, a weighted average of the value vectors at positions $$j\le i$$, with weights $$A=\mathrm{softmax}\!\big(QK^\top/\sqrt{d_k}\big)$$ and a mask that zeroes $$A_{ij}$$ for $$j>i$$.
{:.lead}

![Attention as a soft dictionary lookup: a query is matched against keys to produce weights, which retrieve a blend of values (Beyer-style deck).](/assets/figures/day09/llmks_attention_dict.png)

This is the Day 5 mechanism, now made strictly causal. Project each token vector into a **query**, **key**, and **value**:

$$Q = X W_Q,\quad K = X W_K,\quad V = X W_V,\qquad \mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.$$

Read it as a **soft dictionary lookup**: the query for position $$i$$ is compared (dot product) against the key of every position, the similarities are softmaxed into weights, and the output is the weighted blend of values. The $$1/\sqrt{d_k}$$ scaling keeps the softmax in a healthy gradient regime (Day 5).

For autoregressive modeling we add a **causal mask**: before the softmax, set the logits for $$j>i$$ to $$-\infty$$, so position $$i$$ can only attend to itself and the past. This is exactly what makes the output at position $$i$$ a function of $$x_{\le i}$$ only — the network's realization of the conditional $$p(x_{i+1}\mid x_{\le i})$$. Because the mask is just an upper-triangular pattern, all positions are computed in one parallel pass during training.

### 3.4 Multi-head and grouped-query attention

> **Multi-head attention** runs $$H$$ attention operations in parallel with separate projections and concatenates them. **Grouped-query attention (GQA)** lets several query heads **share** one key/value head, shrinking the KV cache.
{:.lead}

![Multi-head attention: several heads attend in parallel, each capturing a different relation, then are concatenated and projected (Beyer-style deck).](/assets/figures/day09/llmks_mha.png)

One attention map is a bottleneck — tokens can be related in many ways at once (syntax, coreference, topic). **Multi-head attention** runs $$H$$ heads in parallel, each with its own $$W_{Q,h},W_{K,h},W_{V,h}$$ projecting into a $$d/H$$-dimensional subspace, then concatenates the head outputs and mixes them with an output projection $$W_O$$. Each head is a separate "relevance map," directly analogous to multiple convolution filters in a CNN (Day 4).

A scaling refinement matters for Day 10. In standard multi-head attention every head has its own keys and values, so the per-token **KV cache** (the keys/values we store to avoid recomputation at inference) grows with $$H$$. **Multi-query attention** shares a single K/V across all heads; **grouped-query attention (GQA)** is the middle ground, sharing one K/V head per *group* of query heads. GQA keeps almost all of the quality of full multi-head attention while sharply reducing KV-cache memory and inference bandwidth — which is why it is now standard in large models.

### 3.5 Rotary position embeddings (RoPE)

> **Rotary position embeddings (RoPE)** encode position by rotating the query and key vectors by an angle proportional to their position, so that the attention score between positions $$m$$ and $$n$$ depends only on the **relative** offset $$m-n$$.
{:.lead}

![Rotary embeddings rotate queries and keys by a position-dependent angle, injecting *relative* position directly into the attention dot product (Su et al., 2021).](/assets/figures/day09/llmks_rope.png)

Instead of *adding* a position vector to the input (Day 5), RoPE *rotates* the query and key vectors. Split each into 2D pairs and rotate the pair at index $$k$$ by an angle $$m\theta_k$$ that scales with position $$m$$ (with frequencies $$\theta_k$$ decreasing across dimensions). Because a dot product of two rotated vectors depends on the difference of their rotation angles, the resulting attention logit between positions $$m$$ and $$n$$ depends only on the **relative** offset $$m-n$$:

$$\langle R_m q,\; R_n k\rangle = g(q,k,\,m-n).$$

This has two big advantages: there is no learned position table to size in advance, and the relative encoding **extrapolates** more gracefully to sequences longer than those seen in training. To push context length even further, **YaRN** rescales the RoPE frequencies so a model trained at one context length keeps working at a longer one — useful but a detail we only name here. RoPE is the default in LLaMA, Mistral, and Qwen-class models.

### 3.6 The position-wise MLP (GeGLU / SwiGLU)

> The block's second sublayer is a **position-wise MLP** applied to each token independently. Modern variants are **gated**: $$\mathrm{MLP}(x) = W_2\big(\mathrm{GELU}(W_0 x)\odot (W_1 x)\big).$$
{:.lead}

![Most of the parameters and compute live in the position-wise feed-forward network applied to every token (Beyer-style deck).](/assets/figures/day09/llmks_ffn.png)

After attention mixes information *across* the sequence, the MLP refines each token's vector *independently* — the same MLP weights at every position. The classic form expands to a wide hidden layer and contracts back:

$$\mathrm{MLP}(x) = W_2\,\mathrm{GELU}(W_1 x),$$

with hidden width typically $$4d$$. Modern models use a **gated linear unit** variant (GeGLU / SwiGLU): two parallel projections, one passed through a nonlinearity, multiplied elementwise, then projected down:

$$\mathrm{MLP}(x) = W_2\big(\mathrm{GELU}(W_0 x)\odot (W_1 x)\big),$$

usually with a hidden width near $$\tfrac{8}{3}d$$ to keep the parameter count comparable. The gate lets the network modulate which features pass, and consistently improves quality. This MLP is where **most of the parameters and FLOPs** of a Transformer live.

### 3.7 Stacking the block into a full GPT

> A GPT is $$L$$ identical pre-norm blocks applied in sequence, followed by a final norm and a linear **`lm_head`** that maps each hidden vector to $$V$$ logits, turned into a next-token distribution by softmax.
{:.lead}

![The Transformer block, stacked $$N\times$$ with embeddings and positions, produces context-aware representations that feed the output head (Beyer-style deck).](/assets/figures/day09/llmks_architecture.png)

Put the pieces together. Token ids are embedded into $$X^{(0)}\in\mathbb{R}^{T\times d}$$; then $$L$$ blocks each apply causal multi-head attention and a gated MLP through pre-norm residuals:

$$h^{(l)} = \mathrm{Block}\big(h^{(l-1)}\big),\qquad l = 1,\dots,L.$$

A final RMSNorm precedes the output head $$\mathrm{logits} = h^{(L)} W_{\text{lm}}^\top\in\mathbb{R}^{T\times V}$$ (often with $$W_{\text{lm}} = E$$ via weight tying), and a softmax over the vocabulary gives, **at every position simultaneously**, the predicted distribution over the next token. That single forward pass — embed, $$L$$ blocks, norm, head, softmax — is a complete GPT, and is exactly what the next section trains.

## 4. Training One Model End-to-End

### 4.1 The objective: next-token cross-entropy

> Training minimizes the **next-token cross-entropy** $$\mathcal{L} = -\sum_{i} \log p_\theta(x_i\mid x_{<i}),$$ with **teacher forcing** (ground-truth context) and labels equal to the inputs shifted by one. **Perplexity** is $$\exp(\mathcal{L})$$.
{:.lead}

![Training computes a next-token distribution at every position and compares it, via cross-entropy, to the shifted targets (Beyer-style deck).](/assets/figures/day09/llmks_training.png)

Because the model emits a distribution at *every* position in one pass, training is wonderfully simple. Take a block of tokens $$x_{1:T}$$, run the forward pass, and compare the predicted distribution at position $$i$$ to the actual next token $$x_{i+1}$$ with cross-entropy:

$$\mathcal{L} = -\frac{1}{T}\sum_{i=1}^{T} \log p_\theta(x_{i+1}\mid x_{\le i}).$$

Two standard pieces of vocabulary: **teacher forcing** means we always condition on the *ground-truth* prefix during training (not the model's own guesses), which makes the loss above an exact maximum-likelihood objective and lets every position train in parallel. Implementation-wise, the labels are just the inputs **shifted by one position**. We monitor **perplexity** $$=\exp(\mathcal{L})$$, the effective branching factor of the predictor; lower is better. (The mismatch between teacher-forced training and free-running generation, *exposure bias*, is a Day 10 topic.)

### 4.2 The optimization stack

> LLMs are trained with **AdamW**, a **warmup-then-cosine** learning-rate schedule, **gradient clipping**, and **mixed precision** (bf16), often with gradient accumulation to reach very large effective batch sizes.
{:.lead}

The recipe is remarkably stable across scales:

- **Optimizer:** AdamW (Adam with decoupled weight decay) on all weights, typically excluding norms and biases from decay.
- **Schedule:** a short linear **warmup** (to avoid early divergence) followed by **cosine decay** to a small final learning rate.
- **Gradient clipping:** clip the global gradient norm (e.g. to 1.0) to tame occasional spikes.
- **Mixed precision:** compute in **bf16** for speed and memory, keeping a master copy of weights in higher precision.
- **Large effective batches:** **gradient accumulation** sums gradients over several micro-batches before stepping, so a modest GPU can emulate a huge batch.

None of these are LM-specific inventions — they are the Day 3 optimization toolkit, tuned for runs over billions of tokens.

### 4.3 Counting parameters and FLOPs

> A Transformer's parameters scale as $$\mathcal{O}(L\,d^2)$$, and training compute is approximately $$6N$$ FLOPs per token for a model with $$N$$ parameters (≈ $$2N$$ forward, $$4N$$ backward).
{:.lead}

A back-of-the-envelope that every practitioner should be able to reproduce. Per block, attention has four $$d\times d$$ projections ($$Q,K,V$$, and output) $$\approx 4d^2$$ parameters, and the MLP has roughly $$8d^2$$ (two $$d\times 4d$$ matrices), so a block is about $$12d^2$$ and the full stack about

$$N \approx 12\,L\,d^2 + V d,$$

with the $$Vd$$ embedding term often non-trivial for small models. For compute, a matrix multiply with $$N$$ parameters costs about $$2N$$ FLOPs per token in the forward pass; the backward pass costs about twice that, giving the standard rule of thumb

$$\text{training FLOPs} \approx 6N \times (\text{tokens}).$$

These two estimates let you reason about model and dataset sizing *before* launching a run — for instance, predicting how doubling $$d$$ roughly quadruples cost, or estimating the compute for a target token budget.

### 4.4 nanoGPT: the code you can read in an evening

> **nanoGPT** is a ~300-line implementation of GPT training and sampling. Its four files map cleanly onto the concepts above: `model.py` (architecture), `train.py` (the loop), `config` (hyperparameters), and `sample.py` (generation).
{:.lead}

![A decoder generating with masked self-attention — the structure nanoGPT's `CausalSelfAttention` implements (Beyer-style deck).](/assets/figures/day09/llmks_decoder_masked.png)

Everything in this lecture is concretely realized in [nanoGPT](https://github.com/karpathy/nanoGPT). The file-to-concept map:

```
model.py     GPT, Block, CausalSelfAttention, MLP  →  the forward pass above
train.py     data loader, training loop, AdamW, schedule, grad clip, bf16
config/      n_layer (L), n_head (H), n_embd (d), block_size (context length)
sample.py    autoregressive generation  (previews Day 10)
```

Reading `model.py` is the fastest way to make the math tangible: you will find the embedding table, the pre-norm `Block`, the causal mask built once and applied in `CausalSelfAttention`, the MLP, the final norm, and the tied `lm_head`. The practical for today has you build the same structure and train it on a character-level corpus, then sample from it — closing the loop from tokens to a working generator. (nanoGPT uses classic post-/pre-norm LayerNorm and learned positions for simplicity; we note where production models swap in RMSNorm and RoPE.)

### 4.5 Scaling, and the handoff to Day 10

> **Scaling laws** show test loss falling predictably as a power law in parameters, data, and compute; **Chinchilla** balances model size against training tokens. A trained base LM is then *served* and *aligned* (Day 10).
{:.lead}

Why does any of this scale? Empirically, the test loss of a well-trained Transformer falls as a smooth **power law** in model size, dataset size, and compute, over many orders of magnitude. The **Chinchilla** finding refined this: for a fixed compute budget there is an optimal balance between making the model bigger and training on more tokens, and earlier models were often undertrained for their size. Reaching the frontier requires **distributed training** — splitting the batch (data parallel), the layers (pipeline parallel), or individual tensors (tensor parallel) across many GPUs.

The output of all this is a **base language model**: a next-token predictor trained on internet-scale text. It is not yet a helpful assistant. Day 10 picks up exactly here: how to *serve* this model efficiently (the KV cache, sampling, systems), and how a small amount of *post-training* turns a base GPT into something like ChatGPT.

## Checkpoint summary

Before moving to the practical, confirm you can:

- Place autoregressive models on the Day 6 generative taxonomy and contrast their training and sampling with diffusion.
- Write the autoregressive factorization and explain why a causal mask lets one forward pass compute every conditional.
- Explain BPE tokenization, special tokens, and document packing, and why subword vocabularies avoid out-of-vocabulary failures.
- Describe a pre-norm residual block and why RMSNorm and pre-norm are used in modern LLMs.
- Derive causal self-attention as a masked soft dictionary lookup, and explain multi-head and grouped-query attention.
- Explain how RoPE injects relative position and contrast it with additive position embeddings.
- State the next-token cross-entropy objective with teacher forcing, and estimate a model's parameters ($$\sim 12Ld^2$$) and training FLOPs ($$\sim 6N$$ per token).
- Map the nanoGPT files onto the components of a GPT forward pass and training loop.
