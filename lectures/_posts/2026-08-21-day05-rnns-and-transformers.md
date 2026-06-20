---
layout: post
title: Day 5 - Sequence Models: RNNs, Attention, and Transformers
image: /assets/img/lessons/day05.png
description: >
  From recurrence and the seq2seq bottleneck to attention and the Transformer architecture.
invert_sidebar: true
---

# Day 5 - Sequence Models: RNNs, Attention, and Transformers

### Optional reading for this lesson
- [Turner — An Introduction to Transformers (arXiv:2304.10557)](https://arxiv.org/abs/2304.10557)
- [UCL x DeepMind DL2020 — L6: Sequences and Recurrent Networks](https://www.youtube.com/watch?v=87kLfzmYBy8)
- [UCL x DeepMind DL2020 — L8: Attention and Memory in Deep Learning](https://www.youtube.com/watch?v=AIiwuClvH6k)
- [Vaswani et al. — Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762)
- [Structural Bioinformatics — Lesson 4: Evolution, Language and Bioinformatics](https://structural-bioinformatics.netlify.app/blog/proteins/2023-08-02-lesson4/)
- [The Illustrated Transformer — Jay Alammar](https://jalammar.github.io/illustrated-transformer/)

### [Slides](/assets/slides/day05.pdf)

### Exercise

[Download the notebook](/notebooks/practicals/day05.ipynb) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kierandidi/ml-dl-course/blob/main/notebooks/practicals/day05.ipynb)

Images have a fixed grid; language, audio, protein sequences, and time series do not. We start with recurrent networks and the seq2seq bottleneck, then build the Transformer from first principles: attention is nothing more than a *weighted linear combination* across the sequence, with weights learned from query–key similarity. We assemble the full block (multi-head attention, MLP, residuals, LayerNorm, position encoding), connect it to the standard $$Q,K,V$$ notation, and explain why this architecture replaced recurrence for large-scale sequence modelling.

* toc
{:toc}

## 1. Sequence Modeling

### 1.1 What makes sequences hard

> A **sequence model** maps variable-length, ordered data $$\boldsymbol{x}_{1:T}$$ to outputs, sharing parameters across positions so it can handle any length and exploit order.
{:.lead}

**Why this matters.** Most real signals are sequences: a sentence is a sequence of words, speech a sequence of audio frames, a stock price a sequence of values. Three properties make them different from the fixed-size inputs of Days 3–4:

1. **Variable length.** Sentences are not all the same length, so we cannot use a fixed-size input layer.
2. **Order matters.** "dog bites man" and "man bites dog" use the same words but mean opposite things.
3. **Long-range dependencies.** "The keys that I left on the kitchen table this morning *are* gone" — agreement spans many words.

Just as CNNs share parameters across **space**, sequence models must share parameters across **time/position**, so that a pattern learned at one position transfers to others and the parameter count does not grow with length.

### 1.2 Autoregressive factorization

> The **chain rule of probability** factorizes any joint distribution over a sequence into a product of conditionals: $$p(\boldsymbol{x}) = \prod_{i=1}^{n} p(x_i \mid x_1,\dots,x_{i-1}).$$
{:.lead}

![Treating an image as a sequence of pixels and factorizing it autoregressively, $$p(\boldsymbol{x})=\prod_i p(x_i\mid x_{<i})$$ (PixelRNN, UCL L6).](/assets/figures/day05/rnn_pixelrnn.png)

This identity is exact and completely general — it even applies to images if we impose an ordering on pixels (PixelRNN). It turns *generative modeling* into a sequence of *prediction* problems: predict the next element given all previous ones. We model each conditional $$p(x_i\mid x_{<i})$$ with a single shared network and train by **maximum likelihood**, i.e. minimizing the negative log-likelihood

$$\mathcal{L} = -\sum_{i=1}^{n}\log p_\theta(x_i\mid x_{<i}),$$

which for discrete tokens is exactly the **next-token cross-entropy** loss. Generation is then **ancestral sampling**: draw $$x_1$$, feed it back to get $$x_2$$, and so on. This "predict the next token" recipe is the through-line from RNNs to GPT.

## 2. Recurrent Neural Networks

### 2.1 The recurrence and the hidden state

> A **recurrent neural network (RNN)** maintains a hidden state updated at each step: $$\boldsymbol{h}_t = \tanh\!\big(W_h \boldsymbol{h}_{t-1} + W_x \boldsymbol{x}_t + \boldsymbol{b}\big),$$ and predicts $$\boldsymbol{y}_t = \mathrm{softmax}(W_y \boldsymbol{h}_t).$$
{:.lead}

![An RNN unrolled in time: a shared cell updates the hidden state $$\boldsymbol{h}_t$$ and predicts the next token from it.](/assets/figures/day05/rnn_unrolled.png)

The hidden state $$\boldsymbol{h}_t$$ is a fixed-size **summary of everything seen so far**, $$x_1,\dots,x_t$$. The same weight matrices $$W_h, W_x, W_y$$ are applied at *every* time step — this is the parameter sharing across time. If we **unroll** the recurrence, an RNN processing a length-$$T$$ sequence is just a $$T$$-layer feedforward network in which every layer shares weights, with one input and (optionally) one output per layer.

This gives RNNs their flexibility — the same model handles any length — but also their core weakness: information from early steps must survive being repeatedly multiplied by $$W_h$$ and squashed by $$\tanh$$ to reach a late step.

### 2.2 Backpropagation through time

> **Backpropagation through time (BPTT)** is ordinary backprop applied to the unrolled RNN. The gradient w.r.t. shared weights sums contributions from every time step.
{:.lead}

Because $$W_h$$ is reused at every step, the loss depends on it through all of $$\boldsymbol{h}_1,\dots,\boldsymbol{h}_T$$. By the chain rule, the gradient that flows from the loss at step $$t$$ back to an earlier hidden state $$\boldsymbol{h}_k$$ ($$k<t$$) passes through a **product of Jacobians**:

$$\frac{\partial \boldsymbol{h}_t}{\partial \boldsymbol{h}_k} = \prod_{j=k+1}^{t} \frac{\partial \boldsymbol{h}_j}{\partial \boldsymbol{h}_{j-1}} = \prod_{j=k+1}^{t} \operatorname{diag}\!\big(\tanh'(\cdot)\big)\,W_h^{\top}.$$

The total weight gradient sums such terms over all step pairs. Two practical consequences:

- **Sequential and expensive.** The unrolled graph is as deep as the sequence is long, so memory and compute grow with $$T$$, and the steps cannot be parallelized over time. In practice we use **truncated BPTT**, backpropagating only over a window.
- **The product is the problem.** That chain of Jacobians is exactly what makes long-range learning fragile, as we quantify next.

### 2.3 Vanishing gradients, LSTMs, and GRUs

> Because $$\partial \boldsymbol{h}_t/\partial \boldsymbol{h}_k$$ is a product of $$t-k$$ Jacobians, gradients shrink or grow geometrically with distance. **LSTMs** and **GRUs** fix this with a gated, additive memory.
{:.lead}

Roughly, if the Jacobian factors have norm $$\approx\gamma$$, then $$\Vert \partial \boldsymbol{h}_t/\partial \boldsymbol{h}_k\Vert \sim\gamma^{\,t-k}$$. With $$\gamma<1$$ (typical for $$\tanh$$, whose derivative is $$\le 1$$) the gradient **vanishes** over long ranges, so a plain RNN cannot learn dependencies more than a few dozen steps apart; with $$\gamma>1$$ it **explodes**.

**LSTM (Long Short-Term Memory).** Introduce a separate **cell state** $$\boldsymbol{c}_t$$ that is updated *additively* and protected by multiplicative **gates**:

$$\boldsymbol{c}_t = \textcolor{teal}{\boldsymbol{f}_t}\odot \boldsymbol{c}_{t-1} + \textcolor{purple}{\boldsymbol{i}_t}\odot \tilde{\boldsymbol{c}}_t, \qquad \boldsymbol{h}_t = \boldsymbol{o}_t \odot \tanh(\boldsymbol{c}_t),$$

where the forget gate $$\boldsymbol{f}_t$$, input gate $$\boldsymbol{i}_t$$, and output gate $$\boldsymbol{o}_t$$ are sigmoids of the input and previous state. The key is the additive update: when $$\boldsymbol{f}_t\approx 1$$, the cell state is a near-identity "conveyor belt", so gradients flow across many steps without vanishing — a temporal analogue of ResNet's skip connection.

**GRU** is a streamlined variant with two gates and no separate cell state — fewer parameters, often comparable performance. For exploding gradients, **gradient clipping** is the standard remedy. These gated RNNs were state-of-the-art for years, but they remain *sequential*. Attention removes that limitation.

## 3. Sequence to Sequence and the Bottleneck

### 3.1 The encoder–decoder architecture

> A **sequence-to-sequence (seq2seq)** model uses an **encoder** RNN to read the input into a context vector, and a **decoder** RNN to generate the output one token at a time, conditioned on that context.
{:.lead}

![Neural machine translation closed much of the gap between older systems and human-quality translation.](/assets/figures/day05/seq2seq_nmt.png)

Seq2seq made end-to-end neural machine translation work: the encoder compresses the source sentence into a fixed-length vector $$\boldsymbol{c}$$ (its final hidden state), and the decoder is a conditional language model that generates the translation token by token, starting from $$\boldsymbol{c}$$. The same template handles summarization, dialogue, and speech recognition — anything that maps one sequence to another. As the figure shows, neural systems substantially closed the gap to human-quality translation.

### 3.2 The fixed-vector bottleneck

> Forcing **all** information about the source into a single fixed-size vector $$\boldsymbol{c}$$ is the **bottleneck problem**: quality degrades as inputs grow longer, because one vector cannot hold everything.
{:.lead}

Imagine summarizing a 40-word sentence into one 512-dimensional vector and reconstructing a fluent translation from it alone. Empirically, translation quality falls off sharply as sentence length grows: the lone context vector simply cannot retain every relevant detail, and the decoder has **no way to look back** at specific source words.

The fix is to give the decoder access to **all** encoder hidden states, and to learn, at each output step, **which** of them to use. That mechanism is **attention** — and once we have it, we will find that the recurrence was never strictly necessary.

## 4. Building Intuition for Attention and Transformers

### 4.1 Tokens as a feature-by-sequence matrix

> A Transformer ingests $$N$$ tokens $$\boldsymbol{x}_n^{(0)}\in\mathbb{R}^D$$ arranged as a $$D\times N$$ matrix $$X^{(0)}=[\boldsymbol{x}_1^{(0)},\dots,\boldsymbol{x}_N^{(0)}]$$: features run *down*, the sequence runs *across*.
{:.lead}

**Why this notation.** Most Transformer tutorials use batch notation $$X\in\mathbb{R}^{N\times d}$$ (one row per token). We instead write $$X\in\mathbb{R}^{D\times N}$$ so that **stage 1** of the block acts *horizontally* (mixing across the sequence, row by row) and **stage 2** acts *vertically* (refining features within each token). Both are equivalent up to transpose; this layout makes the geometry of the block visually obvious.

**Tokenisation is universal.** Words embedded into vectors, image patches flattened and linearly projected (ViT), or amino-acid residues encoded for protein language models — all become columns of $$X^{(0)}$$. The Transformer iterates $$X^{(m)}=\text{transformer-block}(X^{(m-1)})$$ for $$m=1,\dots,M$$, producing a refined $$D\times N$$ representation usable for next-token prediction, classification (pool or a special token), or sequence-to-sequence heads.

This generic "bag of tokens" view is why one architecture serves text, vision, and biology — a theme also stressed in the [structural bioinformatics course](https://structural-bioinformatics.netlify.app/blog/proteins/2023-08-02-lesson4/) when connecting sequence alignment and language models to protein design.

### 4.2 Attention is a weighted linear combination

> The first stage of a Transformer block forms, at each sequence position $$n$$, a **convex combination** of all input token vectors: $$\boldsymbol{y}_n^{(m)} = \sum_{n'=1}^{N} \boldsymbol{x}_{n'}^{(m-1)}\,A_{n',n}^{(m)},\quad A_{n',n}\ge 0,\;\sum_{n'}A_{n',n}=1.$$
{:.lead}

This is the core intuition: before queries, keys, or multi-head anything, **attention is just a weighted average**. Each column $$n$$ of the output is a linear combination of the input columns, with weights that sum to one — a soft, learned "which positions matter for updating position $$n$$?"

**Matrix form.** Stacking all positions,

$$Y^{(m)} = X^{(m-1)} A^{(m)},\qquad Y,\,X\in\mathbb{R}^{D\times N},\;A\in\mathbb{R}^{N\times N}.$$

Entry $$A_{n',n}$$ is large when token $$n'$$ is *relevant* for representing token $$n$$; near zero when irrelevant. In machine translation this becomes an **alignment matrix** (see below); in vision, patches from the same object often attend to each other.

![An alignment matrix: learned weights $$A_{n',n}$$ show soft correspondence between input and output positions.](/assets/figures/day05/attn_alignment.png)

Nothing here requires neural networks beyond choosing the weights $$A$$. The entire design question is: **where do the weights come from?**

### 4.3 Self-attention: learning the weights from the input

> **Self-attention** generates $$A$$ from the input itself. Start with dot-product similarity, apply softmax for normalisation, then introduce **queries** $$\boldsymbol{q}_n=U_q\boldsymbol{x}_n$$ and **keys** $$\boldsymbol{k}_n=U_k\boldsymbol{x}_n$$ for asymmetric, content-based weights.
{:.lead}

We build the attention matrix in three deliberate steps.

**Step 1 — naive similarity.** Measure how much position $$n'$$ should contribute to $$n$$ by the dot product of their feature vectors, then normalise with a softmax:

$$A_{n,n'} \propto \exp(\boldsymbol{x}_n^{\top}\boldsymbol{x}_{n'}).$$

This already gives self-attention, but the similarity is **entangled with content**: the same features must simultaneously represent *what a token is* and *how it relates to others*.

**Step 2 — linear projection.** Apply a shared linear map $$U$$ before comparing: $$A_{n,n'} \propto \exp\big((U\boldsymbol{x}_n)^{\top}(U\boldsymbol{x}_{n'})\big)$$. Now only a $$K$$-dimensional subspace (typically $$K\ll D$$) is used for similarity; the rest of the features are free to carry content.

**Step 3 — queries and keys (asymmetry).** Symmetric similarity cannot express directional associations. A classic example: *"caulking iron"* should strongly attend to *"tool"*, but *"tool"* need not strongly attend to *"caulking iron"*. Use **two** linear maps:

$$\boldsymbol{q}_n = U_q \boldsymbol{x}_n,\qquad \boldsymbol{k}_n = U_k \boldsymbol{x}_n,\qquad A_{n,n'} = \frac{\exp(\boldsymbol{q}_n^{\top}\boldsymbol{k}_{n'})}{\sum_{n''}\exp(\boldsymbol{q}_n^{\top}\boldsymbol{k}_{n''})}.$$

Together with $$Y=XA$$, this is the complete self-attention mechanism; the only learnable parameters are $$U_q, U_k\in\mathbb{R}^{K\times D}$$.

![Content-based addressing: a query is compared to keys by similarity, then normalised with a softmax.](/assets/figures/day05/attn_content.png)

In the standard $$Q,K,V$$ presentation, we additionally introduce **values** $$\boldsymbol{v}_n=U_v\boldsymbol{x}_n$$ and output $$\sum_{n'} A_{n,n'}\boldsymbol{v}_{n'}$$. This is equivalent to the formulation above with extra projection matrices $$V_h$$ in the multi-head case — both are correct; the value matrix is not redundant once multi-head projection is included.

### 4.4 Multi-head attention, the MLP stage, and the full block

> A Transformer **block** = (multi-head self-attention across the sequence) + (shared MLP across features), each wrapped in **residual connections** and **LayerNorm**: $$X^{(m)} = X^{(m-1)} + \mathrm{res}_\text{MLP}\!\big(\mathrm{LN}(X^{(m-1)} + \mathrm{res}_\text{MHSA}(\mathrm{LN}(X^{(m-1)})))\big).$$
{:.lead}

**Multi-head self-attention (MHSA).** One $$N\times N$$ attention matrix can be a bottleneck — pairs of tokens might be similar in one sense (syntax) but not another (semantics). The Transformer runs $$H$$ self-attention operations in parallel with different $$(U_{q,h}, U_{k,h})$$, then projects down:

$$Y^{(m)} = \sum_{h=1}^{H} V_h\, X^{(m-1)} A_h,\qquad A_{h,n,n'} = \mathrm{softmax}_{n'}\big(\boldsymbol{q}_{h,n}^{\top}\boldsymbol{k}_{h,n'}\big).$$

This is directly analogous to **multiple convolution filters** in a CNN: several "relevance maps" in parallel, each capturing a different relation.

**Stage 2 — MLP across features.** After mixing across the sequence, each token's $$D$$-dimensional vector is refined independently by the *same* MLP: $$\boldsymbol{x}_n \leftarrow \mathrm{MLP}(\boldsymbol{y}_n)$$. This acts down each column of $$X$$. After $$M$$ stacked blocks, information at token $$n$$ and feature $$d$$ can depend on any $$(n', d')$$ in the input — horizontal then vertical mixing, repeated.

**Residuals and LayerNorm.** Rather than $$X^{(m)}=f(X^{(m-1)})$$, the block learns a **residual** correction $$X^{(m)}=X^{(m-1)}+\mathrm{res}(X^{(m-1)})$$ — the same identity-path idea as ResNet (Day 4). **LayerNorm** standardises each feature dimension across the sequence:

$$\bar{x}_{d,n} = \frac{x_{d,n}-\mathrm{mean}_n(x_d)}{\sqrt{\mathrm{var}_n(x_d)}},$$

preventing activations from blowing up through depth. Both MHSA and MLP stages use pre-norm or post-norm variants of this pattern.

### 4.5 Position encoding, causal masking, and the RNN comparison

> Because $$Y=XA$$ is **permutation-equivariant**, order must be injected via **position encodings** added to $$X^{(0)}$$. For autoregressive training, an **upper-triangular mask** on $$A$$ prevents future tokens from affecting past ones.
{:.lead}

**Position.** Without position information, "herbivores eat plants" and "plants eat herbivores" receive the same representation up to permutation. The fix is to add (or concatenate) a position embedding to each token — fixed sinusoids $$\mathrm{PE}_{(p,2i)}=\sin(p/10000^{2i/D})$$ or learned vectors.

**Causal masking (the batching trick).** Training an autoregressive model naïvely requires $$N$$ forward passes. Instead, apply the Transformer to the *whole* sequence and zero out $$A_{n,n'}=0$$ whenever $$n'>n$$ (upper-triangular $$A$$). Every next-token prediction is evaluated in **one parallel forward pass** — critical for LLM training at scale.

**Transformers vs RNNs.** An RNN updates $$\boldsymbol{h}_n=f(\boldsymbol{h}_{n-1},\boldsymbol{x}_n)$$: nearby observations are treated differently from distant ones because information must propagate through $$f$$ repeatedly. Self-attention treats **all** time-points identically regardless of distance — one reason Transformers learn long-range dependencies more easily. The trade-off is $$O(N^2)$$ memory and compute in sequence length, versus $$O(N)$$ for RNNs.

![Attention in translation: the model learns to reorder words by attending across the sentence.](/assets/figures/day05/attn_implicit.png)

### 4.6 Standard batch notation and scaled dot-product attention

> In the usual ML convention with $$X\in\mathbb{R}^{N\times d}$$ (rows = tokens), self-attention is $$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V,$$ identical to the $$Y=XA$$ form above once transposes and values are included.
{:.lead}

The $$Y=XA$$ form with $$\boldsymbol{q}_n=U_q\boldsymbol{x}_n$$, $$\boldsymbol{k}_n=U_k\boldsymbol{x}_n$$ is exactly the **scaled dot-product attention** of the original Transformer, written in $$D\times N$$ layout instead of $$N\times D$$. Stacking $$\boldsymbol{q}_n$$ into $$Q$$, $$\boldsymbol{k}_n$$ into $$K$$, and values $$\boldsymbol{v}_n=U_v\boldsymbol{x}_n$$ into $$V$$:

$$\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V.$$

**Why scale by $$\sqrt{d_k}$$?** If entries of $$\boldsymbol{q},\boldsymbol{k}$$ have unit variance, then $$\boldsymbol{q}^{\top}\boldsymbol{k}=\sum_{i=1}^{d_k} q_i k_i$$ has variance $$d_k$$. Large logits saturate the softmax (gradient $$\approx 0$$). Dividing by $$\sqrt{d_k}$$ restores unit variance and keeps training stable.

**Seq2seq as a special case.** Encoder–decoder attention uses queries from the decoder and keys/values from the encoder — the same weighted-average mechanism, but $$A$$ compares *two different* sequences. This resolves the fixed-vector bottleneck from the previous section: the decoder forms $$\boldsymbol{c}_t=\sum_j A_{j,t}\boldsymbol{h}_j^{\text{enc}}$$ at each step instead of relying on a single $$\boldsymbol{c}$$.

## 5. Why Transformers Won

### 5.1 Parallelism, path length, and scale

> Transformers replaced RNNs because they **parallelise** over sequence length, give **constant interaction depth** between any two tokens, and **scale** predictably with data and compute — the foundation of modern LLMs.
{:.lead}

The three practical advantages:

1. **Parallel training.** No step waits for the previous hidden state; the whole sequence is processed at once (with causal masking for autoregressive models). GPUs utilisation is far higher than for RNNs.
2. **Short paths.** Any two tokens interact directly in one layer; an RNN needs $$O(N)$$ steps. Long-range agreement ("The keys that I left … *are* gone") is no harder than local agreement.
3. **Scaling laws.** Performance keeps improving with more parameters, data, and compute — the empirical basis of GPT, PaLM, and protein language models such as ESM.

The cost is $$O(N^2)$$ attention, motivating sparse attention, linear attention, and state-space models for very long contexts. But for the sequence lengths that dominated NLP and protein modelling (hundreds to a few thousand tokens), the Transformer remains the default.

**Connections.** Day 9–10 cover LLM pretraining and inference. For sequence alignment, evolutionary conservation, and protein language models, the [structural bioinformatics course](https://structural-bioinformatics.netlify.app/blog/proteins/2023-08-02-lesson4/) (Lesson 4: Evolution, Language and Bioinformatics; Lesson 7: Generative Modelling) provides complementary biological motivation and references.

**Bridge to Week 2.** Today we built the Transformer block as a general-purpose *mechanism* and showed how causal masking turns it into an autoregressive sequence model. We deliberately stop here: the details that turn this block into a *production* large language model belong to Day 9, where we return to autoregressive modelling (named on Day 6, then set aside for diffusion on Days 6–8) and build one end to end. Specifically, we defer to **Day 9** the input pipeline (subword **tokenisation**/BPE, document packing), the modern block internals (**RMSNorm**, **GeGLU**, **rotary position embeddings (RoPE)**, **grouped-query attention**), the full GPT forward pass and **cross-entropy training loop**, and a **nanoGPT** code walkthrough; and to **Day 10** efficient **KV-cache** inference, sampling strategies, and the light GPT→ChatGPT post-training arc. Keep the picture from today — attention as a weighted average, the block as horizontal-then-vertical mixing — as the scaffold those details hang on.

## Checkpoint summary

Before moving to the practical, confirm you can:

- Explain the autoregressive factorization and how next-token prediction trains a sequence model.
- Write the RNN recurrence, unroll it, and explain backpropagation through time.
- Show why repeated Jacobian products cause vanishing/exploding gradients, and how LSTM gates fix it.
- Describe the seq2seq bottleneck and how attention removes it.
- Starting from $$Y=XA$$, explain attention as a weighted linear combination and build up to query–key self-attention.
- Assemble a Transformer block (multi-head attention, MLP, residual + LayerNorm, position encoding) and explain causal masking.
- Connect the $$D\times N$$ layout to standard $$Q,K,V$$ batch notation and derive the $$1/\sqrt{d_k}$$ scaling.
- Contrast RNNs and Transformers on path length, parallelism, and scalability.
- List what Day 9 adds on top of today's block (tokenisation, RoPE, RMSNorm/GeGLU, GQA, the GPT training loop, nanoGPT) to turn it into a production autoregressive LM.
