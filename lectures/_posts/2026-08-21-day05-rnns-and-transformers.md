---
layout: post
title: Day 5 - Sequence Models: RNNs, Attention, and Transformers
image: /assets/img/sampling_space.png
accent_image: 
  background: url('/assets/img/sampling_space.png') center/cover
  overlay: false
accent_color: '#ccc'
theme_color: '#ccc'
description: >
  From recurrence and the seq2seq bottleneck to attention and the Transformer architecture.
invert_sidebar: true
---

# Day 5 - Sequence Models: RNNs, Attention, and Transformers

### Optional reading for this lesson
- [UCL x DeepMind DL2020 — L6: Sequences and Recurrent Networks](https://www.youtube.com/watch?v=87kLfzmYBy8)
- [UCL x DeepMind DL2020 — L8: Attention and Memory in Deep Learning](https://www.youtube.com/watch?v=AIiwuClvH6k)
- [Vaswani et al. — Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer — Jay Alammar](https://jalammar.github.io/illustrated-transformer/)

### [Slides](/assets/slides/day05.pdf)

### [Practical](/projects/day05-practical/)

Images have a fixed grid; language, audio, and time series do not. Today we build models for variable-length, ordered data. We start with recurrent networks, which share parameters across time and carry a hidden state, and we see why backpropagation through time makes them forget long-range context. Attention removes the recurrent bottleneck by letting the model look directly at any position, and the Transformer takes attention to its logical conclusion: a fully parallel architecture, built entirely from attention and MLPs, that now underpins nearly all of modern deep learning.

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

![An RNN unrolled in time: a shared cell updates the hidden state $$\boldsymbol{h}_t$$ and predicts the next token from it (UCL L6).](/assets/figures/day05/rnn_unrolled.png)

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

![Neural machine translation closed much of the gap between older systems and human-quality translation (UCL L6).](/assets/figures/day05/seq2seq_nmt.png)

Seq2seq made end-to-end neural machine translation work: the encoder compresses the source sentence into a fixed-length vector $$\boldsymbol{c}$$ (its final hidden state), and the decoder is a conditional language model that generates the translation token by token, starting from $$\boldsymbol{c}$$. The same template handles summarization, dialogue, and speech recognition — anything that maps one sequence to another. As the figure shows, neural systems substantially closed the gap to human-quality translation.

### 3.2 The fixed-vector bottleneck

> Forcing **all** information about the source into a single fixed-size vector $$\boldsymbol{c}$$ is the **bottleneck problem**: quality degrades as inputs grow longer, because one vector cannot hold everything.
{:.lead}

Imagine summarizing a 40-word sentence into one 512-dimensional vector and reconstructing a fluent translation from it alone. Empirically, translation quality falls off sharply as sentence length grows: the lone context vector simply cannot retain every relevant detail, and the decoder has **no way to look back** at specific source words.

The fix is to give the decoder access to **all** encoder hidden states, and to learn, at each output step, **which** of them to use. That mechanism is **attention** — and once we have it, we will find that the recurrence was never strictly necessary.

## 4. Attention

### 4.1 Attention as a differentiable, soft lookup

> **Attention** lets a model, at each step, compute a weighted combination of a set of vectors, where the weights measure relevance. It is a **soft, differentiable dictionary lookup**.
{:.lead}

![An alignment matrix: at each output position the model places most weight on the relevant input positions — a learned, soft correspondence (UCL L8).](/assets/figures/day05/attn_alignment.png)

Instead of compressing the source into one vector, the decoder keeps **all** encoder states $$\boldsymbol{h}_1,\dots,\boldsymbol{h}_T$$ available and, at each output step, forms a context as a **weighted sum** of them. The weights are large for relevant positions and near zero elsewhere, producing an **alignment** between output and input. Because the weights are computed by a differentiable softmax, the whole thing trains end-to-end with backprop, and the alignment matrix is directly interpretable — you can read off which source words the model used for each output word.

### 4.2 Content-based addressing: query, key, value

> In **content-based attention**, a **query** $$\boldsymbol{q}$$ is compared to each **key** $$\boldsymbol{k}_j$$ by a similarity; a softmax turns similarities into weights $$a_j$$; the output is the weighted sum of **values** $$\sum_j a_j \boldsymbol{v}_j.$$
{:.lead}

![Addressing by content: compare a query to each key by similarity, normalize with a softmax (with sharpness $$\beta$$), and read out the closest values (UCL L8).](/assets/figures/day05/attn_content.png)

The query/key/value vocabulary is the heart of attention:

- **Query** $$\boldsymbol{q}$$: what the current position is looking for.
- **Keys** $$\boldsymbol{k}_j$$: an index describing each candidate item.
- **Values** $$\boldsymbol{v}_j$$: the content actually retrieved.

With a dot-product similarity, the weights are

$$a_j = \mathrm{softmax}_j\big(\boldsymbol{q}^{\top}\boldsymbol{k}_j\big) = \frac{\exp(\boldsymbol{q}^{\top}\boldsymbol{k}_j)}{\sum_{l}\exp(\boldsymbol{q}^{\top}\boldsymbol{k}_l)},\qquad \text{output} = \sum_j a_j \boldsymbol{v}_j.$$

This is a **soft** version of looking up the key most similar to the query and returning its value — but differentiable, so it can be learned. Everything else in the Transformer is built from this primitive.

### 4.3 Derivation: scaled dot-product attention

> Stacking queries, keys, and values into matrices $$Q,K,V$$ gives $$\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V,$$ where the $$1/\sqrt{d_k}$$ scaling keeps the softmax in a healthy gradient regime.
{:.lead}

Batch the $$n$$ queries into rows of $$Q\in\mathbb{R}^{n\times d_k}$$, the keys into $$K\in\mathbb{R}^{m\times d_k}$$, and the values into $$V\in\mathbb{R}^{m\times d_v}$$. Then $$QK^{\top}\in\mathbb{R}^{n\times m}$$ holds **all** query–key dot products at once, a softmax over each row gives the attention weights, and multiplying by $$V$$ reads out the values — all as two matrix multiplies (highly parallel on a GPU).

**Why divide by $$\sqrt{d_k}$$?** Suppose the entries of $$\boldsymbol{q}$$ and $$\boldsymbol{k}$$ are independent with mean $$0$$ and variance $$1$$. Then their dot product

$$\boldsymbol{q}^{\top}\boldsymbol{k} = \sum_{i=1}^{d_k} q_i k_i$$

has mean $$0$$ and variance $$\textcolor{teal}{d_k}$$ (a sum of $$d_k$$ independent unit-variance terms), so its **standard deviation grows like $$\sqrt{d_k}$$**. For large $$d_k$$ the logits become huge in magnitude, pushing the softmax into a saturated, nearly one-hot regime where its gradient is almost zero — learning stalls. Dividing the scores by $$\textcolor{purple}{\sqrt{d_k}}$$ rescales the variance back to $$1$$:

$$\mathrm{Var}\!\left(\frac{\boldsymbol{q}^{\top}\boldsymbol{k}}{\sqrt{d_k}}\right) = \frac{d_k}{d_k} = 1,$$

keeping the softmax in a well-conditioned region. That single normalization is the difference between a Transformer that trains and one that does not.

## 5. The Transformer

### 5.1 Self-attention

> In **self-attention**, the queries, keys, and values are all linear projections of the *same* sequence: every position attends to every position, mixing information across the whole sequence in a single layer.
{:.lead}

![Attention learned by a translation model implicitly reorders words, attending across the sentence as needed (UCL L8).](/assets/figures/day05/attn_implicit.png)

Given an input sequence $$X\in\mathbb{R}^{n\times d}$$, self-attention forms $$Q=XW_Q$$, $$K=XW_K$$, $$V=XW_V$$ and applies scaled dot-product attention. Two enormous advantages over recurrence:

- **Constant path length.** Any two positions interact *directly* in one layer, whereas an RNN must pass information through $$O(n)$$ intermediate steps. Long-range dependencies are no harder than short-range ones.
- **Parallelism.** There is no sequential dependence across positions — the whole sequence is processed at once, so training parallelizes across positions on modern hardware. This is what made training on internet-scale data feasible.

The cost is $$O(n^2)$$ in sequence length (every pair of positions), the main scaling concern for long contexts.

### 5.2 Multi-head attention

> **Multi-head attention** runs $$h$$ attention operations in parallel on different learned projections, then concatenates and projects the results: $$\mathrm{MHA}(X) = [\,\mathrm{head}_1,\dots,\mathrm{head}_h\,]\,W_O.$$
{:.lead}

A single attention layer produces one weighting per position — one "kind" of relationship. But language has many simultaneous relations (subject–verb agreement, coreference, local phrasing). Multi-head attention lets the model attend to several at once:

$$\mathrm{head}_i = \mathrm{Attention}(XW_Q^i,\,XW_K^i,\,XW_V^i),\qquad \mathrm{MHA}(X)=[\,\mathrm{head}_1,\dots,\mathrm{head}_h\,]\,W_O.$$

Each head works in a lower-dimensional subspace ($$d/h$$), so the total cost is similar to a single full-dimensional attention, but the model can specialize different heads to different patterns. Inspecting individual heads often reveals interpretable behavior (one tracks the previous token, another links pronouns to their referents).

### 5.3 Positional encoding and the Transformer block

> Because attention is **permutation-equivariant**, the Transformer injects order via **positional encodings** added to the inputs, and stacks **blocks** of (multi-head attention + MLP), each wrapped in a residual connection and LayerNorm.
{:.lead}

**Position.** Self-attention treats its input as a *set* — shuffle the tokens and the outputs shuffle identically, with no notion of order. We restore order by adding a **positional encoding** to each token embedding, either fixed sinusoids of varying frequency,

$$\mathrm{PE}_{(p,2i)} = \sin\!\big(p/10000^{2i/d}\big),\qquad \mathrm{PE}_{(p,2i+1)} = \cos\!\big(p/10000^{2i/d}\big),$$

or learned position embeddings. The sinusoidal choice lets the model express relative offsets through linear combinations and extrapolate to unseen lengths.

**The block.** A Transformer layer is

$$X' = \mathrm{LayerNorm}\big(X + \mathrm{MHA}(X)\big),\qquad X'' = \mathrm{LayerNorm}\big(X' + \mathrm{MLP}(X')\big).$$

The **residual connections** (Day 4's idea) keep gradients flowing through deep stacks, and **LayerNorm** (Day 3) stabilizes the activations. The **encoder** stacks $$N$$ such blocks; the **decoder** adds *masked* self-attention — masking future positions so that, during training, a position can only attend to earlier ones, preserving the autoregressive property — plus cross-attention over the encoder outputs.

### 5.4 Why Transformers won

> The Transformer replaced recurrence entirely, trading sequential computation for parallel attention. Its scalability with data and compute made it the universal backbone of modern AI.
{:.lead}

Putting the pieces together, the Transformer dominates because it is:

- **Parallelizable.** No step waits for the previous one, so training uses hardware far more efficiently than RNNs.
- **Short-pathed.** Constant interaction distance between any two tokens makes long-range dependencies learnable.
- **Scalable.** Performance keeps improving predictably with more data, parameters, and compute — the empirical "scaling laws" behind large language models.
- **General.** The same architecture powers text (BERT, GPT), vision (ViT splits an image into patch "tokens"), audio, and biology (AlphaFold). One toolbox, many domains.

This sets up the rest of the course: **Days 9–10** scale this architecture into pretrained language models, while the generative-modeling thread (Days 6–8) shows how to *sample* high-dimensional data — where Transformers also serve as the workhorse denoiser.

## Checkpoint summary

Before moving to the practical, confirm you can:

- Explain the autoregressive factorization and how next-token prediction trains a sequence model.
- Write the RNN recurrence, unroll it, and explain backpropagation through time.
- Show why repeated Jacobian products cause vanishing/exploding gradients, and how LSTM gates fix it.
- Describe the seq2seq bottleneck and how attention removes it.
- Define query/key/value attention and derive why scores are divided by sqrt(d_k).
- Assemble a Transformer block (multi-head attention, MLP, residual + LayerNorm, positional encoding) and explain masking.
- Contrast RNNs and Transformers on path length, parallelism, and scalability.
