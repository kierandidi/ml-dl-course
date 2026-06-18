#!/usr/bin/env python3
"""Generate week-2 lecture posts (days 9–10; days 6–8 are authored in generate_lectures.py) under lectures/_posts/."""
from __future__ import annotations

import re
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
POSTS = ROOT / "lectures" / "_posts"

# (day, date, slug, short title, description, optional reading lines, body markdown)
LECTURES: list[tuple[int, str, str, str, str, list[str], str]] = [
    (
        9,
        "2026-08-25",
        "day09-autoregressive-llms",
        "Autoregressive Language Models",
        "Encoder–decoder vs decoder-only stacks, training with cross-entropy, RoPE, and the transformer block.",
        [
            "[Vaswani et al. — Attention Is All You Need](https://arxiv.org/abs/1706.03762)",
            "[Gordić — Inside the Transformer: The Life of a Token](https://www.aleksagordic.com/blog/transformer)",
            "[Su et al. — RoFormer / RoPE](https://arxiv.org/abs/2104.09864)",
        ],
        dedent(
            r"""
            Large language models are **autoregressive**: they factorize sequences with causal conditioning. We map architectural
            families, the standard training loop, and the mathematical core of modern decoder-only transformers.

            ## 1. Model families: encoder, decoder, and hybrids

            > **Autoregressive factorization.**
            > \(p(\mathbf{x}) = \prod_{i=1}^{L} p(x_i \mid x_{<i})\) with causal masking so token \(i\) never sees the future.
            {:.lead}

            | Architecture | Attention | Typical use |
            |--------------|-----------|-------------|
            | **Encoder-only** | Bidirectional | Classification, embeddings (BERT) |
            | **Decoder-only** | Causal | GPT-style LMs, chat models |
            | **Encoder–decoder** | Cross + causal | Translation, summarization (T5) |

            **Encoder–decoder** maps input tokens to memory with a bidirectional encoder; the decoder attends to memory with
            cross-attention and to past outputs with a causal mask.

            **Decoder-only** treats prompt and completion as one sequence—today's default for general-purpose LLMs.

            ![Architecture families](/assets/figures/day09/pdf0_page025.png)
            *Figure: where information flows in each stack.*

            ### 1.1 Causal mask

            For positions \(i, j\), attention logits satisfy \(A_{ij} = -\infty\) when \(j > i\). After softmax, token \(i\)
            only aggregates keys/values from positions \(\le i\).

            ## 2. Training loop and cross-entropy

            Given tokenized sequence \(\mathbf{x} = (x_1, \ldots, x_L)\), the model outputs logits \(\mathbf{z}_i \in \mathbb{R}^{|\mathcal{V}|}\)
            for each position. **Next-token prediction** maximizes

            $$
            \mathcal{L}(\theta) = -\sum_{i=1}^{L} \log p_\theta(x_i \mid x_{<i})
            = -\sum_{i=1}^{L} \log \mathrm{softmax}(\mathbf{z}_i)_{x_i}.
            $$

            This is **multiclass cross-entropy** averaged over non-masked positions (padding masked out in the loss).

            > **Teacher forcing.** During training, the model always conditions on ground-truth prefixes \(x_{<i}\),
            > not its own previous predictions—stable gradients, train/inference mismatch handled at decode time (day 10).
            {:.lead}

            ![Training batch](/assets/figures/day09/pdf1_page020.png)
            *Figure: packed sequences and label shift by one.*

            ### 2.1 Optimization stack

            - AdamW with weight decay on matrices (not biases/LayerNorm).
            - Learning-rate warmup + cosine decay.
            - Gradient clipping and mixed-precision (bf16/fp16).

            ## 3. RoPE positional embeddings

            **Sinusoidal** absolute positions add to embeddings; **RoPE (rotary position embedding)** encodes relative position
            in attention by rotating query/key pairs in 2D subspaces.

            For head dimension pairs \((2k, 2k+1)\) and position \(m\),

            $$
            \mathrm{RoPE}(\mathbf{q}, m) =
            \begin{pmatrix}
            \cos m\theta_k & -\sin m\theta_k \\
            \sin m\theta_k & \cos m\theta_k
            \end{pmatrix}
            \begin{pmatrix} q_{2k} \\ q_{2k+1} \end{pmatrix},
            \qquad \theta_k = 10000^{-2k/d_{\mathrm{head}}}.
            $$

            Attention score \(\langle \mathrm{RoPE}(\mathbf{q}, m), \mathrm{RoPE}(\mathbf{k}, n)\rangle\) depends on \(m-n\),
            improving length extrapolation (YaRN scales frequencies for very long contexts).

            ![RoPE intuition](/assets/figures/day09/pdf1_page030.png)
            *Figure: relative position via rotation.*

            ## 4. Transformer block internals

            A **pre-norm** decoder block (schematic):

            $$
            \mathbf{h}' = \mathbf{h} + \mathrm{MHA}(\mathrm{LN}(\mathbf{h})),\qquad
            \mathbf{h}'' = \mathbf{h}' + \mathrm{MLP}(\mathrm{LN}(\mathbf{h}')).
            $$

            **Multi-head attention (MHA):**

            $$
            \mathrm{Attention}(Q,K,V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_{\mathrm{head}}}}\right) V,
            $$

            with \(Q = XW_Q\), \(K = XW_K\), \(V = XW_V\), split into heads. **GQA/MQA** share key/value heads to cut memory at inference.

            **MLP (SwiGLU / GeGLU):** gated feed-forward expands dimension (e.g. \(4\times\)) then projects back.

            ### 4.1 Life of a token (forward pass)

            1. Embed token IDs → vectors.
            2. For each layer: causal self-attention + MLP with residuals.
            3. Final linear **lm_head** → logits; softmax for loss or sampling.

            Stack depth $$L$$, hidden size $$d_{\mathrm{model}}$$, and head count set capacity; FLOPs scale roughly $$\mathcal{O}(L\, d_{\mathrm{model}}^2)$$ per token.

            ### 4.2 Parameter and FLOP scaling (sketch)

            Per layer, attention matrices contribute $$\approx 4 d_{\mathrm{model}}^2$$ parameters (Q,K,V,O projections) and
            MLP another $$\approx 8 d_{\mathrm{model}}^2$$ with expansion ratio 4. Total parameters scale
            $$\mathcal{O}(L_{\mathrm{layers}}\, d_{\mathrm{model}}^2)$$—the basis for Chinchilla-style compute–data trade-offs.

            ### 4.3 Tokenization and packing

            Subword tokenizers (BPE, SentencePiece) map bytes/characters to a vocabulary $$\mathcal{V}$$.
            **Document packing** concatenates multiple examples with attention masks so padding waste drops in training batches.

            ![Transformer block](/assets/figures/day09/pdf0_page035.png)
            *Figure: residual stream through attention and MLP.*

            ## Checkpoint summary

            - **Decoder-only** LMs dominate general text generation; encoder–decoder remains strong for fixed input→output tasks.
            - Training = sum of cross-entropies on next tokens with causal masking.
            - **RoPE** bakes relative position into attention via rotations.
            - A block = LN + attention + LN + MLP with residuals; depth stacks identical blocks with distinct weights.
            """
        ).strip(),
    ),
    (
        10,
        "2026-08-26",
        "day10-ar-inference",
        "Autoregressive Inference",
        "KV caching, temperature, nucleus (top-p) sampling, and batched decoding.",
        [
            "[Attention Is All You Need](https://arxiv.org/abs/1706.03762) — §3 (complexity)",
            "[Holtzman et al. — The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)",
            "[Gordić — KV cache discussion](https://www.aleksagordic.com/blog/transformer)",
        ],
        dedent(
            r"""
            Inference generates one token at a time. Efficiency hinges on **KV caching**, while **temperature** and **top-p**
            shape output quality. Production systems batch requests under memory and latency constraints.

            ## 1. Autoregressive decoding

            At step \(t\), we have prefix \(x_{\le t}\). The model outputs distribution \(p_\theta(x_{t+1}\mid x_{\le t})\).
            We sample or argmax \(x_{t+1}\), append, and repeat until EOS or max length.

            > **Exposure bias.** Training uses teacher forcing; inference feeds the model its own samples—errors compound.
            > Mitigations: scheduled sampling, distillation, RL fine-tuning (out of scope here).
            {:.lead}

            ![Decoding loop](/assets/figures/day10/pdf0_page000.png)
            *Figure: single-token extension per step.*

            ### 1.1 Complexity without cache

            Recomputing keys/values for all past tokens each step costs \(\mathcal{O}(t)\) attention per step and
            \(\mathcal{O}(L^2)\) over full length \(L\)—prohibitive for long contexts.

            ## 2. KV cache

            For each layer \(\ell\) and head, projections produce queries, keys, values. **Keys and values for past tokens are fixed**
            under causal attention, so we store them.

            > **KV cache memory (per layer, rough).**
            > \(2 \times L_{\mathrm{heads}} \times L_{\mathrm{seq}} \times d_{\mathrm{head}} \times \texttt{bytes\_per\_elem}\)
            > for keys plus values, times number of layers.
            {:.lead}

            Let \(C_{\mathrm{KV}}\) denote cached tensors. At step \(t\),

            $$
            K^{(\ell)} = \big[ K^{(\ell)}_{\mathrm{cache}} \;\|\; k^{(\ell)}_t \big], \qquad
            V^{(\ell)} = \big[ V^{(\ell)}_{\mathrm{cache}} \;\|\; v^{(\ell)}_t \big],
            $$

            and only \(q^{(\ell)}_t\) attends to the concatenated length-\(t\) sequence. Per-step cost becomes \(\mathcal{O}(t)\)
            attention per layer instead of recomputing from scratch.

            ![KV cache growth](/assets/figures/day10/pdf0_page005.png)
            *Figure: cache size grows linearly with context.*

            ### 2.1 Multi-request batching

            Batching \(B\) sequences pads to a common length (or uses ragged/paged attention). Cache is indexed per sequence;
            **PagedAttention** stores KV blocks in non-contiguous pages to reduce fragmentation on GPUs.

            ## 3. Temperature and top-p sampling

            Logits \(\mathbf{z}\) become probabilities

            $$
            p_i = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}, \qquad \tau > 0 \;\text{(temperature)}.
            $$

            - \(\tau \to 0^+\): distribution sharpens → greedy / near-argmax behavior.
            - \(\tau = 1\): training-scale probabilities.
            - \(\tau > 1\): flatter, more random outputs.

            **Top-p (nucleus) sampling:** sort probabilities \(p_{(1)} \ge p_{(2)} \ge \cdots\) and keep the smallest set
            \(V_p \subset \mathcal{V}\) such that \(\sum_{i \in V_p} p_i \ge p_{\mathrm{cut}}\) (e.g. \(p_{\mathrm{cut}}=0.9\)).
            Renormalize and sample within \(V_p\). This adapts the support size to model confidence.

            ![Sampling trade-offs](/assets/figures/day10/pdf1_page010.png)
            *Figure: temperature vs diversity.*

            ### 3.1 Other decoding knobs

            - **Top-k:** restrict to \(k\) largest logits before softmax.
            - **Repetition penalty:** down-weight logits of tokens already generated.
            - **Stop sequences:** user-defined EOS strings.

            ## 4. Batching, throughput, and serving

            **Static batching** waits until \(B\) requests fill—simple but increases latency.

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
            """
        ).strip(),
    ),
]


def front_matter(day: int, title: str, description: str) -> str:
    return f"""---
layout: post
title: Day {day} - {title}
image: /assets/img/sampling_space.png
accent_image: 
  background: url('/assets/img/sampling_space.png') center/cover
  overlay: false
accent_color: '#ccc'
theme_color: '#ccc'
description: >
  {description}
invert_sidebar: true
---
"""


def body_header(day: int, title: str, reading: list[str]) -> str:
    nn = f"{day:02d}"
    lines = [
        f"# Day {day} - {title}",
        "",
        f"### [Slides](/assets/slides/day{nn}.pdf)",
        "",
        f"### [Practical](/projects/day{nn}-practical/)",
        "",
        "### Optional reading for this lesson",
    ]
    lines.extend(f"- {r}" for r in reading)
    lines.extend(["", "* toc", "{:toc}", "", ""])
    return "\n".join(lines)


def normalize_math(text: str) -> str:
    """Use $$ delimiters for KaTeX (Hydejack convention)."""
    return re.sub(r"\\\((.+?)\\\)", r"$$\1$$", text, flags=re.DOTALL)


def write_post(day: int, date: str, slug: str, title: str, description: str, reading: list[str], body: str) -> Path:
    POSTS.mkdir(parents=True, exist_ok=True)
    path = POSTS / f"{date}-{slug}.md"
    body = normalize_math(body)
    content = front_matter(day, title, description) + "\n" + body_header(day, title, reading) + body + "\n"
    path.write_text(content, encoding="utf-8")
    return path


def main() -> None:
    written: list[Path] = []
    for day, date, slug, title, description, reading, body in LECTURES:
        path = write_post(day, date, slug, title, description, reading, body)
        n_lines = len(path.read_text(encoding="utf-8").splitlines())
        print(f"Wrote {path.relative_to(ROOT)} ({n_lines} lines)")
        written.append(path)
    print(f"\nGenerated {len(written)} posts in {POSTS.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
