#!/usr/bin/env python3
"""Generate Touying slide decks for the 10-day ML & DL course."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from content_day01 import FIGURES as DAY01_FIGURES, SLIDES as DAY01_SLIDES
from content_day02 import FIGURES as DAY02_FIGURES, SLIDES as DAY02_SLIDES
from content_day03 import FIGURES as DAY03_FIGURES, SLIDES as DAY03_SLIDES
from content_day04 import FIGURES as DAY04_FIGURES, SLIDES as DAY04_SLIDES
from content_day05 import FIGURES as DAY05_FIGURES, SLIDES as DAY05_SLIDES
from content_day06 import FIGURES as DAY06_FIGURES, SLIDES as DAY06_SLIDES
from content_day07 import FIGURES as DAY07_FIGURES, SLIDES as DAY07_SLIDES
from content_day08 import FIGURES as DAY08_FIGURES, SLIDES as DAY08_SLIDES
from content_day09 import FIGURES as DAY09_FIGURES, SLIDES as DAY09_SLIDES
from content_day10 import FIGURES as DAY10_FIGURES, SLIDES as DAY10_SLIDES

# Days with hand-curated figure lists (aligned one-per-slide, None allowed).
CURATED_FIGURES = {1: DAY01_FIGURES, 2: DAY02_FIGURES, 3: DAY03_FIGURES, 4: DAY04_FIGURES, 5: DAY05_FIGURES, 6: DAY06_FIGURES, 7: DAY07_FIGURES, 8: DAY08_FIGURES, 9: DAY09_FIGURES, 10: DAY10_FIGURES}

ROOT = Path(__file__).resolve().parents[1]
DAYS_DIR = ROOT / "slides" / "days"
FIGURES_DIR = ROOT / "assets" / "figures"


def find_figures(day: int, limit: int = 12) -> list[str]:
    """Return relative paths (from slides/days/) to the first PNGs for a day."""
    fig_dir = FIGURES_DIR / f"day{day:02d}"
    if not fig_dir.is_dir():
        return []
    paths = sorted(fig_dir.glob("*.png"))[:limit]
    return [f"/assets/figures/day{day:02d}/{p.name}" for p in paths]


def typst_escape(text: str) -> str:
    """Escape characters that break Typst content blocks."""
    return text.replace("\\", "\\\\")


def fix_typst_math(text: str) -> str:
    """Normalize a few LaTeX-isms for Typst math mode."""
    text = text.replace("$$", "$")
    replacements = [
        (" ge ", " >= "),
        ("$propto ", "$prop "),
        (" propto ", " prop "),
        (" odot ", " dot.op "),
        ("arg max", '"arg max"'),
        ("arg min", '"arg min"'),
        ("bar(alpha)", "overline(alpha)"),
        ("d bar(w)", "dif overline(w)"),
        (" cdot ", " dot "),
        ("$prod_", "$product_"),
        ("$softmax(", '$"softmax"('),
        (" to RR", " arrow.r RR"),
        (" to RR^", " arrow.r RR^"),
        (" sim N", " tilde N"),
        (" sim ", " tilde "),
        ("dot(x)", "dot(x)"),  # keep
        ("dif X", "dif X"),
        ("dif t", "dif t"),
        ("dif W", "dif W"),
        (" in RR", " in RR"),
        ("max_i", "max_i"),
        ("sqrt(", "sqrt("),
        ("langle", "chevron.l"),
        ("rangle", "chevron.r"),
        (" circ ", " compose "),
        ("O(ops)", 'O("ops")'),
        ("O(h)", "O(h)"),
        ("O(h^2)", "O(h^2)"),
        ("O(h^4)", "O(h^4)"),
        ("O(1/", "O(1/"),
        ("$EE", "$EE"),
        ("\"Var\"", '"Var"'),
        ("\"Uniform\"", '"Uniform"'),
        ("D_\"KL\"", 'D_"KL"'),
        ("cos theta", "cos theta"),
        ("lambda", "lambda"),
        ("Sigma", "Sigma"),
        ("mu", "mu"),
        ("theta", "theta"),
        ("alpha", "alpha"),
        ("xi", "xi"),
        ("det ", "det "),
        ("log ", "log "),
        ("exp(", "exp("),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


# (title, subtitle fragment, parts)
# Each part: (section title, [(slide_title, [bullet_or_equation_lines]), ...])
COURSE: list[tuple[str, str, list[tuple[str, list[tuple[str, list[str]]]]]]] = [
    DAY01_SLIDES,
    (
        "Statistical Learning",
        "Regression, classification, regularization",
        [
            (
                "Learning Setup",
                [
                    (
                        "Supervised Learning",
                        [
                            "Dataset $D = {(x_i, y_i)}_(i=1)^n$",
                            "Hypothesis class $cal(H)$ and capacity",
                            "Train / validation / test splits",
                            "Generalization: performance on unseen data",
                        ],
                    ),
                    (
                        "Regression",
                        [
                            "Targets $y in RR$ (continuous)",
                            "Linear model: $f_w(x) = w^T phi(x)$",
                            "MSE: $L(w) = (1/n) sum_i (y_i - f_w(x_i))^2$",
                            "Polynomial features and basis expansion",
                        ],
                    ),
                    (
                        "Classification",
                        [
                            "Binary labels $y in {0, 1}$ or $y in {-1, +1}$",
                            "Linear classifier: $f_w(x) = w^T x$",
                            "Decision boundary where $f_w(x) = 0$",
                        ],
                    ),
                    (
                        "Evaluation Metrics",
                        [
                            "Regression: MSE, MAE, $R^2$",
                            "Classification: accuracy, precision, recall, F1",
                            "Confusion matrix and class imbalance",
                        ],
                    ),
                ],
            ),
            (
                "Linear & Logistic Models",
                [
                    (
                        "Ordinary Least Squares",
                        [
                            "Normal equations: $w^* = (X^T X)^(-1) X^T y$",
                            "Geometric interpretation: orthogonal residual",
                            "When $X^T X$ is ill-conditioned",
                        ],
                    ),
                    (
                        "Logistic Regression",
                        [
                            "$p(y=1|x) = sigma(w^T x)$ with $sigma(z) = 1/(1+e^(-z))$",
                            "Cross-entropy loss for Bernoulli labels",
                            "Decision boundary remains linear in feature space",
                        ],
                    ),
                    (
                        "Softmax Multiclass",
                        [
                            "$p(y=k|x) = exp(w_k^T x) / sum_j exp(w_j^T x)$",
                            "One-vs-rest vs multinomial logistic",
                            "Cross-entropy over $K$ classes",
                        ],
                    ),
                    (
                        "Probabilistic View",
                        [
                            "Model specifies $p(y|x, w)$",
                            "MLE chooses $w$ maximizing $prod_i p(y_i|x_i, w)$",
                            "Log-likelihood turns products into sums",
                        ],
                    ),
                ],
            ),
            (
                "Regularization",
                [
                    (
                        "Overfitting",
                        [
                            "Low train error, high test error",
                            "Model too flexible for data size",
                            "Memorization vs learning structure",
                        ],
                    ),
                    (
                        "L2 (Ridge)",
                        [
                            "$L(w) = sum_i ell_i + lambda ||w||_2^2$",
                            "Shrinks weights toward zero",
                            "Closed form for ridge regression",
                        ],
                    ),
                    (
                        "L1 (Lasso)",
                        [
                            "$L(w) = sum_i ell_i + lambda ||w||_1$",
                            "Promotes sparsity — feature selection",
                            "Non-smooth at zero; subgradient methods",
                        ],
                    ),
                    (
                        "Model Selection",
                        [
                            "Validation curves vs $lambda$",
                            "Early stopping as implicit regularization",
                            "Bias–variance decomposition intuition",
                        ],
                    ),
                ],
            ),
            (
                "Beyond Linearity (Preview)",
                [
                    (
                        "Kernel Trick",
                        [
                            "Implicit feature map $phi(x)$ via $k(x, x')$",
                            "Representer theorem: $w = sum_i alpha_i phi(x_i)$",
                            "RBF kernel for non-linear boundaries",
                        ],
                    ),
                    (
                        "Feature Engineering",
                        [
                            "Domain knowledge beats raw pixels (often)",
                            "Normalization and standardization",
                            "Handling missing data and outliers",
                        ],
                    ),
                    (
                        "Bridge to Deep Learning",
                        [
                            "Neural nets learn features end-to-end (Day 3)",
                            "Same loss + regularization story, richer $cal(H)$",
                            "Practical today: start simple, add complexity",
                        ],
                    ),
                ],
            ),
        ],
    ),
    (
        "Deep Neural Networks",
        "Backpropagation, activations, optimizers",
        [
            (
                "From Linear to Deep",
                [
                    (
                        "Limitations of Linear Models",
                        [
                            "XOR and non-linearly separable data",
                            "Need composed non-linear transformations",
                            "Universal approximation intuition",
                        ],
                    ),
                    (
                        "Layer Composition",
                        [
                            "$h^(l) = sigma(W^(l) h^(l-1) + b^(l))$",
                            "Depth vs width tradeoffs",
                            "Parameter count scales with layer sizes",
                        ],
                    ),
                    (
                        "Activations",
                        [
                            "ReLU: $max(0, z)$ — sparse, fast",
                            "Sigmoid / tanh — saturating, vanishing gradients",
                            "GELU, SiLU in modern transformers",
                        ],
                    ),
                    (
                        "Forward Pass",
                        [
                            "Cache intermediates for backward pass",
                            "Batching: tensor shape $(N, d)$",
                            "Numerical stability: log-sum-exp trick",
                        ],
                    ),
                ],
            ),
            (
                "Backpropagation",
                [
                    (
                        "Computational Graph",
                        [
                            "Nodes = ops; edges = tensors",
                            "Local gradients multiply via chain rule",
                            "Reverse-mode AD = backprop",
                        ],
                    ),
                    (
                        "Output Layer Gradients",
                        [
                            "MSE: $partial L / partial hat(y) = 2(hat(y) - y)$",
                            "Softmax + CE: gradient simplifies to $hat(y) - y$",
                            "Sigmoid + BCE similarly clean",
                        ],
                    ),
                    (
                        "Hidden Layer Gradients",
                        [
                            "$delta^(l) = (W^(l+1))^T delta^(l+1) odot sigma'(z^(l))$",
                            "Vanishing / exploding gradients in deep nets",
                            "Skip connections mitigate (ResNet, Day 4+)",
                        ],
                    ),
                    (
                        "Implementation Notes",
                        [
                            "Autograd in PyTorch / JAX",
                            "Detach, stop_gradient, custom Function",
                            "Check gradients with finite differences",
                        ],
                    ),
                ],
            ),
            (
                "Training Loop",
                [
                    (
                        "Mini-batch SGD",
                        [
                            "Sample batch $B subset D$ each step",
                            "Loss averaged over batch",
                            "Epoch = one pass over training set",
                        ],
                    ),
                    (
                        "Learning Rate Schedules",
                        [
                            "Step decay, cosine annealing, warmup",
                            "$eta_t$ often largest hyperparameter",
                            "Monitor train vs val loss curves",
                        ],
                    ),
                    (
                        "Initialization",
                        [
                            "Xavier / He scaling for variance preservation",
                            "Bad init → dead ReLUs or blow-up",
                            "LayerNorm reduces sensitivity (Day 5)",
                        ],
                    ),
                    (
                        "Batch Normalization",
                        [
                            "Normalize activations per mini-batch",
                            "Learnable scale and shift $gamma, beta$",
                            "Regularization side effect",
                        ],
                    ),
                ],
            ),
            (
                "Optimizers",
                [
                    (
                        "Momentum",
                        [
                            "$v_(t+1) = beta v_t + nabla L$; $w_(t+1) = w_t - eta v_(t+1)$",
                            "Accumulates consistent gradient direction",
                            "Nesterov lookahead variant",
                        ],
                    ),
                    (
                        "Adam",
                        [
                            "Adaptive per-parameter learning rates",
                            "First and second moment estimates",
                            "Default choice for many DL experiments",
                        ],
                    ),
                    (
                        "Weight Decay",
                        [
                            "Decoupled WD vs L2 in AdamW",
                            "Regularization interacts with normalization",
                        ],
                    ),
                    (
                        "Debugging Training",
                        [
                            "Loss not decreasing → LR, init, bugs",
                            "Overfit small batch sanity check",
                            "Gradient clipping for RNNs / LLMs",
                        ],
                    ),
                ],
            ),
        ],
    ),
    (
        "Convolutional Networks",
        "Computer vision and spatial inductive bias",
        [
            (
                "Convolutions",
                [
                    (
                        "Motivation",
                        [
                            "Images: local structure + translation symmetry",
                            "Fully connected layers ignore spatial layout",
                            "Parameter sharing across locations",
                        ],
                    ),
                    (
                        "2D Convolution",
                        [
                            "$(f * k)(i,j) = sum_(u,v) f(i-u,j-v) k(u,v)$",
                            "Output size: $(H - k + 2p)/s + 1$",
                            "Multiple filters → channel depth",
                        ],
                    ),
                    (
                        "Pooling & Stride",
                        [
                            "Max / average pooling downsamples",
                            "Stride reduces spatial resolution",
                            "Receptive field grows with depth",
                        ],
                    ),
                    (
                        "CNN Building Blocks",
                        [
                            "Conv → BN → ReLU → Pool stacks",
                            "1×1 conv for channel mixing",
                            "Depthwise separable conv (MobileNet)",
                        ],
                    ),
                ],
            ),
            (
                "Classic Architectures",
                [
                    (
                        "LeNet → AlexNet",
                        [
                            "Historical milestones on MNIST / ImageNet",
                            "ReLU + GPU training breakthrough",
                            "Data augmentation becomes standard",
                        ],
                    ),
                    (
                        "VGG & Inception",
                        [
                            "Small 3×3 filters, deeper stacks",
                            "Multi-scale Inception modules",
                            "Computational cost vs accuracy",
                        ],
                    ),
                    (
                        "ResNet",
                        [
                            "Residual: $y = F(x) + x$",
                            "Eases optimization of very deep nets",
                            "Skip connections and identity paths",
                        ],
                    ),
                    (
                        "Modern Backbones",
                        [
                            "EfficientNet, ConvNeXt",
                            "ViT hybrid models (Day 5 link)",
                            "Transfer learning from ImageNet",
                        ],
                    ),
                ],
            ),
            (
                "Training for Vision",
                [
                    (
                        "Data Augmentation",
                        [
                            "Random crop, flip, color jitter",
                            "Mixup / CutMix regularization",
                            "Test-time augmentation",
                        ],
                    ),
                    (
                        "Losses & Heads",
                        [
                            "Softmax cross-entropy for classification",
                            "Multi-task: detection, segmentation heads",
                            "Focal loss for hard examples",
                        ],
                    ),
                    (
                        "Object Detection Preview",
                        [
                            "Bounding boxes, IoU metric",
                            "Two-stage vs one-stage detectors",
                            "Feature pyramids (FPN)",
                        ],
                    ),
                    (
                        "Segmentation Preview",
                        [
                            "Per-pixel class labels",
                            "U-Net encoder–decoder",
                            "Dice / IoU losses",
                        ],
                    ),
                ],
            ),
            (
                "Practice & Pitfalls",
                [
                    (
                        "Input Pipeline",
                        [
                            "Normalize with dataset mean/std",
                            "Efficient dataloaders, prefetch",
                            "Resolution vs batch size memory tradeoff",
                        ],
                    ),
                    (
                        "Overfitting in Vision",
                        [
                            "Heavy aug + WD + early stopping",
                            "Small datasets: freeze backbone",
                            "Monitor val accuracy gap",
                        ],
                    ),
                    (
                        "Interpretability",
                        [
                            "Grad-CAM saliency maps",
                            "Adversarial examples",
                            "Dataset bias and spurious cues",
                        ],
                    ),
                ],
            ),
        ],
    ),
    (
        "Sequence Models",
        "RNNs, attention, and Transformers",
        [
            (
                "Sequential Data",
                [
                    (
                        "Motivation",
                        [
                            "Text, speech, time series — ordered inputs",
                            "Variable length sequences",
                            "Markov assumptions and history",
                        ],
                    ),
                    (
                        "Fixed-window Baselines",
                        [
                            "Bag-of-words loses order",
                            "N-gram language models",
                            "Need latent state summarizing past",
                        ],
                    ),
                    (
                        "RNN Recurrence",
                        [
                            "$h_t = sigma(W_h h_(t-1) + W_x x_t + b)$",
                            "Same weights applied at each time step",
                            "Unroll graph for BPTT",
                        ],
                    ),
                    (
                        "Vanishing Gradients",
                        [
                            "Long-range dependencies are hard",
                            "LSTM / GRU gating mechanisms",
                            "Truncated BPTT for long sequences",
                        ],
                    ),
                ],
            ),
            (
                "Attention",
                [
                    (
                        "Seq2seq Bottleneck",
                        [
                            "Encoder final state must encode everything",
                            "Attention reads all encoder states",
                        ],
                    ),
                    (
                        "Scaled Dot-Product Attention",
                        [
                            "$\"Attention\"(Q,K,V) = \"softmax\"(Q K^T / sqrt(d_k)) V$",
                            "Query, Key, Value interpretations",
                            "Soft alignment weights over positions",
                        ],
                    ),
                    (
                        "Multi-Head Attention",
                        [
                            "Parallel heads in subspaces",
                            "Concatenate and project",
                            "Expressivity vs compute",
                        ],
                    ),
                    (
                        "Self-Attention",
                        [
                            "Q, K, V from same sequence",
                            "Direct long-range links $O(n^2)$",
                            "Positional information required",
                        ],
                    ),
                ],
            ),
            (
                "Transformer Architecture",
                [
                    (
                        "Encoder Block",
                        [
                            "MHA → Add&Norm → FFN → Add&Norm",
                            "FFN: two linear layers with non-linearity",
                            "Pre-norm vs post-norm variants",
                        ],
                    ),
                    (
                        "Positional Encoding",
                        [
                            "Sinusoidal or learned position embeddings",
                            "RoPE in modern LLMs (Day 9–10)",
                            "Relative position bias",
                        ],
                    ),
                    (
                        "Decoder & Masking",
                        [
                            "Causal mask: token $t$ sees $<= t$ only",
                            "Cross-attention to encoder (MT)",
                            "Decoder-only for language modeling",
                        ],
                    ),
                    (
                        "Complexity",
                        [
                            "Self-attention: $O(n^2 d)$ time and memory",
                            "Motivates sparse / linear attention research",
                            "KV cache for inference (Day 10)",
                        ],
                    ),
                ],
            ),
            (
                "Applications & Scaling",
                [
                    (
                        "Language Modeling",
                        [
                            "Next-token prediction objective",
                            "Perplexity metric: $exp(-(1/N) sum log p)$",
                            "BPE / SentencePiece tokenization",
                        ],
                    ),
                    (
                        "Pre-training + Fine-tuning",
                        [
                            "Self-supervised pretrain on large corpus",
                            "Supervised fine-tune on downstream task",
                            "Instruction tuning and RLHF preview",
                        ],
                    ),
                    (
                        "Scaling Laws",
                        [
                            "Loss improves predictably with compute/data/params",
                            "Chinchilla-optimal token budgets",
                            "Emergent capabilities (debated)",
                        ],
                    ),
                ],
            ),
        ],
    ),
    (
        "Generative Modeling",
        "Likelihoods, KL, ELBO, model families",
        [
            (
                "What Is Generative Modeling?",
                [
                    (
                        "Discriminative vs Generative",
                        [
                            "Discriminative: model $p(y|x)$",
                            "Generative: model $p(x)$ or $p(x|z)$",
                            "Sample generation, density estimation, editing",
                        ],
                    ),
                    (
                        "Explicit vs Implicit",
                        [
                            "Explicit likelihood: VAEs, flows, autoregressive",
                            "Implicit: GANs (no tractable $p(x)$)",
                            "Tradeoffs: training stability, evaluation",
                        ],
                    ),
                    (
                        "Latent Variable Models",
                        [
                            "$p(x) = integral p(x|z) p(z) d z$",
                            "Latent $z$ captures factors of variation",
                            "Posterior $p(z|x)$ usually intractable",
                        ],
                    ),
                    (
                        "Model Families Roadmap",
                        [
                            "Week 2: flows, diffusion, autoregressive LLMs",
                            "Each defines different tractability / quality",
                        ],
                    ),
                ],
            ),
            (
                "Likelihood & Divergences",
                [
                    (
                        "Maximum Likelihood",
                        [
                            "$theta^* = arg max_theta sum_i log p_theta(x_i)$",
                            "Equivalent to minimizing cross-entropy",
                            "MLE is consistent under mild conditions",
                        ],
                    ),
                    (
                        "KL Divergence",
                        [
                            "$D_\"KL\"(q || p) = EE_q[log q - log p]$",
                            "Non-negative; zero iff $q = p$",
                            "Not symmetric — direction matters",
                        ],
                    ),
                    (
                        "Forward vs Reverse KL",
                        [
                            "Forward KL: mode-covering (mean-field)",
                            "Reverse KL: mode-seeking",
                            "VAE uses reverse KL to approximate posterior",
                        ],
                    ),
                    (
                        "Evidence Lower Bound",
                        [
                            "$log p(x) >= EE_(q(z|x))[log p(x|z)] - D_\"KL\"(q(z|x) || p(z))$",
                            "ELBO tight when $q = p(z|x)$",
                            "Reparameterization trick for gradients",
                        ],
                    ),
                ],
            ),
            (
                "VAEs & GANs (Context)",
                [
                    (
                        "Variational Autoencoder",
                        [
                            "Encoder $q_phi(z|x)$, decoder $p_theta(x|z)$",
                            "Train by maximizing ELBO",
                            "Blurry samples — Gaussian assumption",
                        ],
                    ),
                    (
                        "GAN Objective",
                        [
                            "Min-max game: generator vs discriminator",
                            "No explicit likelihood",
                            "Mode collapse and training instability",
                        ],
                    ),
                    (
                        "Evaluation Challenges",
                        [
                            "FID, IS for images; human eval for text",
                            "Likelihood can misalign with sample quality",
                            "Coverage vs fidelity",
                        ],
                    ),
                    (
                        "Modern Landscape",
                        [
                            "Diffusion dominates image generation",
                            "Autoregressive dominates language",
                            "Hybrid and unified models emerging",
                        ],
                    ),
                ],
            ),
            (
                "Building Blocks for Week 2",
                [
                    (
                        "Score Functions",
                        [
                            "Score: $nabla_x log p(x)$",
                            "Score matching and denoising connections",
                            "Foundation for diffusion (Days 7–8)",
                        ],
                    ),
                    (
                        "Normalizing Flows Preview",
                        [
                            "Invertible maps with tractable Jacobian",
                            "$log p(x) = log p(z) + log |det partial f / partial z|$",
                            "Day 7 training details",
                        ],
                    ),
                    (
                        "Autoregressive Factorization",
                        [
                            "$log p(x) = sum_t log p(x_t | x_(<t))$",
                            "Causal masks enforce ordering",
                            "GPT family (Days 9–10)",
                        ],
                    ),
                ],
            ),
        ],
    ),
    (
        "Training Flow & Diffusion Models",
        "Continuous transforms and denoising objectives",
        [
            (
                "Normalizing Flows",
                [
                    (
                        "Change of Variables",
                        [
                            "$p_X(x) = p_Z(f^(-1)(x)) |det J_(f^(-1))(x)|$",
                            "Bijective $f$ with tractable inverse",
                            "Composition of simple coupling layers",
                        ],
                    ),
                    (
                        "Coupling Layers",
                        [
                            "Split dimensions; transform one part conditioned on other",
                            "Affine coupling: scale and shift",
                            "RealNVP, Glow architectures",
                        ],
                    ),
                    (
                        "Training Flows",
                        [
                            "Maximize log-likelihood directly",
                            "Jacobian log-determinant cost",
                            "Exact sampling and density",
                        ],
                    ),
                    (
                        "Limitations",
                        [
                            "Architectural constraints for invertibility",
                            "Scaling to high-res images is hard",
                            "Diffusion trades exact likelihood for flexibility",
                        ],
                    ),
                ],
            ),
            (
                "Diffusion Intuition",
                [
                    (
                        "Forward Process",
                        [
                            "Gradually add Gaussian noise: $q(x_t|x_(t-1))$",
                            "Closed form $q(x_t|x_0) = cal(N)(sqrt(bar(alpha)_t) x_0, (1-bar(alpha)_t) I)$",
                            "Ends at pure noise",
                        ],
                    ),
                    (
                        "Reverse Process",
                        [
                            "Learn to denoise step by step",
                            "$p_theta(x_(t-1)|x_t)$ parameterized by neural net",
                            "Sampling walks from noise to data",
                        ],
                    ),
                    (
                        "DDPM Objective",
                        [
                            "Predict noise $epsilon$ added at step $t$",
                            "Simple MSE on $epsilon$ with random $t$",
                            "Equivalent variants: predict $x_0$ or score",
                        ],
                    ),
                    (
                        "Noise Schedules",
                        [
                            "Linear, cosine $bar(alpha)_t$ schedules",
                            "Affects training stability and sample quality",
                            "Signal-to-noise ratio view",
                        ],
                    ),
                ],
            ),
            (
                "Score Matching View",
                [
                    (
                        "Denoising Score Matching",
                        [
                            "Learn $s_theta(x_t, t) approx nabla_(x_t) log p(x_t)$",
                            "Tweedie's formula links $epsilon$ and score",
                            "Unifies diffusion training objectives",
                        ],
                    ),
                    (
                        "SDE Formulation Preview",
                        [
                            "Forward SDE adds noise continuously",
                            "Reverse SDE uses score function",
                            "Day 8 inference details",
                        ],
                    ),
                    (
                        "Classifier-Free Guidance",
                        [
                            "Train conditional model with dropped labels",
                            "Guidance scale trades diversity vs fidelity",
                            "Standard in text-to-image systems",
                        ],
                    ),
                    (
                        "Latent Diffusion",
                        [
                            "Diffuse in VAE latent space (Stable Diffusion)",
                            "Lower dimension → cheaper training",
                            "Text encoder provides conditioning",
                        ],
                    ),
                ],
            ),
            (
                "Training Practice",
                [
                    (
                        "Network Architecture",
                        [
                            "U-Net with time embedding $t$",
                            "Attention at lower resolutions",
                            "GroupNorm + SiLU activations",
                        ],
                    ),
                    (
                        "Compute & Data",
                        [
                            "Large-scale image-text pairs",
                            "Mixed precision and EMA weights",
                            "Checkpointing long runs",
                        ],
                    ),
                    (
                        "Monitoring",
                        [
                            "Loss per noise level",
                            "Periodic sample grids",
                            "FID on small val set",
                        ],
                    ),
                ],
            ),
        ],
    ),
    (
        "Diffusion Inference",
        "SDEs, probability-flow ODEs, and samplers",
        [
            (
                "Continuous-Time View",
                [
                    (
                        "Forward SDE",
                        [
                            "$d x = f(x,t) d t + g(t) d w$",
                            "Variance-preserving (VP) and VE variants",
                            "Marginal $p_t(x)$ approaches noise",
                        ],
                    ),
                    (
                        "Reverse SDE",
                        [
                            "$d x = [f(x,t) - g(t)^2 nabla_x log p_t(x)] d t + g(t) d bar(w)$",
                            "Score replaces unknown drift correction",
                            "Anderson's reverse-time SDE theorem",
                        ],
                    ),
                    (
                        "Probability Flow ODE",
                        [
                            "Same marginals without stochastic term",
                            "$d x = [f(x,t) - (1/2) g(t)^2 nabla_x log p_t(x)] d t$",
                            "Deterministic sampling path",
                        ],
                    ),
                    (
                        "Discretization",
                        [
                            "Euler–Maruyama for SDEs",
                            "DDIM as non-Markovian deterministic integrator",
                            "Step count vs quality tradeoff",
                        ],
                    ),
                ],
            ),
            (
                "Samplers",
                [
                    (
                        "DDPM Sampling",
                        [
                            "Markovian ancestral sampling",
                            "$T$ steps — slow at high resolution",
                            "Stochasticity helps diversity",
                        ],
                    ),
                    (
                        "DDIM",
                        [
                            "Skip timesteps with adjusted updates",
                            "eta=0 fully deterministic",
                            "10–50 steps often sufficient",
                        ],
                    ),
                    (
                        "Higher-Order Solvers",
                        [
                            "DPM-Solver, Heun, Runge–Kutta on ODE",
                            "Fewer function evaluations (NFE)",
                            "Active research area",
                        ],
                    ),
                    (
                        "Guidance at Inference",
                        [
                            "Classifier guidance (separate classifier grad)",
                            "CFG combines cond and uncond score",
                            "Large guidance → artifacts",
                        ],
                    ),
                ],
            ),
            (
                "Practical Inference",
                [
                    (
                        "Scheduler Choice",
                        [
                            "Timestep spacing: uniform vs SNR-based",
                            "Respacing pretrained models",
                            "Distillation for 1–4 step models",
                        ],
                    ),
                    (
                        "Memory & Speed",
                        [
                            "Attention at 512² vs 1024²",
                            "VAE decode bottleneck",
                            "Batch size 1 for interactive apps",
                        ],
                    ),
                    (
                        "Editing & Control",
                        [
                            "Img2img: partial noise then denoise",
                            "Inpainting with masked regions",
                            "ControlNet auxiliary conditioning",
                        ],
                    ),
                    (
                        "Failure Modes",
                        [
                            "Mode averaging at low steps",
                            "Text neglect with weak guidance",
                            "Watermark and safety filters",
                        ],
                    ),
                ],
            ),
            (
                "Connections & Outlook",
                [
                    (
                        "Flow Matching",
                        [
                            "Learn vector field transporting noise → data",
                            "Rectified flows and straight paths",
                            "Unified view with diffusion/flows",
                        ],
                    ),
                    (
                        "Consistency Models",
                        [
                            "Single-step or few-step generation",
                            "Distill iterative sampler",
                        ],
                    ),
                    (
                        "Bridge to Autoregressive",
                        [
                            "Different inductive bias: parallel vs sequential",
                            "Multimodal systems combine both",
                            "Days 9–10: language side",
                        ],
                    ),
                ],
            ),
        ],
    ),
    (
        "Autoregressive Models & LLM Training",
        "Causal language modeling at scale",
        [
            (
                "Autoregressive LM",
                [
                    (
                        "Factorization",
                        [
                            "$p(x) = product_(t=1)^T p(x_t | x_(<t); theta)$",
                            "Causal masking enforces ordering",
                            "Teacher forcing during training",
                        ],
                    ),
                    (
                        "Tokenization",
                        [
                            "BPE merges frequent pairs",
                            "Vocabulary size vs sequence length",
                            "Special tokens: BOS, EOS, PAD",
                        ],
                    ),
                    (
                        "Architecture Choices",
                        [
                            "Decoder-only Transformer (GPT)",
                            "RMSNorm, SwiGLU FFN, RoPE positions",
                            "Grouped-query attention (GQA)",
                        ],
                    ),
                    (
                        "Loss & Metrics",
                        [
                            "Cross-entropy over next token",
                            "Perplexity $PP = exp(H)$",
                            "Bits-per-byte for comparison across vocabs",
                        ],
                    ),
                ],
            ),
            (
                "Training at Scale",
                [
                    (
                        "Data Pipeline",
                        [
                            "Web crawl filtering and deduplication",
                            "Quality heuristics and safety filters",
                            "Mixture-of-sources ratios matter",
                        ],
                    ),
                    (
                        "Optimization",
                        [
                            "AdamW + cosine LR + warmup",
                            "Gradient accumulation for large effective batch",
                            "Loss spikes: skip step, reduce LR",
                        ],
                    ),
                    (
                        "Distributed Training",
                        [
                            "Data parallel: replicate model, shard batch",
                            "Tensor / pipeline / sequence parallel",
                            "ZeRO optimizer state sharding",
                        ],
                    ),
                    (
                        "Checkpointing",
                        [
                            "Save model, optimizer, RNG, step",
                            "Resume long runs after failure",
                            "HF format interoperability",
                        ],
                    ),
                ],
            ),
            (
                "Alignment & Fine-tuning",
                [
                    (
                        "Supervised Fine-Tuning",
                        [
                            "Instruction-response pairs",
                            "Catastrophic forgetting mitigation",
                            "LoRA: low-rank adapter updates",
                        ],
                    ),
                    (
                        "RLHF Overview",
                        [
                            "Reward model from human preferences",
                            "PPO fine-tune policy against reward",
                            "DPO direct preference optimization",
                        ],
                    ),
                    (
                        "Evaluation",
                        [
                            "Benchmark suites: MMLU, GSM8K, etc.",
                            "Human eval for chat quality",
                            "Contamination concerns",
                        ],
                    ),
                    (
                        "Safety",
                        [
                            "Red teaming and refusal behavior",
                            "System prompts and moderation",
                            "Alignment is ongoing, not solved",
                        ],
                    ),
                ],
            ),
            (
                "From Training to Inference",
                [
                    (
                        "Train vs Inference Workloads",
                        [
                            "Training: parallel across sequence (with mask)",
                            "Inference: sequential token generation",
                            "Memory dominated by activations vs KV",
                        ],
                    ),
                    (
                        "Model Compression Preview",
                        [
                            "Quantization INT8/INT4 weights",
                            "Knowledge distillation",
                            "Day 10: serving optimizations",
                        ],
                    ),
                    (
                        "Open vs Closed Ecosystem",
                        [
                            "Open weights: Llama, Mistral, etc.",
                            "API-only: GPT-4 class models",
                            "Responsible release considerations",
                        ],
                    ),
                ],
            ),
        ],
    ),
    (
        "Autoregressive Inference",
        "Decoding strategies and KV cache",
        [
            (
                "Text Generation Loop",
                [
                    (
                        "Autoregressive Decoding",
                        [
                            "Start from prompt tokens",
                            "Repeat: forward pass → logits → sample/argmax → append",
                            "Stop at EOS or max length",
                        ],
                    ),
                    (
                        "Greedy & Beam Search",
                        [
                            "Greedy: $x_t = arg max p(x_t|x_(<t))$",
                            "Beam search keeps top-$k$ partial sequences",
                            "Deterministic but often dull",
                        ],
                    ),
                    (
                        "Sampling Methods",
                        [
                            "Temperature $tau$: soften $\"softmax\"(\"logits\"/tau)$",
                            "Top-$k$ and nucleus (top-$p$) filtering",
                            "Repetition penalty",
                        ],
                    ),
                    (
                        "Latency Metrics",
                        [
                            "Time to first token (TTFT)",
                            "Tokens per second (TPS)",
                            "Prefill vs decode phases",
                        ],
                    ),
                ],
            ),
            (
                "KV Cache",
                [
                    (
                        "Motivation",
                        [
                            "Self-attention recomputes keys/values for all past tokens",
                            "At step $t$, past $K,V$ are unchanged",
                            "Cache avoids $O(t^2)$ redundant work per step",
                        ],
                    ),
                    (
                        "Cache Structure",
                        [
                            "Store $K,V$ per layer per head",
                            "Memory $O(L cdot H cdot T cdot d_h)$ grows with context",
                            "Batching pads to max length in batch",
                        ],
                    ),
                    (
                        "Prefill vs Decode",
                        [
                            "Prefill: process prompt in parallel (compute-bound)",
                            "Decode: one token at a time (memory-bandwidth)",
                            "Continuous batching in serving systems",
                        ],
                    ),
                    (
                        "Multi-Query / GQA",
                        [
                            "Share K,V heads across query heads",
                            "Reduces cache size with minimal quality loss",
                            "Standard in modern LLM inference",
                        ],
                    ),
                ],
            ),
            (
                "Efficient Attention",
                [
                    (
                        "FlashAttention",
                        [
                            "Tiled softmax without materializing full $n times n$",
                            "IO-aware — faster on GPU memory hierarchy",
                            "Training and prefill benefit most",
                        ],
                    ),
                    (
                        "PagedAttention (vLLM)",
                        [
                            "Non-contiguous KV blocks like virtual memory",
                            "Reduces fragmentation in batched serving",
                            "Higher GPU utilization",
                        ],
                    ),
                    (
                        "Speculative Decoding",
                        [
                            "Draft model proposes several tokens",
                            "Target model verifies in parallel",
                            "Acceptance rate determines speedup",
                        ],
                    ),
                    (
                        "Long Context",
                        [
                            "RoPE scaling, YaRN, ALiBi",
                            "Ring attention for very long sequences",
                            "Context window vs true reasoning",
                        ],
                    ),
                ],
            ),
            (
                "Serving & Systems",
                [
                    (
                        "Quantization for Inference",
                        [
                            "Weight-only INT4 (GPTQ, AWQ)",
                            "KV cache quantization",
                            "Accuracy vs throughput tradeoffs",
                        ],
                    ),
                    (
                        "Batching Strategies",
                        [
                            "Static vs continuous batching",
                            "Request scheduling and preemption",
                            "Multi-tenant SLA constraints",
                        ],
                    ),
                    (
                        "Course Recap",
                        [
                            "Week 1: ML/DL foundations → Transformers",
                            "Week 2: generative models → production LLM inference",
                            "Final assessment ties math + code together",
                        ],
                    ),
                ],
            ),
        ],
    ),
]


# Use hand-authored decks for upgraded days (override the inline placeholders).
COURSE[1] = DAY02_SLIDES  # Day 2 — Statistical Learning (MML Ch. 8–12)
COURSE[2] = DAY03_SLIDES  # Day 3 — Deep Neural Networks
COURSE[3] = DAY04_SLIDES  # Day 4 — Convolutional Neural Networks
COURSE[4] = DAY05_SLIDES  # Day 5 — Sequence Models & Transformers
COURSE[5] = DAY06_SLIDES  # Day 6 — Generative Modeling & DDPM
COURSE[6] = DAY07_SLIDES  # Day 7 — Score, SDEs & Flow Matching
COURSE[7] = DAY08_SLIDES  # Day 8 — Guidance, Solvers & Fast Sampling
COURSE[8] = DAY09_SLIDES  # Day 9 — Autoregressive Language Models
COURSE[9] = DAY10_SLIDES  # Day 10 — LLM Inference & Alignment


def render_bullets(lines: list[str]) -> str:
    return "\n".join(f"- {typst_escape(fix_typst_math(line))}" for line in lines)


def render_slide(title: str, lines: list[str]) -> str:
    return f"== {typst_escape(title)}\n\n{render_bullets(lines)}\n"


def render_figure_slide(path: str, title: str) -> str:
    """Figure slide: image scaled to *fit* the slide (never cropped/overflowing).

    `fit: "contain"` with both a width and height cap guarantees the whole
    figure is visible regardless of its aspect ratio. No grey caption line
    (it tended to spill onto an otherwise-empty slide).
    """
    return (
        f"== {typst_escape(title)}\n\n"
        f"#align(center + horizon)[#image(\"{path}\", width: 92%, height: 82%, fit: \"contain\")]\n"
    )


def render_day(day: int, title: str, subtitle: str, parts: list) -> str:
    lines: list[str] = [
        '#import "../lib.typ": *',
        "",
        f"#show: course-theme.with(title: [{typst_escape(title)}], subtitle: [Day {day} | Aug 2026])",
        "",
        f"= Day {day}: {typst_escape(title)}",
        "",
        "== Welcome",
        "",
        f"- *{typst_escape(title)}* — {typst_escape(subtitle)}",
        "- 3 hours lecture + practical",
        "- Slides, notes, and code on the course site",
        "",
        "== Outline",
        "",
    ]
    for part_title, _ in parts:
        lines.append(f"- {typst_escape(part_title)}")
    lines.append("")

    figures = CURATED_FIGURES.get(day) or find_figures(day)
    fig_idx = 0

    for part_i, (part_title, slides) in enumerate(parts, start=1):
        # Numbered section divider: "1 · Linear Algebra".
        lines.append(f"= {part_i} · {typst_escape(part_title)}")
        lines.append("")
        for slide_i, (slide_title, bullets) in enumerate(slides, start=1):
            num = f"{part_i}.{slide_i}"
            lines.append(render_slide(f"{num}  {slide_title}", bullets))
            # Insert the curated figure for this slide when one is available.
            fig = figures[fig_idx] if fig_idx < len(figures) else None
            fig_idx += 1
            if fig:
                lines.append(render_figure_slide(fig, f"{num}  {slide_title}"))

    # Remaining figures (only for days using the auto-find list) go in an appendix.
    if fig_idx < len(figures):
        remaining = [f for f in figures[fig_idx:] if f]
        if remaining:
            lines.append("= Additional Figures")
            lines.append("")
            for k, fig in enumerate(remaining, start=1):
                lines.append(render_figure_slide(fig, f"Supplementary figure {k}"))

    lines.extend(
        [
            "== Summary",
            "",
            f"- Day {day}: *{typst_escape(title)}*",
            f"- {typst_escape(subtitle)}",
            "- Questions welcome — practical follows",
            "",
            "== Questions?",
            "",
            "#align(center + horizon)[",
            "  #text(size: 44pt, weight: \"bold\", fill: course-primary)[Questions?]",
            "",
            "  #v(1em)",
            "  Practical session follows — see course site.",
            "]",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    DAYS_DIR.mkdir(parents=True, exist_ok=True)
    for day, (title, subtitle, parts) in enumerate(COURSE, start=1):
        content = render_day(day, title, subtitle, parts)
        out = DAYS_DIR / f"day{day:02d}.typ"
        out.write_text(content, encoding="utf-8")
        slide_count = content.count("\n== ") + content.count("\n= ")
        print(f"Wrote {out} ({slide_count} slide markers)")
    print(f"Done — {len(COURSE)} decks in {DAYS_DIR}")


if __name__ == "__main__":
    main()
