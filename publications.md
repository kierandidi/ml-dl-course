---
layout: welcome
title: Resources
buttons:
  print: false
---

Here you will find the **reading list** for the Machine Learning & Deep Learning course (Aug 2026). Start with the **essential** materials below, then use the **per-day** sections as you follow the lectures. Each day links to [notes](/blog/), [slides](/assets/slides/), and [practicals](/projects/).

---

## Essential reading (crucial for the whole course) {: #essential}

These are the highest-value references — if you only read a few things, read these.

### Textbooks & course notes

| Resource | Why it matters |
|----------|----------------|
| [Prince — *Understanding Deep Learning* (UDL)](https://udlbook.github.io/udlbook/) | Clear unified treatment of ML, deep learning, and generative models; [companion notebooks](https://github.com/udlbook/udlbook/tree/main/Notebooks) |
| [UCL × DeepMind Deep Learning Lectures (2020)](https://www.deepmind.com/learning-resources/deep-learning-lecture-series-2020) | Video + slide intuition for Week 1 (nets, CNNs, RNNs, attention) |
| [MIT 6.S184 — Flow Matching & Diffusion (2026)](https://diffusion.csail.mit.edu/2026/index.html) | Week 2 backbone: SDEs, flow matching, score matching, labs |
| [Lai et al. — *Principles of Diffusion Models*](https://arxiv.org/abs/2510.21890) | Modern unified book on diffusion / score / flow views ([book site](https://the-principles-of-diffusion-models.github.io/)) |

### Papers (foundational)

| Paper | Topic |
|-------|--------|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al.) | Transformers — architecture of modern LLMs |
| [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (Ho et al.) | DDPM — discrete diffusion training & sampling |
| [Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456) (Song et al.) | Continuous diffusion, score matching, reverse SDE |
| [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) (Kingma & Welling) | VAE — ELBO and latent-variable generation |
| [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) (Lipman et al.) | Direct training of velocity fields |

### Implementations & labs (hands-on)

- [MIT 6.S184 labs](https://diffusion.csail.mit.edu/2026/index.html) — ODE/SDE simulation, flow matching ([`lab_one.ipynb`](/notebooks/practicals/) in repo is adapted from this track)
- [Karpathy — micrograd](https://github.com/karpathy/micrograd) — minimal autograd / backprop intuition
- [Karpathy — nanoGPT](https://github.com/karpathy/nanoGPT) — small-scale transformer training reference

---

## Core textbooks (extended)

- [Goodfellow, Bengio & Courville — *Deep Learning*](https://www.deeplearningbook.org/) — comprehensive DL reference (Ch. 2–9 for Week 1)
- [Bishop — *Pattern Recognition and Machine Learning*](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) — probabilistic foundations (Ch. 1–4, 9)
- [Murphy — *Probabilistic Machine Learning: An Introduction*](https://probml.github.io/pml-book/book1.html) — modern Bayesian / ML perspective (Ch. 11 regression)
- [Hastie, Tibshirani & Friedman — *Elements of Statistical Learning*](https://hastie.su.domains/ElemStatLearn/) — classical statistical learning (Ch. 3–4)
- [Boyd & Vandenberghe — *Convex Optimization*](https://web.stanford.edu/~boyd/cvxbook/) — optimization background for Day 1

---

## Reading by day

Use these alongside each lecture. **Notes · Slides · Practical** are on the [About](/index/) page.

### Day 1 — Math foundations (17 Aug) {: #day-1}

**[Lecture notes](/blog/lectures/2026/08/17/day01-math-foundations/) · [Slides](/assets/slides/day01.pdf) · [Exercise (Colab)](https://colab.research.google.com/github/kierandidi/ml-dl-course/blob/main/notebooks/practicals/day01.ipynb)**

- [Deisenroth, Faisal & Ong — *Mathematics for Machine Learning*](https://mml-book.com) (Ch. 2–6: linear algebra through probability)
- [Deisenroth & Ong — *There and Back Again: A Tale of Slopes and Expectations*](https://mml-book.github.io/slopes-expectations.html) (NeurIPS 2020 tutorial on integration & differentiation)
- [Modern Integration Methods in ML](https://mml-book.github.io/book/additional_chapters/integration-methods.pdf) (MML supplementary chapter)
- [3Blue1Brown — Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3tZaY) (geometric intuition)
- [3Blue1Brown — Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3tZaY) (derivatives and integrals)

---

### Day 2 — Statistical learning (18 Aug) {: #day-2}

**[Lecture notes](/blog/lectures/2026/08/18/day02-statistical-learning/) · [Slides](/assets/slides/day02.pdf) · [Exercise (Colab)](https://colab.research.google.com/github/kierandidi/ml-dl-course/blob/main/notebooks/practicals/day02.ipynb)**

- [Hastie et al. — ESL](https://hastie.su.domains/ElemStatLearn/), Ch. 3–4 (linear methods, classification)
- [Murphy — PML](https://probml.github.io/pml-book/book1.html), Ch. 11 (linear / logistic regression)
- [scikit-learn — Linear models](https://scikit-learn.org/stable/modules/linear_model.html) (practical API reference)
- [Prince — UDL](https://udlbook.github.io/udlbook/), Ch. 2 (supervised learning framework)

---

### Day 3 — Deep neural networks (19 Aug) {: #day-3}

**[Lecture notes](/blog/lectures/2026/08/19/day03-deep-neural-networks/) · [Slides](/assets/slides/day03.pdf) · [Exercise (Colab)](https://colab.research.google.com/github/kierandidi/ml-dl-course/blob/main/notebooks/practicals/day03.ipynb)**

- [Goodfellow et al. — Deep Learning](https://www.deeplearningbook.org/), Ch. 6–8 (MLPs, backprop, optimization)
- [UCL × DeepMind — Lecture 2 (feedforward & optimization)](https://www.deepmind.com/learning-resources/deep-learning-lecture-series-2020)
- [Karpathy — micrograd](https://github.com/karpathy/micrograd) (build backprop from scratch)
- [Prince — UDL](https://udlbook.github.io/udlbook/), Ch. 7–9 (training, gradients, activations)

---

### Day 4 — CNNs & computer vision (20 Aug) {: #day-4}

**[Lecture notes](/blog/lectures/2026/08/20/day04-convolutional-networks/) · [Slides](/assets/slides/day04.pdf) · [Exercise (Colab)](https://colab.research.google.com/github/kierandidi/ml-dl-course/blob/main/notebooks/practicals/day04.ipynb)**

- [Goodfellow et al. — Deep Learning, Ch. 9](https://www.deeplearningbook.org/contents/convnets.html) (convolutional networks)
- [CS231n — Convolutional Neural Networks](https://cs231n.github.io/convolutional-networks/) (Stanford vision course notes)
- [He et al. — ResNet](https://arxiv.org/abs/1512.03385) (skip connections, very deep nets)
- [UCL × DeepMind — Lectures 3–4 (CNNs)](https://www.deepmind.com/learning-resources/deep-learning-lecture-series-2020)
- [Prince — UDL](https://udlbook.github.io/udlbook/), Ch. 10 (convolutional networks)

---

### Day 5 — RNNs & Transformers (21 Aug) {: #day-5}

**[Lecture notes](/blog/lectures/2026/08/21/day05-rnns-and-transformers/) · [Slides](/assets/slides/day05.pdf) · [Exercise (Colab)](https://colab.research.google.com/github/kierandidi/ml-dl-course/blob/main/notebooks/practicals/day05.ipynb)**

- [Vaswani et al. — Attention Is All You Need](https://arxiv.org/abs/1706.03762) (transformer architecture)
- [Karpathy — The Unreasonable Effectiveness of RNNs](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) (sequence modeling intuition)
- [Jay Alammar — The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) (visual walkthrough)
- [NVIDIA — Intro to Transformers (slides)](https://docs.google.com/presentation/d/1ZXFIhYczos679r70Yu8vV9uO6B1J0ztzeDxbnBxD1S0/edit) (production-scale context)
- [UCL × DeepMind — Lectures 6–8 (sequences & attention)](https://www.deepmind.com/learning-resources/deep-learning-lecture-series-2020)
- [Prince — UDL](https://udlbook.github.io/udlbook/), Ch. 12 (transformers)

---

### Day 6 — Generative modeling (24 Aug) {: #day-6}

**[Lecture notes](/blog/lectures/2026/08/22/day06-generative-modeling-diffusion/) · [Slides](/assets/slides/day06.pdf) · [Exercise (Colab)](https://colab.research.google.com/github/kierandidi/ml-dl-course/blob/main/notebooks/practicals/day06.ipynb)**

- [Principles of Diffusion Models](https://arxiv.org/abs/2510.21890) — Ch. 1–2 (generative modeling, variational view)
- [Kingma & Welling — VAE](https://arxiv.org/abs/1312.6114) (ELBO, reparameterization)
- [Rezende & Mohamed — Normalizing Flows](https://arxiv.org/abs/1505.05770) (exact likelihood via change of variables)
- [Goodfellow et al. — GANs (Ch. 20)](https://www.deeplearningbook.org/contents/generative_models.html) (implicit models)
- [Prince — UDL](https://udlbook.github.io/udlbook/), Ch. 16 (VAEs and latent variables)

---

### Day 7 — Training diffusion & flow models (25 Aug) {: #day-7}

**[Lecture notes](/blog/lectures/2026/08/23/day07-score-sde-flow-matching/) · [Slides](/assets/slides/day07.pdf) · [Exercise (Colab)](https://colab.research.google.com/github/kierandidi/ml-dl-course/blob/main/notebooks/practicals/day07.ipynb)**

- [MIT 6.S184 — 2026 lectures 2–3](https://diffusion.csail.mit.edu/2026/index.html) (flow matching, score matching)
- [MIT course notes PDF](https://diffusion.csail.mit.edu/) (companion to lectures)
- [Song et al. — Score-Based SDEs](https://arxiv.org/abs/2011.13456) (training objectives)
- [Lipman et al. — Flow Matching](https://arxiv.org/abs/2210.02747) (conditional flow paths)
- [Ho et al. — DDPM](https://arxiv.org/abs/2006.11239) (discrete denoising diffusion)
- [Prince — UDL](https://udlbook.github.io/udlbook/), Ch. 18 (diffusion models)

---

### Day 8 — Inference: SDEs & ODEs (26 Aug) {: #day-8}

**[Lecture notes](/blog/lectures/2026/08/24/day08-guidance-solvers-fast-sampling/) · [Slides](/assets/slides/day08.pdf) · [Exercise (Colab)](https://colab.research.google.com/github/kierandidi/ml-dl-course/blob/main/notebooks/practicals/day08.ipynb)**

- [Song et al. — Score-Based SDEs](https://arxiv.org/abs/2011.13456) — § sampling and reverse-time SDE
- [Ho et al. — DDPM](https://arxiv.org/abs/2006.11239) — sampling algorithms
- [Ho & Salimans — Classifier-Free Guidance](https://arxiv.org/abs/2207.12598) (conditional generation without a separate classifier)
- [MIT 6.S184 — Lab 1 (ODE/SDE simulation)](https://diffusion.csail.mit.edu/2026/index.html)
- [Yang Song — score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch) (reference sampler code)
- [Principles of Diffusion Models](https://arxiv.org/abs/2510.21890) — sampling & guidance chapters

---

### Day 9 — Autoregressive LLMs (27 Aug) {: #day-9}

**[Lecture notes](/blog/lectures/2026/08/25/day09-autoregressive-llms/) · [Slides](/assets/slides/day09.pdf) · [Exercise (Colab)](https://colab.research.google.com/github/kierandidi/ml-dl-course/blob/main/notebooks/practicals/day09.ipynb)**

- [Vaswani et al. — Attention Is All You Need](https://arxiv.org/abs/1706.03762) (full architecture)
- [Gordić — Inside the Transformer: Life of a Token](https://www.aleksagordic.com/blog/transformer) (modern training stack: RoPE, RMSNorm, GQA)
- [Su et al. — RoFormer / RoPE](https://arxiv.org/abs/2104.09864) (rotary position embeddings)
- [Karpathy — nanoGPT](https://github.com/karpathy/nanoGPT) (minimal GPT training)
- [UCL × DeepMind — Lecture 7–8](https://www.deepmind.com/learning-resources/deep-learning-lecture-series-2020)

---

### Day 10 — AR inference & KV cache (28 Aug) {: #day-10}

**[Lecture notes](/blog/lectures/2026/08/26/day10-ar-inference/) · [Slides](/assets/slides/day10.pdf) · [Exercise (Colab)](https://colab.research.google.com/github/kierandidi/ml-dl-course/blob/main/notebooks/practicals/day10.ipynb) · [Assessment](/projects/final-assessment/)**

- [Vaswani et al.](https://arxiv.org/abs/1706.03762) — §3 (complexity, autoregressive decoding)
- [Holtzman et al. — Neural Text Degeneration](https://arxiv.org/abs/1904.09751) (nucleus / top-*p* sampling)
- [Gordić — Life of a Token](https://www.aleksagordic.com/blog/transformer) — KV cache & FLOPs section
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — decoder-only inference patterns
- [vLLM / inference systems blog posts](https://docs.vllm.ai/) (optional: production serving)

---

## Course practicals (this website)

| Day | Notebook |
|-----|----------|
| 1 | [`day01.ipynb`](/notebooks/practicals/day01.ipynb) |
| 2 | [`day02.ipynb`](/notebooks/practicals/day02.ipynb) |
| 3 | [`day03.ipynb`](/notebooks/practicals/day03.ipynb) |
| 4 | [`day04.ipynb`](/notebooks/practicals/day04.ipynb) |
| 5 | [`day05.ipynb`](/notebooks/practicals/day05.ipynb) |
| 6 | [`day06.ipynb`](/notebooks/practicals/day06.ipynb) |
| 7 | [`day07.ipynb`](/notebooks/practicals/day07.ipynb) |
| 8 | [`day08.ipynb`](/notebooks/practicals/day08.ipynb) |
| 9 | [`day09.ipynb`](/notebooks/practicals/day09.ipynb) |
| 10 | [`day10.ipynb`](/notebooks/practicals/day10.ipynb) |
| — | [Final assessment](/notebooks/assessment/final_assessment.ipynb) |

---

## Supplementary videos & tutorials

- [3Blue1Brown](https://www.3blue1brown.com/) — linear algebra, calculus, neural networks
- [fast.ai](https://www.fast.ai/) — practical deep learning (top-down)
- [CS231n](https://cs231n.stanford.edu/) — computer vision (CNN focus)
- [CVPR 2022 Diffusion Tutorial](https://cvpr2022-tutorial-diffusion-models.github.io/) — diffusion foundations
- [Illustrated Diffusion (Jay Alammar)](https://jalammar.github.io/illustrated-stable-diffusion/) — intuitive diffusion pipeline

---

## Figure credits

Slide and note figures under `assets/figures/` are taken from course materials. Please retain attribution when reusing.

| Figure / topic | Source |
|----------------|--------|
| Week 1 slides & notes | UCL × DeepMind DL Lectures 2020; ML Materials PPTX |
| Week 2 diffusion / flow | MIT 6.S184 2026 lectures & course notes (CC BY-NC-SA) |
| Diffusion three-view diagrams | *Principles of Diffusion Models* (Lai et al., arXiv:2510.21890) |
| Transformer / KV cache | UCL L7–L8; [Life of a Token](https://www.aleksagordic.com/blog/transformer) (Aleksa Gordić) |
| DDPM / score SDE | Ho et al.; Song et al. |
| ELBO / VAE | Prince UDL; Kingma & Welling |

Contact [kieran.didi@gmail.com](mailto:kieran.didi@gmail.com) if a credit is missing.
