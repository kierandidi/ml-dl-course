---
layout: post
title: Day 4 - Convolutional Networks
image: /assets/img/lessons/day04.png
accent_image: 
  background: url('/assets/img/sampling_space.png') center/cover
  overlay: false
accent_color: '#ccc'
theme_color: '#ccc'
description: >
  Convolution, pooling, and canonical CNN architectures for vision.
invert_sidebar: true
---

# Day 4 - Convolutional Networks

### Optional reading for this lesson
- [Goodfellow et al. — Deep Learning, Ch. 9](https://www.deeplearningbook.org/contents/convnets.html)
- [CS231n — Convolutional Neural Networks](https://cs231n.github.io/convolutional-networks/)
- [He et al. — Deep Residual Learning (ResNet)](https://arxiv.org/abs/1512.03385)
- [Complete reading list for Day 4](/publications/#day-4) (all resources for this lecture)


### [Slides](/assets/slides/day04.pdf)

### [Practical](/projects/day04-practical/)

Convolutional neural networks exploit translation equivariance and local connectivity to process images efficiently. We build up from the 2D convolution operation through pooling layers to ResNet-style architectures used in modern vision systems.

* toc
{:toc}

## 1. The Convolution Operation

### 1.1 Discrete 2D convolution

> The **2D convolution** of input $$\mathbf{X}$$ and kernel $$\mathbf{K}$$ produces output $$\mathbf{Y}$$ with entries $$Y_{ij} = \sum_{u,v} K_{uv}\, X_{i+u,\,j+v}$$ (cross-correlation is used in most DL frameworks).
{:.lead}

For input $$\mathbf{X} \in \mathbb{R}^{H \times W \times C_{\mathrm{in}}}$$ and $$C_{\mathrm{out}}$$ filters of size $$k \times k$$,

$$\mathbf{Y}_{h,w,c'} = \sum_{c=1}^{C_{\mathrm{in}}} \sum_{i=0}^{k-1}\sum_{j=0}^{k-1} W_{i,j,c,c'}\, X_{h+i,\,w+j,\,c} + b_{c'}.$$

**Parameter sharing**: one filter slides across the entire spatial grid — far fewer weights than a fully connected layer on flattened pixels.

![Convolution sliding window](/assets/figures/day04/pdf0_page005.png)

Output spatial size with padding $$p$$ and stride $$s$$:

$$H_{\mathrm{out}} = \left\lfloor \frac{H + 2p - k}{s} \right\rfloor + 1.$$

### 1.2 Translation equivariance

> A layer is **translation equivariant** if shifting the input shifts the output: $$f(\mathrm{shift}(\mathbf{x})) = \mathrm{shift}(f(\mathbf{x}))$$. Conv layers satisfy this; fully connected layers do not.
{:.lead}

Equivariance is ideal for detection and segmentation: a cat in the corner activates the same filter as a cat in the center.

**1×1 convolutions** mix channels without changing spatial size — used in Inception and pointwise expansions in MobileNet.

Depthwise separable conv factorizes into depthwise (spatial) + pointwise (channel) steps, reducing FLOPs:

$$\text{standard} \; O(k^2 C_{\mathrm{in}} C_{\mathrm{out}}), \quad \text{separable} \; O(k^2 C_{\mathrm{in}} + C_{\mathrm{in}} C_{\mathrm{out}}).$$

## 2. Pooling and Normalization

### 2.1 Max and average pooling

> **Max pooling** takes the maximum in each $$p \times p$$ window; **average pooling** takes the mean. Both reduce spatial resolution and provide local translation invariance.
{:.lead}

Max pool with stride $$s = p$$ halves spatial dimensions (typical $$p=2$$):

$$Y_{i,j} = \max_{0 \leq u,v < p} X_{s i + u,\, s j + v}.$$

Pooling has no learnable parameters but downsamples activations, expanding receptive field in deeper layers.

![Max pool 2×2 stride 2](/assets/figures/day04/pdf0_page010.png)

**Global average pooling** (GAP) averages each feature map to a scalar — common before classification head in ResNet, replacing large fully connected layers.

### 2.2 Batch normalization

> **BatchNorm** normalizes pre-activations across the mini-batch: $$\hat{z} = (z - \mu_B)/\sqrt{\sigma_B^2 + \epsilon}$$, then applies learnable scale $$\gamma$$ and shift $$\beta$$.
{:.lead}

During training, batch statistics $$\mu_B, \sigma_B^2$$; at inference, use running averages $$\mu_{\mathrm{EMA}}, \sigma_{\mathrm{EMA}}^2$$.

BatchNorm reduces internal covariate shift, allows higher learning rates, and acts as mild regularization.

**LayerNorm** normalizes across features per example — preferred in transformers and RNNs where batch statistics are unreliable.

## 3. Canonical Architectures

### 3.1 LeNet, AlexNet, and VGG

> **LeNet-5** (1998): conv → pool → conv → pool → FC. **AlexNet** (2012): deeper, ReLU, dropout, GPU training. **VGG**: stacks of 3×3 convs — simplicity over large filters.
{:.lead}

AlexNet showed that depth + ReLU + data augmentation could dominate ImageNet:

$$\text{top-5 error: } 15.3\% \; (2012) \; \text{vs } 26\% \; \text{(previous best)}.$$

VGG-16 uses only 3×3 convs: two 3×3 layers have receptive field 5×5 but more nonlinearities and fewer parameters than one 5×5 layer.

![Architecture comparison timeline](/assets/figures/day04/pdf0_page015.png)

### 3.2 ResNet and skip connections

> A **residual block** learns $$\mathcal{F}(\mathbf{x})$$ with $$\mathbf{y} = \mathbf{x} + \mathcal{F}(\mathbf{x})$$. Gradients flow directly through the skip, enabling 100+ layer nets.
{:.lead}

If optimal mapping is close to identity, residual form lets $$\mathcal{F} \approx 0$$ rather than learning identity explicitly.

Bottleneck block (1×1 → 3×3 → 1×1) reduces compute in deep models:

$$\mathbf{y} = \mathbf{x} + \mathcal{F}_{1\times1}(\mathcal{F}_{3\times3}(\mathcal{F}_{1\times1}(\mathbf{x}))).$$

**Receptive field** grows with depth: stacking $$L$$ layers of 3×3 conv gives RF $$\approx 1 + 2L$$ (without dilation).

## 4. Training CNNs for Vision

### 4.1 Data augmentation

> **Data augmentation** applies label-preserving transforms (random crop, flip, color jitter) to expand effective training set size and improve robustness.
{:.lead}

Standard ImageNet pipeline: random resized crop to 224×224, horizontal flip, normalize with dataset mean/std.

**Mixup** blends pairs of examples:

$$\tilde{\mathbf{x}} = \lambda \mathbf{x}_i + (1-\lambda)\mathbf{x}_j, \quad \tilde{y} = \lambda y_i + (1-\lambda) y_j.$$

**Cutout / Random Erasing** drops random patches, forcing context reasoning.

![Augmentation examples](/assets/figures/day04/pdf0_page020.png)

### 4.2 Transfer learning

> **Transfer learning** initializes from pretrained weights (ImageNet) and fine-tunes on a target task — essential when target data is limited.
{:.lead}

Typical recipe: replace final FC layer, train head with frozen backbone, then unfreeze top layers with small $$\eta$$.

Feature maps from early layers capture edges/textures (general); late layers capture class-specific semantics.

**Linear probe** vs **full fine-tune**: probe trains only the head (fast baseline); full fine-tune adapts all weights (better with enough data).

Compute FLOPs for conv layer: $$\approx 2\, H_{\mathrm{out}} W_{\mathrm{out}} C_{\mathrm{out}} k^2 C_{\mathrm{in}}$$ multiply-adds.

## Checkpoint summary

Before moving to the practical, confirm you can:

- Conv layers share weights spatially and are translation equivariant.
- Pooling downsamples; BatchNorm stabilizes training; GAP feeds compact classifiers.
- ResNet skip connections solve degradation and enable very deep nets.
- Augmentation + transfer learning are standard for practical vision pipelines.
