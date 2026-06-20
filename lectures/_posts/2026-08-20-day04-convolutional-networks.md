---
layout: post
title: Day 4 - Convolutional Neural Networks
image: /assets/img/lessons/day04.png
description: >
  Locality, weight sharing, and hierarchy: the convolution operation, classic architectures, and dense prediction.
invert_sidebar: true
---

# Day 4 - Convolutional Neural Networks

### Optional reading for this lesson
- [UCL x DeepMind DL2020 — L3: Convolutional Neural Networks](https://www.youtube.com/watch?v=shVKhOmT0HE)
- [UCL x DeepMind DL2020 — L4: Advanced Models for Computer Vision](https://www.youtube.com/watch?v=_aUq7lmMfxo)
- [Goodfellow, Bengio & Courville — Deep Learning](https://www.deeplearningbook.org/), Ch. 9
- [CS231n — Convolutional Neural Networks](https://cs231n.github.io/convolutional-networks/)

### [Slides](/assets/slides/day04.pdf)

### Exercise

[Download the notebook](/notebooks/practicals/day04.ipynb) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kierandidi/ml-dl-course/blob/main/notebooks/practicals/day04.ipynb)

Fully-connected networks treat an image as an unordered bag of pixels — throwing away the very structure that makes vision tractable. Convolutional neural networks instead bake three priors directly into the architecture: locality, weight sharing, and hierarchy. Today we build the convolution operation from these priors, count its parameters and receptive field, assemble the conv–norm–nonlinearity block, trace the architectures that won ImageNet (AlexNet, VGG, ResNet), and see how the same backbone extends to detection and segmentation.

* toc
{:toc}

## 1. Why Convolutions

### 1.1 The failure of fully-connected vision

> A **fully-connected** layer connects every output to every input, so it has no notion of spatial structure: permuting the input pixels leaves it unchanged. For images this is both statistically wasteful and computationally explosive.
{:.lead}

**Why this matters.** Consider a modest $$224\times224\times3$$ image — that is $$150{,}528$$ input numbers. A single fully-connected layer with $$1000$$ units would need $$150{,}528\times1000\approx1.5\times10^{8}$$ weights *in the first layer alone*. Three problems follow:

1. **Too many parameters.** The model overfits and is expensive to store and train.
2. **No use of structure.** A fully-connected layer is invariant to permuting the inputs — it cannot exploit the fact that pixel $$(i,j)$$ is next to $$(i,j{+}1)$$. Shuffle the pixels with a fixed permutation and the layer learns equally well, which means it is ignoring everything we know about images.
3. **No reuse across space.** A feature detector learned for the top-left corner must be re-learned, independently, for every other location.

The cure is to **constrain** the architecture using two facts that are true of essentially all natural images. Those constraints turn the unwieldy fully-connected layer into the convolution.

### 1.2 Locality and translation invariance

> **Locality**: pixels close together are statistically far more correlated than distant ones, so a useful feature can be computed from a small **local patch**. **Translation invariance**: a meaningful pattern (an edge, an eye) means the same thing wherever it appears.
{:.lead}

![Two priors about images: nearby pixels are correlated (locality), and patterns are meaningful regardless of position (translation invariance).](/assets/figures/day04/cnn_locality.png)

These two observations translate directly into two architectural constraints:

- **Locality $$\Rightarrow$$ local connectivity.** Connect each output unit only to a small patch of the input (its *receptive field*), say $$3\times3$$, instead of to the whole image. This alone slashes the parameter count.
- **Translation invariance $$\Rightarrow$$ weight sharing.** Use the *same* small set of weights at every spatial location. A pattern detector learned in one place is then automatically available everywhere.

A layer that connects locally **and** shares weights across positions is exactly a **convolution**. So the convolution is not an arbitrary trick — it is the minimal layer consistent with locality and translation invariance.

### 1.3 Weight sharing and feature hierarchy

> **Weight sharing** means one filter (kernel) scans the entire image, giving **translation equivariance**: shift the input, and the feature map shifts the same way. Stacking such layers builds a **hierarchy** of increasingly abstract features.
{:.lead}

![Weight sharing reuses one set of parameters across the image; stacking layers composes low-level features (edges, textures) into high-level ones (parts, objects).](/assets/figures/day04/cnn_topology.png)

Two payoffs:

- **Parameter efficiency.** A $$3\times3$$ filter has $$9$$ weights regardless of image size, versus the millions a fully-connected layer would need. The same nine numbers detect the pattern everywhere.
- **Equivariance.** Convolution commutes with translation: $$\text{conv}(\text{shift}(x)) = \text{shift}(\text{conv}(x))$$. The network does not have to learn each pattern separately at each location.

The second payoff is **compositional**. Early layers detect oriented edges; the next layers combine edges into corners and textures; deeper layers combine those into object parts, and finally whole objects. Because each layer's units only look at a local window of the previous layer's feature map, *depth* is what lets a unit eventually "see" a large region of the original image — the receptive field we quantify later.

## 2. The Convolution Operation

### 2.1 From local connectivity to convolution

> A **convolutional layer** computes each output by applying one shared kernel $$\boldsymbol{w}$$ over a sliding local window of the input: $$y = \boldsymbol{w} * \boldsymbol{x} + b.$$
{:.lead}

![Going from a locally-connected layer (local but with position-specific weights) to a convolution (local *and* weight-shared): a single $$3\times3$$ kernel with a $$3\times3$$ receptive field.](/assets/figures/day04/cnn_receptive.png)

Start from a fully-connected layer and impose the two priors in turn. First **locality**: zero out all connections except those to a small patch — now each output depends on a $$3\times3$$ receptive field. Second **weight sharing**: force the patch weights to be identical at every position. What remains is a convolution with a single small kernel.

(Deep-learning libraries actually compute **cross-correlation** — they do not flip the kernel — but the term "convolution" has stuck. Since the kernel is learned, the distinction does not matter for training.)

### 2.2 The convolution step by step and channels

> **2-D convolution** slides a kernel over the input; at each position it computes an elementwise product with the underlying patch and sums: $$(\boldsymbol{w} * \boldsymbol{x})_{i,j} = \sum_{u,v} w_{u,v}\,x_{i+u,\,j+v}.$$ With multiple channels the kernel spans all of them.
{:.lead}

![The kernel slides across the image and produces one output value at each position, forming a feature map.](/assets/figures/day04/cnn_convop.png)

The output is a **feature map**: a 2-D array recording how strongly the kernel's pattern is present at each location. Bright spots are where the feature "fires". Because the same kernel is used everywhere, the feature map is translation-equivariant.

Real images have **channels** (RGB has 3), so inputs and outputs are *tensors* of shape height $$\times$$ width $$\times$$ channels:

![Inputs and outputs are tensors with a channel dimension; a filter spans all input channels.](/assets/figures/day04/cnn_tensors.png)

- A single filter has shape $$K\times K\times C_{\text{in}}$$ — it spans **all** input channels and produces **one** output channel.
- Using $$C_{\text{out}}$$ such filters yields $$C_{\text{out}}$$ output channels, each a different learned feature.
- The full weight tensor is $$K\times K\times C_{\text{in}}\times C_{\text{out}}$$ (plus $$C_{\text{out}}$$ biases).
- A **$$1\times1$$ convolution** ($$K=1$$) mixes channels at each pixel — a per-pixel linear layer, used to cheaply change channel count.

### 2.3 Derivation: output size and parameter count

> For input width $$W$$, kernel size $$K$$, padding $$P$$, and stride $$S$$, the output width is $$O = \left\lfloor\dfrac{W - K + 2P}{S}\right\rfloor + 1,$$ and a conv layer has $$K^2 C_{\text{in}} C_{\text{out}}$$ weights, **independent of image size**.
{:.lead}

**Output size.** Pad the input on each side by $$P$$, giving effective width $$\textcolor{teal}{W+2P}$$. The kernel's left edge can sit at positions $$0,S,2S,\dots$$ as long as the whole kernel fits, i.e. up to $$\textcolor{teal}{W+2P}-\textcolor{purple}{K}$$. The number of such positions is $$\big\lfloor (\textcolor{teal}{W+2P}-\textcolor{purple}{K})/S\big\rfloor + 1$$, which is the formula above (the same holds for height).

- **"Valid" padding** ($$P=0$$) shrinks the map by $$K-1$$ each layer.
- **"Same" padding** with $$S=1$$ uses $$P=(K-1)/2$$ to keep the spatial size fixed — the usual choice for $$3\times3$$ kernels ($$P=1$$).

*Worked example.* Input $$32\times32$$, $$K=3$$, $$P=1$$, $$S=1$$: $$O=\lfloor(32-3+2)/1\rfloor+1 = 32$$ (size preserved). With $$S=2$$: $$O=\lfloor(32-3+2)/2\rfloor+1 = 16$$ (halved).

**Parameter count.** A conv filter is $$K\times K\times C_{\text{in}}$$, and there are $$C_{\text{out}}$$ of them, so the layer has $$K^2 C_{\text{in}} C_{\text{out}}$$ weights. Crucially this does **not** depend on $$W$$ or $$H$$ — the same kernel is reused at every location. Compare a fully-connected layer between two $$32\times32\times64$$ tensors: $$\textcolor{purple}{(32^2\cdot64)^2\approx4.3\times10^9}$$ weights, versus a $$3\times3$$ conv with $$\textcolor{teal}{3^2\cdot64\cdot64\approx3.7\times10^4}$$ — five orders of magnitude fewer.

### 2.4 Stride, padding, dilation, and variants

> **Stride** controls how far the kernel hops (and thus downsampling); **padding** controls border handling and output size; **dilation** spreads the kernel to enlarge the receptive field without extra parameters.
{:.lead}

![Variants of the convolution: strided convolution downsamples, dilated convolution enlarges the receptive field at the same parameter cost.](/assets/figures/day04/cnn_variants.png)

- **Strided convolution** ($$S>1$$) computes outputs at every $$S$$-th position, downsampling the map — an alternative to pooling that *learns* its downsampling.
- **Dilated (atrous) convolution** inserts gaps of size $$d$$ between kernel taps. A $$3\times3$$ kernel with dilation $$d$$ covers a $$(2d+1)\times(2d+1)$$ region but still has only $$9$$ weights — ideal for segmentation where large context matters at full resolution.
- **Transposed convolution** is "learnable upsampling": it increases spatial size and is used in decoders and generators.
- **Depthwise-separable convolution** factorizes a standard conv into a per-channel spatial conv followed by a $$1\times1$$ channel mix, cutting cost dramatically — the backbone of mobile architectures (MobileNet, EfficientNet).

## 3. Building Blocks

### 3.1 Pooling and downsampling

> **Pooling** aggregates each small window of a feature map into a single value (max or mean), reducing spatial resolution while retaining the strongest responses.
{:.lead}

![Pooling computes a mean or max over small windows to reduce resolution.](/assets/figures/day04/cnn_pooling.png)

Why downsample at all? Three reasons:

- **Efficiency.** Fewer spatial positions means less compute and memory in deeper layers.
- **Larger receptive field.** Each later unit summarizes a bigger region of the original image.
- **Robustness.** Max-pooling gives a small amount of **local translation invariance** — a feature still registers if it moves by a pixel or two.

**Max** vs **average**: max-pooling keeps the strongest activation (good for "is the feature present?"), while average-pooling smooths. Many modern networks drop explicit pooling in favor of **strided convolutions** (learned downsampling), and end with a **global average pool** that collapses the final $$H\times W\times C$$ map to a $$C$$-vector fed to the classifier head.

### 3.2 The conv block and normalization

> The basic repeating unit of a CNN is the **conv block**: a convolution, a normalization layer, and a nonlinearity, stacked and periodically downsampled.
{:.lead}

![Batch normalization speeds and stabilizes training, reaching higher accuracy faster than the same network without it.](/assets/figures/day04/cnn_batchnorm.png)

A typical block is `conv → BatchNorm → ReLU`, repeated, with the channel count growing as spatial resolution shrinks. **Batch normalization** standardizes each channel's activations over the minibatch, $$\hat z = (z-\mu_{\mathcal B})/\sqrt{\sigma_{\mathcal B}^2+\epsilon}$$, then rescales with learned $$\gamma,\beta$$. The figure shows the practical effect: faster convergence, tolerance of larger learning rates, and higher final accuracy. (Variants — GroupNorm, LayerNorm — are used when batches are small.)

Regularization in CNNs leans less on dropout (weight sharing already constrains the model) and more on **data augmentation**: random crops, flips, color jitter, mixup/cutmix. These encode the invariances we want and are often the biggest single lever on generalization.

### 3.3 Derivation: how the receptive field grows

> The **receptive field** of a unit is the region of the input that can influence it. For a stack of $$L$$ stride-1 convolutions with kernel $$K$$, it grows **linearly**: $$\text{RF} = 1 + L\,(K-1).$$
{:.lead}

Each stride-1 convolution with kernel $$K$$ lets a unit see $$K-1$$ more input pixels than a unit in the layer below (the kernel reaches $$(K-1)/2$$ further on each side). Adding one such layer therefore increases the receptive field by $$K-1$$:

$$\text{RF}_{\ell} = \text{RF}_{\ell-1} + \textcolor{teal}{(K-1)},\qquad \text{RF}_0 = 1 \;\Rightarrow\; \text{RF}_L = 1 + \textcolor{purple}{L}\,\textcolor{teal}{(K-1)}.$$

*Example.* Ten $$3\times3$$ conv layers give $$\text{RF}=1+10\cdot2 = 21$$ — still small relative to a $$224\times224$$ image, which is why downsampling matters.

**Downsampling multiplies it.** A stride-$$s$$ layer (or pooling) scales the contribution of all *subsequent* layers by $$s$$, so receptive field then grows **geometrically** with depth. This is precisely how a deep CNN turns local edge detectors into units that respond to whole objects.

**Dilation adds reach cheaply.** A dilation-$$d$$ layer contributes $$d(K-1)$$ instead of $$K-1$$, so dilated stacks reach large context at full resolution — the trick behind dense-prediction networks.

*Two $$3\times3$$ = one $$5\times5$$.* Two stacked $$3\times3$$ convs have the same $$5\times5$$ receptive field as a single $$5\times5$$ conv, but use $$2\cdot9=18$$ weights instead of $$25$$ **and** insert an extra nonlinearity — the design principle behind VGG.

## 4. Classic Architectures

### 4.1 The depth revolution on ImageNet

> The ImageNet benchmark drove a rapid drop in top-5 error — from ~26% in 2011 to under 3% by 2017 — almost entirely by making convolutional networks **deeper** (and learning how to train them).
{:.lead}

![ImageNet top-5 error rate of the winning entries by year; each major drop came from a deeper, better-trained CNN, culminating in ResNet.](/assets/figures/day04/cnn_imagenet.png)

Milestones:

- **AlexNet (2012)** was the breakthrough: a deep CNN trained on GPUs with **ReLU** activations and **dropout**, halving the error of the best classical computer-vision pipeline.
- **VGG (2014)** showed that a very deep, uniform stack of $$3\times3$$ convolutions works extremely well — establishing the small-kernel design.
- **GoogLeNet/Inception (2014)** processed multiple scales in parallel and was far more parameter-efficient.
- **ResNet (2015)** broke the depth barrier with residual connections, reaching 100+ layers and sub-4% error.

The throughline is that **depth, trained correctly, is what wins** — with each architecture contributing a piece of "trained correctly" (ReLU, small kernels, normalization, skip connections).

### 4.2 From LeNet to AlexNet to VGG

> The classic CNN template is a sequence of **conv–pool stages** that progressively reduce spatial size and increase channels, followed by a small fully-connected **head** for classification.
{:.lead}

![AlexNet as a computational graph: alternating convolution and pooling stages feeding a fully-connected classifier.](/assets/figures/day04/cnn_alexnet.png)

**LeNet-5 (1998)** introduced the template for handwritten-digit recognition: two conv+pool stages then fully-connected layers. **AlexNet (2012)** is essentially LeNet scaled up — more layers, more channels, ReLU, dropout, and GPU training — and it won ImageNet by a wide margin.

**VGG (2014)** refined the design with a single principle: replace large kernels by **stacks of $$3\times3$$ convolutions**. As derived above, two $$3\times3$$ layers match a $$5\times5$$ receptive field with fewer parameters and an extra nonlinearity; three match a $$7\times7$$. The result is a deep, remarkably uniform network that remains a popular feature extractor.

### 4.3 ResNet and residual learning

> A **residual block** computes $$\boldsymbol{y} = \boldsymbol{x} + \mathcal{F}(\boldsymbol{x})$$, adding a **skip connection** that lets the layer learn a correction to the identity rather than a full transformation.
{:.lead}

A surprising empirical fact motivated ResNet: stacking *more* plain layers eventually made both training **and** test error **worse**. Since training error rose too, this was an **optimization** failure, not overfitting — very deep plain nets are simply hard to optimize (echoing the vanishing-gradient story from Day 3).

The fix is the residual block:

$$\boldsymbol{y} = \boldsymbol{x} + \mathcal{F}(\boldsymbol{x};\theta).$$

Two reasons it works:

1. **Gradient flow.** The identity path provides a shortcut: $$\frac{\partial \boldsymbol{y}}{\partial \boldsymbol{x}} = I + \frac{\partial \mathcal{F}}{\partial \boldsymbol{x}}$$, so gradients reach early layers undiminished even when $$\mathcal{F}$$'s Jacobian is small.
2. **Easier target.** If the optimal mapping is close to the identity, learning the *residual* $$\mathcal{F}\approx 0$$ is far easier than learning the identity from scratch through nonlinear layers.

Residual connections made networks with hundreds of layers trainable, and the idea — an additive identity path — reappears everywhere, including the Transformer blocks of Day 5.

## 5. Beyond Classification

### 5.1 Object detection

> **Object detection** predicts both *what* (a class) and *where* (a bounding box) for every object in an image — multiple structured outputs per image.
{:.lead}

![Object detection: localize and classify every object with bounding boxes.](/assets/figures/day04/cnn_detection.png)

Detection reuses a CNN **backbone** (often an ImageNet-pretrained ResNet) and adds task-specific heads. Two families:

- **Two-stage** (R-CNN → Fast/Faster R-CNN): first propose candidate regions, then classify and refine each box. Accurate but slower.
- **One-stage** (YOLO, SSD, RetinaNet): predict classes and boxes densely over the feature map in a single pass. Faster, increasingly accurate.

Common ingredients are **anchors** (reference boxes of various sizes/aspect ratios) and **feature pyramids** (combining feature maps at several resolutions) to handle objects at different scales. The backbone does the heavy representational work; the heads turn features into boxes and labels.

### 5.2 Semantic segmentation

> **Semantic segmentation** assigns a class label to **every pixel**, producing a dense map at (near) the input resolution.
{:.lead}

![Dense pixel-level prediction (semantic segmentation) on a street scene.](/assets/figures/day04/cnn_segmentation.png)

The challenge is that classification CNNs *downsample* aggressively, but segmentation needs full-resolution output. Two standard solutions:

- **Encoder–decoder (U-Net).** An encoder downsamples to capture context, then a decoder upsamples (via transposed convolution) back to full resolution. **Skip connections** carry high-resolution detail from encoder to decoder, recovering sharp boundaries.
- **Dilated convolutions.** Keep resolution high throughout while still enlarging the receptive field, avoiding the lossy down/up cycle.

The lesson generalizes: a single strong CNN backbone, combined with a task-appropriate output head (classification, boxes, or dense maps), covers a huge range of vision problems.

### 5.3 Transfer learning and modern CNNs

> **Transfer learning** reuses features learned on a large dataset (e.g. ImageNet) for a new task, fine-tuning rather than training from scratch — the default recipe when data is limited.
{:.lead}

Because early convolutional features (edges, textures) are **generic**, a network pretrained on a large dataset provides an excellent starting point. The workflow:

1. Take a backbone pretrained on ImageNet.
2. Replace the head with one for your task.
3. Fine-tune — either just the head (little data) or the whole network (more data, smaller learning rate).

This routinely beats training from scratch and needs far less data and compute.

**Modern backbones.** ResNet remains a robust default; **EfficientNet** balances depth/width/resolution by a principled scaling rule; **ConvNeXt** modernizes the ResNet recipe to match Vision Transformers. Speaking of which — **Vision Transformers (ViT)** apply the attention machinery of Day 5 to image patches and, given enough data, rival or beat CNNs. The current consensus is that with strong training recipes both families are excellent; convolutions remain especially data-efficient thanks to their built-in priors. That sets up Day 5, where attention takes center stage for sequences.

## Checkpoint summary

Before moving to the practical, confirm you can:

- Explain locality, weight sharing, and hierarchy, and how each becomes part of the convolution.
- Compute a convolution by hand and give the output size for given kernel, stride, and padding.
- Count the parameters of a conv layer and contrast with a fully-connected layer on the same tensors.
- Derive how the receptive field grows with depth, stride, and dilation, and explain 'two 3x3 = one 5x5'.
- Describe the conv block and why batch normalization helps; identify the role of pooling.
- Summarize AlexNet, VGG, and ResNet, and explain why residual connections enable very deep networks.
- Outline how a CNN backbone extends to detection and segmentation, and when to use transfer learning.
