#import "../lib.typ": *

#show: course-theme.with(title: [Convolutional Neural Networks], subtitle: [Day 4 | Aug 2026])

= Day 4: Convolutional Neural Networks

== Welcome

- *Convolutional Neural Networks* — Priors for images: locality, sharing, and hierarchy
- 3 hours lecture + practical
- Slides, notes, and code on the course site

== Outline

- Why Convolutions
- The Convolution Operation
- Building Blocks
- Classic Architectures
- Beyond Classification

= 1 · Why Convolutions

== 1.1  The Curse of Fully-Connected Vision

- A $224 times 224 times 3$ image = 150,528 inputs
- One fully-connected layer of 1000 units $arrow.r$ 150 million weights
- Ignores image structure: shuffle the pixels and it can't tell
- No reuse: a cat in the corner must be relearned in the center
- We need an architecture that *bakes in* what we know about images

== 1.2  Locality & Translation Invariance

- Locality: nearby pixels are strongly correlated; far ones less so
- Translation invariance: a pattern means the same anywhere
- $arrow.r$ connect each unit only to a local patch (receptive field)
- $arrow.r$ reuse the *same* weights at every location
- These two priors *are* the convolution

== 1.2  Locality & Translation Invariance

#align(center + horizon)[#image("/assets/figures/day04/cnn_locality.png", width: 92%, height: 82%, fit: "contain")]

== 1.3  Parameter Sharing & Hierarchy

- Weight sharing: one filter scans the whole image
- Huge parameter savings + built-in equivariance to shifts
- Hierarchy: edges $arrow.r$ textures $arrow.r$ parts $arrow.r$ objects
- Stacking conv layers grows the receptive field
- Compose simple local detectors into global concepts

== 1.3  Parameter Sharing & Hierarchy

#align(center + horizon)[#image("/assets/figures/day04/cnn_topology.png", width: 92%, height: 82%, fit: "contain")]

= 2 · The Convolution Operation

== 2.1  From Local Connectivity to Convolution

- Locally-connected: each output sees a $3 times 3$ patch
- Convolutional: *tie* those local weights across positions
- Output $y = w * x + b$ (cross-correlation, by convention)
- A $3 times 3$ receptive field, one shared kernel
- Far fewer parameters than fully-connected

== 2.1  From Local Connectivity to Convolution

#align(center + horizon)[#image("/assets/figures/day04/cnn_receptive.png", width: 92%, height: 82%, fit: "contain")]

== 2.2  The Convolution, Step by Step

- Slide a small kernel over the image
- At each position: elementwise multiply + sum $arrow.r$ one output
- $ (w * x)_(i,j) = sum_(u,v) w_(u,v) x_(i+u, j+v) $
- The result is a *feature map* — where the pattern fires
- Same kernel everywhere = translation equivariance

== 2.2  The Convolution, Step by Step

#align(center + horizon)[#image("/assets/figures/day04/cnn_convop.png", width: 92%, height: 82%, fit: "contain")]

== 2.3  Channels: Inputs & Outputs are Tensors

- Image = tensor (height $times$ width $times$ channels)
- A filter spans *all* input channels: $K times K times C_"in"$
- Use $C_"out"$ filters $arrow.r$ output has $C_"out"$ channels
- Weights: $K times K times C_"in" times C_"out"$ (+ biases)
- $1 times 1$ conv = per-pixel channel mixing

== 2.3  Channels: Inputs & Outputs are Tensors

#align(center + horizon)[#image("/assets/figures/day04/cnn_tensors.png", width: 92%, height: 82%, fit: "contain")]

== 2.4  Derivation: Output Size & Parameter Count

- Input $W$, kernel $K$, padding $P$, stride $S$
- Output: $O = floor((W - K + 2P)\\/S) + 1$
- 'same' padding ($S=1$): $P = (K-1)\\/2$ keeps size
- Conv params: $K^2 C_"in" C_"out"$ — independent of image size
- FC params would be (num pixels)$times$(num units) — orders larger

== 2.5  Stride, Padding & Dilation

- Stride $S$: step size $arrow.r$ downsamples by $S$
- Padding $P$: add a border to control output size / edges
- Dilation: spread the kernel $arrow.r$ larger field, same params
- Transposed conv: learnable upsampling (decoders, GANs)
- Depthwise-separable: factorize for cheap mobile nets

== 2.5  Stride, Padding & Dilation

#align(center + horizon)[#image("/assets/figures/day04/cnn_variants.png", width: 92%, height: 82%, fit: "contain")]

= 3 · Building Blocks

== 3.1  Pooling & Downsampling

- Pool = mean or max over small windows
- Reduces resolution $arrow.r$ cheaper, larger receptive field
- Max-pool: small translation invariance
- Modern nets often downsample with strided conv instead
- Global average pool collapses a map to a vector for the head

== 3.1  Pooling & Downsampling

#align(center + horizon)[#image("/assets/figures/day04/cnn_pooling.png", width: 92%, height: 82%, fit: "contain")]

== 3.2  The Conv Block & Normalization

- Block = conv $arrow.r$ norm $arrow.r$ nonlinearity (repeat)
- BatchNorm: standardize activations $arrow.r$ faster, higher LR
- Big accuracy/speed gains vs no normalization
- Dropout less common in convs; augmentation dominates
- Stack blocks; periodically downsample, grow channels

== 3.2  The Conv Block & Normalization

#align(center + horizon)[#image("/assets/figures/day04/cnn_batchnorm.png", width: 92%, height: 82%, fit: "contain")]

== 3.3  Derivation: Receptive Field Growth

- Receptive field = input region one unit depends on
- Stack of $L$ conv layers (kernel $K$, stride 1):
- $ "RF" = 1 + L(K-1) $ — grows *linearly* with depth
- Stride/pooling multiply it $arrow.r$ grows *geometrically*
- Dilation $d$: each layer adds $d(K-1)$

= 4 · Classic Architectures

== 4.1  The Depth Revolution (ImageNet)

- ImageNet top-5 error: 26% (2011) $arrow.r$ under 3% (2017)
- AlexNet (2012): deep convs + ReLU + GPUs + dropout
- VGG: stacks of $3 times 3$ convs, very deep & uniform
- GoogLeNet/Inception: multi-scale, parameter-efficient
- Each jump came from *more depth*, done right

== 4.1  The Depth Revolution (ImageNet)

#align(center + horizon)[#image("/assets/figures/day04/cnn_imagenet.png", width: 92%, height: 82%, fit: "contain")]

== 4.2  LeNet -> AlexNet -> VGG

- LeNet-5 (1998): convs + pooling for digits
- AlexNet (2012): scaled it up, won ImageNet
- Pattern: conv-pool stages then fully-connected head
- VGG: replace big kernels with stacks of $3 times 3$
- Two $3 times 3$ = one $5 times 5$ field, fewer params + more nonlinearity

== 4.2  LeNet -> AlexNet -> VGG

#align(center + horizon)[#image("/assets/figures/day04/cnn_alexnet.png", width: 92%, height: 82%, fit: "contain")]

== 4.3  ResNet & Residual Learning

- Very deep plain nets get *worse* — optimization, not overfitting
- Residual block: $y = x + cal(F)(x)$ (skip connection)
- Identity path lets gradients flow undiminished
- Learn a *residual* correction, easier than the full map
- Enabled 100+ layer nets; still a default backbone

= 5 · Beyond Classification

== 5.1  Object Detection

- Predict *what* and *where*: class + bounding box
- Two-stage: R-CNN family (propose then classify)
- One-stage: YOLO/SSD (dense predictions, fast)
- Anchors / feature pyramids handle scale
- Shared CNN backbone + task-specific heads

== 5.1  Object Detection

#align(center + horizon)[#image("/assets/figures/day04/cnn_detection.png", width: 92%, height: 82%, fit: "contain")]

== 5.2  Semantic Segmentation

- Label *every pixel* $arrow.r$ dense prediction
- Encoder-decoder: downsample then upsample (U-Net)
- Skip connections recover spatial detail
- Transposed conv / dilated conv keep resolution
- Same backbone, different output head

== 5.2  Semantic Segmentation

#align(center + horizon)[#image("/assets/figures/day04/cnn_segmentation.png", width: 92%, height: 82%, fit: "contain")]

== 5.3  Transfer Learning & Modern CNNs

- Pretrain on ImageNet, fine-tune on your task
- Early features are generic & reusable
- ResNet/EfficientNet/ConvNeXt: strong backbones
- Depthwise-separable convs for mobile/edge
- CNNs vs Vision Transformers: convs still strong with good training

== Summary

- Day 4: *Convolutional Neural Networks*
- Priors for images: locality, sharing, and hierarchy
- Questions welcome — practical follows

== Questions?

#align(center + horizon)[
  #text(size: 44pt, weight: "bold", fill: course-primary)[Questions?]

  #v(1em)
  Practical session follows — see course site.
]
