#import "../lib.typ": *

#show: course-theme.with(title: [Convolutional Networks], subtitle: [Day 4 | Aug 2026])

= Day 4: Convolutional Networks

== Welcome

- *Convolutional Networks* — Computer vision and spatial inductive bias
- 3 hours lecture + practical
- Slides, notes, and code on the course site

== Outline

- Convolutions
- Classic Architectures
- Training for Vision
- Practice & Pitfalls

= Convolutions

== Motivation

- Images: local structure + translation symmetry
- Fully connected layers ignore spatial layout
- Parameter sharing across locations

== Motivation — illustration

#align(center)[#image("/assets/figures/day04/ucl0_p010.png", width: 80%)]

#text(size: 14pt, fill: gray)[Convolutions — Motivation (source: course materials)]

== 2D Convolution

- $(f * k)(i,j) = sum_(u,v) f(i-u,j-v) k(u,v)$
- Output size: $(H - k + 2p)/s + 1$
- Multiple filters → channel depth

== 2D Convolution — illustration

#align(center)[#image("/assets/figures/day04/ucl0_p017.png", width: 80%)]

#text(size: 14pt, fill: gray)[Convolutions — 2D Convolution (source: course materials)]

== Pooling & Stride

- Max / average pooling downsamples
- Stride reduces spatial resolution
- Receptive field grows with depth

== Pooling & Stride — illustration

#align(center)[#image("/assets/figures/day04/ucl0_p019.png", width: 80%)]

#text(size: 14pt, fill: gray)[Convolutions — Pooling & Stride (source: course materials)]

== CNN Building Blocks

- Conv → BN → ReLU → Pool stacks
- 1×1 conv for channel mixing
- Depthwise separable conv (MobileNet)

== CNN Building Blocks — illustration

#align(center)[#image("/assets/figures/day04/ucl0_p022.png", width: 80%)]

#text(size: 14pt, fill: gray)[Convolutions — CNN Building Blocks (source: course materials)]

= Classic Architectures

== LeNet → AlexNet

- Historical milestones on MNIST / ImageNet
- ReLU + GPU training breakthrough
- Data augmentation becomes standard

== LeNet → AlexNet — illustration

#align(center)[#image("/assets/figures/day04/ucl0_p025.png", width: 80%)]

#text(size: 14pt, fill: gray)[Classic Architectures — LeNet → AlexNet (source: course materials)]

== VGG & Inception

- Small 3×3 filters, deeper stacks
- Multi-scale Inception modules
- Computational cost vs accuracy

== VGG & Inception — illustration

#align(center)[#image("/assets/figures/day04/ucl0_p026.png", width: 80%)]

#text(size: 14pt, fill: gray)[Classic Architectures — VGG & Inception (source: course materials)]

== ResNet

- Residual: $y = F(x) + x$
- Eases optimization of very deep nets
- Skip connections and identity paths

== ResNet — illustration

#align(center)[#image("/assets/figures/day04/ucl0_p027.png", width: 80%)]

#text(size: 14pt, fill: gray)[Classic Architectures — ResNet (source: course materials)]

== Modern Backbones

- EfficientNet, ConvNeXt
- ViT hybrid models (Day 5 link)
- Transfer learning from ImageNet

== Modern Backbones — illustration

#align(center)[#image("/assets/figures/day04/ucl1_p008.png", width: 80%)]

#text(size: 14pt, fill: gray)[Classic Architectures — Modern Backbones (source: course materials)]

= Training for Vision

== Data Augmentation

- Random crop, flip, color jitter
- Mixup / CutMix regularization
- Test-time augmentation

== Data Augmentation — illustration

#align(center)[#image("/assets/figures/day04/ucl1_p075.png", width: 80%)]

#text(size: 14pt, fill: gray)[Training for Vision — Data Augmentation (source: course materials)]

== Losses & Heads

- Softmax cross-entropy for classification
- Multi-task: detection, segmentation heads
- Focal loss for hard examples

== Losses & Heads — illustration

#align(center)[#image("/assets/figures/day04/ucl1_p099.png", width: 80%)]

#text(size: 14pt, fill: gray)[Training for Vision — Losses & Heads (source: course materials)]

== Object Detection Preview

- Bounding boxes, IoU metric
- Two-stage vs one-stage detectors
- Feature pyramids (FPN)

== Object Detection Preview — illustration

#align(center)[#image("/assets/figures/day04/ucl1_p100.png", width: 80%)]

#text(size: 14pt, fill: gray)[Training for Vision — Object Detection Preview (source: course materials)]

== Segmentation Preview

- Per-pixel class labels
- U-Net encoder–decoder
- Dice / IoU losses

== Segmentation Preview — illustration

#align(center)[#image("/assets/figures/day04/ucl1_p103.png", width: 80%)]

#text(size: 14pt, fill: gray)[Training for Vision — Segmentation Preview (source: course materials)]

= Practice & Pitfalls

== Input Pipeline

- Normalize with dataset mean/std
- Efficient dataloaders, prefetch
- Resolution vs batch size memory tradeoff

== Overfitting in Vision

- Heavy aug + WD + early stopping
- Small datasets: freeze backbone
- Monitor val accuracy gap

== Interpretability

- Grad-CAM saliency maps
- Adversarial examples
- Dataset bias and spurious cues

== Summary

- Day 4: *Convolutional Networks*
- Computer vision and spatial inductive bias
- Questions welcome — practical follows

== Questions?

#align(center + horizon)[
  #text(size: 44pt, weight: "bold", fill: course-primary)[Questions?]

  #v(1em)
  Practical session follows — see course site.
]
