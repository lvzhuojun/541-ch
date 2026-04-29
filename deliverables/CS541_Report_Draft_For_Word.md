# CS541 Challenge Spring 2026 Report

**Student:** Zhuojun Lyu  
**BU ID:** U06761622  
**Email:** lzj2729@bu.edu  
**Kaggle Name:** Zhuojun Lyu

## Overview

In this challenge, I built and evaluated three image classification models on CIFAR-100 and compared their behavior on an out-of-distribution evaluation set with 19 distortion types. The three parts progress from a CNN trained from scratch, to a fine-tuned pretrained CNN, to a fine-tuned pretrained Vision Transformer. The final direction of the project was to prioritize not only clean accuracy but also robustness under distribution shift.

## Experiment Summary

| Part | Model | Best Val Acc | Clean Test Acc | Notes |
|---|---|---:|---:|---|
| Part 1 | Custom CNN | 0.6134 | 0.6065 | CNN from scratch with more than 2 blocks |
| Part 2 | ResNet-18 fine-tune | 0.8127 | 0.8127 | Pretrained CNN on ImageNet |
| Part 3 | ViT-B/16 fine-tune | 0.8902 | 0.8894 | Final selected leaderboard CSV restored from `part3_v089` export of the best ViT checkpoint |

## Part 1: CNN From Scratch

### Model Design

For Part 1, I built a custom 4-block CNN, which satisfies the assignment requirement that the model contain more than 2 blocks. Each block uses convolution, normalization, nonlinear activation, pooling, and dropout. The design increases channel width gradually so the model can learn richer feature hierarchies while keeping the network practical to train on CIFAR-100.

### Training Setup

- Optimizer: AdamW
- Learning rate: 3e-4
- Weight decay: 1e-4
- Batch size: 128
- Epochs: 20
- Image size: 32x32
- Augmentation: random crop, random horizontal flip, AutoAugment, random erasing
- Regularization: dropout and label smoothing

### Results

- Best validation accuracy: 0.6134
- Clean CIFAR-100 test accuracy: 0.6065

### Discussion

This model established a solid baseline without pretrained weights. It showed that a deeper hand-built CNN can learn CIFAR-100 reasonably well, but it still lagged behind transfer learning methods in both final accuracy and robustness.

## Part 2: Fine-Tuned Pretrained CNN

### Model Choice

For Part 2, I used ResNet-18 pretrained on ImageNet. This was a strong and efficient baseline because pretrained convolutional features transfer well to standard image classification problems.

### Fine-Tuning Strategy

I replaced the final classification layer to output 100 classes and fine-tuned the network on CIFAR-100. I used 224x224 input resolution, AdamW, a learning rate of 1e-4, weight decay of 1e-4, batch size 64, and 8 epochs of training.

### Results

- Best validation accuracy: 0.8127
- Clean CIFAR-100 test accuracy: 0.8127

### Discussion

The jump from Part 1 to Part 2 was substantial. This confirms the value of transfer learning: pretrained visual features dramatically improve performance with limited task-specific training. However, the pretrained CNN was still more sensitive to distorted evaluation images than the transformer-based model used in Part 3.

## Part 3: Fine-Tuned Pretrained Transformer

### Model Choice

For Part 3, I used ViT-B/16 pretrained on ImageNet. The motivation was that a transformer-based vision model may generalize better under the challenge's distorted test conditions because it uses global self-attention rather than only local convolutions.

### Fine-Tuning Strategy

I fine-tuned the pretrained ViT with a lower learning rate than the CNN setup because transformer models are more sensitive during adaptation. The practical setup used 224x224 resolution, AdamW, learning rate 5e-5, weight decay 1e-4, batch size 32, cosine annealing, label smoothing, and the same augmentation pipeline used elsewhere in the project.

### Results

- Best validation accuracy: 0.8902
- Clean CIFAR-100 test accuracy: 0.8894

### Discussion

Part 3 produced the strongest clean accuracy among the three parts. The transformer outperformed both the scratch CNN and the fine-tuned ResNet on the clean CIFAR-100 evaluation pipeline. For the final Kaggle choice, I kept the exported `part3_v089` CSV because it was the better practical submission among the late-stage Part 3 variants I compared.

## Comparison Across Parts

The comparison across parts was clear:

- Part 1 was useful as a baseline and demonstrated end-to-end design from scratch.
- Part 2 showed a major improvement from transfer learning.
- Part 3 produced the highest clean accuracy and the best overall robustness.

From an engineering perspective, Part 1 was the simplest to interpret, Part 2 was the fastest route to strong performance, and Part 3 required the most compute and tuning effort but produced the best results.

## Error Analysis

The main challenge in this assignment is distribution shift. Models that perform well on the clean validation split do not automatically remain robust on distorted images. Based on the model comparisons, convolutional models appear more vulnerable to distortions such as blur, corruption, or other local image degradations. The transformer-based model handled this shift better, likely because its representation integrates information more globally across the image.

The supporting files for Part 3 error analysis are included under `report_artifacts/part3_best/`, including:

- top 3 worst-performing classes
- top 3 largest prediction errors
- per-class accuracy CSV
- example error images

## AI Disclosure

AI tools were used in a limited support role during this assignment. The assistance was mainly used for:

- discussing implementation ideas
- debugging Python training and submission scripts
- improving code organization and comments
- polishing written documentation

The student was responsible for:

- choosing the final modeling approach for each part
- running experiments and checking outputs
- selecting the final submission files
- reviewing and keeping only the final repository contents

The codebase was manually reviewed and adjusted before final submission.

## Conclusion

This project showed a clear progression from a custom CNN baseline to stronger pretrained models. The custom CNN achieved 0.6065 clean test accuracy, ResNet-18 improved that to 0.8127, and the final ViT-B/16 checkpoint reached 0.8894 clean test accuracy. The final selected Kaggle submission used the restored `part3_v089` export. The main lesson is that pretrained large-scale vision models, especially transformer-based ones, provide a strong advantage, but clean validation accuracy and distorted-distribution leaderboard behavior are not always perfectly aligned.
