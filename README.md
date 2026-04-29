# CS541 Challenge Spring 2026

**Student:** Zhuojun Lyu  
**BU ID:** U06761622  
**Email:** lzj2729@bu.edu  
**Kaggle Name:** Zhuojun Lyu

## Overview

This repository contains my final submission package for the CS541 image classification challenge. The assignment has three required parts:

1. build a CNN from scratch on CIFAR-100
2. fine-tune a pretrained CNN
3. fine-tune a pretrained transformer

The main goal of the challenge is not only to achieve strong clean accuracy on CIFAR-100, but also to build models that generalize better to the distorted out-of-distribution evaluation set used on the Kaggle leaderboard.

## Repository Scope

This repository was intentionally cleaned to keep only the files needed for submission and review. The tracked files are:

- `README.md`
- `run_cs541_assignment.py`
- `cs541_challenge_utils.py`
- `requirements.txt`
- `deliverables/cs541_challenge_three_parts.ipynb`
- `deliverables/CS541_Report_Draft_For_Word.md`
- `deliverables/kaggle_csv/`
- `deliverables/report_assets/`

Local experiment folders, cached datasets, checkpoints, and temporary packaging directories are not part of the submission repo.

## Final Submission Files

Upload these three CSV files to Kaggle:

- `deliverables/kaggle_csv/Zhuojun Lyu_Part_1.csv`
- `deliverables/kaggle_csv/Zhuojun Lyu_Part_2.csv`
- `deliverables/kaggle_csv/Zhuojun Lyu_Part_3.csv`

Final selected versions:

- Part 1: final custom CNN submission
- Part 2: final pretrained CNN submission
- Part 3: restored `part3_v089` submission

The Part 3 file was kept from `part3_v089` because it matched the stronger practical leaderboard result among my late-stage exported variants.

## Project Idea And Strategy

My overall strategy was to move from a simple baseline to stronger transfer learning models while keeping the pipeline comparable across all three parts.

The logic was:

- first build a scratch CNN to establish a baseline and satisfy the architecture requirement
- then move to a pretrained CNN to test how much transfer learning improves accuracy
- finally use a pretrained Vision Transformer to test whether a stronger global representation helps robustness under image distortions

I tried to keep the training pipeline reasonably consistent across the three parts, while still adjusting image resolution, learning rate, and batch size to fit each model family.

## Experimental Summary

| Part | Model | Best Val Acc | Clean Test Acc | Main Takeaway |
|---|---|---:|---:|---|
| Part 1 | Custom CNN | 0.6134 | 0.6065 | A deeper scratch model can work, but it is clearly weaker than transfer learning |
| Part 2 | ResNet-18 fine-tune | 0.8127 | 0.8127 | Pretrained convolutional features provide a large jump in performance |
| Part 3 | ViT-B/16 fine-tune | 0.8902 | 0.8894 | The strongest clean model and the best final practical Part 3 submission source |

## Part 1: CNN From Scratch

For Part 1, I built a 4-block convolutional neural network. Each block contains convolution, normalization, activation, pooling, and dropout. I intentionally used more than 2 blocks to satisfy the assignment requirement while also making the model deep enough to learn richer image features.

Training choices:

- image size: `32 x 32`
- batch size: `128`
- epochs: `20`
- optimizer: `AdamW`
- learning rate: `3e-4`
- weight decay: `1e-4`
- regularization: dropout, label smoothing, random crop, random horizontal flip, AutoAugment, random erasing

This model gave a reasonable baseline, but its main limitation was capacity. It could learn CIFAR-100, but not nearly as well as the pretrained models.

## Part 2: Fine-Tuned Pretrained CNN

For Part 2, I fine-tuned an ImageNet-pretrained ResNet-18. I replaced the final classification layer to predict 100 CIFAR-100 classes, resized the inputs to `224 x 224`, and trained with a smaller learning rate than the scratch CNN.

Training choices:

- image size: `224 x 224`
- batch size: `64`
- epochs: `8`
- optimizer: `AdamW`
- learning rate: `1e-4`
- weight decay: `1e-4`

This model showed the clearest improvement over Part 1. The main lesson from Part 2 was that transfer learning provides a very strong baseline quickly, especially when training data is limited relative to model capacity.

## Part 3: Fine-Tuned Pretrained Transformer

For Part 3, I fine-tuned a pretrained ViT-B/16 model. My reasoning was that a transformer-based vision model might better handle distortions and distribution shift because it builds a more global representation of the image than a standard CNN.

Training choices:

- image size: `224 x 224`
- batch size: `32`
- epochs: `6`
- optimizer: `AdamW`
- learning rate: `5e-5`
- weight decay: `1e-4`
- scheduler: cosine annealing
- regularization: label smoothing and the same general augmentation pipeline

Part 3 gave the strongest clean performance among all three parts. In practice, I found that strong clean validation accuracy and leaderboard behavior were related, but not perfectly aligned. For the final submission package, I selected the restored `part3_v089` CSV because it was the better practical result among the exported Part 3 files I compared late in the project.

## What Worked

- Moving from scratch training to pretrained models gave the largest gains.
- ResNet-18 was a strong and stable middle ground between simplicity and accuracy.
- ViT-B/16 produced the strongest clean metrics and the best final Part 3 source.
- AdamW with moderate weight decay was stable across all three parts.

## What Did Not Work As Well

- Late-stage Part 3 variants with altered submission-time behavior did not always improve leaderboard performance.
- Strong clean validation accuracy did not guarantee stronger distorted-test performance.
- Part 3 training and submission export were much slower than the CNN-based models.

## Error Analysis

The supporting error-analysis files used for the report are stored in:

- `deliverables/report_assets/part3_top3_worst_classes.csv`
- `deliverables/report_assets/part3_top3_error_samples.csv`
- `deliverables/report_assets/part3_largest_error_1.png`
- `deliverables/report_assets/part3_largest_error_2.png`
- `deliverables/report_assets/part3_largest_error_3.png`

The broader lesson from the challenge is that robustness under distortion is meaningfully harder than standard clean-image classification. A model can look very strong on clean CIFAR-100 and still lose performance on the leaderboard due to distribution shift.

## AI Disclosure

AI assistance was used in a limited support role during this project. The main uses were:

- discussing implementation ideas and experiment directions
- debugging Python training and export scripts
- cleaning repository structure and documentation wording

The student remained responsible for:

- choosing the modeling strategy
- running experiments
- checking outputs and comparing variants
- selecting the final submitted files
- reviewing the final code and write-up

## Conclusion

This challenge showed a clear progression in model strength. The custom CNN established a valid baseline, the pretrained ResNet-18 gave a large accuracy improvement, and the pretrained ViT-B/16 gave the strongest clean results. The final submission package reflects that progression and keeps the final Part 3 choice aligned with the stronger practical leaderboard variant I observed during the last round of selection.
