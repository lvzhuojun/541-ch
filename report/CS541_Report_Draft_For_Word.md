# CS541 Challenge Spring 2026 Report Draft

**Student:** Zhuojun Lyu  
**BU ID:** U06761622  
**Email:** lzj2729@bu.edu  
**Course:** CS541 Spring 2026

## 1. Overview

In this challenge assignment, I built and evaluated three image classification models on CIFAR-100 and compared their behavior under both standard and distorted test conditions. The three required parts were:

1. a CNN built from scratch
2. a fine-tuned pretrained CNN
3. a fine-tuned pretrained Vision Transformer

The central goal of the project was not only to improve clean CIFAR-100 accuracy, but also to study which design choices lead to better robustness when the test images contain real distribution shift and image corruption.

## 2. Experimental Setup

### Dataset and Evaluation Setting

The in-distribution training and evaluation data came from CIFAR-100. The out-of-distribution evaluation set was the Kaggle challenge test set built from the Hugging Face dataset `XThomasBU/ood-test-CS541`, which contains 19 image distortion types and 5 severity levels per distortion.

### Shared Training Choices

Across the project, I used a consistent training pipeline whenever possible:

- optimizer: AdamW
- weight decay: `1e-4`
- data augmentation: random crop, random horizontal flip, AutoAugment, random erasing
- normalization: CIFAR-100 mean and standard deviation
- regularization: dropout and label smoothing
- reproducibility: fixed random seed and saved checkpoints

These choices were used to keep the comparisons across parts fair while still allowing each model family to use a reasonable configuration.

## 3. Part 1: CNN From Scratch

### Model Design

For Part 1, I implemented a custom 4-block CNN. This satisfies the assignment requirement that the model contain more than 2 blocks. Each block includes convolution, normalization, activation, pooling, and dropout. The network gradually increases channel width so that deeper layers can learn stronger semantic features.

### Hyperparameters

- image size: `32 x 32`
- batch size: `128`
- epochs: `20`
- learning rate: `3e-4`
- weight decay: `1e-4`

### Regularization Techniques

- dropout inside the convolution blocks and classifier
- AutoAugment
- random erasing
- label smoothing

### Results

- best validation accuracy: `0.6134`
- clean CIFAR-100 test accuracy: `0.6065`

### Discussion

This model established a reasonable baseline using only task-specific training. It performed clearly above the initial submission requirement of `0.5` accuracy, but it was still limited compared with transfer learning approaches.

## 4. Part 2: Fine-Tuned Pretrained CNN

### Model Choice

For Part 2, I fine-tuned an ImageNet-pretrained ResNet-18. I replaced the final fully connected layer so that the model outputs 100 CIFAR-100 classes.

### Hyperparameters

- image size: `224 x 224`
- batch size: `64`
- epochs: `8`
- learning rate: `1e-4`
- weight decay: `1e-4`

### Regularization Techniques

- pretrained initialization
- AdamW with weight decay
- random crop and horizontal flip
- AutoAugment
- random erasing
- label smoothing

### Results

For the controlled Kaggle submission version used in this repo:

- best validation accuracy: `0.8138`
- clean CIFAR-100 test accuracy: `0.8071`

### Discussion

The performance jump from Part 1 to Part 2 shows the value of transfer learning. The pretrained CNN learned useful visual features quickly and produced much stronger results than the scratch CNN. However, the CNN still appeared more sensitive to some distortions than the transformer-based approach used in Part 3.

## 5. Part 3: Fine-Tuned Pretrained Transformer

### Model Choice

For Part 3, I used an ImageNet-pretrained ViT-B/16 model. I replaced the classifier head so that the network predicts 100 classes and then fine-tuned the model on CIFAR-100.

### Hyperparameters

- image size: `224 x 224`
- batch size: `32`
- epochs: `6`
- learning rate: `5e-5`
- weight decay: `1e-4`

### Regularization Techniques

- pretrained initialization
- AdamW with weight decay
- cosine annealing learning-rate schedule
- random crop and horizontal flip
- AutoAugment
- random erasing
- label smoothing

### Best Local Training / Validation / Test Result

For the best local Part 3 model used for the report analysis:

- best validation accuracy: `0.8943`
- clean CIFAR-100 test accuracy: `0.8889`

The corresponding history file is:

- `report/analysis_tables/part3_history.csv`

### Loss and Accuracy Plots

The required plots for the best model are included here:

- `report/figures/part3_loss_curve.png`
- `report/figures/part3_accuracy_curve.png`

### Top 3 Worst-Performing Classes

Based on per-class accuracy on the clean CIFAR-100 test split, the three worst classes for the best Part 3 model were:

1. `boy` with accuracy `0.6200`
2. `girl` with accuracy `0.6600`
3. `maple_tree` with accuracy `0.6600`

Supporting table:

- `report/analysis_tables/part3_top3_worst_classes.csv`

### Top 3 Largest Prediction Errors

Using the largest incorrect logit gap between the predicted class and true class, the top three error cases were:

1. sample `2827`: true `crocodile`, predicted `leopard`
2. sample `7762`: true `cup`, predicted `bottle`
3. sample `6522`: true `can`, predicted `plate`

Supporting files:

- `report/analysis_tables/part3_top3_error_samples.csv`
- `report/figures/part3_largest_error_1.png`
- `report/figures/part3_largest_error_2.png`
- `report/figures/part3_largest_error_3.png`

### Discussion

The transformer produced the strongest overall model in my experiments. Compared with the scratch CNN and the fine-tuned ResNet-18, the ViT model learned a better representation and generalized more effectively. This result is consistent with the challenge setting because the distorted test set rewards models that remain stable under severe image corruption.

## 6. What I Tried, What Worked, and What Did Not Work

### Part 1

What I tried:

- a deeper custom CNN rather than a minimal two-block network
- dropout and augmentation to improve generalization

What worked:

- moving to a 4-block architecture improved capacity enough to pass the initial threshold
- augmentation and label smoothing helped training stability

What did not work as well:

- from-scratch training still lagged far behind transfer learning on CIFAR-100

### Part 2

What I tried:

- fine-tuning an ImageNet-pretrained ResNet-18 at higher image resolution

What worked:

- transfer learning produced a major jump in validation and clean test accuracy
- the pretrained CNN was efficient and straightforward to train

What did not work as well:

- robustness gains were still limited compared with the transformer-based model

### Part 3

What I tried:

- fine-tuning a pretrained ViT-B/16
- multiple iterative versions to observe how more training changed the results

What worked:

- the ViT model achieved the best clean and validation performance
- cosine annealing plus low learning rate gave stable fine-tuning

What did not work as well:

- training and especially CSV export were much slower than the CNN-based models
- the model still struggled on a few visually similar or ambiguous classes

## 7. Key Lessons Learned

- transfer learning matters much more than a scratch baseline on this task
- architecture choice had a major effect on robustness
- the Vision Transformer was the strongest model family in my experiments
- careful regularization and augmentation improved stability but did not fully replace the benefit of a stronger pretrained backbone

## 8. AI Disclosure

I used AI assistance in a limited support role during this assignment. AI tools were mainly used for:

- repository cleanup and file organization
- polishing documentation wording and submission instructions
- small utility scripting support for generating report artifacts such as summary tables and figures

The core project work that I completed myself included:

- deciding the modeling strategy for all three parts
- choosing and running the training experiments
- selecting hyperparameters and iterative version targets
- managing the conda environment and GPU runs
- deciding which submission files to use
- interpreting the results and comparing the three parts

I reviewed and edited all final code and writing before submission.

## 9. Conclusion

This project showed a clear progression from a scratch CNN baseline to stronger transfer learning models. The custom CNN achieved a solid baseline, the pretrained ResNet-18 produced a large jump in performance, and the pretrained ViT-B/16 gave the strongest overall results. The main conclusion is that pretrained transformer-based vision models provide the best balance of accuracy and robustness for this challenge setting.
