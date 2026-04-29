# CS541 Challenge Spring 2026

**Student:** Zhuojun Lyu  
**BU ID:** U06761622  
**Email:** lzj2729@bu.edu  
**Kaggle Name:** Zhuojun Lyu

This repository contains the final submission materials for the CS541 image classification challenge. The project includes the three required parts:

1. Part 1: a CNN built from scratch on CIFAR-100
2. Part 2: a fine-tuned pretrained CNN
3. Part 3: a fine-tuned pretrained transformer

The goal of the project was to improve both clean CIFAR-100 performance and robustness on the distorted out-of-distribution evaluation set used in the Kaggle challenge.

## Repository Layout

This repo is intentionally small and only keeps the files needed for the course submission.

- `deliverables/`
  Contains the notebook, report, Kaggle CSV files, and report figures/tables.
- `run_cs541_assignment.py`
  Main script for training runs and CSV export.
- `cs541_challenge_utils.py`
  Shared data loading, model definitions, training utilities, and submission generation.
- `requirements.txt`
  Minimal Python package list used in the project.

## What To Submit

### Kaggle

Upload the three CSV files in `deliverables/kaggle_csv/`, one file at a time:

- `Zhuojun Lyu_Part_1.csv`
- `Zhuojun Lyu_Part_2.csv`
- `Zhuojun Lyu_Part_3.csv`

The selected submission versions in this repo are:

- Part 1: root Part 1 submission
- Part 2: `part2_v070` submission
- Part 3: `part3_v089` submission

### Colab / Notebook

Use:

- `deliverables/cs541_challenge_three_parts.ipynb`

### Report

Use:

- `deliverables/CS541_Report_Draft_For_Word.md`

The supporting figures and analysis tables referenced by the report are stored in:

- `deliverables/report_assets/`
