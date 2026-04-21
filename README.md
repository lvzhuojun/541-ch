# CS541 Challenge Spring 2026

**Student:** Zhuojun Lyu  
**BU ID:** U06761622  
**Email:** lzj2729@bu.edu  
**Kaggle Name:** Zhuojun Lyu

This repository contains the final course submission materials for the CS541 image classification challenge. The project covers three required parts:

1. Part 1: a CNN built from scratch on CIFAR-100
2. Part 2: a fine-tuned pretrained CNN
3. Part 3: a fine-tuned pretrained transformer

The work focuses on both clean CIFAR-100 accuracy and robustness on the distorted out-of-distribution evaluation set used by the Kaggle competition.

## Repository Contents

- `cs541_challenge_three_parts.ipynb`: main notebook for the Colab-style workflow
- `run_cs541_assignment.py`: command-line runner for training and CSV export
- `cs541_challenge_utils.py`: shared model, data, training, and submission utilities
- `build_report_artifacts.py`: helper script used to generate report tables and figures
- `submissions/`: Kaggle CSV files prepared for the final submission set
- `report/`: report draft, figures, and analysis tables for the final PDF/Word write-up

## Final Kaggle CSV Files

These are the three CSV files intended for Kaggle submission, uploaded one at a time:

- `submissions/Zhuojun Lyu_Part_1.csv`
- `submissions/Zhuojun Lyu_Part_2.csv`
- `submissions/Zhuojun Lyu_Part_3.csv`

For score control, the selected files in this repo are:

- Part 1: root Part 1 submission
- Part 2: `part2_v070` submission
- Part 3: `part3_v080` submission

## Report Materials

The main report draft for conversion into Word/PDF is:

- `report/CS541_Report_Draft_For_Word.md`

Supporting figures and tables used in that report are stored in:

- `report/figures/`
- `report/analysis_tables/`

## Submission Mapping

- Kaggle: upload the three CSV files in `submissions/`, one submission per file
- Colab / notebook deliverable: use `cs541_challenge_three_parts.ipynb`
- Gradescope / repo report deliverable: use the report content in `report/CS541_Report_Draft_For_Word.md` and export it to PDF in Word later

## Reproducibility Notes

The main local environment used for training was the Windows conda environment `kaggle311` with GPU-enabled PyTorch. Large datasets, checkpoints, Hugging Face cache files, and temporary training outputs are intentionally excluded from GitHub to keep the repository clean and submission-ready.
