from __future__ import annotations

import nbformat as nbf


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


nb = nbf.v4.new_notebook()
nb.cells = [
    md(
        """# CS541 Challenge Spring 2026

This notebook keeps the starter workflow, but extends it to satisfy all three required parts:

- Part 1: CNN from scratch
- Part 2: Fine-tune a pretrained CNN
- Part 3: Fine-tune a pretrained transformer

The original starter notebook is kept unchanged in the repo as `sp2026_midterm_student_baseline_colab.ipynb`.
"""
    ),
    code(
        """# Install required packages (safe to re-run)
import importlib.util
import subprocess
import sys

required = [
    "torch",
    "torchvision",
    "tqdm",
    "numpy",
    "pandas",
    "matplotlib",
    "huggingface_hub",
]
missing = [pkg for pkg in required if importlib.util.find_spec(pkg) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", *missing])

print("Environment ready")
"""
    ),
    code(
        """# Runtime configuration
import os

FAST_DEV_RUN = False
SEED = 42
STUDENT_NAME = "Zhuojun Lyu"
STUDENT_ID = "U06761622"
STUDENT_EMAIL = "lzj2729@bu.edu"
PARTS_TO_RUN = [1, 2, 3]

IN_COLAB = False
try:
    import google.colab  # type: ignore
    IN_COLAB = True
except Exception:
    IN_COLAB = False

WORK_ROOT = "/content" if IN_COLAB else os.path.abspath("./cs541_workspace")
DATA_ROOT = os.path.join(WORK_ROOT, "data")
OOD_DIR = os.path.join(WORK_ROOT, "ood-test-CS541")
CHECKPOINT_DIR = os.path.join(WORK_ROOT, "checkpoints")
SUBMISSION_DIR = os.path.join(WORK_ROOT, "submissions")
LOG_DIR = os.path.join(WORK_ROOT, "logs")

os.makedirs(WORK_ROOT, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

print("IN_COLAB:", IN_COLAB)
print("WORK_ROOT:", WORK_ROOT)
print("DATA_ROOT:", DATA_ROOT)
print("OOD_DIR:", OOD_DIR)
print("CHECKPOINT_DIR:", CHECKPOINT_DIR)
print("SUBMISSION_DIR:", SUBMISSION_DIR)
print("LOG_DIR:", LOG_DIR)
"""
    ),
    code(
        """import json
import matplotlib.pyplot as plt
import pandas as pd
import torch

from cs541_challenge_utils import (
    build_submission,
    create_part1_model,
    create_part2_model,
    create_part3_model,
    get_device,
    make_cifar100_loaders,
    set_seed,
    train_model,
)

set_seed(SEED)
device = get_device()
print("Device:", device)
print("Student:", STUDENT_NAME, STUDENT_ID, STUDENT_EMAIL)
"""
    ),
    md(
        """## Student Information

- Name: Zhuojun Lyu
- BU ID: U06761622
- Email: lzj2729@bu.edu

Use your full name on Kaggle so it matches the course requirement.
"""
    ),
    md(
        """## Shared Configuration

These settings are intentionally conservative so the notebook stays runnable on Colab.
You can increase epochs or batch size for final experiments before your April 30, 2026 submission.
"""
    ),
    code(
        """PART_CONFIGS = {
    1: {
        "title": "Part 1 - CNN from scratch",
        "builder": create_part1_model,
        "image_size": 32,
        "batch_size": 128,
        "epochs": 20 if not FAST_DEV_RUN else 2,
        "lr": 3e-4,
        "weight_decay": 1e-4,
    },
    2: {
        "title": "Part 2 - Pretrained CNN",
        "builder": create_part2_model,
        "image_size": 224,
        "batch_size": 64,
        "epochs": 8 if not FAST_DEV_RUN else 1,
        "lr": 1e-4,
        "weight_decay": 1e-4,
    },
    3: {
        "title": "Part 3 - Pretrained Transformer",
        "builder": create_part3_model,
        "image_size": 224,
        "batch_size": 32,
        "epochs": 6 if not FAST_DEV_RUN else 1,
        "lr": 5e-5,
        "weight_decay": 1e-4,
    },
}

NUM_WORKERS = 0 if IN_COLAB else 2
results = {}
EXPERIMENT_LOG_PATH = os.path.join(LOG_DIR, "experiment_log.csv")
"""
    ),
    code(
        """def append_experiment_log(row: dict):
    if os.path.exists(EXPERIMENT_LOG_PATH):
        df = pd.read_csv(EXPERIMENT_LOG_PATH)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(EXPERIMENT_LOG_PATH, index=False)
    return df


def run_part(part_id: int, notes: str = ""):
    cfg = PART_CONFIGS[part_id]
    print("=" * 80)
    print(cfg["title"])
    print("=" * 80)

    train_loader, val_loader, test_loader = make_cifar100_loaders(
        data_root=DATA_ROOT,
        batch_size=cfg["batch_size"],
        num_workers=NUM_WORKERS,
        image_size=cfg["image_size"],
        seed=SEED,
        fast_dev_run=FAST_DEV_RUN,
    )

    model = cfg["builder"]().to(device)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"part_{part_id}_best.pt")

    result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        epochs=cfg["epochs"],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        checkpoint_path=checkpoint_path,
    )

    output_csv = os.path.join(SUBMISSION_DIR, f"{STUDENT_NAME}_Part_{part_id}.csv")
    submission = build_submission(
        model=model,
        ood_dir=OOD_DIR,
        output_csv=output_csv,
        device=device,
        batch_size=64 if device.type == "cuda" else 32,
        image_size=cfg["image_size"],
    )

    results[part_id] = {
        "config": cfg,
        "best_val_acc": result.best_val_acc,
        "clean_test_acc": result.clean_test_acc,
        "history": result.history,
        "checkpoint_path": checkpoint_path,
        "submission_path": output_csv,
        "submission_rows": len(submission),
    }

    log_row = {
        "student_name": STUDENT_NAME,
        "student_id": STUDENT_ID,
        "student_email": STUDENT_EMAIL,
        "part": part_id,
        "title": cfg["title"],
        "image_size": cfg["image_size"],
        "batch_size": cfg["batch_size"],
        "epochs": cfg["epochs"],
        "lr": cfg["lr"],
        "weight_decay": cfg["weight_decay"],
        "best_val_acc": result.best_val_acc,
        "clean_test_acc": result.clean_test_acc,
        "checkpoint_path": checkpoint_path,
        "submission_path": output_csv,
        "notes": notes,
    }
    append_experiment_log(log_row)

    print(f"Best validation accuracy: {result.best_val_acc:.4f}")
    print(f"Clean CIFAR-100 test accuracy: {result.clean_test_acc:.4f}")
    print(f"Submission saved to: {output_csv}")
    print(f"Experiment log updated: {EXPERIMENT_LOG_PATH}")
    return model, result
"""
    ),
    md(
        """## Experiment Logging

Each run appends one row to `logs/experiment_log.csv` so that hyperparameters and results are recorded in a simple format for your report.
Use the optional `notes` argument to record what changed in that run.
"""
    ),
    md(
        """## Part 1

Requirement reminder: this must be your own CNN from scratch with more than 2 blocks.
The implementation below uses a 4-block CNN with convolution, normalization, activation, pooling, and dropout.
"""
    ),
    code(
        """if 1 in PARTS_TO_RUN:
    model_part1, result_part1 = run_part(1, notes="Initial Part 1 run")
"""
    ),
    code(
        """if 1 in results:
    plt.figure(figsize=(6, 4))
    plt.plot(results[1]["history"]["train_acc"], label="train")
    plt.plot(results[1]["history"]["val_acc"], label="val")
    plt.title("Part 1 Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
"""
    ),
    md(
        """## Part 2

Requirement reminder: this must fine-tune a pretrained CNN.
The implementation below fine-tunes an ImageNet-pretrained `ResNet18`.
"""
    ),
    code(
        """if 2 in PARTS_TO_RUN:
    model_part2, result_part2 = run_part(2, notes="Initial Part 2 run")
"""
    ),
    code(
        """if 2 in results:
    plt.figure(figsize=(6, 4))
    plt.plot(results[2]["history"]["train_acc"], label="train")
    plt.plot(results[2]["history"]["val_acc"], label="val")
    plt.title("Part 2 Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
"""
    ),
    md(
        """## Part 3

Requirement reminder: this must fine-tune a pretrained transformer.
The implementation below fine-tunes an ImageNet-pretrained `ViT-B/16`.
"""
    ),
    code(
        """if 3 in PARTS_TO_RUN:
    model_part3, result_part3 = run_part(3, notes="Initial Part 3 run")
"""
    ),
    code(
        """if 3 in results:
    plt.figure(figsize=(6, 4))
    plt.plot(results[3]["history"]["train_acc"], label="train")
    plt.plot(results[3]["history"]["val_acc"], label="val")
    plt.title("Part 3 Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
"""
    ),
    md(
        """## Final Summary

This cell gives a compact summary of the three required outputs.
"""
    ),
    code(
        """summary_rows = []
for part_id in sorted(results):
    item = results[part_id]
    summary_rows.append(
        {
            "part": part_id,
            "best_val_acc": item["best_val_acc"],
            "clean_test_acc": item["clean_test_acc"],
            "checkpoint_path": item["checkpoint_path"],
            "submission_path": item["submission_path"],
            "submission_rows": item["submission_rows"],
        }
    )

summary_df = pd.DataFrame(summary_rows)
summary_df
"""
    ),
    code(
        """if os.path.exists(EXPERIMENT_LOG_PATH):
    pd.read_csv(EXPERIMENT_LOG_PATH)
else:
    print("No experiment log has been created yet.")
"""
    ),
    md(
        """## Submission Checklist

- Gradescope by **April 30, 2026**:
  - Shared Colab link to this runnable notebook containing all 3 parts
  - PDF report
- Kaggle:
  - `{Name}_Part_1.csv`
  - `{Name}_Part_2.csv`
  - `{Name}_Part_3.csv`
- Initial submission by **April 21, 2026**:
  - At least one leaderboard submission with accuracy **>= 0.5**
"""
    ),
]

nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb.metadata["language_info"] = {
    "name": "python",
    "version": "3.11",
}

with open("cs541_challenge_three_parts.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Wrote cs541_challenge_three_parts.ipynb")
