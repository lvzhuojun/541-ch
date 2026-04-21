from __future__ import annotations

import argparse
import os
import tarfile
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from cs541_challenge_utils import (
    CIFAR100ArrayDataset,
    _make_transforms,
    create_part1_model,
    create_part2_model,
    create_part3_model,
    get_device,
    load_cifar100_arrays,
)


MODEL_BUILDERS = {
    1: create_part1_model,
    2: create_part2_model,
    3: create_part3_model,
}

IMAGE_SIZES = {
    1: 32,
    2: 224,
    3: 224,
}


def parse_args() -> argparse.Namespace:
    """Collect the checkpoint and output paths needed for report artifact generation."""
    parser = argparse.ArgumentParser(description="Build report artifacts for a trained checkpoint.")
    parser.add_argument("--part", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="./cs541_workspace/data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def load_class_names(data_root: str) -> list[str]:
    """Read the CIFAR-100 fine label names from the original archive metadata."""
    archive_path = os.path.join(data_root, "cifar-100-python.tar.gz")
    with tarfile.open(archive_path, "r:gz") as tar:
        member = tar.getmember("cifar-100-python/meta")
        extracted = tar.extractfile(member)
        if extracted is None:
            raise FileNotFoundError("cifar-100-python/meta")
        meta = pd.read_pickle(BytesIO(extracted.read()))
    return list(meta["fine_label_names"])


def build_test_dataset(data_root: str, image_size: int) -> tuple[CIFAR100ArrayDataset, np.ndarray]:
    """Build the clean CIFAR-100 test dataset and keep raw images for figure export."""
    _, _, x_test, y_test = load_cifar100_arrays(data_root)
    test_ds = CIFAR100ArrayDataset(
        x_test,
        y_test,
        transform=_make_transforms(image_size=image_size, train=False),
    )
    return test_ds, x_test


def main() -> None:
    """Compute report-ready tables and hardest-example figures for a saved checkpoint."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = get_device()
    builder = MODEL_BUILDERS[args.part]
    image_size = IMAGE_SIZES[args.part]

    model = builder().to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    test_ds, raw_test_images = build_test_dataset(args.data_root, image_size=image_size)
    class_names = load_class_names(args.data_root)
    loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    per_class_correct = np.zeros(100, dtype=np.int64)
    per_class_total = np.zeros(100, dtype=np.int64)
    error_rows: list[dict] = []
    sample_offset = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            for i in range(yb.size(0)):
                idx = sample_offset + i
                true_label = int(yb[i].item())
                pred_label = int(preds[i].item())
                per_class_total[true_label] += 1
                if pred_label == true_label:
                    per_class_correct[true_label] += 1
                    continue

                pred_prob = float(probs[i, pred_label].item())
                true_prob = float(probs[i, true_label].item())
                pred_logit = float(logits[i, pred_label].item())
                true_logit = float(logits[i, true_label].item())
                error_rows.append(
                    {
                        "sample_index": idx,
                        "true_label": true_label,
                        "true_class": class_names[true_label],
                        "pred_label": pred_label,
                        "pred_class": class_names[pred_label],
                        "pred_prob": pred_prob,
                        "true_prob": true_prob,
                        "pred_logit": pred_logit,
                        "true_logit": true_logit,
                        "logit_gap": pred_logit - true_logit,
                    }
                )

            sample_offset += yb.size(0)

    per_class_acc = per_class_correct / np.maximum(per_class_total, 1)
    per_class_df = pd.DataFrame(
        {
            "class_id": np.arange(100),
            "class_name": class_names,
            "correct": per_class_correct,
            "total": per_class_total,
            "accuracy": per_class_acc,
        }
    ).sort_values(["accuracy", "class_id"], ascending=[True, True])
    per_class_csv = os.path.join(args.output_dir, f"part_{args.part}_per_class_accuracy.csv")
    per_class_df.to_csv(per_class_csv, index=False)

    top3_worst = per_class_df.head(3).copy()
    top3_worst.to_csv(
        os.path.join(args.output_dir, f"part_{args.part}_top3_worst_classes.csv"),
        index=False,
    )

    errors_df = pd.DataFrame(error_rows).sort_values("logit_gap", ascending=False)
    errors_csv = os.path.join(args.output_dir, f"part_{args.part}_largest_errors.csv")
    errors_df.to_csv(errors_csv, index=False)

    top3_errors = errors_df.head(3).copy()
    top3_errors.to_csv(
        os.path.join(args.output_dir, f"part_{args.part}_top3_error_samples.csv"),
        index=False,
    )

    for rank, row in enumerate(top3_errors.itertuples(index=False), start=1):
        image = raw_test_images[row.sample_index]
        plt.figure(figsize=(3.2, 3.2))
        plt.imshow(image)
        plt.axis("off")
        plt.title(
            f"Rank {rank}\ntrue={row.true_class} ({row.true_prob:.3f})\n"
            f"pred={row.pred_class} ({row.pred_prob:.3f})"
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(args.output_dir, f"part_{args.part}_largest_error_{rank}.png"),
            dpi=180,
        )
        plt.close()

    summary_lines = [
        f"Part {args.part} checkpoint: {args.checkpoint}",
        "Top 3 worst classes:",
    ]
    for row in top3_worst.itertuples(index=False):
        summary_lines.append(
            f"- {row.class_name} (class {row.class_id}): {row.accuracy:.4f} "
            f"[{row.correct}/{row.total}]"
        )
    summary_lines.append("Top 3 largest incorrect prediction gaps:")
    for row in top3_errors.itertuples(index=False):
        summary_lines.append(
            f"- sample {row.sample_index}: true={row.true_class}, pred={row.pred_class}, "
            f"logit_gap={row.logit_gap:.4f}, pred_prob={row.pred_prob:.4f}, true_prob={row.true_prob:.4f}"
        )
    with open(os.path.join(args.output_dir, f"part_{args.part}_analysis_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")


if __name__ == "__main__":
    main()
