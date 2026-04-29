from __future__ import annotations

import argparse
import os

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


PART_CONFIGS = {
    1: {
        "title": "Part 1 - CNN from scratch",
        "builder": create_part1_model,
        "image_size": 32,
        "batch_size": 128,
        "epochs": 20,
        "lr": 3e-4,
        "weight_decay": 1e-4,
    },
    2: {
        "title": "Part 2 - Pretrained CNN",
        "builder": create_part2_model,
        "image_size": 224,
        "batch_size": 64,
        "epochs": 8,
        "lr": 1e-4,
        "weight_decay": 1e-4,
    },
    3: {
        "title": "Part 3 - Pretrained Transformer",
        "builder": create_part3_model,
        "image_size": 224,
        "batch_size": 32,
        "epochs": 6,
        "lr": 5e-5,
        "weight_decay": 1e-4,
    },
}


def append_experiment_log(log_path: str, row: dict) -> pd.DataFrame:
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    df.to_csv(log_path, index=False)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CS541 challenge workflow.")
    parser.add_argument("--parts", type=int, nargs="+", default=[1, 2, 3], choices=[1, 2, 3])
    parser.add_argument("--student-name", default="Zhuojun Lyu")
    parser.add_argument("--student-id", default="U06761622")
    parser.add_argument("--student-email", default="lzj2729@bu.edu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--work-root", default=os.path.abspath("./cs541_workspace"))
    parser.add_argument("--shared-data-root", default="")
    parser.add_argument("--shared-ood-dir", default="")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--skip-submission", action="store_true")
    parser.add_argument("--submission-only", action="store_true")
    parser.add_argument("--submission-batch-size", type=int, default=0)
    parser.add_argument("--override-epochs", type=int, default=0)
    parser.add_argument("--override-batch-size", type=int, default=0)
    parser.add_argument("--override-lr", type=float, default=0.0)
    parser.add_argument("--override-weight-decay", type=float, default=0.0)
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    data_root = args.shared_data_root or os.path.join(args.work_root, "data")
    ood_dir = args.shared_ood_dir or os.path.join(args.work_root, "ood-test-CS541")
    checkpoint_dir = os.path.join(args.work_root, "checkpoints")
    submission_dir = os.path.join(args.work_root, "submissions")
    log_dir = os.path.join(args.work_root, "logs")
    experiment_log_path = os.path.join(log_dir, "experiment_log.csv")
    summary_path = os.path.join(log_dir, "latest_summary.csv")

    for path in [args.work_root, data_root, checkpoint_dir, submission_dir, log_dir]:
        os.makedirs(path, exist_ok=True)

    print("Device:", device)
    print("Work root:", args.work_root)

    summary_rows = []

    for part_id in args.parts:
        cfg = dict(PART_CONFIGS[part_id])
        if args.fast_dev_run:
            cfg["epochs"] = min(cfg["epochs"], 1 if part_id in (2, 3) else 2)
            cfg["batch_size"] = min(cfg["batch_size"], 32)
        if args.override_epochs > 0:
            cfg["epochs"] = args.override_epochs
        if args.override_batch_size > 0:
            cfg["batch_size"] = args.override_batch_size
        if args.override_lr > 0:
            cfg["lr"] = args.override_lr
        if args.override_weight_decay > 0:
            cfg["weight_decay"] = args.override_weight_decay

        print("=" * 80)
        print(cfg["title"])
        print("=" * 80)

        checkpoint_path = os.path.join(checkpoint_dir, f"part_{part_id}_best.pt")
        model = cfg["builder"]().to(device)

        if args.submission_only:
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found for part {part_id}: {checkpoint_path}")
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            result_best_val_acc = float("nan")
            result_clean_test_acc = float("nan")
        else:
            train_loader, val_loader, test_loader = make_cifar100_loaders(
                data_root=data_root,
                batch_size=cfg["batch_size"],
                num_workers=args.num_workers,
                image_size=cfg["image_size"],
                seed=args.seed,
                fast_dev_run=args.fast_dev_run,
            )

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
            result_best_val_acc = result.best_val_acc
            result_clean_test_acc = result.clean_test_acc

        submission_path = ""
        submission_rows = 0
        if not args.skip_submission:
            submission_path = os.path.join(submission_dir, f"{args.student_name}_Part_{part_id}.csv")
            submission_batch_size = args.submission_batch_size or (64 if device.type == "cuda" else 32)
            submission = build_submission(
                model=model,
                ood_dir=ood_dir,
                output_csv=submission_path,
                device=device,
                batch_size=submission_batch_size,
                image_size=cfg["image_size"],
            )
            submission_rows = len(submission)

        log_row = {
            "student_name": args.student_name,
            "student_id": args.student_id,
            "student_email": args.student_email,
            "part": part_id,
            "title": cfg["title"],
            "image_size": cfg["image_size"],
            "batch_size": cfg["batch_size"],
            "epochs": cfg["epochs"],
            "lr": cfg["lr"],
            "weight_decay": cfg["weight_decay"],
            "best_val_acc": result_best_val_acc,
            "clean_test_acc": result_clean_test_acc,
            "checkpoint_path": checkpoint_path,
            "submission_path": submission_path,
            "submission_rows": submission_rows,
            "notes": args.notes,
        }
        append_experiment_log(experiment_log_path, log_row)
        summary_rows.append(log_row)

        if not args.submission_only:
            print(f"Best validation accuracy: {result_best_val_acc:.4f}")
            print(f"Clean CIFAR-100 test accuracy: {result_clean_test_acc:.4f}")
        if submission_path:
            print(f"Submission saved to: {submission_path}")

    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print("Summary written to:", summary_path)
    print("Experiment log written to:", experiment_log_path)


if __name__ == "__main__":
    main()
