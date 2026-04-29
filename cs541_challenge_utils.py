from __future__ import annotations

import copy
import os
import pickle
import random
import tarfile
from io import BytesIO
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights, ViT_B_16_Weights
from torchvision.datasets.utils import download_url
from tqdm.auto import tqdm


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
os.environ.setdefault("TORCH_HOME", os.path.abspath("./.torch-cache"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = torch.cuda.is_available()


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_transforms(image_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
                transforms.RandomErasing(p=0.15, scale=(0.02, 0.15)),
            ]
        )
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )


def ensure_cifar100_archive(data_root: str) -> str:
    os.makedirs(data_root, exist_ok=True)
    archive_name = "cifar-100-python.tar.gz"
    archive_path = os.path.join(data_root, archive_name)
    if not os.path.exists(archive_path):
        download_url(
            url="https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
            root=data_root,
            filename=archive_name,
            md5="eb9058c3a382ffc7106e4002c42a8d85",
        )
    return archive_path


def _load_pickle_from_tar(tar: tarfile.TarFile, member_name: str) -> dict:
    member = tar.getmember(member_name)
    f = tar.extractfile(member)
    if f is None:
        raise FileNotFoundError(member_name)
    data = f.read()
    return pickle.load(BytesIO(data), encoding="latin1")


def load_cifar100_arrays(data_root: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    archive_path = ensure_cifar100_archive(data_root)
    with tarfile.open(archive_path, "r:gz") as tar:
        train_obj = _load_pickle_from_tar(tar, "cifar-100-python/train")
        test_obj = _load_pickle_from_tar(tar, "cifar-100-python/test")

    x_train = train_obj["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_train = np.array(train_obj["fine_labels"], dtype=np.int64)
    x_test = test_obj["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_test = np.array(test_obj["fine_labels"], dtype=np.int64)
    return x_train, y_train, x_test, y_test


class CIFAR100ArrayDataset(torch.utils.data.Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None) -> None:
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        image = self.images[idx]
        label = int(self.labels[idx])
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def make_cifar100_loaders(
    data_root: str,
    batch_size: int,
    num_workers: int,
    image_size: int,
    seed: int,
    fast_dev_run: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    x_train, y_train, x_test, y_test = load_cifar100_arrays(data_root)
    train_full_plain = CIFAR100ArrayDataset(x_train, y_train, transform=None)
    test_ds = CIFAR100ArrayDataset(
        x_test, y_test, transform=_make_transforms(image_size=image_size, train=False)
    )

    n_train_total = len(train_full_plain)
    n_train = int(0.8 * n_train_total)
    n_val = n_train_total - n_train
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = torch.utils.data.random_split(
        train_full_plain, [n_train, n_val], generator=generator
    )

    train_subset.dataset.transform = _make_transforms(image_size=image_size, train=True)
    val_subset.dataset = copy.copy(val_subset.dataset)
    val_subset.dataset.transform = _make_transforms(image_size=image_size, train=False)

    if fast_dev_run:
        if torch.cuda.is_available():
            train_subset = torch.utils.data.Subset(train_subset, range(min(2048, len(train_subset))))
            val_subset = torch.utils.data.Subset(val_subset, range(min(512, len(val_subset))))
            test_ds = torch.utils.data.Subset(test_ds, range(min(512, len(test_ds))))
        else:
            train_subset = torch.utils.data.Subset(train_subset, range(min(256, len(train_subset))))
            val_subset = torch.utils.data.Subset(val_subset, range(min(128, len(val_subset))))
            test_ds = torch.utils.data.Subset(test_ds, range(min(128, len(test_ds))))

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ScratchCNN(nn.Module):
    """CNN from scratch with more than 2 blocks, satisfying Part 1."""

    def __init__(self, num_classes: int = 100) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 64, 0.05),
            ConvBlock(64, 128, 0.08),
            ConvBlock(128, 256, 0.12),
            ConvBlock(256, 512, 0.15),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def create_part1_model(num_classes: int = 100) -> nn.Module:
    return ScratchCNN(num_classes=num_classes)


def create_part2_model(num_classes: int = 100) -> nn.Module:
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def create_part3_model(num_classes: int = 100) -> nn.Module:
    model = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model


@dataclass
class TrainingResult:
    best_val_acc: float
    clean_test_acc: float
    history: Dict[str, List[float]]


def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            total += yb.numel()
    return correct / max(total, 1)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    checkpoint_path: str,
) -> TrainingResult:
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history: Dict[str, List[float]] = {"train_acc": [], "val_acc": []}
    best_val_acc = -1.0
    best_state: Dict[str, torch.Tensor] | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_correct = 0
        train_total = 0

        for xb, yb in tqdm(train_loader, desc=f"train {epoch}/{epochs}", leave=False):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_correct += (logits.argmax(dim=1) == yb).sum().item()
            train_total += yb.numel()

        val_acc = evaluate_accuracy(model, val_loader, device)
        train_acc = train_correct / max(train_total, 1)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        scheduler.step()

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.6f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(best_state, checkpoint_path)

    if best_state is None:
        raise RuntimeError("Training finished without producing a checkpoint.")

    model.load_state_dict(best_state)
    clean_test_acc = evaluate_accuracy(model, test_loader, device)
    return TrainingResult(
        best_val_acc=best_val_acc,
        clean_test_acc=clean_test_acc,
        history=history,
    )


def ensure_ood_files(ood_dir: str) -> None:
    os.makedirs(ood_dir, exist_ok=True)
    hf_home = os.path.join(ood_dir, ".hf-home")
    os.makedirs(hf_home, exist_ok=True)
    os.environ["HF_HOME"] = hf_home
    os.environ["HF_HUB_CACHE"] = os.path.join(hf_home, "hub")
    os.environ["HF_XET_CACHE"] = os.path.join(hf_home, "xet")
    os.environ["HF_ASSETS_CACHE"] = os.path.join(hf_home, "assets")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    print("Downloading OOD files from Hugging Face dataset...")
    snapshot_download(
        repo_id="XThomasBU/ood-test-CS541",
        repo_type="dataset",
        local_dir=ood_dir,
        local_dir_use_symlinks=False,
    )
    print("OOD files ready in", ood_dir)


@torch.no_grad()
def predict_file(
    model: nn.Module,
    npy_path: str,
    severity: int,
    batch_size: int,
    device: torch.device,
    image_size: int,
) -> np.ndarray:
    images = np.load(npy_path, mmap_mode="r")
    start = (severity - 1) * 10000
    end = severity * 10000

    normalize = transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
    preds: List[np.ndarray] = []
    model.eval()
    for b0 in tqdm(range(start, end, batch_size), desc=f"{os.path.basename(npy_path)} sev{severity}", leave=False):
        b1 = min(b0 + batch_size, end)
        xb_np = np.array(images[b0:b1], copy=True)
        xb = torch.from_numpy(xb_np).permute(0, 3, 1, 2).float().div(255.0)
        if image_size != 32:
            xb = torch.nn.functional.interpolate(
                xb, size=(image_size, image_size), mode="bilinear", align_corners=False
            )
        xb = normalize(xb).to(device, non_blocking=True)
        logits = model(xb)
        preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds, axis=0)


def build_submission(
    model: nn.Module,
    ood_dir: str,
    output_csv: str,
    device: torch.device,
    batch_size: int,
    image_size: int,
) -> pd.DataFrame:
    ensure_ood_files(ood_dir)
    distortion_files = sorted(
        [p for p in os.listdir(ood_dir) if p.startswith("distortion") and p.endswith(".npy")]
    )
    print("Distortion files found:", len(distortion_files))

    rows = []
    for fname in distortion_files:
        dname = os.path.splitext(fname)[0]
        path = os.path.join(ood_dir, fname)
        for severity in [1, 2, 3, 4, 5]:
            pred = predict_file(model, path, severity, batch_size, device, image_size=image_size)
            for i, y in enumerate(pred.tolist()):
                rows.append((f"{dname}_{severity}_{i}", int(y)))
            print(f"{dname}_{severity} done")

    submission = pd.DataFrame(rows, columns=["id", "label"])
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    submission.to_csv(output_csv, index=False)
    print("Wrote", output_csv, "rows:", len(submission))
    return submission
