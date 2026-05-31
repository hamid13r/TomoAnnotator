"""
Train a 3D CNN patch classifier on extracted patches.

Architecture: small 3D ConvNet with global average pooling → softmax.
Input:  (B, 1, P, P, P) patch
Output: (B, N_classes) class probabilities

Training is fast — typically 5-15 minutes on a single GPU for 50 epochs.

Usage:
    python train_patch_classifier.py --patches patches.npz --output-dir models/
    python train_patch_classifier.py --patches patches.npz --output-dir models/ --epochs 80
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PatchCNN(nn.Module):
    """
    Lightweight 3D CNN for patch classification.
    ~500k parameters — trains in minutes on a single GPU.
    """
    def __init__(self, n_classes: int, base_channels: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv3d(1, base_channels, 3, padding=1), nn.BatchNorm3d(base_channels), nn.ReLU(),
            nn.Conv3d(base_channels, base_channels, 3, padding=1), nn.BatchNorm3d(base_channels), nn.ReLU(),
            nn.MaxPool3d(2),

            # Block 2
            nn.Conv3d(base_channels, base_channels * 2, 3, padding=1), nn.BatchNorm3d(base_channels * 2), nn.ReLU(),
            nn.Conv3d(base_channels * 2, base_channels * 2, 3, padding=1), nn.BatchNorm3d(base_channels * 2), nn.ReLU(),
            nn.MaxPool3d(2),

            # Block 3
            nn.Conv3d(base_channels * 2, base_channels * 4, 3, padding=1), nn.BatchNorm3d(base_channels * 4), nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),    # global average pool → (B, C, 1, 1, 1)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(base_channels * 4, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_patches(path: Path):
    data = np.load(path)
    patches = data["patches"].astype(np.float32)   # (N, 1, P, P, P)
    labels = data["labels"].astype(np.int64)        # (N,)
    class_names = list(data["class_names"])
    return patches, labels, class_names


def make_loaders(patches: np.ndarray, labels: np.ndarray,
                 val_fraction: float, batch_size: int, num_workers: int = 4):
    X_train, X_val, y_train, y_val = train_test_split(
        patches, labels, test_size=val_fraction, random_state=42,
        stratify=labels if len(np.unique(labels)) > 1 else None,
    )
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, len(X_train), len(X_val)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device, class_names):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    per_class_correct = np.zeros(len(class_names))
    per_class_total = np.zeros(len(class_names))

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * len(y)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        n += len(y)
        for c in range(len(class_names)):
            mask = y == c
            per_class_correct[c] += (preds[mask] == c).sum().item()
            per_class_total[c] += mask.sum().item()

    # Only divide where the class actually has validation samples; classes with
    # zero samples stay NaN (undefined accuracy) without triggering a 0/0 warning.
    per_class_acc = np.full(len(class_names), np.nan)
    nonzero = per_class_total > 0
    per_class_acc[nonzero] = per_class_correct[nonzero] / per_class_total[nonzero]
    return total_loss / n, correct / n, per_class_acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patches", default="patches.npz")
    parser.add_argument("--output-dir", default="models/")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--push-s3", action="store_true")
    parser.add_argument("--bucket", default=None)
    parser.add_argument("--profile", default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    train_cfg = cfg["training"]

    epochs = args.epochs or train_cfg["epochs"]
    batch_size = args.batch_size or train_cfg["batch_size"]
    lr = args.lr or train_cfg["lr"]
    val_fraction = train_cfg["val_fraction"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    patches, labels, class_names = load_patches(Path(args.patches))
    n_classes = len(class_names)
    print(f"Classes ({n_classes}): {class_names}")
    print(f"Patches: {len(patches)}  shape={patches.shape[1:]}")

    # Class weights to handle imbalance
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    weights = (counts.sum() / (n_classes * counts + 1e-6)).astype(np.float32)
    weights_tensor = torch.from_numpy(weights).to(device)

    train_loader, val_loader, n_train, n_val = make_loaders(
        patches, labels, val_fraction, batch_size)
    print(f"Train: {n_train}  Val: {n_val}")

    model = PatchCNN(n_classes=n_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=train_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save class names alongside the model
    (out_dir / "class_names.txt").write_text("\n".join(class_names))

    best_val_acc = 0.0
    print(f"\n{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>8}  {'Val Acc':>7}")
    print("-" * 55)

    for epoch in range(1, epochs + 1):
        tl, ta = train_epoch(model, train_loader, optimizer, criterion, device)
        vl, va, per_class_acc = evaluate(model, val_loader, criterion, device, class_names)
        scheduler.step()

        print(f"{epoch:5d}  {tl:10.4f}  {ta:9.3f}  {vl:8.4f}  {va:7.3f}")

        if va > best_val_acc:
            best_val_acc = va
            torch.save({
                "model_state": model.state_dict(),
                "class_names": class_names,
                "n_classes": n_classes,
                "patch_size": patches.shape[-1],
            }, out_dir / "patch_classifier.pth")

        # Detailed per-class accuracy every 10 epochs.
        # "n/a" = no samples of that class in the validation split (it was not
        # painted, got 0 patches, or is too rare to land in the 15% val set).
        if epoch % 10 == 0:
            acc_str = "  ".join(
                f"{name}={'n/a' if np.isnan(acc) else f'{acc:.2f}'}"
                for name, acc in zip(class_names, per_class_acc))
            print(f"       Per-class val acc: {acc_str}")

    print(f"\nBest val accuracy: {best_val_acc:.3f}")
    print(f"Checkpoint: {out_dir / 'patch_classifier.pth'}")

    if args.push_s3:
        from aws_utils import ensure_bucket, upload, DEFAULT_BUCKET
        bucket = args.bucket or DEFAULT_BUCKET
        ensure_bucket(bucket, args.profile)
        for f in ["patch_classifier.pth", "class_names.txt"]:
            upload(out_dir / f, f"models/{f}", bucket=bucket, profile=args.profile)

    print("Next: python detect_features.py --tomogram <path>")


if __name__ == "__main__":
    main()
