"""
Detect which features are present in a tomogram using the trained patch classifier.

Slides a window across the full tomogram, classifies each patch, and aggregates
results into a presence/absence report with confidence scores and Z-ranges.

Output:
  - Console table: feature → YES/no + confidence + where found
  - Optional CSV with per-feature details
  - Optional heatmap .npy files showing where each feature was detected

Usage:
    python detect_features.py --tomogram data/processed/new_run/tomogram.npy
    python detect_features.py --data-dir data/processed/ --output-csv results/predictions.csv
    python detect_features.py --tomogram data/processed/run1/tomogram.npy --save-heatmaps
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from train_patch_classifier import PatchCNN


def load_model(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    class_names = ckpt["class_names"]
    n_classes = ckpt["n_classes"]
    patch_size = ckpt["patch_size"]

    model = PatchCNN(n_classes=n_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, class_names, patch_size


@torch.no_grad()
def detect(tomo: np.ndarray, model: nn.Module, patch_size: int,
           stride_fraction: float, device: torch.device, batch_size: int = 64
           ) -> np.ndarray:
    """
    Slide a window over the tomogram and return a probability heatmap.
    Shape: (n_classes, n_z, n_y, n_x) where each cell is a patch-center probability.
    """
    stride = max(1, int(patch_size * stride_fraction))
    h = patch_size // 2

    zs = np.arange(h, tomo.shape[0] - patch_size + h + 1, stride)
    ys = np.arange(h, tomo.shape[1] - patch_size + h + 1, stride)
    xs = np.arange(h, tomo.shape[2] - patch_size + h + 1, stride)

    # Build list of all patch centers
    centers = [(z, y, x) for z in zs for y in ys for x in xs]

    n_classes = None
    all_probs = []

    # Process in batches
    batch_patches = []
    batch_centers = []

    def flush_batch():
        if not batch_patches:
            return
        arr = np.stack(batch_patches)[:, None].astype(np.float32)  # (B, 1, P, P, P)
        tensor = torch.from_numpy(arr).to(device)
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        batch_patches.clear()
        batch_centers.clear()

    for z, y, x in tqdm(centers, desc="Detecting", leave=False):
        patch = tomo[z-h:z-h+patch_size, y-h:y-h+patch_size, x-h:x-h+patch_size]
        if patch.shape != (patch_size, patch_size, patch_size):
            continue
        batch_patches.append(patch)
        batch_centers.append((z, y, x))
        if len(batch_patches) >= batch_size:
            flush_batch()
    flush_batch()

    if not all_probs:
        raise RuntimeError("No valid patches found. Tomogram may be too small for the patch size.")

    probs_flat = np.concatenate(all_probs, axis=0)  # (N_patches, n_classes)
    n_classes = probs_flat.shape[1]

    # Build 4D heatmap (n_classes, nz, ny, nx)
    nz, ny, nx = len(zs), len(ys), len(xs)
    heatmap = probs_flat.reshape(nz, ny, nx, n_classes).transpose(3, 0, 1, 2)
    return heatmap, (zs, ys, xs)


def summarize(heatmap: np.ndarray, zs: np.ndarray, class_names: list[str],
              confidence_threshold: float, min_patches: int) -> list[dict]:
    """Aggregate heatmap → per-feature presence report."""
    results = []
    for c, name in enumerate(class_names):
        if name == "background":
            continue
        prob_map = heatmap[c]  # (nz, ny, nx)
        above = prob_map > confidence_threshold
        n_above = above.sum()
        present = n_above >= min_patches
        max_conf = float(prob_map.max())
        mean_conf = float(prob_map[above].mean()) if n_above > 0 else 0.0

        # Z-range where detected
        if n_above > 0:
            z_indices = np.where(above.any(axis=(1, 2)))[0]
            z_min, z_max = int(zs[z_indices.min()]), int(zs[z_indices.max()])
        else:
            z_min, z_max = None, None

        results.append({
            "feature": name,
            "present": present,
            "max_confidence": max_conf,
            "mean_confidence_above_threshold": mean_conf,
            "patches_detected": int(n_above),
            "z_range": f"{z_min}–{z_max}" if z_min is not None else "n/a",
        })
    return results


def print_results(run_name: str, results: list[dict]):
    print(f"\n{'─'*60}")
    print(f"  {run_name}")
    print(f"{'─'*60}")
    print(f"  {'Feature':<22} {'Present':>7}  {'Max conf':>8}  {'Z range':>14}  {'Patches':>7}")
    print(f"  {'':22} {'':>7}  {'':>8}  {'':>14}  {'':>7}")
    for r in results:
        present_str = "YES" if r["present"] else "no"
        conf_str = f"{r['max_confidence']:.2f}"
        print(f"  {r['feature']:<22} {present_str:>7}  {conf_str:>8}  "
              f"{r['z_range']:>14}  {r['patches_detected']:>7}")
    print(f"{'─'*60}\n")


def process_tomogram(tomo_path: Path, model, class_names, patch_size, device,
                     stride_fraction, confidence_threshold, min_patches,
                     save_heatmaps: bool = False) -> list[dict]:
    tomo = np.load(tomo_path).astype(np.float32)
    heatmap, (zs, ys, xs) = detect(tomo, model, patch_size, stride_fraction, device)
    results = summarize(heatmap, zs, class_names, confidence_threshold, min_patches)

    if save_heatmaps:
        hm_dir = tomo_path.parent / "heatmaps"
        hm_dir.mkdir(exist_ok=True)
        for c, name in enumerate(class_names):
            if name != "background":
                np.save(hm_dir / f"{name}.npy", heatmap[c])
        print(f"  Heatmaps saved: {hm_dir}/")

    return results


import torch.nn as nn  # needed for type hint in detect signature


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tomogram", help="Single tomogram.npy to classify")
    group.add_argument("--data-dir", help="Directory of processed runs to classify")
    parser.add_argument("--model-dir", default="models/")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--save-heatmaps", action="store_true",
                        help="Save per-feature probability maps alongside tomogram")
    parser.add_argument("--push-s3", action="store_true")
    parser.add_argument("--bucket", default=None)
    parser.add_argument("--aws-profile", default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    det_cfg = cfg["detection"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = Path(args.model_dir) / "patch_classifier.pth"
    model, class_names, patch_size = load_model(ckpt_path, device)
    print(f"Model loaded: {ckpt_path}  classes={class_names}  patch={patch_size}^3")

    if args.tomogram:
        tomo_paths = [Path(args.tomogram)]
    else:
        data_dir = Path(args.data_dir)
        tomo_paths = sorted(data_dir.rglob("tomogram.npy"))
        print(f"Found {len(tomo_paths)} tomogram(s)")

    all_rows = []
    for tomo_path in tomo_paths:
        run_name = tomo_path.parent.name
        results = process_tomogram(
            tomo_path, model, class_names, patch_size, device,
            stride_fraction=det_cfg["stride_fraction"],
            confidence_threshold=det_cfg["confidence_threshold"],
            min_patches=det_cfg["min_patches_detected"],
            save_heatmaps=args.save_heatmaps,
        )
        print_results(run_name, results)
        for r in results:
            all_rows.append({"tomogram_id": run_name, **r})

    if args.output_csv:
        import pandas as pd
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_rows).to_csv(out, index=False)
        print(f"Saved: {out}")

        if args.push_s3:
            from aws_utils import ensure_bucket, upload, DEFAULT_BUCKET
            bucket = args.bucket or DEFAULT_BUCKET
            ensure_bucket(bucket, args.aws_profile)
            upload(out, f"results/{out.name}", bucket=bucket, profile=args.aws_profile)

    print("Next: python report.py --predictions results/predictions.csv")


if __name__ == "__main__":
    main()
