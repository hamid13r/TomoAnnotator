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
import torch.nn as nn
import yaml
from tqdm import tqdm

from train_patch_classifier import PatchCNN, PatchCNN2D


def load_model(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    class_names = ckpt["class_names"]
    n_classes = ckpt["n_classes"]
    patch_size = ckpt["patch_size"]
    # Older checkpoints had no model_type → they were 3D.
    model_type = ckpt.get("model_type", "3d")
    in_channels = ckpt.get("in_channels", 1)

    if model_type == "3d":
        model = PatchCNN(n_classes=n_classes).to(device)
    else:
        model = PatchCNN2D(n_classes=n_classes, in_channels=in_channels).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, class_names, patch_size, model_type, in_channels


@torch.no_grad()
def detect(tomo: np.ndarray, model: nn.Module, patch_size: int,
           stride_fraction: float, device: torch.device,
           model_type: str = "3d", in_channels: int = 1,
           batch_size: int = 64) -> np.ndarray:
    """
    Slide a window over the tomogram and return a probability heatmap.
    Shape: (n_classes, n_z, n_y, n_x) where each cell is a patch-center probability.

    Geometry depends on model_type:
      3d        cube patch (patch_size^3), strided in all three axes.
      2d/2.5d   in-plane patch (patch_size^2) with a stack of `in_channels`
                adjacent Z-slices; strided in-plane, and along Z by the slice depth.
    """
    if model_type == "3d":
        extent = (patch_size, patch_size, patch_size)
        z_stride = max(1, int(patch_size * stride_fraction))
    else:
        extent = (in_channels, patch_size, patch_size)
        z_stride = max(1, int(in_channels * stride_fraction))
    ez, ey, ex = extent
    hz, hy, hx = (e // 2 for e in extent)
    xy_stride = max(1, int(patch_size * stride_fraction))

    zs = np.arange(hz, tomo.shape[0] - ez + hz + 1, z_stride)
    ys = np.arange(hy, tomo.shape[1] - ey + hy + 1, xy_stride)
    xs = np.arange(hx, tomo.shape[2] - ex + hx + 1, xy_stride)

    centers = [(z, y, x) for z in zs for y in ys for x in xs]

    all_probs = []
    batch_patches = []
    is_3d = (model_type == "3d")

    def flush_batch():
        if not batch_patches:
            return
        arr = np.stack(batch_patches).astype(np.float32)
        if is_3d:
            arr = arr[:, None]                 # (B, 1, P, P, P)
        # else: (B, C, H, W) already — the Z-stack axis is the channel axis
        tensor = torch.from_numpy(arr).to(device)
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        batch_patches.clear()

    for z, y, x in tqdm(centers, desc="Detecting", leave=False):
        patch = tomo[z-hz:z-hz+ez, y-hy:y-hy+ey, x-hx:x-hx+ex]
        if patch.shape != extent:
            continue
        batch_patches.append(patch)
        if len(batch_patches) >= batch_size:
            flush_batch()
    flush_batch()

    if not all_probs:
        raise RuntimeError("No valid patches found. Tomogram may be too small for the patch size.")

    probs_flat = np.concatenate(all_probs, axis=0)  # (N_patches, n_classes)
    n_classes = probs_flat.shape[1]

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


def build_segmentation(heatmap: np.ndarray, zs, ys, xs, tomo_shape,
                       confidence_threshold: float) -> np.ndarray:
    """Turn the per-class heatmap into a full-resolution label volume.

    For each patch-center grid cell we take the argmax class (0 = background,
    1..N = features, matching the config order and the painting annotations).
    Cells whose top probability is below `confidence_threshold` are set to
    background. The coarse grid is then expanded to the tomogram's full shape by
    nearest-neighbour assignment, so the output is a uint8 volume the same shape
    as the tomogram — directly comparable to a painted annotations.npy.
    """
    n_classes = heatmap.shape[0]
    coarse = heatmap.argmax(axis=0).astype(np.uint8)      # (nz, ny, nx)
    top_prob = heatmap.max(axis=0)
    coarse[top_prob < confidence_threshold] = 0           # low confidence -> background

    def nearest_index(full_n, centers):
        coords = np.arange(full_n)
        pos = np.clip(np.searchsorted(centers, coords), 1, len(centers) - 1)
        left, right = centers[pos - 1], centers[pos]
        return np.where(np.abs(coords - left) <= np.abs(coords - right), pos - 1, pos)

    zi = nearest_index(tomo_shape[0], np.asarray(zs))
    yi = nearest_index(tomo_shape[1], np.asarray(ys))
    xi = nearest_index(tomo_shape[2], np.asarray(xs))
    seg = coarse[np.ix_(zi, yi, xi)]
    return seg.astype(np.uint8)


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
                     model_type: str = "3d", in_channels: int = 1,
                     save_heatmaps: bool = False,
                     save_segmentation: bool = True) -> list[dict]:
    tomo = np.load(tomo_path).astype(np.float32)
    heatmap, (zs, ys, xs) = detect(tomo, model, patch_size, stride_fraction, device,
                                   model_type=model_type, in_channels=in_channels)
    results = summarize(heatmap, zs, class_names, confidence_threshold, min_patches)

    if save_segmentation:
        seg = build_segmentation(heatmap, zs, ys, xs, tomo.shape, confidence_threshold)
        seg_path = tomo_path.parent / "segmentation.npy"
        np.save(seg_path, seg)
        present = ", ".join(f"{class_names[c]}={int((seg == c).sum())}"
                            for c in np.unique(seg) if c != 0) or "nothing above threshold"
        print(f"  Segmentation saved: {seg_path}  [{present}]")

    if save_heatmaps:
        hm_dir = tomo_path.parent / "heatmaps"
        hm_dir.mkdir(exist_ok=True)
        for c, name in enumerate(class_names):
            if name != "background":
                np.save(hm_dir / f"{name}.npy", heatmap[c])
        print(f"  Heatmaps saved: {hm_dir}/")

    return results


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
    parser.add_argument("--save-segmentation", action="store_true", default=True,
                        help="Save a full-size label volume segmentation.npy (default: on)")
    parser.add_argument("--no-segmentation", dest="save_segmentation",
                        action="store_false")
    parser.add_argument("--push-s3", action="store_true")
    parser.add_argument("--bucket", default=None)
    parser.add_argument("--aws-profile", default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    det_cfg = cfg["detection"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = Path(args.model_dir) / "patch_classifier.pth"
    model, class_names, patch_size, model_type, in_channels = load_model(ckpt_path, device)
    geom_str = (f"{patch_size}^3" if model_type == "3d"
                else f"{in_channels}×{patch_size}² ({model_type})")
    print(f"Model loaded: {ckpt_path}  classes={class_names}  patch={geom_str}")

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
            model_type=model_type, in_channels=in_channels,
            save_heatmaps=args.save_heatmaps,
            save_segmentation=args.save_segmentation,
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
