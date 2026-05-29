"""
Extract training patches from painted annotations.

For each annotated run:
  - For each feature class, randomly sample patch centers from painted voxels
  - Extract 3D cubic patches from the tomogram around those centers
  - Also sample background patches from unpainted regions
  - Apply augmentation to balance and diversify the dataset

Output:
    patches.npz — {'patches': (N, 1, P, P, P) float32,
                   'labels':  (N,) int64,
                   'class_names': [...]}

Usage:
    python extract_patches.py --data-dir data/processed/ --output patches.npz
    python extract_patches.py --data-dir data/processed/ --output patches.npz --push-s3
"""

import argparse
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm


def load_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def extract_patch(vol: np.ndarray, z: int, y: int, x: int, size: int) -> np.ndarray | None:
    """Extract a cubic patch centered at (z, y, x). Returns None if out of bounds."""
    h = size // 2
    z0, z1 = z - h, z + size - h
    y0, y1 = y - h, y + size - h
    x0, x1 = x - h, x + size - h
    if z0 < 0 or y0 < 0 or x0 < 0:
        return None
    if z1 > vol.shape[0] or y1 > vol.shape[1] or x1 > vol.shape[2]:
        return None
    return vol[z0:z1, y0:y1, x0:x1].copy()


def augment(patch: np.ndarray) -> list[np.ndarray]:
    """Return the patch plus several augmented versions."""
    results = [patch]

    # All 3 flip axes
    for ax in range(3):
        results.append(np.flip(patch, axis=ax).copy())

    # 90-degree rotations in XY plane
    for k in (1, 2, 3):
        results.append(np.rot90(patch, k=k, axes=(1, 2)).copy())

    # Gaussian noise
    noisy = patch + np.random.normal(0, np.random.uniform(0.02, 0.08), patch.shape).astype(np.float32)
    results.append(noisy)

    # Intensity scale + shift
    scaled = patch * np.random.uniform(0.85, 1.15) + np.random.uniform(-0.1, 0.1)
    results.append(scaled.astype(np.float32))

    return results


def sample_centers(mask: np.ndarray, n: int, patch_size: int) -> np.ndarray:
    """
    Randomly sample n voxel coordinates from mask==True,
    ensuring each center is far enough from the volume boundary for a full patch.
    """
    h = patch_size // 2
    zs, ys, xs = np.where(mask)

    # Filter to valid centers (enough margin for a full patch)
    valid = (
        (zs >= h) & (zs < mask.shape[0] - patch_size + h) &
        (ys >= h) & (ys < mask.shape[1] - patch_size + h) &
        (xs >= h) & (xs < mask.shape[2] - patch_size + h)
    )
    zs, ys, xs = zs[valid], ys[valid], xs[valid]

    if len(zs) == 0:
        return np.empty((0, 3), dtype=int)

    idx = np.random.choice(len(zs), size=min(n, len(zs)), replace=False)
    return np.stack([zs[idx], ys[idx], xs[idx]], axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="patches.npz")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--augment", action="store_true", default=True,
                        help="Apply augmentation (default: on)")
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--push-s3", action="store_true")
    parser.add_argument("--bucket", default=None)
    parser.add_argument("--profile", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    features = cfg["features"]
    feature_names = ["background"] + [f["name"] for f in features]
    patch_size = cfg["patches"]["size"]
    per_class = cfg["patches"]["per_class"]
    bg_ratio = cfg["patches"]["background_ratio"]
    n_bg = int(per_class * bg_ratio)

    data_dir = Path(args.data_dir)
    annotated_runs = sorted([
        p for p in data_dir.iterdir()
        if p.is_dir() and (p / "tomogram.npy").exists() and (p / "annotations.npy").exists()
    ])

    if not annotated_runs:
        print(f"No annotated runs found in {data_dir}")
        print("Run paint_annotations.py first.")
        return

    print(f"Found {len(annotated_runs)} annotated run(s): {[p.name for p in annotated_runs]}")

    all_patches, all_labels = [], []

    for run_dir in tqdm(annotated_runs, desc="Runs"):
        tomo = np.load(run_dir / "tomogram.npy").astype(np.float32)
        ann = np.load(run_dir / "annotations.npy").astype(np.uint8)

        n_classes = len(features)

        # Feature classes (1..N)
        for class_id in range(1, n_classes + 1):
            class_mask = ann == class_id
            n_painted = class_mask.sum()
            if n_painted == 0:
                print(f"  [{run_dir.name}] No painted voxels for class {class_id} "
                      f"({feature_names[class_id]}) — skipping")
                continue

            centers = sample_centers(class_mask, per_class, patch_size)
            print(f"  [{run_dir.name}] {feature_names[class_id]}: "
                  f"{n_painted} painted voxels → {len(centers)} patch centers")

            for z, y, x in centers:
                patch = extract_patch(tomo, z, y, x, patch_size)
                if patch is None:
                    continue
                variants = augment(patch) if args.augment else [patch]
                all_patches.extend(variants)
                all_labels.extend([class_id] * len(variants))

        # Background class (0): sample from regions painted as 0 AND far from any annotation
        from scipy.ndimage import binary_dilation
        any_annotation = ann > 0
        # Dilate to avoid sampling too close to painted regions
        dilated = binary_dilation(any_annotation, iterations=patch_size // 2)
        bg_mask = ~dilated
        bg_centers = sample_centers(bg_mask, n_bg, patch_size)
        print(f"  [{run_dir.name}] background: {bg_mask.sum()} valid voxels → {len(bg_centers)} patches")

        for z, y, x in bg_centers:
            patch = extract_patch(tomo, z, y, x, patch_size)
            if patch is None:
                continue
            variants = augment(patch) if args.augment else [patch]
            all_patches.extend(variants)
            all_labels.extend([0] * len(variants))

    if not all_patches:
        print("No patches extracted. Check annotations.")
        return

    patches_arr = np.stack(all_patches)[:, None, ...]  # (N, 1, P, P, P)
    labels_arr = np.array(all_labels, dtype=np.int64)

    # Shuffle
    idx = np.random.permutation(len(patches_arr))
    patches_arr = patches_arr[idx]
    labels_arr = labels_arr[idx]

    # Stats
    print(f"\nTotal patches: {len(patches_arr)}  shape={patches_arr.shape}")
    for c, name in enumerate(feature_names):
        n = (labels_arr == c).sum()
        print(f"  class {c} ({name}): {n} patches")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, patches=patches_arr, labels=labels_arr,
                        class_names=np.array(feature_names))
    print(f"\nSaved: {out}")

    if args.push_s3:
        from aws_utils import ensure_bucket, upload, DEFAULT_BUCKET
        bucket = args.bucket or DEFAULT_BUCKET
        ensure_bucket(bucket, args.profile)
        upload(out, f"training/{out.name}", bucket=bucket, profile=args.profile)

    print("Next: python train_patch_classifier.py --patches patches.npz")


if __name__ == "__main__":
    main()
