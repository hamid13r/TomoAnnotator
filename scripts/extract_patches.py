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


def resolve_geometry(cfg: dict) -> dict:
    """Work out patch geometry from the config `model` section.

    Returns a dict with:
        type     : "3d" | "2d" | "2.5d"
        extent   : (ez, ey, ex)  full patch size per axis (voxels)
        half     : (hz, hy, hx)  offset of patch start from the center
    For 2d / 2.5d the in-plane size is box_2d and the Z depth is n_slices
    (1 for pure 2d). For 3d it is a cube of patches.size.
    """
    model_cfg = cfg.get("model", {}) or {}
    mtype = str(model_cfg.get("type", "3d")).lower()

    if mtype == "3d":
        p = cfg["patches"]["size"]
        extent = (p, p, p)
    elif mtype in ("2d", "2.5d"):
        box = int(model_cfg.get("box_2d", 96))
        n_slices = 1 if mtype == "2d" else int(model_cfg.get("n_slices", 5))
        n_slices = max(1, n_slices)
        extent = (n_slices, box, box)
    else:
        raise ValueError(f"Unknown model.type {mtype!r} (use 3d, 2d, or 2.5d)")

    half = tuple(e // 2 for e in extent)
    return {"type": mtype, "extent": extent, "half": half}


def extract_patch(vol: np.ndarray, z: int, y: int, x: int,
                  extent, half) -> np.ndarray | None:
    """Extract a patch centered at (z, y, x).

    Returns a 3D array of shape `extent`. For 2d / 2.5d the first axis is the
    Z-slice stack (depth n_slices); for 3d it is a cube. None if out of bounds.
    """
    ez, ey, ex = extent
    hz, hy, hx = half
    z0, z1 = z - hz, z - hz + ez
    y0, y1 = y - hy, y - hy + ey
    x0, x1 = x - hx, x - hx + ex
    if z0 < 0 or y0 < 0 or x0 < 0:
        return None
    if z1 > vol.shape[0] or y1 > vol.shape[1] or x1 > vol.shape[2]:
        return None
    return vol[z0:z1, y0:y1, x0:x1].copy()


def augment(patch: np.ndarray) -> list[np.ndarray]:
    """Return the patch plus several augmented versions.

    Works for both cube patches (3d) and slice-stack patches (2.5d/2d): the
    last two axes are always the in-plane (Y, X) dimensions, so flips and
    rotations are applied there. The Z/stack axis is never flipped (its
    ordering is physically meaningful and anisotropic in cryoET).
    """
    results = [patch]

    # In-plane flips (Y, X)
    for ax in (1, 2):
        results.append(np.flip(patch, axis=ax).copy())

    # 90-degree rotations in the in-plane (Y, X)
    for k in (1, 2, 3):
        results.append(np.rot90(patch, k=k, axes=(1, 2)).copy())

    # Gaussian noise
    noisy = patch + np.random.normal(0, np.random.uniform(0.02, 0.08), patch.shape).astype(np.float32)
    results.append(noisy)

    # Intensity scale + shift
    scaled = patch * np.random.uniform(0.85, 1.15) + np.random.uniform(-0.1, 0.1)
    results.append(scaled.astype(np.float32))

    return results


def sample_centers(mask: np.ndarray, n: int, extent, half) -> np.ndarray:
    """Randomly sample up to n voxel coordinates from mask==True that leave
    enough margin on every axis for a full patch of the given extent."""
    ez, ey, ex = extent
    hz, hy, hx = half
    zs, ys, xs = np.where(mask)

    valid = (
        (zs >= hz) & (zs < mask.shape[0] - ez + hz + 1) &
        (ys >= hy) & (ys < mask.shape[1] - ey + hy + 1) &
        (xs >= hx) & (xs < mask.shape[2] - ex + hx + 1)
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
    per_class = cfg["patches"]["per_class"]
    bg_ratio = cfg["patches"]["background_ratio"]
    n_bg = int(per_class * bg_ratio)

    geom = resolve_geometry(cfg)
    extent, half = geom["extent"], geom["half"]
    print(f"Model type: {geom['type']}   patch extent (Z,Y,X)={extent}")

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

    # How many variants augment() produces per patch (probe once with a dummy).
    aug_factor = len(augment(np.zeros(extent, dtype=np.float32))) if args.augment else 1
    voxels_per_patch = int(np.prod(extent))
    bytes_per_patch = voxels_per_patch * 4  # float32

    # Rough UPPER bound: assumes every class hits its per_class cap in every run.
    n_feat = len(features)
    max_patches = len(annotated_runs) * (n_feat * per_class + n_bg) * aug_factor
    est_gb = max_patches * bytes_per_patch / 1e9
    print(f"Augmentation factor: {aug_factor}×   patch size: {voxels_per_patch:,} voxels "
          f"({bytes_per_patch/1e6:.2f} MB each)")
    print(f"Upper-bound estimate: ≤ {max_patches:,} patches  →  ~{est_gb:.2f} GB in memory "
          f"(actual depends on how much you painted)")
    if est_gb > 8:
        print(f"  ⚠ That is large. Consider lowering patches.per_class / background_ratio, "
              f"disabling --no-augment, or reducing box size.")

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

            centers = sample_centers(class_mask, per_class, extent, half)
            print(f"  [{run_dir.name}] {feature_names[class_id]}: "
                  f"{n_painted} painted voxels → {len(centers)} patch centers")

            for z, y, x in centers:
                patch = extract_patch(tomo, z, y, x, extent, half)
                if patch is None:
                    continue
                variants = augment(patch) if args.augment else [patch]
                all_patches.extend(variants)
                all_labels.extend([class_id] * len(variants))

        # Background class (0): sample from regions painted as 0 AND far from any annotation
        from scipy.ndimage import binary_dilation
        any_annotation = ann > 0
        # Dilate to avoid sampling too close to painted regions
        dilated = binary_dilation(any_annotation, iterations=max(half))
        bg_mask = ~dilated
        bg_centers = sample_centers(bg_mask, n_bg, extent, half)
        print(f"  [{run_dir.name}] background: {bg_mask.sum()} valid voxels → {len(bg_centers)} patches")

        for z, y, x in bg_centers:
            patch = extract_patch(tomo, z, y, x, extent, half)
            if patch is None:
                continue
            variants = augment(patch) if args.augment else [patch]
            all_patches.extend(variants)
            all_labels.extend([0] * len(variants))

    if not all_patches:
        print("No patches extracted. Check annotations.")
        return

    # 3d  -> add a channel axis: (N, 1, P, P, P)
    # 2d/2.5d -> the Z-stack axis IS the channel axis: (N, C, H, W)
    stacked = np.stack(all_patches)
    if geom["type"] == "3d":
        patches_arr = stacked[:, None, ...]        # (N, 1, P, P, P)
    else:
        patches_arr = stacked                       # (N, n_slices, box, box)
    labels_arr = np.array(all_labels, dtype=np.int64)

    # Shuffle
    idx = np.random.permutation(len(patches_arr))
    patches_arr = patches_arr[idx]
    labels_arr = labels_arr[idx]

    # Stats
    mem_gb = patches_arr.nbytes / 1e9
    print(f"\nTotal patches: {len(patches_arr)}  shape={patches_arr.shape}  "
          f"({mem_gb:.2f} GB float32 in memory)")
    for c, name in enumerate(feature_names):
        n = (labels_arr == c).sum()
        print(f"  class {c} ({name}): {n} patches")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, patches=patches_arr, labels=labels_arr,
                        class_names=np.array(feature_names),
                        model_type=np.array(geom["type"]),
                        extent=np.array(extent),
                        n_slices=np.array(extent[0]),
                        box=np.array(extent[1]))
    print(f"\nSaved: {out}  (model_type={geom['type']}, shape={patches_arr.shape})")

    if args.push_s3:
        from aws_utils import ensure_bucket, upload, DEFAULT_BUCKET
        bucket = args.bucket or DEFAULT_BUCKET
        ensure_bucket(bucket, args.profile)
        upload(out, f"training/{out.name}", bucket=bucket, profile=args.profile)

    print("Next: python train_patch_classifier.py --patches patches.npz")


if __name__ == "__main__":
    main()
