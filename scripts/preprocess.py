"""
Normalize raw MRC/NRRD tomograms and save as .npy for fast loading.

Input layout (--input-dir):
    <run_name>/tomogram.mrc    (or .rec, .nrrd)
    -- OR --
    flat directory of .mrc files

Output (--output-dir):
    <run_name>/tomogram.npy    float32, z-scored and clipped

Usage:
    python preprocess.py --input-dir data/raw/ --output-dir data/processed/
    python preprocess.py --input-dir data/raw/ --output-dir data/processed/ --dry-run
"""

import argparse
from pathlib import Path

import mrcfile
import numpy as np


CLIP_PERCENTILE = (5, 95)


def load_mrc(path: Path) -> np.ndarray:
    with mrcfile.open(str(path), mode="r", permissive=True) as mrc:
        return mrc.data.copy()


def load_nrrd(path: Path) -> np.ndarray:
    import nrrd
    data, _ = nrrd.read(str(path))
    return data


def load_volume(path: Path) -> np.ndarray:
    ext = path.suffix.lower()
    if ext in (".mrc", ".rec"):
        return load_mrc(path)
    elif ext == ".nrrd":
        return load_nrrd(path)
    else:
        raise ValueError(f"Unsupported format: {ext}")


def normalize(vol: np.ndarray) -> np.ndarray:
    vol = vol.astype(np.float32)
    lo, hi = np.percentile(vol, CLIP_PERCENTILE)
    vol = np.clip(vol, lo, hi)
    std = vol.std()
    return (vol - vol.mean()) / std if std > 0 else vol - vol.mean()


def find_tomogram(run_dir: Path) -> Path | None:
    for name in ("tomogram.mrc", "tomogram.rec", "tomogram.nrrd"):
        if (run_dir / name).exists():
            return run_dir / name
    # Fall back: any .mrc in the directory
    candidates = list(run_dir.glob("*.mrc")) + list(run_dir.glob("*.rec"))
    return candidates[0] if candidates else None


def process(tomo_path: Path, out_path: Path, dry_run: bool):
    vol = normalize(load_volume(tomo_path))
    print(f"  {tomo_path.name}  shape={vol.shape}  mean={vol.mean():.3f}  std={vol.std():.3f}")
    if not dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, vol)
        print(f"  → {out_path}")
    else:
        print(f"  → [dry-run] {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Support two layouts: subdirectory-per-run OR flat directory of .mrc files
    run_dirs = sorted([p for p in input_dir.iterdir() if p.is_dir()])
    flat_mrc = sorted(input_dir.glob("*.mrc")) + sorted(input_dir.glob("*.rec"))

    tasks: list[tuple[Path, Path]] = []

    if run_dirs:
        for run_dir in run_dirs:
            tomo = find_tomogram(run_dir)
            if tomo:
                tasks.append((tomo, output_dir / run_dir.name / "tomogram.npy"))
    elif flat_mrc:
        for mrc in flat_mrc:
            tasks.append((mrc, output_dir / mrc.stem / "tomogram.npy"))
    else:
        print(f"No tomograms found in {input_dir}")
        return

    print(f"Found {len(tasks)} tomogram(s)")
    for tomo_path, out_path in tasks:
        process(tomo_path, out_path, dry_run=args.dry_run)

    print("\nDone." if not args.dry_run else "\nDry run complete.")


if __name__ == "__main__":
    main()
