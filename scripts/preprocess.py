"""
Low-pass filter and normalize raw MRC/NRRD tomograms, saving as .npy.

Processing order (per tomogram):
    1. load raw volume
    2. low-pass filter   (parameters from configs/config.yaml -> preprocess.lowpass)
    3. normalize         (clip to percentiles, then z-score)

Input layout (--input-dir):
    <run_name>/tomogram.mrc    (or .rec, .nrrd)
    -- OR --
    flat directory of .mrc files

Output (--output-dir):
    <run_name>/tomogram.npy    float32, low-pass filtered then z-scored and clipped

Usage:
    python preprocess.py --input-dir data/raw/ --output-dir data/processed/
    python preprocess.py --input-dir data/raw/ --output-dir data/processed/ --dry-run
"""

import argparse
from pathlib import Path

import mrcfile
import numpy as np
import yaml


# Fallback used only if the config has no preprocess section.
DEFAULT_CLIP_PERCENTILE = (5, 95)


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        print(f"Config not found ({path}); using defaults (no low-pass filter).")
        return {}
    return yaml.safe_load(p.read_text()) or {}


def lowpass_filter(vol: np.ndarray, cfg: dict) -> np.ndarray:
    """Apply a low-pass filter to the raw volume according to config.

    cfg is the `preprocess.lowpass` mapping. Returns float32. If disabled or
    absent, the volume is returned unchanged (as float32).
    """
    vol = vol.astype(np.float32)
    if not cfg or not cfg.get("enabled", False):
        return vol

    method = str(cfg.get("method", "gaussian")).lower()

    if method == "gaussian":
        from scipy.ndimage import gaussian_filter
        sigma = float(cfg.get("sigma", 1.5))
        if sigma <= 0:
            return vol
        return gaussian_filter(vol, sigma=sigma).astype(np.float32)

    if method == "fourier":
        # Butterworth low-pass in the Fourier domain. `cutoff` is a fraction of
        # Nyquist (0-1); `order` controls how sharp the roll-off is.
        cutoff = float(cfg.get("cutoff", 0.25))
        order = float(cfg.get("order", 2))
        cutoff = min(max(cutoff, 1e-3), 1.0)

        # Radial frequency grid (0 at DC, 1.0 at Nyquist along each axis).
        axes = [np.fft.fftfreq(n) * 2.0 for n in vol.shape]  # *2 -> Nyquist = 1.0
        grids = np.meshgrid(*axes, indexing="ij")
        radius = np.sqrt(sum(g ** 2 for g in grids))
        butter = 1.0 / (1.0 + (radius / cutoff) ** (2 * order))

        spec = np.fft.fftn(vol)
        filtered = np.fft.ifftn(spec * butter).real
        return filtered.astype(np.float32)

    raise ValueError(f"Unknown lowpass method: {method!r} (use 'gaussian' or 'fourier')")


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


def normalize(vol: np.ndarray, clip_percentile=DEFAULT_CLIP_PERCENTILE) -> np.ndarray:
    vol = vol.astype(np.float32)
    lo, hi = np.percentile(vol, clip_percentile)
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


def process(tomo_path: Path, out_path: Path, dry_run: bool,
            lowpass_cfg: dict, clip_percentile):
    raw = load_volume(tomo_path)
    # 1) low-pass filter the raw volume, then 2) normalize.
    filtered = lowpass_filter(raw, lowpass_cfg)
    vol = normalize(filtered, clip_percentile)
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
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pre_cfg = cfg.get("preprocess", {}) or {}
    lowpass_cfg = pre_cfg.get("lowpass", {}) or {}
    clip_percentile = tuple(pre_cfg.get("clip_percentile", DEFAULT_CLIP_PERCENTILE))

    if lowpass_cfg.get("enabled", False):
        method = lowpass_cfg.get("method", "gaussian")
        detail = (f"sigma={lowpass_cfg.get('sigma', 1.5)}" if method == "gaussian"
                  else f"cutoff={lowpass_cfg.get('cutoff', 0.25)}, order={lowpass_cfg.get('order', 2)}")
        print(f"Low-pass filter: {method} ({detail})  →  normalize (clip {clip_percentile})")
    else:
        print(f"Low-pass filter: disabled  →  normalize (clip {clip_percentile})")

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
        process(tomo_path, out_path, dry_run=args.dry_run,
                lowpass_cfg=lowpass_cfg, clip_percentile=clip_percentile)

    print("\nDone." if not args.dry_run else "\nDry run complete.")


if __name__ == "__main__":
    main()
