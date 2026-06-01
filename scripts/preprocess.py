"""
Rescale, low-pass filter, and normalize raw MRC/NRRD tomograms, saving as .npy.

Processing order (per tomogram):
    1. load raw volume (and its voxel/pixel size)
    2. rescale           (resample to a common output pixel size, default 20 A)
    3. low-pass filter   (parameters from configs/config.yaml -> preprocess.lowpass)
    4. normalize         (clip to percentiles, then z-score)

Rescaling brings tomograms collected at different magnifications onto a common
voxel size so a single model sees features at a consistent physical scale. The
input pixel size is read from the MRC header by default; override it (or supply
it for headerless .npy inputs) via config `preprocess.rescale.input_pixel_size`
or the --input-pixel-size flag. The target is `preprocess.rescale.output_pixel_size`
(default 20 A), overridable with --output-pixel-size.

Input layout (--input-dir):
    <run_name>/tomogram.mrc    (or .rec, .nrrd)
    -- OR --
    flat directory of .mrc files

Output (--output-dir):
    <run_name>/tomogram.npy    float32, rescaled, low-pass filtered, z-scored/clipped

Usage:
    python preprocess.py --input-dir data/raw/ --output-dir data/processed/
    python preprocess.py --input-dir data/raw/ --output-dir data/processed/ --output-pixel-size 20
    python preprocess.py --input-dir data/raw/ --output-dir data/processed/ --input-pixel-size 13.5
    python preprocess.py --input-dir data/raw/ --output-dir data/processed/ --dry-run
"""

import argparse
from pathlib import Path

import mrcfile
import numpy as np
import yaml


# Fallback used only if the config has no preprocess section.
DEFAULT_CLIP_PERCENTILE = (5, 95)
DEFAULT_OUTPUT_PIXEL_SIZE = 20.0  # Angstroms; common target voxel size


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


def load_mrc(path: Path) -> tuple[np.ndarray, tuple | None]:
    """Return (volume, voxel_size) where voxel_size is (z, y, x) in Angstroms.

    voxel_size is None if the header carries no usable scale (all zeros).
    """
    with mrcfile.open(str(path), mode="r", permissive=True) as mrc:
        data = mrc.data.copy()
        vs = mrc.voxel_size  # record with .x .y .z in Angstroms
        voxel = (float(vs.z), float(vs.y), float(vs.x))
    if not all(v > 0 for v in voxel):
        voxel = None
    return data, voxel


def load_nrrd(path: Path) -> tuple[np.ndarray, tuple | None]:
    import nrrd
    data, header = nrrd.read(str(path))
    voxel = None
    # NRRD stores spacing on the diagonal of "space directions" when present.
    dirs = header.get("space directions")
    if dirs is not None:
        try:
            spac = [float(np.linalg.norm(np.asarray(d, dtype=float)))
                    for d in dirs if d is not None]
            if len(spac) == 3 and all(s > 0 for s in spac):
                voxel = (spac[0], spac[1], spac[2])
        except Exception:
            voxel = None
    return data, voxel


def load_volume(path: Path) -> tuple[np.ndarray, tuple | None]:
    ext = path.suffix.lower()
    if ext in (".mrc", ".rec"):
        return load_mrc(path)
    elif ext == ".nrrd":
        return load_nrrd(path)
    else:
        raise ValueError(f"Unsupported format: {ext}")


def rescale(vol: np.ndarray, input_px, output_px: float, order: int = 1) -> np.ndarray:
    """Resample `vol` from `input_px` to `output_px` (Angstroms/voxel).

    `input_px` may be a scalar (isotropic) or a per-axis (z, y, x) tuple. The
    zoom factor per axis is input_px / output_px (>1 upsamples, <1 downsamples).
    Returns float32.
    """
    from scipy.ndimage import zoom
    vol = vol.astype(np.float32)
    if np.isscalar(input_px):
        in_px = (float(input_px),) * 3
    else:
        in_px = tuple(float(v) for v in input_px)
    factors = tuple(ip / float(output_px) for ip in in_px)
    if all(abs(f - 1.0) < 1e-3 for f in factors):
        return vol  # already at target scale
    return zoom(vol, factors, order=order).astype(np.float32)


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
            lowpass_cfg: dict, clip_percentile,
            rescale_cfg: dict, input_px_override=None, output_px_override=None):
    raw, voxel = load_volume(tomo_path)

    # 1) rescale to a common output pixel size (if enabled)
    rescale_enabled = rescale_cfg.get("enabled", False)
    output_px = output_px_override or rescale_cfg.get("output_pixel_size", DEFAULT_OUTPUT_PIXEL_SIZE)
    # Resolve the input pixel size: explicit override > config > MRC/NRRD header.
    input_px = input_px_override or rescale_cfg.get("input_pixel_size", None) or voxel
    if rescale_enabled:
        if input_px is None:
            print(f"  {tomo_path.name}  ⚠ no input pixel size (header has none and "
                  f"none provided) — skipping rescale. Set preprocess.rescale.input_pixel_size "
                  f"or pass --input-pixel-size.")
        else:
            in_str = (f"{input_px:.3f}" if np.isscalar(input_px)
                      else "(" + ",".join(f"{v:.3f}" for v in input_px) + ")")
            before = raw.shape
            raw = rescale(raw, input_px, output_px, order=int(rescale_cfg.get("order", 1)))
            print(f"  {tomo_path.name}  rescale {in_str}A → {output_px}A/vox   "
                  f"{before} → {raw.shape}")

    # 2) low-pass filter the (rescaled) volume, then 3) normalize.
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
    parser.add_argument("--input-pixel-size", type=float, default=None,
                        help="Input voxel size in Angstroms (overrides header/config). "
                             "Required for headerless .npy inputs when rescaling.")
    parser.add_argument("--output-pixel-size", type=float, default=None,
                        help=f"Target voxel size in Angstroms (default from config or "
                             f"{DEFAULT_OUTPUT_PIXEL_SIZE}).")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pre_cfg = cfg.get("preprocess", {}) or {}
    lowpass_cfg = pre_cfg.get("lowpass", {}) or {}
    clip_percentile = tuple(pre_cfg.get("clip_percentile", DEFAULT_CLIP_PERCENTILE))
    rescale_cfg = pre_cfg.get("rescale", {}) or {}

    if rescale_cfg.get("enabled", False):
        out_px = args.output_pixel_size or rescale_cfg.get("output_pixel_size", DEFAULT_OUTPUT_PIXEL_SIZE)
        in_src = ("--input-pixel-size" if args.input_pixel_size
                  else ("config" if rescale_cfg.get("input_pixel_size") else "MRC/NRRD header"))
        print(f"Rescale: enabled  →  {out_px} A/voxel  (input pixel size from {in_src})")
    else:
        print("Rescale: disabled")

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
                lowpass_cfg=lowpass_cfg, clip_percentile=clip_percentile,
                rescale_cfg=rescale_cfg,
                input_px_override=args.input_pixel_size,
                output_px_override=args.output_pixel_size)

    print("\nDone." if not args.dry_run else "\nDry run complete.")


if __name__ == "__main__":
    main()
