# TomoAnnotator — From Idea to Working Tool

**Grotjahn Lab · Scripps Research Hackathon 2026 (May 29 – June 1)**
Built with Claude: an idea-to-execution log for detecting organelles in cryo-ET tomograms.

This document walks the project from the original planning, through how it was
executed, to concrete examples you can run.

---

## 1. Planning

### The problem

Cryo-electron tomography (cryo-ET) produces noisy 3D maps of frozen cells — each
tomogram here is roughly a 250 × 720 × 512 voxel volume. Today a biologist scrolls
through slices by hand asking, for every tomogram: *is there mitochondria here? ER?
microtubules?* That manual screening is slow, requires expert eyes, and is the
bottleneck before any downstream analysis. With dozens of tomograms per session, it
does not scale.

### The goal

A tool that reports **which organelles are present in each tomogram, automatically** —
without requiring complete, pixel-perfect segmentations to get started.

### The core idea

Instead of full segmentation, lean on weak supervision:

1. **Paint** — a biologist paints a few examples of each feature in just 1–2 tomograms.
2. **Learn** — extract patches around the painted voxels and train a small CNN in minutes.
3. **Detect** — slide the trained model over new tomograms to produce a presence/absence report.
4. **Report** — Claude (on Bedrock) summarizes which tomograms are most interesting.

A few dozen brush strokes per class is enough to bootstrap detection.

### Key design decisions

- **Config-driven feature classes.** The list of organelles lives in
  `configs/config.yaml`; adding or removing a class requires no code changes. Ships
  with 7: mitochondria, ER, microtubules, vesicles, ribosomes, nuclear envelope, filaments.
- **Three model modes from one config line.** `2d`, `2.5d` (default), and `3d`. The
  2.5D mode stacks adjacent Z-slices as input channels — it keeps large in-plane
  context while sidestepping the worst of the cryo-ET missing-wedge artifact along Z.
- **GPU-optional.** Everything but training runs on a CPU laptop. Training runs on a
  GPU (Garibaldi via SLURM, or an EC2 `g4dn` instance).
- **Memory-conscious storage.** Patches are stored as **float16** by default to roughly
  halve memory and disk, so ~2× more patches fit for the same footprint. A
  `--dtype float32` flag restores full precision. Training upcasts each batch to
  float32 on the fly, so either choice trains identically.

---

## 2. Execution

### The pipeline

```
Raw MRC tomograms
      │
      ▼
preprocess.py             → data/processed/<run>/tomogram.npy
      │                      (low-pass filter, then normalize)
      ▼
paint_annotations.py      → data/processed/<run>/annotations.npy
      │                      (lightweight matplotlib painting viewer)
      ▼
extract_patches.py        → patches.npz
      │                      (balanced patch sampling + augmentation; 2D/2.5D/3D)
      ▼
train_patch_classifier.py → models/patch_classifier.pth
      │                      (small CNN, ~5–15 min on one GPU)
      ▼
detect_features.py        → results/predictions.csv  (+ segmentation.npy)
      │                      (sliding window over new tomograms)
      ▼
report.py                 → results/report.md
                             (Claude on Bedrock summarizes findings)
```

### What each stage does

- **preprocess.py** — Low-pass filters each raw volume (Gaussian blur or Butterworth
  Fourier low-pass), then clips to percentiles and z-scores. Writes one run folder per
  tomogram.
- **paint_annotations.py** — A matplotlib viewer (no napari needed). Scroll/slider to
  move through slices, left-click-drag to paint the selected class, number keys to pick
  a class, brush-radius slider, erase and undo. Saves a uint8 `annotations.npy` the same
  shape as the tomogram (`0` = background, `1..N` = classes).
- **extract_patches.py** — Samples patch centers from painted voxels (balanced per class)
  plus background patches drawn far from any annotation, applies in-plane flips/rotations
  and intensity/noise augmentation, and stores the result. Patch shape follows the model
  mode. **Stored as float16 by default** (`--dtype float32` to override).
- **train_patch_classifier.py** — Picks the CNN architecture from the patch tensor rank
  (3D CNN for cubic patches, 2D CNN with stacked-slice channels for 2D/2.5D). Class-
  weighted cross-entropy handles imbalance. Keeps patches in their stored dtype and
  upcasts each batch to float32 on device.
- **detect_features.py** — Slides the model across new tomograms, thresholds confidence,
  and writes a `predictions.csv` plus a `segmentation.npy` label volume (same format as a
  painted annotation, so you can re-open it in the painter to correct and retrain).
- **view_segmentation.py** — Read-only overlay viewer to inspect what was classified as
  what, in the same style as the painter.
- **report.py** — Sends the predictions to Claude on Bedrock for a plain-English summary.

### How Claude helped

- Scaffolded the full 9-script pipeline and the config-driven class system.
- Wrote the harder pieces: the matplotlib painting viewer, balanced patch sampling with
  margin-aware center selection, and the 2D/2.5D/3D CNNs that swap with one config line.
- Set up the infrastructure: GPU and CPU conda envs, SLURM job scripts, S3 sync, and an
  EC2 GPU launch recipe.
- Added the float16 storage optimization (with the `--dtype` flag) so more patches fit
  in memory, and verified the round-trip end-to-end on real lab data.

---

## 3. Examples

> Run all commands from the project root (the folder containing `configs/`).

### Setup

```bash
# GPU machine (NVIDIA):
conda env create -f environment.yml
# CPU-only laptop:
conda env create -f environment-cpu.yml
conda activate tomoannotator
```

### 1. Preprocess

```bash
python scripts/preprocess.py --input-dir data/raw/ --output-dir data/processed/
# Preview without writing:
python scripts/preprocess.py --input-dir data/raw/ --output-dir data/processed/ --dry-run
```

### 2. Paint annotations

```bash
python scripts/paint_annotations.py --data-dir data/processed/ --run run_001
```

Paint a few examples of each feature in 1–2 tomograms. Controls: scroll / Z slider to
move through slices, left-click-drag to paint, number keys `0–N` to pick a class
(`0` = erase), `[` / `]` for brush size, `u` to undo, Save to write `annotations.npy`.

### 3. Extract patches

```bash
# Default: float16 storage (fits ~2x more patches)
python scripts/extract_patches.py --data-dir data/processed/ --output patches.npz

# Full precision instead:
python scripts/extract_patches.py --data-dir data/processed/ --output patches.npz --dtype float32

# No augmentation:
python scripts/extract_patches.py --data-dir data/processed/ --output patches.npz --no-augment
```

The script prints per-class patch counts and an estimated memory footprint — check that
every class you painted has patches before training. Example output on real lab data
(`MIM019_2_lam11_ts_003`, 2.5D mode):

```
Model type: 2.5d   patch extent (Z,Y,X)=(5, 96, 96)   storage dtype: float16
Total patches: 1024  shape=(1024, 5, 96, 96)  dtype=float16  (0.09 GB in memory)
```

The same run at `--dtype float32` produces a 0.19 GB array — exactly 2× the size.

### 4. Train

```bash
# Local GPU:
python scripts/train_patch_classifier.py --patches patches.npz --output-dir models/

# On Garibaldi (recommended):
sbatch slurm/train_gpu.slurm
```

Watch per-class validation accuracy. If a class stays near 0, paint more examples of it.

### 5. Detect

```bash
# All runs → CSV:
python scripts/detect_features.py --data-dir data/processed/ --output-csv results/predictions.csv

# Single tomogram with probability heatmaps (shows WHERE each feature was found):
python scripts/detect_features.py --tomogram data/processed/new_run/tomogram.npy --save-heatmaps

# Inspect the overlay:
python scripts/view_segmentation.py --run-dir data/processed/new_run
```

### 6. Generate the Claude report

```bash
# Laptop (SSO profile):
python scripts/report.py --predictions results/predictions.csv --profile <your-sso-profile>

# EC2 (instance profile):
python scripts/report.py --predictions results/predictions.csv --push-s3
```

### A/B the model modes on the same painted data

Change `model.type` in `configs/config.yaml` (`2d` | `2.5d` | `3d`), then re-run
`extract_patches.py → train_patch_classifier.py → detect_features.py`. No code or flag
changes needed — the choice is baked into `patches.npz` and carried through automatically.

---

## Results so far

- Working end-to-end pipeline on real Grotjahn Lab tomograms.
- 7 extensible organelle classes, configurable without code changes.
- Training in **minutes**, not days; paint **1–2** tomograms, auto-detect the rest.
- float16 patch storage lets you hold ~2× more training data per machine.

## Next steps

- More painted data and validation against expert labels.
- napari integration for painting and review.
- Quantitative accuracy benchmarks per organelle class.

---

*Reference: Medina, Rahmani et al. "Surface Morphometrics reveals local membrane
thickness variation in organellar subcompartments." J Cell Biol 2025. PMID: 41474626*
