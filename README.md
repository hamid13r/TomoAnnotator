# Grotjahn Lab — Tomogram Feature Detection

**Hackathon project, Scripps Research 2026 (May 29 – June 1)**
**Project lead:** Michaela Medina, Grotjahn Lab

Automatically flag which cellular features (mitochondria, ER, microtubules, etc.)
are present in cryoET tomograms. A biologist paints examples of each feature in
1–2 tomograms; the model learns what they look like and detects them in new data.

---

## How it works

1. **Paint** — open 1–2 tomograms in the built-in matplotlib viewer, paint a few examples of each feature
2. **Learn** — extract 3D patches from painted regions, train a small 3D CNN (minutes)
3. **Detect** — slide the trained model across new tomograms → presence/absence report
4. **Report** — Claude (via Bedrock) summarizes which tomograms are most interesting

You don't need complete segmentations. A few dozen brush strokes per class is enough.

---

## Pipeline

```
Raw MRC tomograms
      │
      ▼
preprocess.py             → data/processed/<run>/tomogram.npy

      │
      ▼
paint_annotations.py      → data/processed/<run>/annotations.npy
  (matplotlib painting viewer, paint once)

      │
      ▼
extract_patches.py        → patches.npz
  (3D patches from painted regions + background)

      │
      ▼
train_patch_classifier.py → models/patch_classifier.pth
  (3D CNN, ~5–15 min on GPU)

      │
      ▼
detect_features.py        → results/predictions.csv
  (sliding window over new tomograms)

      │
      ▼
report.py                 → results/report.md
  (Claude on Bedrock summarizes findings)
```

---

## Setup

```bash
# GPU machine (NVIDIA):
conda env create -f environment.yml
# CPU-only laptop (no NVIDIA GPU):
conda env create -f environment-cpu.yml

conda activate tomoannotator
```

The env files use only the `pytorch`, `nvidia`, and `conda-forge` channels
(plus `nodefaults`) — no access to the Anaconda `defaults` channel is required.
The GPU env pins `pytorch-cuda=12.1`; if your NVIDIA driver is older, change it
to `11.8`. If you have no GPU, use `environment-cpu.yml` — only training needs a
GPU (run that on Garibaldi or EC2); everything else works fine on CPU.

---

## Step-by-step

### 1. Preprocess tomograms

```bash
python scripts/preprocess.py --input-dir data/raw/ --output-dir data/processed/
# Dry run first:
python scripts/preprocess.py --input-dir data/raw/ --output-dir data/processed/ --dry-run
```

Supports subdirectory-per-run or flat directory of .mrc files.

Each tomogram is **low-pass filtered first, then normalized** (clipped to
percentiles and z-scored). Filter settings live in `configs/config.yaml` under
`preprocess.lowpass` — choose `gaussian` (blur by `sigma` voxels) or `fourier`
(Butterworth low-pass by `cutoff` fraction of Nyquist and `order`), or set
`enabled: false` to skip filtering.

### 2. Paint annotations (matplotlib viewer)

Open a tomogram and paint examples of each feature. You only need to annotate
**1–2 tomograms** — you don't need to paint everything, just representative examples.
The viewer is a lightweight matplotlib window (no napari required).

```bash
python scripts/paint_annotations.py --data-dir data/processed/ --run run_001
```

The right-hand panel has a color-coded button per class (configured in
`configs/config.yaml`):
- Label **1** = mitochondria (orange)
- Label **2** = ER (cyan)
- Label **3** = microtubules (green)
- etc.

**Controls:**
- **Scroll wheel** or the **Z slider** — move through slices
- **Left-click + drag** — paint the selected class onto the slice
- **Number keys 0–N** — pick the class (`0` = background/erase)
- **Brush r slider** (or `[` / `]` keys) — set brush radius in voxels
- **Erase** button — toggle erase mode; **u** — undo last stroke
- **Save** button — writes `annotations.npy` for the next step

Annotations are saved as `data/processed/<run>/annotations.npy` (uint8, same
shape as the tomogram, `0` = background, `1..N` = feature classes) and auto-save
on window close.

### 3. Extract patches

```bash
python scripts/extract_patches.py \
    --data-dir data/processed/ \
    --output patches.npz
```

Balanced patch sampling with augmentation. Prints class counts — check that all
classes have patches before training.

### 4. Train patch classifier

```bash
# Local GPU:
python scripts/train_patch_classifier.py --patches patches.npz --output-dir models/

# On Garibaldi (recommended):
sbatch slurm/train_gpu.slurm
```

Training is fast (~5–15 min for 50 epochs on one GPU). Watch per-class validation
accuracy — if a class stays near 0, you need more painted examples for that class.

### 5. Detect features in new tomograms

```bash
# Single tomogram:
python scripts/detect_features.py --tomogram data/processed/new_run/tomogram.npy

# All runs, save CSV:
python scripts/detect_features.py \
    --data-dir data/processed/ \
    --output-csv results/predictions.csv

# Save probability heatmaps (shows WHERE each feature was found):
python scripts/detect_features.py \
    --tomogram data/processed/new_run/tomogram.npy \
    --save-heatmaps

# Batch on Garibaldi:
sbatch slurm/detect_array.slurm
```

### 6. Generate Bedrock report

```bash
# On laptop (SSO profile):
python scripts/report.py --predictions results/predictions.csv --profile <your-sso-profile>

# On EC2 (instance profile, no --profile needed):
python scripts/report.py --predictions results/predictions.csv --push-s3
```

---

## AWS

### Create your S3 bucket (once)

```bash
aws s3 mb s3://scrippsresearch-tomoannotator \
    --region us-west-2 --profile <your-profile>
```

### Sync data

```bash
# Laptop → S3
python scripts/aws_utils.py up data/processed/ processed/ --profile <your-profile>
python scripts/aws_utils.py up models/ models/ --profile <your-profile>

# S3 → EC2 (no --profile on EC2)
python scripts/aws_utils.py down processed/ data/processed/
python scripts/aws_utils.py down models/ models/
```

### Run feature extraction on EC2

```bash
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id ami-00563078bca04e287 \
  --instance-type g4dn.xlarge \
  --subnet-id subnet-0096ffc9c05bebab3 \
  --security-group-ids sg-09d5ef7889a26f56a \
  --iam-instance-profile Name=hackathon-ec2-profile \
  --metadata-options HttpTokens=required \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=tomoannotator-hackathon}]' \
  --profile <your-profile> --region us-west-2 \
  --query 'Instances[0].InstanceId' --output text)
```

---

## Configuration

All scripts read **`configs/config.yaml`** by default, resolved relative to the
directory you run them from — so run commands from the project root (the folder
containing `configs/`). Every script also accepts `--config <path>` to point at
a different file. Edit it to add/remove feature classes, change patch size, or
adjust detection thresholds; the feature list there defines the paint classes
(label `1..N`) used everywhere downstream — no code changes needed.

The default config ships with 6 feature classes: mitochondria, ER, microtubules,
vesicles, ribosomes, nuclear_envelope.

### Expected data layout

`paint_annotations.py` and `extract_patches.py` expect one folder per tomogram:

```
data/processed/<run_name>/tomogram.npy      # produced by preprocess.py
data/processed/<run_name>/annotations.npy   # produced by paint_annotations.py
```

If your tomograms sit loose in `data/processed/` (not inside `<run_name>/`
subfolders), `paint_annotations.py` will tell you so — re-run `preprocess.py`,
which writes each volume into its own run folder.

---

## Reference

Medina, Rahmani et al. "Surface Morphometrics reveals local membrane thickness
variation in organellar subcompartments." J Cell Biol 2025. PMID: 41474626
