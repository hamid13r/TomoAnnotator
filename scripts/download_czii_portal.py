"""
Download annotated cryoET tomograms from the CZ cryoET Data Portal.
Filters for datasets with mitochondria and/or ER annotations.

Usage:
    python download_czii_portal.py --output-dir /gpfs/scratch/$USER/grotjahn-organelle-seg/raw/czii/
    python download_czii_portal.py --output-dir ./data/czii/ --max-runs 5 --organelles mitochondria er
"""

import argparse
import json
from pathlib import Path

import cryoet_data_portal as cdp


ORGANELLE_KEYWORDS = {
    "mitochondria": ["mitochondri", "mito"],
    "er": ["endoplasmic reticulum", " er ", "ER"],
}


def find_annotated_runs(client, organelles: list[str], max_runs: int):
    """Return runs that have segmentation annotations for the requested organelles."""
    keyword_groups = [ORGANELLE_KEYWORDS.get(o, [o]) for o in organelles]

    datasets = cdp.Dataset.find(client, [])
    runs = []
    for dataset in datasets:
        for run in dataset.runs:
            for ann in run.annotations:
                obj = (ann.object_name or "").lower()
                if any(any(kw.lower() in obj for kw in kws) for kws in keyword_groups):
                    runs.append(run)
                    break
        if len(runs) >= max_runs:
            break
    return runs[:max_runs]


def download_run(run, output_dir: Path, organelles: list[str]):
    """Download tomogram MRC and matching segmentation annotations for a run."""
    run_dir = output_dir / run.name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Download the canonical tomogram
    tomo = next(iter(run.tomograms), None)
    if tomo is None:
        print(f"  [skip] {run.name}: no tomograms")
        return

    tomo_path = run_dir / "tomogram.mrc"
    if not tomo_path.exists():
        print(f"  Downloading tomogram: {tomo_path}")
        tomo.download_mrcfile(str(tomo_path))
    else:
        print(f"  Tomogram already exists: {tomo_path}")

    keyword_groups = [ORGANELLE_KEYWORDS.get(o, [o]) for o in organelles]

    # Download matching segmentation annotations
    for ann in run.annotations:
        obj = (ann.object_name or "").lower()
        if not any(any(kw.lower() in obj for kw in kws) for kws in keyword_groups):
            continue

        ann_dir = run_dir / "annotations" / ann.object_name.replace(" ", "_")
        ann_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = ann_dir / "metadata.json"
        metadata_path.write_text(json.dumps({
            "object_name": ann.object_name,
            "annotation_method": getattr(ann, "annotation_method", ""),
            "annotation_id": ann.id,
        }, indent=2))

        for seg_file in ann.files:
            if getattr(seg_file, "format", "") in ("mrc", "zarr", "nrrd"):
                dest = ann_dir / Path(seg_file.s3_path).name
                if not dest.exists():
                    print(f"  Downloading annotation: {dest}")
                    seg_file.download(str(dest))
                else:
                    print(f"  Annotation already exists: {dest}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, help="Directory to save downloaded data")
    parser.add_argument("--max-runs", type=int, default=10, help="Maximum number of tomogram runs to download")
    parser.add_argument("--organelles", nargs="+", default=["mitochondria"],
                        choices=list(ORGANELLE_KEYWORDS.keys()),
                        help="Organelle classes to filter for")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = cdp.Client()

    print(f"Searching CZ cryoET Data Portal for runs with: {args.organelles}")
    runs = find_annotated_runs(client, args.organelles, args.max_runs)
    print(f"Found {len(runs)} runs. Downloading...")

    for run in runs:
        print(f"\nRun: {run.name}")
        download_run(run, output_dir, args.organelles)

    print(f"\nDone. Data in: {output_dir}")


if __name__ == "__main__":
    main()
