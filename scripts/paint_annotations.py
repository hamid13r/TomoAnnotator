"""
Annotation painting tool — open a tomogram in napari and paint feature labels.

Each feature class gets a color-coded label ID (1=mito, 2=ER, 3=microtubules, etc.).
You don't need to paint everything — just representative examples of each feature.
A few dozen strokes per class across 1-2 tomograms is enough.

Painted labels are saved as:
    data/processed/<run_name>/annotations.npy   (uint8, 0=background, 1..N=features)

Usage:
    python paint_annotations.py --data-dir data/processed/ --run run_001
    python paint_annotations.py --data-dir data/processed/   # picks first unannotated run
"""

import argparse
from pathlib import Path

import napari
import numpy as np
import yaml
from magicgui.widgets import ComboBox, Container, Label, PushButton


def load_config(path: str = "configs/config.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


def load_tomogram(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32)
    import mrcfile
    with mrcfile.open(str(path), mode="r", permissive=True) as f:
        vol = f.data.copy().astype(np.float32)
    lo, hi = np.percentile(vol, (5, 95))
    vol = np.clip(vol, lo, hi)
    std = vol.std()
    return (vol - vol.mean()) / std if std > 0 else vol


def find_unannotated_run(data_dir: Path) -> str | None:
    for run_dir in sorted(data_dir.iterdir()):
        if run_dir.is_dir() and (run_dir / "tomogram.npy").exists():
            if not (run_dir / "annotations.npy").exists():
                return run_dir.name
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="data/processed/ directory")
    parser.add_argument("--run", default=None, help="Run name (default: first unannotated)")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    features = cfg["features"]
    # Class IDs: 0 = background (unpainted), 1..N = features
    feature_names = [f["name"] for f in features]
    feature_colors = {i + 1: tuple(c / 255 for c in f["color"]) + (0.6,)
                      for i, f in enumerate(features)}

    data_dir = Path(args.data_dir)
    run_name = args.run or find_unannotated_run(data_dir)
    if run_name is None:
        print("All runs are already annotated.")
        return

    run_dir = data_dir / run_name
    tomo_path = run_dir / "tomogram.npy"
    ann_path = run_dir / "annotations.npy"

    if not tomo_path.exists():
        print(f"Tomogram not found: {tomo_path}")
        return

    print(f"Annotating: {run_name}")
    print("Feature classes:")
    for i, name in enumerate(feature_names, 1):
        print(f"  Label {i} = {name}")
    print("  Label 0 = background (default, leave unpainted)")
    print("\nPress the number keys in napari to select a label ID.")

    tomo = load_tomogram(tomo_path)

    # Load existing annotations if resuming
    if ann_path.exists():
        existing = np.load(ann_path)
        print(f"Resuming existing annotations from {ann_path}")
    else:
        existing = np.zeros(tomo.shape, dtype=np.uint8)

    viewer = napari.Viewer(title=f"Annotate: {run_name}")

    # Add tomogram
    lo, hi = np.percentile(tomo, (2, 98))
    viewer.add_image(tomo, name="tomogram", colormap="grays",
                     contrast_limits=(lo, hi))

    # Add Labels layer with feature colors
    label_layer = viewer.add_labels(
        existing, name="annotations",
        opacity=0.5,
        color={**{0: None}, **feature_colors},
    )
    # Start with label 1 (first feature) selected
    label_layer.selected_label = 1

    # --- Dock panel ---
    legend_text = "Label IDs:\n  0 = background\n" + \
                  "\n".join(f"  {i+1} = {f['name']}" for i, f in enumerate(features))
    legend = Label(value=legend_text)

    # Run selector to switch between runs without relaunching
    run_dirs = sorted([p.name for p in data_dir.iterdir()
                       if p.is_dir() and (p / "tomogram.npy").exists()])
    run_selector = ComboBox(label="Switch run", choices=run_dirs, value=run_name)

    def on_switch_run(event):
        nonlocal run_name, ann_path
        # Save current before switching
        save_current()
        run_name = run_selector.value
        new_dir = data_dir / run_name
        new_tomo = load_tomogram(new_dir / "tomogram.npy")
        lo2, hi2 = np.percentile(new_tomo, (2, 98))
        viewer.layers["tomogram"].data = new_tomo
        viewer.layers["tomogram"].contrast_limits = (lo2, hi2)
        ann_path = new_dir / "annotations.npy"
        label_layer.data = (np.load(ann_path) if ann_path.exists()
                            else np.zeros(new_tomo.shape, dtype=np.uint8))
        viewer.title = f"Annotate: {run_name}"
        status.value = f"Loaded {run_name}"

    run_selector.changed.connect(on_switch_run)

    status = Label(value="Paint labels, then click Save.")

    def save_current():
        arr = label_layer.data.astype(np.uint8)
        save_path = data_dir / run_name / "annotations.npy"
        np.save(save_path, arr)
        classes, counts = np.unique(arr[arr > 0], return_counts=True)
        summary = ", ".join(f"{feature_names[c-1]}={n}" for c, n in zip(classes, counts))
        status.value = f"Saved: {save_path.name}  [{summary or 'no labels yet'}]"
        print(f"Saved annotations: {save_path}  [{summary}]")

    save_btn = PushButton(label="Save annotations")
    save_btn.clicked.connect(lambda: save_current())

    hints = Label(value=(
        "Brush shortcuts:\n"
        "  Q = paint   E = erase\n"
        "  [ / ] = brush size\n"
        "  Ctrl+Z = undo"
    ))

    panel = Container(widgets=[legend, run_selector, save_btn, hints, status], layout="vertical")
    viewer.window.add_dock_widget(panel, name="Annotation", area="right")

    napari.run()

    # Auto-save on exit
    save_current()


if __name__ == "__main__":
    main()
