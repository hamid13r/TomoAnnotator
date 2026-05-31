"""
View a detection result overlaid on its tomogram.

Loads a processed tomogram and the segmentation.npy produced by
detect_features.py, and shows them in the same style as the painting viewer:
a grayscale slice with the predicted class labels overlaid in the config colors.
This is read-only — it does not modify the segmentation.

The overlay starts hidden so you can inspect the raw tomogram first; toggle it
on with the button or the `o` key (just like comparing against the painting).

Controls:
    scroll / Z slider   move through slices
    o (or button)       toggle the class overlay on/off
    opacity slider      overlay transparency
    [ / ]               not used (read-only viewer)

Usage:
    python view_segmentation.py --run-dir data/processed/run_001
    python view_segmentation.py --tomogram data/processed/run_001/tomogram.npy \
                                --segmentation data/processed/run_001/segmentation.npy
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# Reuse the exact colormap / loaders the painting viewer uses, so the overlay
# colors match the painted annotations one-to-one.
from paint_annotations import (
    load_config, load_tomogram, find_tomogram_in, build_label_cmap,
)


def view(tomo: np.ndarray, seg: np.ndarray, features: list[dict], title_name: str):
    feature_names = [f["name"] for f in features]
    cmap, norm = build_label_cmap(features)
    z_pixels = tomo.shape[0]

    state = {"z": z_pixels // 2, "show": False, "alpha": 0.55}

    lo, hi = np.percentile(tomo, (2, 98))

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_axes([0.06, 0.18, 0.66, 0.76])
    ax_z = fig.add_axes([0.06, 0.09, 0.66, 0.03], facecolor="lightgoldenrodyellow")
    ax_op = fig.add_axes([0.06, 0.04, 0.66, 0.03], facecolor="lightgoldenrodyellow")

    img = ax.imshow(tomo[state["z"]], cmap="gray", origin="lower",
                    aspect="equal", vmin=lo, vmax=hi)
    overlay = ax.imshow(seg[state["z"]], cmap=cmap, norm=norm, origin="lower",
                        aspect="equal", interpolation="nearest", alpha=state["alpha"])
    overlay.set_visible(state["show"])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    slider_z = Slider(ax_z, "Z", 0, z_pixels - 1, valinit=state["z"], valstep=1)
    slider_op = Slider(ax_op, "Opacity", 0.0, 1.0, valinit=state["alpha"])

    def present_summary():
        ids = [c for c in np.unique(seg) if c != 0]
        return ", ".join(feature_names[c - 1] for c in ids) or "none above threshold"

    def title():
        ax.set_title(
            f"{title_name}   Z={state['z']}/{z_pixels - 1}   "
            f"overlay: {'ON' if state['show'] else 'off'}\n"
            f"detected: {present_summary()}   ·   o = toggle overlay · scroll = change slice",
            fontsize=9,
        )

    def refresh():
        img.set_data(tomo[state["z"]])
        overlay.set_data(seg[state["z"]])
        title()
        fig.canvas.draw_idle()

    def on_z(val):
        state["z"] = int(slider_z.val)
        refresh()

    def on_op(val):
        state["alpha"] = float(slider_op.val)
        overlay.set_alpha(state["alpha"])
        fig.canvas.draw_idle()

    slider_z.on_changed(on_z)
    slider_op.on_changed(on_op)

    def on_scroll(event):
        if event.inaxes is not ax:
            return
        step = 1 if getattr(event, "button", None) == "up" else -1
        slider_z.set_val(max(0, min(z_pixels - 1, state["z"] + step)))

    fig.canvas.mpl_connect("scroll_event", on_scroll)

    def toggle_overlay(_=None):
        state["show"] = not state["show"]
        overlay.set_visible(state["show"])
        btn.label.set_text(f"Overlay: {'on' if state['show'] else 'off'}")
        title()
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key and event.key.lower() == "o":
            toggle_overlay()

    fig.canvas.mpl_connect("key_press_event", on_key)

    ax_btn = fig.add_axes([0.78, 0.88, 0.12, 0.05])
    btn = Button(ax_btn, "Overlay: off")
    btn.on_clicked(toggle_overlay)

    # Color-coded class legend
    for i, name in enumerate(feature_names):
        ax_c = fig.add_axes([0.78, 0.80 - i * 0.05, 0.19, 0.04])
        ax_c.axis("off")
        r, g, b = (c / 255.0 for c in features[i]["color"])
        ax_c.add_patch(plt.Rectangle((0, 0), 0.18, 1, color=(r, g, b)))
        ax_c.text(0.24, 0.5, f"{i + 1}  {name}", va="center", fontsize=9,
                  transform=ax_c.transAxes)

    title()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=None,
                        help="A processed run folder containing tomogram.npy + segmentation.npy")
    parser.add_argument("--tomogram", default=None)
    parser.add_argument("--segmentation", default=None)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    features = cfg["features"]

    if args.run_dir:
        run_dir = Path(args.run_dir)
        tomo_path = find_tomogram_in(run_dir)
        seg_path = run_dir / "segmentation.npy"
        name = run_dir.name
    else:
        if not (args.tomogram and args.segmentation):
            print("Provide --run-dir, or both --tomogram and --segmentation.")
            return
        tomo_path = Path(args.tomogram)
        seg_path = Path(args.segmentation)
        name = tomo_path.parent.name

    if tomo_path is None or not tomo_path.exists():
        print(f"Tomogram not found ({tomo_path}).")
        return
    if not seg_path.exists():
        print(f"Segmentation not found ({seg_path}). "
              f"Run detect_features.py first (it saves segmentation.npy by default).")
        return

    tomo = load_tomogram(tomo_path)
    seg = np.load(seg_path).astype(np.uint8)
    if seg.shape != tomo.shape:
        print(f"Shape mismatch: tomogram {tomo.shape} vs segmentation {seg.shape}")
        return

    view(tomo, seg, features, name)


if __name__ == "__main__":
    main()
