"""
Annotation painting tool — a lightweight matplotlib viewer (no napari).

Scroll or drag the slider to move through Z slices. Hold the left mouse button
and drag to paint the currently-selected feature class onto the slice. Press a
number key (0..N) to choose what you're painting:

    0 = background / erase
    1..N = feature classes (from configs/config.yaml)

Three brush sizes are available (small / medium / large) via on-screen buttons
or the keys 1/2/3 on the numpad-style brush row (see buttons). An Erase button
toggles erase mode (equivalent to painting with class 0), and Save writes the
annotation to disk for the next pipeline step.

Painted labels are saved as:
    data/processed/<run_name>/annotations.npy   (uint8, 0=background, 1..N=features)

This is the exact format extract_patches.py expects: a uint8 volume the same
shape as tomogram.npy, with 0 = background and 1..N = feature class IDs.

Usage:
    python paint_annotations.py --data-dir data/processed/ --run run_001
    python paint_annotations.py --data-dir data/processed/   # first unannotated run
"""

import argparse
from pathlib import Path

import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.widgets import Slider, Button


# Three brush radii (in voxels), selectable in the UI.
BRUSH_SIZES = {"S": 3, "M": 7, "L": 14}


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


def build_label_cmap(features: list[dict]):
    """ListedColormap where index 0 is transparent and 1..N are feature colors."""
    colors = [(0, 0, 0, 0.0)]  # background: fully transparent
    for f in features:
        r, g, b = (c / 255.0 for c in f["color"])
        colors.append((r, g, b, 0.55))
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(features) + 1, 1), cmap.N)
    return cmap, norm


def paint_run(tomo: np.ndarray, ann: np.ndarray, features: list[dict],
              run_name: str, save_fn) -> None:
    """Open the painting GUI for one run. save_fn(ann_array) persists labels."""
    feature_names = [f["name"] for f in features]
    n_classes = len(features)
    z_pixels = tomo.shape[0]

    cmap, norm = build_label_cmap(features)

    # Mutable UI state
    state = {
        "z": z_pixels // 2,
        "label": 1 if n_classes else 0,   # currently painting this class
        "brush": "M",
        "erase": False,
        "painting": False,
    }

    lo, hi = np.percentile(tomo, (2, 98))

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_axes([0.06, 0.16, 0.66, 0.78])
    ax_slider = fig.add_axes([0.06, 0.07, 0.66, 0.03], facecolor="lightgoldenrodyellow")

    img = ax.imshow(tomo[state["z"]], cmap="gray", origin="lower",
                    aspect="equal", vmin=lo, vmax=hi)
    overlay = ax.imshow(ann[state["z"]], cmap=cmap, norm=norm,
                        origin="lower", aspect="equal", interpolation="nearest")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    slider_z = Slider(ax_slider, "Z", 0, z_pixels - 1,
                      valinit=state["z"], valstep=1)

    def current_label() -> int:
        return 0 if state["erase"] else state["label"]

    def title():
        name = "background (erase)" if current_label() == 0 else feature_names[current_label() - 1]
        ax.set_title(
            f"{run_name}   Z={state['z']}/{z_pixels - 1}   "
            f"painting: [{current_label()}] {name}   brush={state['brush']} "
            f"(r={BRUSH_SIZES[state['brush']]})\n"
            "drag to paint · scroll/slider to change slice · keys 0-{} pick class · "
            "S/M/L brush · u=undo last stroke".format(n_classes),
            fontsize=9,
        )

    def refresh():
        img.set_data(tomo[state["z"]])
        overlay.set_data(ann[state["z"]])
        title()
        fig.canvas.draw_idle()

    # ---- slice navigation ----
    def on_slider(val):
        state["z"] = int(slider_z.val)
        refresh()

    slider_z.on_changed(on_slider)

    def on_scroll(event):
        if event.inaxes is not ax:
            return
        step = 1 if getattr(event, "button", None) == "up" else -1
        new_z = max(0, min(z_pixels - 1, state["z"] + step))
        slider_z.set_val(new_z)

    fig.canvas.mpl_connect("scroll_event", on_scroll)

    # ---- painting ----
    _undo_stack = []

    def paint_at(xdata, ydata, fresh_stroke=False):
        if xdata is None or ydata is None:
            return
        cx, cy = int(round(xdata)), int(round(ydata))
        r = BRUSH_SIZES[state["brush"]]
        z = state["z"]
        sl = ann[z]
        y0, y1 = max(0, cy - r), min(sl.shape[0], cy + r + 1)
        x0, x1 = max(0, cx - r), min(sl.shape[1], cx + r + 1)
        if y0 >= y1 or x0 >= x1:
            return
        yy, xx = np.ogrid[y0:y1, x0:x1]
        disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        if fresh_stroke:
            _undo_stack.append((z, sl.copy()))
            if len(_undo_stack) > 20:
                _undo_stack.pop(0)
        region = sl[y0:y1, x0:x1]
        region[disk] = current_label()
        overlay.set_data(ann[z])
        fig.canvas.draw_idle()

    def on_press(event):
        if event.inaxes is ax and event.button == 1:
            state["painting"] = True
            paint_at(event.xdata, event.ydata, fresh_stroke=True)

    def on_motion(event):
        if state["painting"] and event.inaxes is ax:
            paint_at(event.xdata, event.ydata)

    def on_release(event):
        state["painting"] = False

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)

    # ---- keyboard ----
    def on_key(event):
        if event.key is None:
            return
        k = event.key
        if k.isdigit():
            v = int(k)
            if 0 <= v <= n_classes:
                state["label"] = v
                state["erase"] = (v == 0)
                title()
                fig.canvas.draw_idle()
        elif k.lower() in ("s", "m", "l"):
            state["brush"] = k.upper()
            title()
            fig.canvas.draw_idle()
        elif k.lower() == "u":
            if _undo_stack:
                z, prev = _undo_stack.pop()
                ann[z] = prev
                if z == state["z"]:
                    overlay.set_data(ann[z])
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)

    # ---- buttons ----
    # Brush size buttons
    bx, bw, bh = 0.78, 0.06, 0.04
    ax_bs = fig.add_axes([bx, 0.84, bw, bh])
    ax_bm = fig.add_axes([bx + 0.065, 0.84, bw, bh])
    ax_bl = fig.add_axes([bx + 0.13, 0.84, bw, bh])
    btn_s = Button(ax_bs, "S")
    btn_m = Button(ax_bm, "M")
    btn_l = Button(ax_bl, "L")

    def set_brush(name):
        state["brush"] = name
        title()
        fig.canvas.draw_idle()

    btn_s.on_clicked(lambda e: set_brush("S"))
    btn_m.on_clicked(lambda e: set_brush("M"))
    btn_l.on_clicked(lambda e: set_brush("L"))

    # Class buttons (0..N)
    class_buttons = []
    for i in range(n_classes + 1):
        ax_c = fig.add_axes([0.78, 0.76 - i * 0.05, 0.19, 0.04])
        lbl = "0  background (erase)" if i == 0 else f"{i}  {feature_names[i - 1]}"
        btn = Button(ax_c, lbl)
        if i > 0:
            r, g, b = (c / 255.0 for c in features[i - 1]["color"])
            btn.ax.set_facecolor((r, g, b, 0.4))

        def make_cb(idx):
            def cb(event):
                state["label"] = idx
                state["erase"] = (idx == 0)
                title()
                fig.canvas.draw_idle()
            return cb

        btn.on_clicked(make_cb(i))
        class_buttons.append(btn)

    # Erase toggle + Save
    ax_erase = fig.add_axes([0.78, 0.76 - (n_classes + 1) * 0.05 - 0.02, 0.09, 0.045])
    ax_save = fig.add_axes([0.88, 0.76 - (n_classes + 1) * 0.05 - 0.02, 0.09, 0.045])
    btn_erase = Button(ax_erase, "Erase: off")
    btn_save = Button(ax_save, "Save")

    def toggle_erase(event):
        state["erase"] = not state["erase"]
        btn_erase.label.set_text(f"Erase: {'on' if state['erase'] else 'off'}")
        title()
        fig.canvas.draw_idle()

    def do_save(event):
        msg = save_fn(ann)
        btn_save.label.set_text("Saved ✓")
        fig.canvas.draw_idle()
        print(msg)

    btn_erase.on_clicked(toggle_erase)
    btn_save.on_clicked(do_save)

    title()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="data/processed/ directory")
    parser.add_argument("--run", default=None, help="Run name (default: first unannotated)")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    features = cfg["features"]
    feature_names = [f["name"] for f in features]

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
    print("  Label 0 = background (default / erase)")
    print("\nPress number keys 0-{} to pick a class; S/M/L for brush size.".format(len(features)))

    tomo = load_tomogram(tomo_path)

    if ann_path.exists():
        ann = np.load(ann_path).astype(np.uint8)
        print(f"Resuming existing annotations from {ann_path}")
    else:
        ann = np.zeros(tomo.shape, dtype=np.uint8)

    def save_fn(arr: np.ndarray) -> str:
        out = arr.astype(np.uint8)
        np.save(ann_path, out)
        classes, counts = np.unique(out[out > 0], return_counts=True)
        summary = ", ".join(f"{feature_names[c - 1]}={n}" for c, n in zip(classes, counts))
        return f"Saved annotations: {ann_path}  [{summary or 'no labels yet'}]"

    paint_run(tomo, ann, features, run_name, save_fn)

    # Auto-save on window close
    print(save_fn(ann))


if __name__ == "__main__":
    main()
