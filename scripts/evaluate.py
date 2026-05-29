"""
Evaluate classifier predictions against ground-truth labels.

Metrics: per-feature accuracy, precision, recall, F1, ROC-AUC.
Also shows a confusion summary and which tomograms were misclassified.

Usage:
    python evaluate.py \\
        --features features.npz \\
        --labels labels.csv \\
        --model-dir models/

    # Save full report to CSV:
    python evaluate.py --features features.npz --labels labels.csv --output results/eval.csv
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    accuracy_score,
)


def load_classifier(model_dir: Path):
    with open(model_dir / "classifier.pkl", "rb") as f:
        clf = pickle.load(f)
    label_names = (model_dir / "label_names.txt").read_text().strip().splitlines()
    return clf, label_names


def align_features_labels(features_path: Path, labels_path: Path):
    data = np.load(features_path, allow_pickle=True)
    X_all = data["X"]
    run_names_feat = list(data["run_names"])

    labels_df = pd.read_csv(labels_path)
    labels_df = labels_df.dropna()
    label_cols = [c for c in labels_df.columns if c != "tomogram_id"]

    feat_index = {name: i for i, name in enumerate(run_names_feat)}
    matched = [(i, feat_index[row.tomogram_id])
               for i, row in labels_df.iterrows()
               if row.tomogram_id in feat_index]

    if not matched:
        raise RuntimeError("No overlap between features.npz and labels.csv")

    label_rows, feat_rows = zip(*matched)
    X = X_all[list(feat_rows)]
    Y = labels_df.loc[list(label_rows), label_cols].values.astype(int)
    names = labels_df.loc[list(label_rows), "tomogram_id"].tolist()
    return X, Y, label_cols, names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="features.npz")
    parser.add_argument("--labels", default="labels.csv")
    parser.add_argument("--model-dir", default="models/")
    parser.add_argument("--output", default=None, help="Save per-tomogram results to CSV")
    args = parser.parse_args()

    clf, label_names = load_classifier(Path(args.model_dir))
    X, Y_true, label_cols, run_names = align_features_labels(
        Path(args.features), Path(args.labels)
    )

    # Use only the features the classifier was trained on
    common = [l for l in label_cols if l in label_names]
    col_idx = [label_cols.index(l) for l in common]
    Y_true = Y_true[:, col_idx]

    Y_pred = clf.predict(X)

    try:
        Y_prob = np.column_stack([
            est.predict_proba(clf["scaler"].transform(X))[:, 1]
            for est in clf["clf"].estimators_
        ])
    except Exception:
        Y_prob = Y_pred.astype(float)

    print("\n=== Per-feature classification report ===\n")
    print(classification_report(Y_true, Y_pred, target_names=label_names, zero_division=0))

    print("=== ROC-AUC per feature ===")
    for i, name in enumerate(label_names):
        if Y_true[:, i].sum() > 0:
            auc = roc_auc_score(Y_true[:, i], Y_prob[:, i])
            print(f"  {name:<25} AUC={auc:.3f}")
        else:
            print(f"  {name:<25} AUC=n/a (no positives in eval set)")

    print(f"\nOverall exact-match accuracy: {accuracy_score(Y_true, Y_pred):.3f}")

    # Per-tomogram breakdown
    print("\n=== Per-tomogram predictions ===")
    header = f"{'Tomogram':<30} " + " ".join(f"{l[:8]:>10}" for l in label_names)
    print(header)
    print("-" * len(header))
    mismatches = []
    for name, gt, pred in zip(run_names, Y_true, Y_pred):
        cells = []
        mismatch = False
        for g, p in zip(gt, pred):
            match = g == p
            if not match:
                mismatch = True
            cells.append(f"{'✓' if match else '✗'}{g}/{p}")
        print(f"{name:<30} " + " ".join(f"{c:>10}" for c in cells))
        if mismatch:
            mismatches.append(name)

    if mismatches:
        print(f"\nMisclassified tomograms: {mismatches}")

    if args.output:
        rows = []
        for name, gt, pred, prob in zip(run_names, Y_true, Y_pred, Y_prob):
            row = {"tomogram_id": name}
            for label, g, p, pr in zip(label_names, gt, pred, prob):
                row[f"{label}_gt"] = g
                row[f"{label}_pred"] = p
                row[f"{label}_conf"] = round(float(pr), 3)
            rows.append(row)
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
