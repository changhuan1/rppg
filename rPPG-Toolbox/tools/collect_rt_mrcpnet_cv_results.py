"""Collect 5-fold subject-level CV metrics for RT-MRCPNet."""

import csv
import glob
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs" / "exp"
OUT_DIR = ROOT / "paper_outputs"
METRICS = ["ACC", "Precision", "Recall", "Specificity", "F1", "AUC"]


def read_metric_row(path):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return next(reader)


def main():
    pattern = str(RUNS_DIR / "**" / "saved_test_outputs" / "UBFCPHYS_RTMRCPNet_full_cv5_fold*" / "metrics.csv")
    metric_paths = sorted(glob.glob(pattern, recursive=True))
    if not metric_paths:
        raise FileNotFoundError("No CV fold metrics found. Run configs/train_configs/rt_mrcpnet_cv5/run_full_cv5.sh first.")

    rows = []
    for path in metric_paths:
        row = read_metric_row(path)
        row["metrics_path"] = path
        rows.append(row)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    detail_path = OUT_DIR / "rt_mrcpnet_cv5_detail.csv"
    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["experiment", "best_epoch"] + METRICS + ["metrics_path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    summary = {}
    for metric in METRICS:
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        summary[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }

    summary_path = OUT_DIR / "rt_mrcpnet_cv5_summary.md"
    lines = [
        "# RT-MRCPNet 5-Fold Subject-Level CV Summary",
        "",
        f"Number of folds: {len(rows)}",
        "",
        "| Metric | Mean | Std |",
        "|---|---:|---:|",
    ]
    for metric in METRICS:
        lines.append(f"| {metric} | {summary[metric]['mean']:.4f} | {summary[metric]['std']:.4f} |")
    lines.extend([
        "",
        "## LaTeX Table Row",
        "",
        (
            "RT-MRCPNet & "
            + " & ".join(
                f"{summary[metric]['mean']:.4f}$\\pm${summary[metric]['std']:.4f}"
                for metric in METRICS
            )
            + r" \\"
        ),
    ])
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Saved CV detail: {detail_path}")
    print(f"Saved CV summary: {summary_path}")


if __name__ == "__main__":
    main()
