"""Collect RT-MRCPNet paper metrics from saved experiment folders.

Run from the rPPG-Toolbox root after one or more controlled experiments:
    python tools/collect_rt_mrcpnet_results.py
"""

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "exp"
OUT = ROOT / "paper_outputs"

MAIN_METHOD_ORDER = [
    "Full-face RGB temporal network",
    "Forehead-only temporal network",
    "Left-cheek-only temporal network",
    "Right-cheek-only temporal network",
    "Nose-only temporal network",
    "Chin-only temporal network",
    "Multi-region fixed fusion",
    "rPPG features + SVM",
    "rPPG features + random forest",
    "Lightweight CNN-LSTM",
    "Lightweight 3D CNN",
    "PhysNet/EfficientPhys + stress head",
    "RT-MRCPNet",
]

ABLATION_ORDER = [
    "Without attention",
    "Temporal attention only",
    "Region attention only",
    "Full RT-MRCPNet",
]


def read_metrics(path):
    with open(path, "r", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    row["metrics_path"] = str(path)
    return row


def format_float(value, digits=4):
    if value in ("", None):
        return "--"
    return f"{float(value):.{digits}f}"


def method_name(row):
    variant = row.get("data_variant", "")
    arch = str(row.get("architecture", "")).lower()
    temporal = str(row.get("use_temporal_attention", "")).lower()
    region = str(row.get("use_region_attention", "")).lower()
    single_roi_names = {
        "single_forehead": "Forehead-only temporal network",
        "single_left_cheek": "Left-cheek-only temporal network",
        "single_right_cheek": "Right-cheek-only temporal network",
        "single_nose": "Nose-only temporal network",
        "single_chin": "Chin-only temporal network",
    }

    if variant == "rppg_features_svm":
        return "rPPG features + SVM"
    if variant == "rppg_features_random_forest":
        return "rPPG features + random forest"
    if arch == "cnn_lstm":
        return "Lightweight CNN-LSTM"
    if arch == "cnn3d":
        return "Lightweight 3D CNN"
    if arch in ("rppg_stress_head", "physnet_stress_head", "efficientphys_stress_head"):
        return "PhysNet/EfficientPhys + stress head"
    if variant in ("roi_mean", "full_face", "mean_fusion"):
        return "Full-face RGB temporal network"
    if variant in single_roi_names:
        return single_roi_names[variant]
    if variant == "multi_roi" and temporal == "false" and region == "false":
        return "Multi-region fixed fusion"
    if variant == "multi_roi" and temporal == "true" and region == "true":
        return "RT-MRCPNet"
    if variant == "multi_roi" and temporal == "true" and region == "false":
        return "Temporal attention only"
    if variant == "multi_roi" and temporal == "false" and region == "true":
        return "Region attention only"
    return row.get("experiment", "Unknown experiment")


def input_description(row):
    variant = row.get("data_variant", "")
    arch = str(row.get("architecture", "")).lower()
    single_roi_inputs = {
        "single_forehead": "Forehead RGB sequence",
        "single_left_cheek": "Left cheek RGB sequence",
        "single_right_cheek": "Right cheek RGB sequence",
        "single_nose": "Nose RGB sequence",
        "single_chin": "Chin RGB sequence",
    }
    if variant in ("rppg_features_svm", "rppg_features_random_forest"):
        return "Handcrafted HR/HRV features"
    if arch in ("cnn_lstm", "cnn3d"):
        return "ROI RGB temporal tensor"
    if arch in ("rppg_stress_head", "physnet_stress_head", "efficientphys_stress_head"):
        return "Face-level RGB/rPPG-oriented features"
    if variant in ("roi_mean", "full_face", "mean_fusion"):
        return "Averaged face-level RGB sequence"
    if variant in single_roi_inputs:
        return single_roi_inputs[variant]
    if variant == "multi_roi":
        temporal = str(row.get("use_temporal_attention", "")).lower()
        region = str(row.get("use_region_attention", "")).lower()
        if temporal == "true" and region == "true":
            return "Five ROI RGB sequences, attention fusion"
        if temporal == "false" and region == "false":
            return "Five ROI RGB sequences, fixed fusion"
        return "Five ROI RGB sequences"
    return row.get("input", "")


def ablation_name(row):
    variant = row.get("data_variant", "")
    arch = str(row.get("architecture", "")).lower()
    temporal = str(row.get("use_temporal_attention", "")).lower()
    region = str(row.get("use_region_attention", "")).lower()
    if arch != "rtmrcpnet" or variant != "multi_roi":
        return None
    if temporal == "false" and region == "false":
        return "Without attention"
    if temporal == "true" and region == "false":
        return "Temporal attention only"
    if temporal == "false" and region == "true":
        return "Region attention only"
    if temporal == "true" and region == "true":
        return "Full RT-MRCPNet"
    return None


def main():
    metric_files = sorted(RUNS.glob("**/saved_test_outputs/*/metrics.csv"))
    rows = [read_metrics(path) for path in metric_files]
    if not rows:
        raise SystemExit("No RT-MRCPNet metrics.csv files found under runs/exp.")

    OUT.mkdir(parents=True, exist_ok=True)
    fields = [
        "experiment",
        "architecture",
        "data_variant",
        "use_temporal_attention",
        "use_region_attention",
        "ACC",
        "Precision",
        "Recall",
        "F1",
        "AUC",
        "TP",
        "TN",
        "FP",
        "FN",
        "param_count",
        "avg_inference_ms_per_clip",
        "best_epoch",
        "model_path",
        "metrics_path",
    ]
    out_csv = OUT / "rt_mrcpnet_all_results.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    out_md = OUT / "rt_mrcpnet_latex_table_rows.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# RT-MRCPNet Controlled Protocol Table Rows\n\n")
        f.write("## Main Comparison\n\n")
        f.write("| Method | Input | ACC | F1 | AUC |\n")
        f.write("|---|---|---:|---:|---:|\n")
        rows_by_method = {}
        for row in rows:
            rows_by_method[method_name(row)] = row
        for method in MAIN_METHOD_ORDER:
            row = rows_by_method.get(method)
            if row is None:
                f.write(f"| {method} | TODO | TODO | TODO | TODO |\n")
                continue
            f.write(
                f"| {method} | {input_description(row)} | "
                f"{format_float(row.get('ACC'))} | {format_float(row.get('F1'))} | "
                f"{format_float(row.get('AUC'))} |\n"
            )

        f.write("\n## Ablation Study\n\n")
        f.write("| Variant | ACC | F1 | AUC |\n")
        f.write("|---|---:|---:|---:|\n")
        rows_by_ablation = {}
        for row in rows:
            name = ablation_name(row)
            if name:
                rows_by_ablation[name] = row
        for name in ABLATION_ORDER:
            row = rows_by_ablation.get(name)
            if row is None:
                f.write(f"| {name} | TODO | TODO | TODO |\n")
            else:
                f.write(
                    f"| {name} | {format_float(row.get('ACC'))} | "
                    f"{format_float(row.get('F1'))} | {format_float(row.get('AUC'))} |\n"
                )

        f.write("\n## Complexity\n\n")
        f.write("| Method | Parameters | Average inference time per clip (ms) |\n")
        f.write("|---|---:|---:|\n")
        for method in MAIN_METHOD_ORDER:
            row = rows_by_method.get(method)
            if row is None:
                f.write(f"| {method} | TODO | TODO |\n")
                continue
            params = row.get("param_count", "") or "--"
            infer = format_float(row.get("avg_inference_ms_per_clip"))
            f.write(f"| {method} | {params} | {infer} |\n")

    print(f"Saved aggregate CSV: {out_csv}")
    print(f"Saved paper table rows: {out_md}")


if __name__ == "__main__":
    main()
