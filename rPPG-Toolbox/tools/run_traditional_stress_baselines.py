"""Run traditional rPPG-feature stress baselines for the paper table.

This script reads the same preprocessed UBFC-Phys ROI RGB clips used by
RT-MRCPNet, extracts compact handcrafted temporal/spectral features, trains
SVM and random forest classifiers, and writes metrics.csv files in the same
location/format as neural experiments.

Run from the rPPG-Toolbox root after preprocessing has been created:
    python tools/run_traditional_stress_baselines.py --config_file configs/train_configs/UBFCPHYS_RTMRCPNET_BASIC.yaml
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import get_config
from evaluation.stress_metrics import calculate_stress_metrics


def load_split(file_list_path):
    file_list = pd.read_csv(file_list_path)["input_files"].tolist()
    x_list, y_list, filenames, chunk_ids = [], [], [], []
    for input_path in sorted(file_list):
        x = np.load(input_path).astype(np.float32)
        y = np.load(input_path.replace("input", "label")).astype(np.int64).reshape(-1)[0]
        item = Path(input_path).name
        split_idx = item.rindex("_")
        filename = item[:split_idx]
        chunk_id = item[split_idx + 6:].split(".")[0]
        x_list.append(x)
        y_list.append(int(y))
        filenames.append(filename)
        chunk_ids.append(chunk_id)
    return np.asarray(x_list), np.asarray(y_list), filenames, chunk_ids


def band_features(signal, fs=30.0, low=0.7, high=3.0):
    signal = np.asarray(signal, dtype=np.float64)
    signal = signal - np.mean(signal)
    std = np.std(signal) + 1e-8
    signal = signal / std
    freq = np.fft.rfftfreq(signal.shape[0], d=1.0 / fs)
    spec = np.abs(np.fft.rfft(signal)) ** 2
    band = (freq >= low) & (freq <= high)
    if not np.any(band):
        return [0.0, 0.0, 0.0, 0.0]
    band_freq = freq[band]
    band_power = spec[band]
    total = float(np.sum(band_power) + 1e-8)
    peak_idx = int(np.argmax(band_power))
    peak_freq = float(band_freq[peak_idx])
    peak_ratio = float(band_power[peak_idx] / total)
    centroid = float(np.sum(band_freq * band_power) / total)
    entropy = float(-np.sum((band_power / total) * np.log((band_power / total) + 1e-8)))
    return [peak_freq, peak_ratio, centroid, entropy]


def chrom_trace(rgb):
    rgb = np.asarray(rgb, dtype=np.float64)
    rgb = rgb / (np.mean(rgb, axis=0, keepdims=True) + 1e-8)
    x = rgb[:, 0] - rgb[:, 1]
    y = rgb[:, 0] + rgb[:, 1] - 2.0 * rgb[:, 2]
    alpha = np.std(x) / (np.std(y) + 1e-8)
    return x - alpha * y


def extract_features(clips, fs=30.0):
    # clips: [N, K, T, 3]
    features = []
    for clip in clips:
        face = np.mean(clip, axis=0, keepdims=True)
        signals = np.concatenate([clip, face], axis=0)
        sample_features = []
        for roi in signals:
            green = roi[:, 1]
            chrom = chrom_trace(roi)
            for trace in (green, chrom):
                sample_features.extend([
                    float(np.mean(trace)),
                    float(np.std(trace)),
                    float(np.min(trace)),
                    float(np.max(trace)),
                    float(np.percentile(trace, 75) - np.percentile(trace, 25)),
                ])
                sample_features.extend(band_features(trace, fs=fs))
        features.append(sample_features)
    return np.asarray(features, dtype=np.float32)


def write_outputs(config, model_name, classifier_name, y_true, y_score, y_pred, filenames, chunk_ids, elapsed_ms):
    metrics = calculate_stress_metrics(y_true, y_score, prefix="Test")
    output_dir = os.path.join(config.TEST.OUTPUT_SAVE_DIR, model_name)
    os.makedirs(output_dir, exist_ok=True)
    row = {
        "experiment": model_name,
        "architecture": "traditional_features",
        "data_variant": classifier_name,
        "use_temporal_attention": False,
        "use_region_attention": False,
        "use_region_residual": False,
        "best_epoch": "",
        "model_path": "sklearn",
        "param_count": "",
        "avg_inference_ms_per_clip": elapsed_ms / max(len(y_true), 1),
    }
    row.update(metrics)

    metrics_path = os.path.join(output_dir, "metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2)

    pred_path = os.path.join(output_dir, "predictions.csv")
    with open(pred_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "chunk_id", "label", "score_stress", "prediction", "correct"])
        for values in zip(filenames, chunk_ids, y_true, y_score, y_pred):
            filename, chunk_id, label, score, pred = values
            writer.writerow([filename, chunk_id, int(label), float(score), int(pred), int(label == pred)])

    todo_path = os.path.join(output_dir, "paper_todo_values.md")
    with open(todo_path, "w", encoding="utf-8") as f:
        f.write("# Paper TODO Values\n\n")
        f.write("| Method | ACC | Precision | Recall | F1 | AUC |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        f.write(
            f"| {model_name} | {metrics['ACC']:.4f} | {metrics['Precision']:.4f} | "
            f"{metrics['Recall']:.4f} | {metrics['F1']:.4f} | {metrics['AUC']:.4f} |\n"
        )
    print("Saved traditional baseline metrics:", metrics_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="configs/train_configs/UBFCPHYS_RTMRCPNET_BASIC.yaml")
    args = parser.parse_args()
    config = get_config(argparse.Namespace(config_file=args.config_file, cached_path=None, preprocess=None))

    if not os.path.exists(config.TRAIN.DATA.FILE_LIST_PATH):
        raise FileNotFoundError(
            f"Missing train file list: {config.TRAIN.DATA.FILE_LIST_PATH}. "
            "Run one neural experiment first to create preprocessing outputs."
        )
    if not os.path.exists(config.VALID.DATA.FILE_LIST_PATH):
        raise FileNotFoundError(f"Missing valid file list: {config.VALID.DATA.FILE_LIST_PATH}")
    if not os.path.exists(config.TEST.DATA.FILE_LIST_PATH):
        raise FileNotFoundError(f"Missing test file list: {config.TEST.DATA.FILE_LIST_PATH}")

    x_train, y_train, _, _ = load_split(config.TRAIN.DATA.FILE_LIST_PATH)
    x_valid, y_valid, _, _ = load_split(config.VALID.DATA.FILE_LIST_PATH)
    x_test, y_test, filenames, chunk_ids = load_split(config.TEST.DATA.FILE_LIST_PATH)
    fs = float(config.TEST.DATA.FS) if config.TEST.DATA.FS else 30.0

    x_fit = np.concatenate([x_train, x_valid], axis=0)
    y_fit = np.concatenate([y_train, y_valid], axis=0)
    feat_fit = extract_features(x_fit, fs=fs)
    feat_test = extract_features(x_test, fs=fs)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    baselines = [
        (
            "UBFCPHYS_RTMRCPNet_rppg_features_svm",
            "rppg_features_svm",
            make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, class_weight="balanced")),
        ),
        (
            "UBFCPHYS_RTMRCPNet_rppg_features_random_forest",
            "rppg_features_random_forest",
            RandomForestClassifier(n_estimators=300, max_depth=None, class_weight="balanced", random_state=100),
        ),
    ]

    for model_name, classifier_name, clf in baselines:
        start = time.perf_counter()
        clf.fit(feat_fit, y_fit)
        if hasattr(clf, "predict_proba"):
            y_score = clf.predict_proba(feat_test)[:, 1]
        else:
            decision = clf.decision_function(feat_test)
            y_score = 1.0 / (1.0 + np.exp(-decision))
        y_pred = (y_score >= 0.5).astype(np.int64)
        elapsed_ms = 1000.0 * (time.perf_counter() - start)
        write_outputs(
            config=config,
            model_name=model_name,
            classifier_name=classifier_name,
            y_true=y_test,
            y_score=y_score,
            y_pred=y_pred,
            filenames=filenames,
            chunk_ids=chunk_ids,
            elapsed_ms=elapsed_ms,
        )


if __name__ == "__main__":
    main()
