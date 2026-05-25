"""Metrics for binary stress recognition."""

import numpy as np


def _binary_auc(y_true, y_score):
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float64)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = np.sum(pos)
    n_neg = np.sum(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)

    # Average ranks for ties.
    sorted_scores = y_score[order]
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = np.mean(np.arange(start + 1, end + 1))
        start = end

    rank_sum_pos = np.sum(ranks[pos])
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def calculate_stress_metrics(y_true, y_score, threshold=0.5, prefix=""):
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float64)
    y_pred = (y_score >= threshold).astype(np.int64)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    auc = _binary_auc(y_true, y_score)

    metrics = {
        "ACC": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1": f1,
        "AUC": auc,
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }

    title = f"{prefix} " if prefix else ""
    print(f"{title}Stress Recognition Metrics")
    print(f"ACC: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"AUC: {auc:.4f}" if not np.isnan(auc) else "AUC: nan")
    print(f"Confusion Matrix [[TN, FP], [FN, TP]]: [[{tn}, {fp}], [{fn}, {tp}]]")
    return metrics
