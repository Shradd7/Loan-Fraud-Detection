import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def business_cost_report(
    y_true,
    probabilities,
    threshold: float,
    false_negative_cost: float = 10000.0,
    false_positive_cost: float = 500.0,
) -> dict:
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    y_true = np.asarray(y_true).astype(int)
    false_negatives = int(((y_true == 1) & (predictions == 0)).sum())
    false_positives = int(((y_true == 0) & (predictions == 1)).sum())
    expected_cost = false_negatives * false_negative_cost + false_positives * false_positive_cost
    return {
        "threshold": float(threshold),
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "false_negative_cost": float(false_negative_cost),
        "false_positive_cost": float(false_positive_cost),
        "expected_cost": float(expected_cost),
    }


def segment_performance(frame: pd.DataFrame, target: str, prediction: str, segment: str) -> list[dict]:
    rows = []
    for value, group in frame.groupby(segment, dropna=False):
        y_true = group[target].astype(int)
        y_pred = group[prediction].astype(int)
        rows.append(
            {
                "segment": str(value),
                "rows": int(len(group)),
                "positive_rate": float(y_true.mean()),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            }
        )
    return rows
