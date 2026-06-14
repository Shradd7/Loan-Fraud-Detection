from pathlib import Path

import joblib
import pandas as pd

from .config import MODEL_PATH
from .features import prepare_credit_features


class ModelNotTrainedError(RuntimeError):
    pass


def load_model(path: Path = MODEL_PATH):
    if not path.exists():
        raise ModelNotTrainedError(
            f"Model artifact not found at {path}. Run: python -m ficoforce.train"
        )
    return joblib.load(path)


def predict_records(records, artifact=None):
    artifact = artifact or load_model()
    frame = pd.DataFrame(records)
    frame = prepare_credit_features(frame)

    feature_columns = artifact["feature_columns"]
    for column in feature_columns:
        if column not in frame.columns:
            frame[column] = None
    frame = frame[feature_columns]

    probabilities = artifact["pipeline"].predict_proba(frame)[:, 1]
    threshold = artifact["threshold"]
    predictions = (probabilities >= threshold).astype(int)
    return [
        {
            "prediction": int(prediction),
            "label": "Fraud Risk" if int(prediction) == 1 else "No Fraud Risk",
            "fraud_probability": round(float(probability), 4),
            "default_probability": round(float(probability), 4),
            "threshold": round(float(threshold), 4),
        }
        for prediction, probability in zip(predictions, probabilities)
    ]
