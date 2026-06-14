import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer

from .config import ARTIFACT_DIR, DATA_PATH, ID_COLUMN, METRICS_PATH, MODEL_PATH, TARGET_COLUMN
from .audit import business_cost_report
from .features import prepare_credit_features


@dataclass
class TrainingResult:
    threshold: float
    f1: float
    precision: float
    recall: float
    accuracy: float
    pr_auc: float
    roc_auc: float
    confusion_matrix: list
    train_rows: int
    validation_rows: int
    test_rows: int
    positive_rate_train: float
    positive_rate_test: float


def log(message: str):
    print(f"[ficoforce] {message}", flush=True)


def build_pipeline(
    numeric_columns,
    categorical_columns,
    scale_pos_weight: float,
    profile: str = "fast",
    resampling: str = "none",
    verbose: bool = True,
):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if profile == "deep":
        numeric_steps.append(("power", PowerTransformer(method="yeo-johnson", standardize=True)))
    numeric = Pipeline(steps=numeric_steps, verbose=verbose)
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
        ],
        verbose=verbose,
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", numeric, numeric_columns),
            ("categorical", categorical, categorical_columns),
        ],
        remainder="drop",
        verbose=verbose,
    )
    model_settings = {
        "fast": {
            "n_estimators": 450,
            "learning_rate": 0.055,
            "num_leaves": 40,
            "min_child_samples": 80,
        },
        "deep": {
            "n_estimators": 800,
            "learning_rate": 0.035,
            "num_leaves": 64,
            "min_child_samples": 60,
        },
    }[profile]

    model = LGBMClassifier(
        objective="binary",
        **model_settings,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.25,
        reg_lambda=1.25,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )
    steps = [("preprocess", preprocess)]
    if resampling == "smote_tomek":
        steps.append(("balance", SMOTETomek(random_state=42)))
    steps.append(("model", model))
    return ImbPipeline(steps=steps, verbose=verbose)


def best_f1_threshold(y_true, probabilities):
    thresholds = np.linspace(0.05, 0.95, 181)
    scores = [f1_score(y_true, probabilities >= threshold, zero_division=0) for threshold in thresholds]
    best_index = int(np.argmax(scores))
    return float(thresholds[best_index]), float(scores[best_index])


def evaluate(y_true, probabilities, threshold, train_rows, validation_rows, positive_rate_train):
    predictions = (probabilities >= threshold).astype(int)
    return TrainingResult(
        threshold=float(threshold),
        f1=float(f1_score(y_true, predictions, zero_division=0)),
        precision=float(precision_score(y_true, predictions, zero_division=0)),
        recall=float(recall_score(y_true, predictions, zero_division=0)),
        accuracy=float(accuracy_score(y_true, predictions)),
        pr_auc=float(average_precision_score(y_true, probabilities)),
        roc_auc=float(roc_auc_score(y_true, probabilities)),
        confusion_matrix=confusion_matrix(y_true, predictions).tolist(),
        train_rows=int(train_rows),
        validation_rows=int(validation_rows),
        test_rows=int(len(y_true)),
        positive_rate_train=float(positive_rate_train),
        positive_rate_test=float(np.mean(y_true)),
    )


def train(
    data_path: Path = DATA_PATH,
    sample_frac: float = 1.0,
    output_path: Path = MODEL_PATH,
    profile: str = "fast",
    resampling: str = "none",
    verbose: bool = True,
):
    started = time.time()
    log(f"Reading data from {data_path}")
    df = pd.read_csv(data_path)
    log(f"Loaded {len(df):,} rows and {len(df.columns):,} columns")
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42)
        log(f"Sampled {len(df):,} rows with sample_frac={sample_frac}")

    log("Engineering fraud-risk features")
    df = prepare_credit_features(df)
    y = df[TARGET_COLUMN].astype(int)
    drop_columns = [TARGET_COLUMN]
    if ID_COLUMN in df.columns:
        drop_columns.append(ID_COLUMN)
    X = df.drop(columns=drop_columns)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full
    )
    log(
        "Split rows: "
        f"train={len(X_train):,}, valid={len(X_valid):,}, test={len(X_test):,}, "
        f"positive_rate={y.mean():.3f}"
    )

    numeric_columns = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [column for column in X_train.columns if column not in numeric_columns]

    positives = max(int(y_train.sum()), 1)
    negatives = max(int((y_train == 0).sum()), 1)
    scale_pos_weight = negatives / positives

    log(
        f"Training LightGBM profile={profile}, resampling={resampling}, "
        f"numeric_features={len(numeric_columns)}, categorical_features={len(categorical_columns)}"
    )
    if resampling == "smote_tomek":
        log(
            "SMOTE-Tomek can be slow on the full dataset. "
            "The next pipeline step may take several minutes while synthetic minority samples are generated and Tomek links are removed."
        )
    pipeline = build_pipeline(
        numeric_columns,
        categorical_columns,
        scale_pos_weight,
        profile,
        resampling,
        verbose,
    )
    pipeline.fit(X_train, y_train)
    log("Model fitting complete")

    log("Tuning threshold on validation split")
    valid_probabilities = pipeline.predict_proba(X_valid)[:, 1]
    threshold, _ = best_f1_threshold(y_valid, valid_probabilities)

    log(f"Evaluating test split at threshold={threshold:.3f}")
    test_probabilities = pipeline.predict_proba(X_test)[:, 1]
    metrics = evaluate(
        y_test,
        test_probabilities,
        threshold,
        len(X_train),
        len(X_valid),
        float(y_train.mean()),
    )
    cost_report = business_cost_report(y_test, test_probabilities, threshold)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "pipeline": pipeline,
        "threshold": threshold,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "metrics": asdict(metrics),
        "business_cost_report": cost_report,
        "feature_columns": X.columns.tolist(),
    }
    joblib.dump(artifact, output_path)
    report = asdict(metrics)
    report["business_cost_report"] = cost_report
    METRICS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"Saved model to {output_path}")
    log(f"Saved metrics to {METRICS_PATH}")
    log(f"Finished in {(time.time() - started) / 60:.1f} minutes")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train the FICOFORCE ML fraud-risk model.")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--sample-frac", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    parser.add_argument("--profile", choices=["fast", "deep"], default="fast")
    parser.add_argument("--resampling", choices=["none", "smote_tomek"], default="none")
    parser.add_argument("--quiet", action="store_true", help="Disable sklearn/imblearn step logs.")
    args = parser.parse_args()
    metrics = train(
        args.data,
        args.sample_frac,
        args.output,
        args.profile,
        args.resampling,
        verbose=not args.quiet,
    )
    print(json.dumps(asdict(metrics), indent=2))


if __name__ == "__main__":
    main()
