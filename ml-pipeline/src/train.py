"""
train.py

LightGBM training script for forecast anomaly classification.
This script runs INSIDE Azure ML on a compute cluster — it is submitted
as a Pipeline Job by run_pipeline.py (or as a Command Job by submit_training.py).

It receives the training and validation dataset paths as arguments,
trains a LightGBM binary classifier, evaluates it, and logs metrics
via MLflow (visible in AML Studio) and writes metrics.json as a
reliable fallback for the pipeline orchestrator's metric gating.

Target metric: F1_weighted >= 0.80
Label: 0 = temporary anomaly, 1 = baseline_shift
"""

import argparse
import json
import os
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

# Feature columns used for training (excludes raw categoricals and the label)
FEATURE_COLS = [
    "forecast_bias",
    "forecast_accuracy",
    "volume_of_error",
    "pct_error",
    "weeks_affected",
    "prior_anomaly_count",
    "dc_region_encoded",
    "product_lifecycle_encoded",
    "is_seasonal_encoded",
]
LABEL_COL = "label"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=str, required=True, help="Path to training CSV")
    parser.add_argument("--validation-data", type=str, required=True, help="Path to validation CSV")
    parser.add_argument("--model-output", type=str, default="outputs", help="Directory to save model artifacts")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=31)
    return parser.parse_args()


def load_data(train_path: str, validation_path: str):
    # AML passes a folder path when using URI_FILE — find the CSV inside it
    def resolve(path):
        if os.path.isdir(path):
            csvs = sorted(f for f in os.listdir(path) if f.endswith(".csv"))
            if not csvs:
                raise FileNotFoundError(f"No CSV files found in directory: {path}")
            return os.path.join(path, csvs[0])
        return path

    train = pd.read_csv(resolve(train_path))
    validation = pd.read_csv(resolve(validation_path))
    return train, validation


def train(args):
    train_df, val_df = load_data(args.train_data, args.validation_data)

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[LABEL_COL]
    X_val = val_df[FEATURE_COLS]
    y_val = val_df[LABEL_COL]

    print(f"Training on {len(X_train)} rows, validating on {len(X_val)} rows")
    print(f"Class distribution (train): {y_train.value_counts().to_dict()}")

    model = lgb.LGBMClassifier(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        class_weight="balanced",   # handles class imbalance (70/30 split)
        random_state=42,
    )

    model.fit(X_train, y_train)

    # Evaluate on validation set
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="weighted")
    auc = roc_auc_score(y_val, y_prob)

    metrics = {"accuracy": round(accuracy, 4), "f1_weighted": round(f1, 4), "auc": round(auc, 4)}
    params = {"n_estimators": args.n_estimators, "learning_rate": args.learning_rate, "num_leaves": args.num_leaves}

    # Log to MLflow (visible in AML Studio). Graceful fallback if unavailable.
    try:
        import mlflow
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        for k, v in params.items():
            mlflow.log_param(k, v)
        print("Metrics logged to MLflow")
    except Exception as e:
        print(f"MLflow logging skipped ({e})")

    print(f"\nValidation results:")
    print(f"  Accuracy:    {accuracy:.4f}")
    print(f"  F1_weighted: {f1:.4f}  (target: >= 0.80)")
    print(f"  AUC:         {auc:.4f}")
    print(f"\nClassification report:\n{classification_report(y_val, y_pred, target_names=['temporary', 'baseline_shift'])}")

    # Save model artifacts to the specified output directory
    output_dir = args.model_output
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "model.txt")
    model.booster_.save_model(model_path)
    print(f"Model saved to {model_path}")

    # Write metrics JSON — read by the pipeline orchestrator for metric gating
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"metrics": metrics, "params": params}, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
