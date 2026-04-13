"""
train.py

LightGBM training script for forecast anomaly classification.
This script runs INSIDE Azure ML on a compute cluster — it is submitted
as a Command Job by submit_training.py.

It receives the training and validation dataset paths as arguments,
trains a LightGBM binary classifier, evaluates it, and logs metrics
to AML Experiments so you can compare runs in AML Studio.

Target metric: F1_weighted >= 0.80
Label: 0 = temporary anomaly, 1 = baseline_shift
"""

import argparse
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

    # Log metrics to stdout — visible in AML Studio job logs
    print(f"##[metric]accuracy={accuracy:.4f}")
    print(f"##[metric]f1_weighted={f1:.4f}")
    print(f"##[metric]auc={auc:.4f}")

    print(f"\nValidation results:")
    print(f"  Accuracy:    {accuracy:.4f}")
    print(f"  F1_weighted: {f1:.4f}  (target: >= 0.80)")
    print(f"  AUC:         {auc:.4f}")
    print(f"\nClassification report:\n{classification_report(y_val, y_pred, target_names=['temporary', 'baseline_shift'])}")

    # Save model — AML picks this up and registers it
    os.makedirs("outputs", exist_ok=True)
    model.booster_.save_model("outputs/model.txt")
    print("Model saved to outputs/model.txt")


if __name__ == "__main__":
    args = parse_args()
    train(args)
