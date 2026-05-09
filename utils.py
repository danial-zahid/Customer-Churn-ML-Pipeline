# utils.py

"""Utility helpers for the churn prediction pipeline."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, RocCurveDisplay,
    PrecisionRecallDisplay
)
import io, sys


def load_data(url: str) -> pd.DataFrame:
    """Load the Telco churn CSV from a URL or local path."""
    df = pd.read_csv(url)
    print(f"✅ Data loaded — shape: {df.shape}")
    return df


def inspect_data(df: pd.DataFrame) -> None:
    """Print basic EDA info."""
    print("\n" + "=" * 60)
    print("DATA INSPECTION")
    print("=" * 60)
    print(f"\nShape: {df.shape}")
    print(f"\nColumn types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nTarget distribution:\n{df['Churn'].value_counts(normalize=True)}")
    print(f"\nFirst 5 rows:\n{df.head()}")


def clean_data(df: pd.DataFrame, drop_columns: list,
               numeric_features: list, target_column: str) -> pd.DataFrame:
    """
    Clean the raw dataframe:
      - Drop non-predictive columns
      - Coerce numeric columns (TotalCharges can have blanks)
      - Encode target as 0/1
    """
    df = df.copy()

    # Drop identifier columns
    df.drop(columns=drop_columns, inplace=True, errors="ignore")

    # Coerce numeric columns that may be read as object
    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with NaN after coercion (very few in Telco dataset)
    n_before = len(df)
    df.dropna(inplace=True)
    n_after = len(df)
    if n_before != n_after:
        print(f"⚠️  Dropped {n_before - n_after} rows with missing values")

    # Encode target: Yes → 1, No → 0
    df[target_column] = df[target_column].map({"Yes": 1, "No": 0})

    print(f"✅ Data cleaned — shape: {df.shape}")
    return df


def evaluate_model(y_true, y_pred, y_prob, model_name: str) -> dict:
    """Return a dict of key classification metrics."""
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob)
    }
    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name: str, save_path: str = None):
    """Plot and optionally save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_roc_curve(y_true, y_prob, model_name: str, save_path: str = None):
    """Plot ROC curve."""
    RocCurveDisplay.from_predictions(y_true, y_prob, name=model_name)
    plt.title(f"ROC Curve — {model_name}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def print_metrics_table(metrics_list: list[dict]) -> None:
    """Pretty-print a comparison table of model metrics."""
    df_metrics = pd.DataFrame(metrics_list)
    df_metrics = df_metrics.set_index("model")
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(df_metrics.to_string(float_format="{:.4f}".format))
    print()