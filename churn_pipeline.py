# churn_pipeline.py

"""
Production-Ready ML Pipeline — Telco Customer Churn Prediction
==============================================================
• Preprocessing via sklearn Pipeline + ColumnTransformer
• Logistic Regression & Random Forest with GridSearchCV
• Full evaluation (metrics, confusion matrix, ROC)
• Export best pipeline with joblib for deployment
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import joblib
import json
from datetime import datetime

# ── Local modules ──
from config import *
from utils import (
    load_data, inspect_data, clean_data,
    evaluate_model, plot_confusion_matrix,
    plot_roc_curve, print_metrics_table
)


# ================================================================
# 1. DATA LOADING & CLEANING
# ================================================================
def get_data():
    """Load and clean the dataset; return X and y."""
    raw = load_data(DATASET_URL)
    inspect_data(raw)

    df = clean_data(
        raw,
        drop_columns=DROP_COLUMNS,
        numeric_features=NUMERIC_FEATURES,
        target_column=TARGET_COLUMN
    )

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


# ================================================================
# 2. PREPROCESSING — ColumnTransformer inside a Pipeline
# ================================================================
def build_preprocessor() -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
      • imputes + scales numeric features
      • imputes + one-hot encodes categorical features
    """
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore",
                                  sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder="drop"          # safety: drop any unexpected columns
    )
    return preprocessor


# ================================================================
# 3. MODEL BUILDING — Full Pipeline + GridSearchCV
# ================================================================
def build_model_pipeline(classifier) -> Pipeline:
    """Return a full sklearn Pipeline: preprocessor → classifier."""
    preprocessor = build_preprocessor()
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier",   classifier)
    ])
    return pipe


def run_grid_search(pipe: Pipeline, param_grid: dict,
                    X_train, y_train, model_name: str) -> GridSearchCV:
    """Fit GridSearchCV and return the fitted search object."""
    print(f"\n🔍 Running GridSearchCV for {model_name} …")
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=CV_FOLDS,
        scoring=SCORING,
        n_jobs=-1,
        verbose=1,
        refit=True,
        return_train_score=True
    )
    grid.fit(X_train, y_train)

    print(f"\n✅ Best params ({model_name}): {grid.best_params_}")
    print(f"✅ Best CV  {SCORING}: {grid.best_score_:.4f}")
    return grid


# ================================================================
# 4. EVALUATION
# ================================================================
def evaluate_on_test(grid: GridSearchCV, X_test, y_test,
                     model_name: str) -> dict:
    """Predict on test set and return evaluation metrics."""
    y_pred = grid.predict(X_test)
    y_prob = grid.predict_proba(X_test)[:, 1]

    metrics = evaluate_model(y_test, y_pred, y_prob, model_name)

    # Visuals
    plot_confusion_matrix(y_test, y_pred, model_name,
                          save_path=f"reports/cm_{model_name.replace(' ', '_')}.png")
    plot_roc_curve(y_test, y_prob, model_name,
                   save_path=f"reports/roc_{model_name.replace(' ', '_')}.png")

    return metrics


# ================================================================
# 5. EXPORT
# ================================================================
def export_pipeline(grid: GridSearchCV, model_name: str,
                    metrics: dict) -> str:
    """
    Export the best pipeline (preprocessor + model) using joblib.
    Also save a metadata JSON alongside it.
    """
    path = MODEL_EXPORT_PATH
    joblib.dump(grid.best_estimator_, path)
    print(f"\n💾 Pipeline exported → {path}")

    # Metadata
    meta = {
        "model_name":   model_name,
        "export_date":  datetime.now().isoformat(),
        "best_params":  grid.best_params_,
        "cv_score":     grid.best_score_,
        "test_metrics": metrics,
        "features": {
            "numeric":     NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES
        }
    }
    meta_path = path.replace(".pkl", "_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"💾 Metadata exported → {meta_path}")

    return path


# ================================================================
# 6. INFERENCE HELPER — demonstrates reusability
# ================================================================
def predict_new(customers: list[dict], model_path: str) -> np.ndarray:
    """
    Load a saved pipeline and predict churn for new customer records.
    
    Parameters
    ----------
    customers : list of dicts  — each dict must contain all feature keys
    model_path : str           — path to the joblib-exported pipeline

    Returns
    -------
    np.ndarray of predictions (0 = No Churn, 1 = Churn)
    """
    model = joblib.load(model_path)
    df_new = pd.DataFrame(customers)
    predictions = model.predict(df_new)
    probabilities = model.predict_proba(df_new)[:, 1]

    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        label = "Churn" if pred == 1 else "No Churn"
        print(f"  Customer {i+1}: {label}  (probability = {prob:.3f})")

    return predictions, probabilities


# ================================================================
# 7. MAIN ORCHESTRATOR
# ================================================================
def main():
    print("=" * 70)
    print("  CUSTOMER CHURN PREDICTION — PRODUCTION ML PIPELINE")
    print("=" * 70)

    # ── Data ──
    X, y = get_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"\n📊 Train: {X_train.shape}  |  Test: {X_test.shape}")

    # ── Model 1: Logistic Regression ──
    lr_pipe = build_model_pipeline(LogisticRegression(random_state=RANDOM_STATE))
    lr_grid = run_grid_search(lr_pipe, LR_PARAM_GRID,
                              X_train, y_train, "Logistic Regression")
    lr_metrics = evaluate_on_test(lr_grid, X_test, y_test,
                                  "Logistic Regression")

    # ── Model 2: Random Forest ──
    rf_pipe = build_model_pipeline(RandomForestClassifier(random_state=RANDOM_STATE))
    rf_grid = run_grid_search(rf_pipe, RF_PARAM_GRID,
                              X_train, y_train, "Random Forest")
    rf_metrics = evaluate_on_test(rf_grid, X_test, y_test,
                                  "Random Forest")

    # ── Compare ──
    all_metrics = [lr_metrics, rf_metrics]
    print_metrics_table(all_metrics)

    # ── Select best by F1 ──
    best_idx = np.argmax([m["f1"] for m in all_metrics])
    best_name  = all_metrics[best_idx]["model"]
    best_grid  = [lr_grid, rf_grid][best_idx]
    best_metrics = all_metrics[best_idx]

    print(f"\n🏆 Best model: {best_name}  (F1 = {best_metrics['f1']:.4f})")

    # ── Export ──
    export_pipeline(best_grid, best_name, best_metrics)

    # ── Quick inference demo ──
    print("\n" + "=" * 70)
    print("  INFERENCE DEMO — Predicting for 2 sample customers")
    print("=" * 70)
    sample_customers = [
        {
            "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
            "Dependents": "No", "tenure": 12, "PhoneService": "Yes",
            "MultipleLines": "No", "InternetService": "DSL",
            "OnlineSecurity": "Yes", "OnlineBackup": "No",
            "DeviceProtection": "No", "TechSupport": "Yes",
            "StreamingTV": "No", "StreamingMovies": "No",
            "Contract": "One year", "PaperlessBilling": "Yes",
            "PaymentMethod": "Bank transfer (automatic)",
            "MonthlyCharges": 56.95, "TotalCharges": 703.35
        },
        {
            "gender": "Male", "SeniorCitizen": 0, "Partner": "No",
            "Dependents": "No", "tenure": 2, "PhoneService": "Yes",
            "MultipleLines": "No", "InternetService": "Fiber optic",
            "OnlineSecurity": "No", "OnlineBackup": "No",
            "DeviceProtection": "No", "TechSupport": "No",
            "StreamingTV": "Yes", "StreamingMovies": "Yes",
            "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 85.70, "TotalCharges": 180.40
        }
    ]
    predict_new(sample_customers, MODEL_EXPORT_PATH)

    print("\n✅ Pipeline execution complete!")


if __name__ == "__main__":
    main()