# config.py

"""Centralized configuration for the churn prediction pipeline."""

# Dataset
DATASET_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
TARGET_COLUMN = "Churn"
DROP_COLUMNS = ["customerID"]  # Non-predictive identifier

# Numeric / Categorical column definitions
NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]

# Train / Test split
TEST_SIZE = 0.20
RANDOM_STATE = 42

# GridSearchCV
CV_FOLDS = 5
SCORING = "f1"            # Optimise for F1 (handles class imbalance better)

# Logistic Regression hyper-parameter grid
LR_PARAM_GRID = {
    "classifier__C": [0.01, 0.1, 1, 10, 100],
    "classifier__penalty": ["l1", "l2"],
    "classifier__solver": ["liblinear"],
    "classifier__max_iter": [100, 200, 500],
    "classifier__class_weight": [None, "balanced"]
}

# Random Forest hyper-parameter grid
RF_PARAM_GRID = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [5, 10, 20, None],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
    "classifier__class_weight": [None, "balanced", "balanced_subsample"]
}

# Export path
MODEL_EXPORT_PATH = "models/churn_model.pkl"