# Customer Churn ML Pipeline

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange)
![Status](https://img.shields.io/badge/Status-Production--Ready-success)

A production-grade machine learning pipeline designed to predict customer churn using the Telco Customer Churn dataset. This project implements a robust, modular, and reproducible workflow from data ingestion to model deployment.

## 🚀 Project Overview

Predicting customer churn (when customers stop using a service) is critical for retention strategies. This pipeline automates:
- **Data Ingestion:** Fetching data directly from IBM's Telco dataset.
- **Preprocessing:** Handling missing values, scaling numeric features, and encoding categorical variables using `ColumnTransformer`.
- **Model Selection:** Comparing Logistic Regression and Random Forest models using `GridSearchCV`.
- **Evaluation:** Generating detailed metrics (F1-score, Precision, Recall, AUC-ROC) and visualizations (Confusion Matrix, ROC Curves).
- **Export:** Saving the best-performing pipeline for real-time inference.

## 🛠️ Tech Stack

- **Language:** Python
- **Data Handling:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn
- **Visualization:** Matplotlib, Seaborn
- **Model Persistence:** Joblib

## 📁 Project Structure

```text
churn_pipeline/
│
├── churn_pipeline.py      # Main orchestrator (builds, trains, and exports)
├── config.py              # Centralized configuration and hyperparameters
├── utils.py               # Data cleaning, evaluation, and plotting helpers
├── requirements.txt       # Project dependencies
├── models/                # Exported model binaries (.pkl) and metadata
└── reports/               # Visualizations (CM, ROC curves)
```

## ⚙️ Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/danial-zahid/Customer-Churn-ML-Pipeline.git
   cd Customer-Churn-ML-Pipeline
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 📈 Running the Pipeline

To run the entire pipeline (load, train, evaluate, and export):

```bash
python churn_pipeline.py
```

### What happens when you run it?
1. **Data Loading:** Downloads the dataset and performs basic EDA.
2. **Preprocessing:** Automatically cleans the data and handles data types.
3. **Training:** Runs exhaustive Grid Searches for Logistic Regression and Random Forest.
4. **Evaluation:** Selects the best model based on F1-score and saves plots to the `reports/` folder.
5. **Deployment:** Exports the best model to `models/churn_model.pkl`.
6. **Inference Demo:** Runs a quick prediction on sample customer data.

## 📊 Model Evaluation

The pipeline optimizes for the **F1-score** to balance Precision and Recall, which is crucial for imbalanced churn datasets. 

- **Confusion Matrix:** Generated for each model in `reports/`.
*   **ROC Curve:** AUC-ROC score is used to measure the model's ability to distinguish between churn and non-churn classes.

## 🔮 Inference

You can load the exported pipeline in any environment to make predictions:

```python
import joblib
import pandas as pd

# Load the pipeline
pipeline = joblib.load('models/churn_model.pkl')

# Make predictions
predictions = pipeline.predict(new_customer_data)
```

---
Developed by [Danial Zahid](https://github.com/danial-zahid)