from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# -------------------------------------------------
# 1. PROJECT PATHS
# -------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "raw" / "leads.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Converted"

# -------------------------------------------------
# 2. LOAD DATA
# -------------------------------------------------

print("\n" + "=" * 60)
print("LEAD INTENT & PRIORITY SCORER")
print("MACHINE LEARNING TRAINING")
print("=" * 60)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_PATH}\n"
        "Place your CSV at data/raw/leads.csv"
    )

df = pd.read_csv(DATA_PATH)

# Remove accidental whitespace from column names.
df.columns = df.columns.str.strip()

print("\nDataset loaded successfully!")
print("Original shape:", df.shape)

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' was not found.\n"
        f"Available columns: {df.columns.tolist()}"
    )

# -------------------------------------------------
# 3. BASIC DATA CLEANING
# -------------------------------------------------

# Remove exact duplicate records.
df = df.drop_duplicates().copy()

# Remove records with no target value.
df = df.dropna(subset=[TARGET]).copy()

# The expected target encoding is 0 and 1.
target_values = set(df[TARGET].unique())

if not target_values.issubset({0, 1, 0.0, 1.0}):
    raise ValueError(
        "The Converted column must contain only 0 and 1. "
        f"Found values: {target_values}"
    )

df[TARGET] = df[TARGET].astype(int)

# Drop records with invalid target values, if any remain.
df = df[df[TARGET].isin([0, 1])].copy()

if df[TARGET].nunique() != 2:
    raise ValueError(
        "The dataset must contain both converted and non-converted leads."
    )

print("Cleaned shape:", df.shape)
print("\nTarget distribution:")
print(df[TARGET].value_counts())

print("\nConversion rate: {:.2f}%".format(df[TARGET].mean() * 100))

# -------------------------------------------------
# 4. REMOVE IDENTIFIERS AND POSSIBLE DATA LEAKAGE
# -------------------------------------------------

# These fields can identify records rather than generalize
# useful patterns to new leads.
identifier_names = {
    "prospect id",
    "lead number",
    "lead id",
    "id",
}

# Outcome-related fields may contain information obtained
# after the conversion decision or be too close to the outcome.
# This is a conservative starting list; inspect your dataset
# and its data dictionary before finalizing your feature policy.
leakage_names = {
    "converted",
    "status",
    "lead quality",
    "tags",
    "payment status",
    "enrollment status",
    "enrolled",
    "admission status",
}

columns_to_drop = []

for column in df.columns:
    normalized = column.strip().lower()

    if normalized in identifier_names:
        columns_to_drop.append(column)

    elif normalized in leakage_names:
        columns_to_drop.append(column)

# Never include the target itself as an input.
if TARGET in df.columns and TARGET not in columns_to_drop:
    columns_to_drop.append(TARGET)

X = df.drop(columns=columns_to_drop, errors="ignore").copy()
y = df[TARGET].copy()

if X.shape[1] == 0:
    raise ValueError("No input features remain after removing excluded columns.")

# Remove columns that contain no information.
constant_columns = [
    column for column in X.columns
    if X[column].nunique(dropna=False) <= 1
]

if constant_columns:
    X = X.drop(columns=constant_columns)

print("\nExcluded columns:", columns_to_drop)
print("Constant columns removed:", constant_columns)
print("Input features:", X.shape[1])

# -------------------------------------------------
# 5. IDENTIFY NUMERICAL AND CATEGORICAL COLUMNS
# -------------------------------------------------

# Convert boolean columns to integers.
for column in X.select_dtypes(include=["bool"]).columns:
    X[column] = X[column].astype(int)

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = [
    column for column in X.columns
    if column not in numeric_features
]

print("\nNumerical features:", len(numeric_features))
print("Categorical features:", len(categorical_features))

# -------------------------------------------------
# 6. PREPROCESSING
# -------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True,
            ),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ],
    remainder="drop",
)

# -------------------------------------------------
# 7. TRAIN / TEST SPLIT
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# -------------------------------------------------
# 8. DEFINE MODELS
# -------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        max_depth=15,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}

results = []
trained_models = {}

# -------------------------------------------------
# 9. TRAIN AND EVALUATE
# -------------------------------------------------

for model_name, classifier in models.items():

    print("\n" + "-" * 60)
    print("Training:", model_name)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC AUC": roc_auc_score(y_test, probabilities),
    }

    results.append(metrics)

    trained_models[model_name] = pipeline

    print("Accuracy :", round(metrics["Accuracy"], 4))
    print("Precision:", round(metrics["Precision"], 4))
    print("Recall   :", round(metrics["Recall"], 4))
    print("F1 Score :", round(metrics["F1 Score"], 4))
    print("ROC AUC  :", round(metrics["ROC AUC"], 4))

# -------------------------------------------------
# 10. COMPARE MODELS
# -------------------------------------------------

results_df = pd.DataFrame(results)

# Select the model with the best ROC AUC on the held-out test set.
# For a production project, model selection should use validation
# data or cross-validation, with the final test set used only once.
best_row = results_df.sort_values(
    by="ROC AUC",
    ascending=False,
).iloc[0]

best_model_name = best_row["Model"]
best_model = trained_models[best_model_name]

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df.round(4).to_string(index=False))

print("\nSelected model:", best_model_name)

# -------------------------------------------------
# 11. SAVE MODEL AND METRICS
# -------------------------------------------------

model_path = MODEL_DIR / "lead_scoring_pipeline.joblib"
results_path = MODEL_DIR / "model_comparison.csv"
metrics_path = MODEL_DIR / "metrics.json"
feature_path = MODEL_DIR / "feature_names.json"

joblib.dump(best_model, model_path)

results_df.to_csv(results_path, index=False)

final_predictions = best_model.predict(X_test)
final_probabilities = best_model.predict_proba(X_test)[:, 1]

final_metrics = {
    "selected_model": best_model_name,
    "accuracy": float(accuracy_score(y_test, final_predictions)),
    "precision": float(
        precision_score(y_test, final_predictions, zero_division=0)
    ),
    "recall": float(
        recall_score(y_test, final_predictions, zero_division=0)
    ),
    "f1_score": float(
        f1_score(y_test, final_predictions, zero_division=0)
    ),
    "roc_auc": float(roc_auc_score(y_test, final_probabilities)),
    "train_records": int(len(X_train)),
    "test_records": int(len(X_test)),
    "feature_count": int(X.shape[1]),
    "conversion_rate": float(y.mean()),
    "confusion_matrix": confusion_matrix(
        y_test, final_predictions
    ).tolist(),
    "classification_report": classification_report(
        y_test,
        final_predictions,
        zero_division=0,
        output_dict=True,
    ),
}

with open(metrics_path, "w", encoding="utf-8") as file:
    json.dump(final_metrics, file, indent=4)

with open(feature_path, "w", encoding="utf-8") as file:
    json.dump(X.columns.tolist(), file, indent=4)

print("\nModel saved to:", model_path)
print("Comparison saved to:", results_path)
print("Metrics saved to:", metrics_path)
print("\nTRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)