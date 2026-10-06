"""Build Table 6: held-out comparison of the 8 trained classifiers.

The pickled estimators in models/ were fitted in 02_train.py against the
relabeled 4-class target 'climate_class' (classes 0-3), not the raw 5-class
'climate_bin' (classes 0-4). This script therefore rebuilds 'climate_class'
with the identical mapping so that the train/test split reproduces the split
used at training time exactly, giving a genuinely held-out test set.

SMOTE is imported for interface parity with the training pipeline but is not
applied: resampling is a training-time concern, and oversampling the test
side would corrupt the evaluation.
"""

import os
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

LABELED_PATH = "data/processed/labeled_data.csv"
MODELS_DIR = "models"
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
REPORT_PATH = os.path.join("reports", "model_comparison.csv")

FEATURE_COLS = [
    "pl_insol",
    "pl_rade",
    "pl_eqt",
    "st_teff",
    "st_mass",
    "co2_frac",
    "h2o_frac",
    "ch4_frac",
    "o2_frac",
    "n2_frac",
    "greenhouse_index",
    "esi",
]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "K-Nearest Neighbors": "knn.pkl",
    "Decision Tree": "decision_tree.pkl",
    "Random Forest": "random_forest.pkl",
    "Gradient Boosting": "gradient_boosting.pkl",
    "Support Vector Machine (RBF)": "svc.pkl",
    "Gaussian Naive Bayes": "gaussian_nb.pkl",
    "Multi-Layer Perceptron": "mlp.pkl",
}

ROW_ORDER = [
    "Logistic Regression",
    "K-Nearest Neighbors",
    "Decision Tree",
    "Random Forest",
    "Gradient Boosting",
    "Support Vector Machine (RBF)",
    "Gaussian Naive Bayes",
    "Multi-Layer Perceptron",
]


def build_climate_class(series):
    """Map the 5 raw climate bins onto the 4 classes used for training.

    4 -> 3 Gas Giant, 0/1 -> 0 Hot Non-Habitable, 2 -> 1 Potentially
    Habitable, 3 -> 2 Cold Non-Habitable.
    """
    mapping = {4: 3, 0: 0, 1: 0, 2: 1, 3: 2}
    unknown = set(series.unique()) - set(mapping)
    if unknown:
        raise ValueError("Unexpected climate_bin values: %s" % sorted(unknown))
    return series.map(mapping)


def safe_roc_auc(model, X_test, y_test):
    """Macro one-vs-rest ROC-AUC, or 'N/A' when the estimator cannot provide it."""
    if not hasattr(model, "predict_proba"):
        return "N/A"
    try:
        proba = model.predict_proba(X_test)
        return round(roc_auc_score(y_test, proba, multi_class="ovr", average="macro"), 4)
    except Exception:
        return "N/A"


def main():
    df = pd.read_csv(LABELED_PATH)
    print("Loaded", LABELED_PATH, "shape:", df.shape)

    existing = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print("Skipping missing columns:", missing)
    X = df[existing]
    y = build_climate_class(df["climate_bin"])
    print("Feature matrix:", X.shape, "| target classes:", sorted(y.unique()))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print("Split -> train:", len(X_train), "test:", len(X_test))

    scaler = joblib.load(SCALER_PATH)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Scaled features with", SCALER_PATH)

    smote = SMOTE(random_state=42)
    print("SMOTE available (", type(smote).__name__, ") but not applied: test set is held out")

    rows = []
    for name in ROW_ORDER:
        model = joblib.load(os.path.join(MODELS_DIR, MODEL_FILES[name]))
        y_pred = model.predict(X_test_scaled)

        accuracy = round(accuracy_score(y_test, y_pred), 4)
        precision = round(precision_score(y_test, y_pred, average="macro", zero_division=0), 4)
        recall = round(recall_score(y_test, y_pred, average="macro", zero_division=0), 4)
        f1 = round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4)
        roc = safe_roc_auc(model, X_test_scaled, y_test)

        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1,
                "ROC-AUC": roc,
            }
        )
        print(
            "  %-30s acc=%s prec=%s rec=%s f1=%s roc_auc=%s"
            % (name, accuracy, precision, recall, f1, roc)
        )

    table = pd.DataFrame(rows)[["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]]

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    table.to_csv(REPORT_PATH, index=False)
    print("\nSaved to", REPORT_PATH)

    print()
    print(table.to_string(index=False))
    print()
    print("TABLE 6 DATA READY")


if __name__ == "__main__":
    main()
