"""Two-stage pipeline: rule-based Gas Giant filter + binary ML on rocky planets.

Stage 1 filters out Gas Giants with the exact rule-based criterion used to
generate the labels (pl_rade >= 1.6). Stage 2 trains binary classifiers on the
remaining rocky planets only, predicting whether a planet is inside the
Habitable Zone (hz_class=1) or outside it (hz_class=0). No oversampling is
used; class imbalance is handled with class_weight='balanced'. The minority
class (In HZ) is the metric of interest.

"""

import joblib
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    balanced_accuracy_score,
    average_precision_score,
    roc_auc_score,
)

LABELED_PATH = "data/processed/labeled_data.csv"
ROCKY_PATH = "data/processed/rocky_only.csv"
BINARY_PATH = "data/processed/binary_data.csv"
MODELS_DIR = "models"
SCALER_PATH = os.path.join(MODELS_DIR, "scaler_v2.pkl")
REPORT_PATH = "reports/two_stage_results.csv"

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

MODEL_NAMES = ["logistic_regression", "decision_tree", "random_forest", "svc"]


def stage1_filter():
    """Split labeled data into gas giants vs rocky planets.

    Returns
    -------
    pandas.DataFrame
        Rocky subset saved to data/processed/rocky_only.csv.
    """
    df = pd.read_csv(LABELED_PATH)
    print("Loaded", LABELED_PATH, "shape:", df.shape)

    gas_giants = df[df["climate_bin"] == 4]
    rocky = df[df["climate_bin"] != 4]
    print("Gas giants: %d, Rocky: %d" % (len(gas_giants), len(rocky)))

    rocky.to_csv(ROCKY_PATH, index=False)
    print("Saved rocky subset to", ROCKY_PATH)
    return rocky


def stage2_relabel(rocky_df):
    """Create binary 'hz_class' target on the rocky subset.

    Returns
    -------
    pandas.DataFrame
        Binary-labeled dataframe saved to data/processed/binary_data.csv.
    """
    df = rocky_df.copy()

    def map_hz(b):
        if b == 0:
            return 0
        if b in (1, 2, 3):
            return 1
        raise ValueError("Unexpected climate_bin in rocky subset: %s" % b)

    df["hz_class"] = df["climate_bin"].map(map_hz)
    print("hz_class value_counts (0=Outside HZ, 1=In HZ):")
    print(df["hz_class"].value_counts().sort_index())

    df.to_csv(BINARY_PATH, index=False)
    print("Saved binary data to", BINARY_PATH)
    return df


def load_features(df):
    """Return feature matrix X and binary target y.

    Returns
    -------
    (pandas.DataFrame, pandas.Series)
    """
    X = df[FEATURE_COLS]
    y = df["hz_class"]
    print("X shape:", X.shape)
    print("y value_counts:")
    print(y.value_counts().sort_index())
    return X, y


def split_and_scale(X, y):
    """Stratified train/test split, fit StandardScaler on train only.

    Returns
    -------
    (X_train_s, X_test_s, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print("Split sizes -> train:", len(X_train), "test:", len(X_test))
    print("Train class distribution:")
    print(y_train.value_counts().sort_index())

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    print("Scaler saved to", SCALER_PATH)
    return X_train_s, X_test_s, y_train, y_test


def train_models(X_train, y_train):
    """Train 4 balanced-weight classifiers, save as models/v2_{name}.pkl.

    Returns
    -------
    dict
        {name: fitted_model}
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    models = [
        LogisticRegression(max_iter=1000, class_weight="balanced"),
        DecisionTreeClassifier(class_weight="balanced", random_state=42),
        RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42
        ),
        SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
    ]
    fitted = {}
    for name, model in zip(MODEL_NAMES, models):
        model.fit(X_train, y_train)
        fitted[name] = model
        path = os.path.join(MODELS_DIR, "v2_" + name + ".pkl")
        joblib.dump(model, path)
        print("Trained and saved v2_" + name + " ->", path)
    return fitted


def evaluate_all(models, X_test, y_test):
    """Print per-model metrics, save results table, return ranked list."""
    rows = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        cm = confusion_matrix(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, labels=[1], average="binary", zero_division=0
        )
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        pr_auc = average_precision_score(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)

        print("\n--- %s ---" % name)
        print("Confusion matrix (rows=true, cols=pred):")
        print(cm)
        print("In-HZ class -> Precision: %.4f  Recall: %.4f  F1: %.4f" % (prec, rec, f1))
        print("Balanced accuracy: %.4f" % bal_acc)
        print("PR-AUC: %.4f  ROC-AUC: %.4f" % (pr_auc, roc_auc))

        rows.append(
            {
                "model": name,
                "precision_in_hz": prec,
                "recall_in_hz": rec,
                "f1_in_hz": f1,
                "balanced_accuracy": bal_acc,
                "pr_auc": pr_auc,
                "roc_auc": roc_auc,
            }
        )

    df = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    df.to_csv(REPORT_PATH, index=False)
    print("\nResults table saved to", REPORT_PATH)
    print("Models ranked by PR-AUC:")
    print(df.to_string(index=False))
    return df


def main():
    rocky_df = stage1_filter()
    binary_df = stage2_relabel(rocky_df)
    X, y = load_features(binary_df)
    X_train_s, X_test_s, y_train, y_test = split_and_scale(X, y)
    models = train_models(X_train_s, y_train)
    results = evaluate_all(models, X_test_s, y_test)
    best = results.iloc[0]
    print("\nBest model by PR-AUC:", best["model"])
    print("TWO-STAGE PIPELINE COMPLETE")


if __name__ == "__main__":
    main()