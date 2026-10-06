"""Feature ablation for the two-stage pipeline.

Ablates pl_insol (and radius/atmosphere) as features to expose how much
predictive signal survives after removing the exact variable used to derive
the hz_class labels. Runs 4 experiments (A-D) with 4 binary classifiers each.
No oversampling; class_weight='balanced' only. Models are saved per experiment
under models/ablation/ for reuse by src/04_evaluate_figures.py.
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
    precision_recall_fscore_support,
    average_precision_score,
    roc_auc_score,
    f1_score,
)

BINARY_PATH = "data/processed/binary_data.csv"
ABLATION_DIR = "models/ablation"
RESULT_PATH = "reports/ablation_results.csv"

FEATURES_ALL = [
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

EXPERIMENTS = {
    "A": {
        "name": "A - All features (baseline)",
        "features": FEATURES_ALL,
    },
    "B": {
        "name": "B - No insolation",
        "features": [f for f in FEATURES_ALL if f != "pl_insol"],
    },
    "C": {
        "name": "C - No insolation, no radius",
        "features": [f for f in FEATURES_ALL if f not in ("pl_insol", "pl_rade")],
    },
    "D": {
        "name": "D - Atmosphere only",
        "features": [
            "co2_frac",
            "h2o_frac",
            "ch4_frac",
            "o2_frac",
            "n2_frac",
            "greenhouse_index",
        ],
    },
}


def _safe(fn):
    """Run fn, returning None (printed as 'N/A') on failure."""
    try:
        return fn()
    except Exception as exc:
        print("    (metric unavailable:", str(exc)[:60], ")")
        return None


def load_binary():
    """Load binary-class rocky-planet dataset.

    Returns
    -------
    pandas.DataFrame
    """
    df = pd.read_csv(BINARY_PATH)
    print("Loaded", BINARY_PATH, "shape:", df.shape)
    return df


def train_with_features(X, y, feature_key, feature_list):
    """Train 4 balanced classifiers on the given feature subset.

    Parameters
    ----------
    X, y : features and binary target (hz_class)
    feature_key : str, experiment id (A/B/C/D) used for model filenames
    feature_list : list of feature columns to use

    Returns
    -------
    dict
        Experiment label, feature count, and per-model metric dicts.
    """
    X_sub = X[feature_list]
    X_train, X_test, y_train, y_test = train_test_split(
        X_sub, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    print("  Split -> train:", len(y_train), "test:", len(y_test))
    print("  Train In-HZ count:", int(y_train.sum()))

    models = [
        LogisticRegression(max_iter=1000, class_weight="balanced"),
        DecisionTreeClassifier(class_weight="balanced", random_state=42),
        RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42
        ),
        SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
    ]

    os.makedirs(ABLATION_DIR, exist_ok=True)
    results = {}
    for name, model in zip(MODEL_NAMES, models):
        model.fit(X_train_s, y_train)
        joblib.dump(
            model, os.path.join(ABLATION_DIR, "exp%s_%s.pkl" % (feature_key, name))
        )

        y_pred = model.predict(X_test_s)
        prec, rec, f1 = _safe(
            lambda: precision_recall_fscore_support(
                y_test, y_pred, labels=[1], average="binary", zero_division=0
            )[:3]
        ) or (None, None, None)
        f1_macro = _safe(lambda: f1_score(y_test, y_pred, average="macro"))
        pr_auc = _safe(
            lambda: average_precision_score(y_test, model.predict_proba(X_test_s)[:, 1])
        )
        roc_auc = _safe(
            lambda: roc_auc_score(y_test, model.predict_proba(X_test_s)[:, 1])
        )

        results[name] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "f1_macro": f1_macro,
        }
        print(
            "  %-20s P=%.3f R=%.3f F1=%.3f F1m=%.3f PR-AUC=%.3f ROC-AUC=%.3f"
            % (
                name,
                prec if prec is not None else -1,
                rec if rec is not None else -1,
                f1 if f1 is not None else -1,
                f1_macro if f1_macro is not None else -1,
                pr_auc if pr_auc is not None else -1,
                roc_auc if roc_auc is not None else -1,
            )
        )

    return {
        "experiment": feature_key,
        "name": EXPERIMENTS[feature_key]["name"],
        "n_features": len(feature_list),
        "models": results,
    }


def _fmt(v):
    return "N/A" if v is None else "%.4f" % v


def main():
    df = load_binary()
    X = df[FEATURES_ALL]
    y = df["hz_class"]
    print("hz_class value_counts:")
    print(y.value_counts().sort_index())

    rows = []
    summaries = {}
    for key, spec in EXPERIMENTS.items():
        print("\n=== EXPERIMENT %s: %s (features: %d) ===" % (key, spec["name"], len(spec["features"])))
        exp = train_with_features(X, y, key, spec["features"])
        summaries[key] = exp
        for model, m in exp["models"].items():
            rows.append(
                {
                    "experiment": key,
                    "name": exp["name"],
                    "n_features": exp["n_features"],
                    "model": model,
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "f1": m["f1"],
                    "f1_macro": m["f1_macro"],
                    "pr_auc": m["pr_auc"],
                    "roc_auc": m["roc_auc"],
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(RESULT_PATH, index=False)
    print("\nConsolidated ablation results saved to", RESULT_PATH)
    print(out.to_string(index=False))

    rf_a = summaries["A"]["models"]["random_forest"]["f1_macro"]
    rf_b = summaries["B"]["models"]["random_forest"]["f1_macro"]
    print("\nAblation story: RF f1_macro drops from %s to %s when pl_insol is removed"
          % (_fmt(rf_a), _fmt(rf_b)))
    print("ABLATION COMPLETE")


if __name__ == "__main__":
    main()