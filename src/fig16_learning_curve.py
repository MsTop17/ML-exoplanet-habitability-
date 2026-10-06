"""Learning curve for the SVM classifier (macro-F1 vs training set size).

Label space note: models/svc.pkl was trained on the 4 merged 'climate_class'
labels produced by 02_train.py, not on the raw 5-bin 'climate_bin'. Applying
the identical merge to 'climate_bin' keeps the target consistent with the
model's own classes_ ([0, 1, 2, 3]) so the curve measures the real task.
"""

import os
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve

LABELED_PATH = "data/processed/labeled_data.csv"
MODELS_DIR = "models"
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
SVC_PATH = os.path.join(MODELS_DIR, "svc.pkl")

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


def merge_climate_bin(b):
    """Map a raw 5-bin label onto the 4 merged classes used to train svc.pkl.

    Returns
    -------
    int
        0 Hot Non-Habitable, 1 Potentially Habitable,
        2 Cold Non-Habitable, 3 Gas Giant.
    """
    if b == 4:
        return 3
    if b == 0 or b == 1:
        return 0
    if b == 2:
        return 1
    if b == 3:
        return 2
    raise ValueError("Unexpected climate_bin: %s" % b)


def load_data():
    """Return scaled feature matrix X and target y.

    Returns
    -------
    (numpy.ndarray, pandas.Series)
    """
    df = pd.read_csv(LABELED_PATH)
    print("Loaded", LABELED_PATH, "shape:", df.shape)

    X = df[FEATURE_COLS]
    y = df["climate_bin"].map(merge_climate_bin)
    print("X shape:", X.shape, "| y shape:", y.shape)
    print("Class distribution:")
    print(y.value_counts().sort_index())

    scaler = joblib.load(SCALER_PATH)
    print("Scaler loaded from", SCALER_PATH)
    X = scaler.transform(X)
    return X, y


def compute_curve(svc, X, y):
    """Run learning_curve with 5-fold CV and macro-F1 scoring.

    Returns
    -------
    tuple of numpy.ndarray
        train_sizes, train_scores_mean, train_scores_std,
        val_scores_mean, val_scores_std
    """
    train_sizes, train_scores, val_scores = learning_curve(
        svc,
        X,
        y,
        cv=5,
        scoring="f1_macro",
        train_sizes=np.linspace(0.1, 1.0, 8),
        n_jobs=-1,
    )
    print("\nTraining set size | train f1_macro | val f1_macro")
    for size, tr, va in zip(train_sizes, train_scores.mean(axis=1), val_scores.mean(axis=1)):
        print("%15d | %13.4f | %11.4f" % (size, tr, va))
    return (
        train_sizes,
        train_scores.mean(axis=1),
        train_scores.std(axis=1),
        val_scores.mean(axis=1),
        val_scores.std(axis=1),
    )


def main():
    X, y = load_data()

    svc = joblib.load(SVC_PATH)
    print("SVM loaded from", SVC_PATH, "| classes:", svc.classes_)

    print("\nComputing learning curve (cv=5, scoring=f1_macro)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        sizes, tr_mean, tr_std, va_mean, va_std = compute_curve(svc, X, y)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        sizes,
        tr_mean,
        "o-",
        color="steelblue",
        linewidth=2,
        markersize=6,
        label="Training score",
    )
    ax.fill_between(sizes, tr_mean - tr_std, tr_mean + tr_std, alpha=0.15, color="steelblue")

    ax.plot(
        sizes,
        va_mean,
        "s-",
        color="crimson",
        linewidth=2,
        markersize=6,
        label="Cross-validation score",
    )
    ax.fill_between(sizes, va_mean - va_std, va_mean + va_std, alpha=0.15, color="crimson")

    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("F1-macro Score")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(loc="best")

    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/fig16_learning_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\nFigure 16 saved")


if __name__ == "__main__":
    main()