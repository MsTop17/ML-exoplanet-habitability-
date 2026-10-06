"""Generate the 7 paper figures for the two-stage pipeline.

Depends on outputs of src/03_ablation.py (models/ablation/*.pkl and
reports/ablation_results.csv). All figures saved at 300 DPI with tight
bounding boxes.
"""

import os
import glob
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
)

BINARY_PATH = "data/processed/binary_data.csv"
ABLATION_DIR = "models/ablation"
RESULT_PATH = "reports/ablation_results.csv"
FIG_DIR = "figures"
DPI = 300

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


def _save(fig, filename):
    path = os.path.join(FIG_DIR, filename)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def fig1_class_distribution(df):
    """Bar chart of raw class counts (Outside HZ vs In HZ), log scale."""
    counts = df["hz_class"].value_counts().sort_index()
    labels = ["Outside HZ\n(1086)", "In HZ\n(32)"]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(labels, [counts.get(0, 0), counts.get(1, 0)], color=["#d62728", "#2ca02c"])
    ax.set_yscale("log")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Binary class distribution (rocky planets, N=%d)" % len(df))
    for i, c in enumerate([counts.get(0, 0), counts.get(1, 0)]):
        ax.text(i, c * 1.3, str(c), ha="center")
    return _save(fig, "fig_class_distribution.png")


def fig2_scatter_hz(df):
    """pl_insol vs pl_rade scatter, boundaries, Earth reference."""
    fig, ax = plt.subplots(figsize=(7, 6))
    outside = df[df["hz_class"] == 0]
    inside = df[df["hz_class"] == 1]
    ax.scatter(
        outside["pl_insol"], outside["pl_rade"], c="#d62728", s=12, alpha=0.6, label="Outside HZ"
    )
    ax.scatter(
        inside["pl_insol"], inside["pl_rade"], c="#2ca02c", s=22, alpha=0.9, label="In HZ"
    )
    ax.axvline(0.35, color="blue", ls="--", lw=1, label="Snowball limit (S_eff=0.35)")
    ax.axvline(1.5, color="orange", ls="--", lw=1, label="Runaway greenhouse (S_eff=1.5)")
    ax.axhline(1.6, color="gray", ls="-.", lw=1, label="Fulton gap (R=1.6 Re)")
    ax.scatter(1, 1, marker="*", s=180, c="black", label="Earth reference")
    ax.set_xscale("log")
    ax.set_xlabel("pl_insol (log scale)")
    ax.set_ylabel("pl_rade (Earth radii)")
    ax.set_title("Habitable-zone classification of rocky planets")
    ax.legend(fontsize=8, loc="upper right")
    return _save(fig, "fig_scatter_hz.png")


def fig3_correlation(df):
    """Correlation heatmap of all 12 features + hz_class."""
    cols = FEATURES_ALL + ["hz_class"]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        corr, annot=True, cmap="coolwarm", fmt=".2f",
        annot_kws={"size": 7}, ax=ax, cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature correlation (12 features + hz_class)")
    ax.tick_params(axis="x", rotation=45)
    return _save(fig, "fig_correlation.png")


def fig4_ablation():
    """Random Forest F1_macro across the 4 ablation experiments."""
    res = pd.read_csv(RESULT_PATH)
    rf = res[res["model"] == "random_forest"].copy()
    order = ["A", "B", "C", "D"]
    rf["exp"] = pd.Categorical(rf["experiment"], categories=order, ordered=True)
    rf = rf.sort_values("exp")
    vals = rf["f1_macro"].fillna(0.0).values
    names = ["A\n(all)", "B\n(no pl_insol)", "C\n(no insol/rade)", "D\n(atmosphere only)"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(names, vals, color=["#1f77b4", "#ff7f0e", "#9467bd", "#8c564b"])
    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 0.01,
            "%.3f" % v,
            ha="center",
            fontsize=9,
        )
    ax.set_ylabel("Random Forest F1_macro")
    ax.set_title("Ablation: predictive signal vs feature set")
    ax.set_ylim(0, 1.15)
    return _save(fig, "fig_ablation.png")


def _load_exp_models(exp):
    out = {}
    for name in MODEL_NAMES:
        path = os.path.join(ABLATION_DIR, "exp%s_%s.pkl" % (exp, name))
        out[name] = joblib.load(path)
    return out


def _get_scaled_split(df, feature_list):
    """Recreate the exact scaled split used in 03 (deterministic)."""
    X = df[feature_list]
    y = df["hz_class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, y_train, y_test


def fig5_confusion_rf(df, models_a, X_test_s, y_test):
    """Confusion matrix for Random Forest, Experiment A."""
    cm = confusion_matrix(y_test, models_a["random_forest"].predict(X_test_s))
    fig, ax = plt.subplots(figsize=(5.5, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Pred 0\n(Outside HZ)", "Pred 1\n(In HZ)"],
        yticklabels=["True 0\n(Outside HZ)", "True 1\n(In HZ)"],
    )
    ax.set_title("Random Forest confusion matrix (Experiment A)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    return _save(fig, "fig_confusion_rf.png")


def fig6_roc(df, models_a, X_test_s, y_test):
    """ROC curves for all 4 Experiment A models."""
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for name, model in models_a.items():
        proba = model.predict_proba(X_test_s)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=1.8, label="%s (AUC=%.3f)" % (name, roc_auc))
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (Experiment A, In-HZ class)")
    ax.legend(fontsize=8)
    return _save(fig, "fig_roc.png")


def fig7_shap(df, models_a, X_train_s):
    """SHAP summary plot for Random Forest, Experiment A."""
    explainer = shap.TreeExplainer(models_a["random_forest"])
    sv = np.asarray(explainer.shap_values(X_train_s))
    if sv.ndim == 3:
        shap_values = sv[:, :, 1]
    elif isinstance(sv, list):
        shap_values = sv[1]
    else:
        shap_values = sv
    ensure = np.asarray(shap_values)
    shap.summary_plot(ensure, X_train_s, feature_names=FEATURES_ALL, show=False)
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    return _save(fig, "fig_shap.png")


def main():
    df = pd.read_csv(BINARY_PATH)
    print("Loaded", BINARY_PATH, "shape:", df.shape)

    os.makedirs(FIG_DIR, exist_ok=True)

    paths = []
    paths.append(fig1_class_distribution(df))
    print("Figure 1 done:", paths[-1])

    paths.append(fig2_scatter_hz(df))
    print("Figure 2 done:", paths[-1])

    paths.append(fig3_correlation(df))
    print("Figure 3 done:", paths[-1])

    paths.append(fig4_ablation())
    print("Figure 4 done:", paths[-1])

    models_a = _load_exp_models("A")
    _, X_test_s, _, y_test = _get_scaled_split(df, FEATURES_ALL)

    paths.append(fig5_confusion_rf(df, models_a, X_test_s, y_test))
    print("Figure 5 done:", paths[-1])

    paths.append(fig6_roc(df, models_a, X_test_s, y_test))
    print("Figure 6 done:", paths[-1])

    X_train_s, _, _, _ = _get_scaled_split(df, FEATURES_ALL)
    paths.append(fig7_shap(df, models_a, X_train_s))
    print("Figure 7 done:", paths[-1])

    print("\nSummary")
    print("Raw class counts: Outside HZ = %d, In HZ = %d" % (
        int((df["hz_class"] == 0).sum()), int((df["hz_class"] == 1).sum())))

    ablation = pd.read_csv(RESULT_PATH)
    for exp in ["A", "B", "C", "D"]:
        rows = ablation[ablation["experiment"] == exp].dropna(subset=["f1"])
        if rows.empty:
            print("Experiment", exp, ": no valid F1")
            continue
        best = rows.loc[rows["f1"].idxmax()]
        print("Experiment", exp, "- best model:", best["model"], "(F1=%.3f)" % best["f1"])

    rf = ablation[(ablation["model"] == "random_forest") & (ablation["experiment"].isin(["A", "B"]))]
    rf = rf.set_index("experiment")["f1_macro"]
    a, b = rf.get("A", None), rf.get("B", None)
    if a is not None and b is not None and not np.isnan(a) and not np.isnan(b):
        print("Ablation story: RF F1_macro drops from %.3f to %.3f when pl_insol is removed" % (a, b))
    else:
        print("Ablation story: unable to compute F1 drop (N/A values)")

    print("\nFigures:")
    for p in paths:
        print("  %-28s %.1f KB" % (os.path.basename(p), os.path.getsize(p) / 1024))
    print("FIGURES COMPLETE")


if __name__ == "__main__":
    main()