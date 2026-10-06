"""Re-label to 4 balanced classes and train 8 classifiers with imbalance handling.

The original 5 climate bins are merged into 4 classes so the tiny Moist (8) and
Snowball (6) bins are never modeled in isolation. Models are trained on
RandomOverSampler-augmented data and compared via 5-fold stratified
cross-validation using macro-averaged F1.

Class mapping:
    4 Gas Giant  -> 3 (Gas Giant)
    0/1          -> 0 (Hot Non-Habitable)
    2            -> 1 (Potentially Habitable)
    3            -> 2 (Cold Non-Habitable)
"""

import joblib
import os
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import RandomOverSampler

LABELED_PATH = "data/processed/labeled_data.csv"
FINAL_PATH = "data/processed/final_data.csv"
MODELS_DIR = "models"
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

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

MODEL_NAMES = [
    "logistic_regression",
    "knn",
    "decision_tree",
    "random_forest",
    "gradient_boosting",
    "svc",
    "gaussian_nb",
    "mlp",
]


def load_and_relabel():
    """Merge rare climate bins into 4 balanced classes and save.

    Returns
    -------
    pandas.DataFrame
        Dataframe with new 'climate_class' target, saved to final_data.csv.
    """
    df = pd.read_csv(LABELED_PATH)
    print("Loaded", LABELED_PATH, "shape:", df.shape)

    def map_bin(b):
        if b == 4:
            return 3
        if b == 0 or b == 1:
            return 0
        if b == 2:
            return 1
        if b == 3:
            return 2
        raise ValueError("Unexpected climate_bin: %s" % b)

    df["climate_class"] = df["climate_bin"].map(map_bin)
    print("climate_class value_counts:")
    print(df["climate_class"].value_counts().sort_index())

    df.to_csv(FINAL_PATH, index=False)
    print("Saved final data to", FINAL_PATH)
    return df


def load_features(df=None):
    """Return feature matrix X and target vector y.

    Returns
    -------
    (pandas.DataFrame, pandas.Series)
    """
    if df is None:
        df = pd.read_csv(FINAL_PATH)
    X = df[FEATURE_COLS]
    y = df["climate_class"]
    print("X shape:", X.shape, "| y shape:", y.shape)
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


def apply_oversampling(X_train, y_train):
    """RandomOverSampler to balance tiny classes before training.

    Returns
    -------
    (X_res, y_res)
    """
    print("Class distribution BEFORE oversampling (y_train):")
    print(y_train.value_counts().sort_index())
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X_train, y_train)
    print("Class distribution AFTER oversampling:")
    print(pd.Series(y_res).value_counts().sort_index())
    return X_res, y_res


def train_all_models(X_res, y_res):
    """Train the 8 configured models and save each to models/{name}.pkl."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    models = [
        LogisticRegression(max_iter=1000, class_weight="balanced"),
        KNeighborsClassifier(n_neighbors=5),
        DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced"
        ),
        GradientBoostingClassifier(random_state=42),
        SVC(kernel="rbf", probability=True, random_state=42, class_weight="balanced"),
        GaussianNB(),
        MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42
        ),
    ]
    for name, model in zip(MODEL_NAMES, models):
        model.fit(X_res, y_res)
        path = os.path.join(MODELS_DIR, name + ".pkl")
        joblib.dump(model, path)
        print("Trained and saved", name, "->", path)


def cross_validate(X_res, y_res):
    """Stratified 5-fold CV with f1_macro, printed sorted best-first."""
    models = [
        LogisticRegression(max_iter=1000, class_weight="balanced"),
        KNeighborsClassifier(n_neighbors=5),
        DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced"
        ),
        GradientBoostingClassifier(random_state=42),
        SVC(kernel="rbf", probability=True, random_state=42, class_weight="balanced"),
        GaussianNB(),
        MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42),
    ]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    for name, model in zip(MODEL_NAMES, models):
        scores = cross_val_score(
            model, X_res, y_res, cv=cv, scoring="f1_macro", n_jobs=-1
        )
        results.append((name, scores.mean(), scores.std()))
        print(
            "CV %s: f1_macro = %.4f (+/- %.4f)"
            % (name, scores.mean(), scores.std())
        )

    print("\nCross-validation results (sorted):")
    for name, mean, std in sorted(results, key=lambda r: r[1], reverse=True):
        print("%-20s %.4f +/- %.4f" % (name, mean, std))
    return results


def main():
    load_and_relabel()
    print("Loading final data for training")
    X, y = load_features()
    print("Splitting and scaling features")
    X_train_s, X_test_s, y_train, y_test = split_and_scale(X, y)
    print("Applying RandomOverSampler")
    X_res, y_res = apply_oversampling(X_train_s, y_train)
    print("Training all 8 models")
    train_all_models(X_res, y_res)
    print("Running 5-fold cross-validation")
    results = cross_validate(X_res, y_res)
    best = max(results, key=lambda r: r[1])
    print("Best model:", best[0], "with f1_macro = %.4f" % best[1])
    print("TRAINING COMPLETE")


if __name__ == "__main__":
    main()