"""Preprocess merged exoplanet data and create 5-class climate labels.

Habitable-zone bin boundaries follow the runaway/moist greenhouse limits of
Kasting et al. (1993) and the updated stellar flux thresholds (S_eff) of
Kopparapu et al. (2013). The radius boundary separating terrestrial planets
from small gas giants uses the Fulton gap at 1.6 Earth radii from
Fulton et al. (2017). The Earth Similarity Index (ESI) formulation follows
Schulze-Makuch et al. (2011).
"""

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

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
]

CLEAN_PATH = "data/processed/clean_data.csv"
LABELED_PATH = "data/processed/labeled_data.csv"


def load_and_clean():
    """Load raw data, drop sparse rows, impute features, derive features, save.

    Returns
    -------
    pandas.DataFrame
        Cleaned dataframe saved to data/processed/clean_data.csv.
    """
    df = pd.read_csv("data/raw/merged_raw.csv")
    print("Original shape:", df.shape)

    # Drop rows where more than half of the feature columns are missing.
    threshold = 0.5 * len(FEATURE_COLS)
    n_missing = df[FEATURE_COLS].isna().sum(axis=1)
    df = df.loc[n_missing <= threshold].copy()
    print("Shape after dropping rows with >50% NaN features:", df.shape)

    # KNN imputation on FEATURE_COLS only (pl_bmasse handled separately).
    imputer = KNNImputer(n_neighbors=5)
    df[FEATURE_COLS] = imputer.fit_transform(df[FEATURE_COLS])
    print("KNNImputer(n_neighbors=5) applied to FEATURE_COLS")

    # pl_bmasse is imputed separately with the column median.
    median_bmasse = df["pl_bmasse"].median()
    df["pl_bmasse"] = df["pl_bmasse"].fillna(median_bmasse)
    print("pl_bmasse median-imputed with:", median_bmasse)

    # Derived features.
    df["log_insol"] = np.log1p(df["pl_insol"].clip(lower=0))
    df["log_rade"] = np.log1p(df["pl_rade"].clip(lower=0))
    df["greenhouse_index"] = df["co2_frac"] + df["h2o_frac"] + df["ch4_frac"]
    print("Derived features created: log_insol, log_rade, greenhouse_index")

    # Earth Similarity Index (Schulze-Makuch et al., 2011).
    df["esi_radius"] = (1 - abs(df["pl_rade"] - 1) / (df["pl_rade"] + 1)) ** 0.57
    df["esi_mass"] = (1 - abs(df["pl_bmasse"] - 1) / (df["pl_bmasse"] + 1)) ** 1.07
    df["esi"] = np.sqrt(df["esi_radius"] * df["esi_mass"])
    df["esi"] = df["esi"].fillna(df["esi_radius"])
    print("ESI computed (radius and mass terms)")

    nan_counts = df[FEATURE_COLS + ["pl_bmasse", "esi"]].isna().sum()
    print("NaN count for feature columns:")
    print(nan_counts)

    df.to_csv(CLEAN_PATH, index=False)
    print("Saved clean data to", CLEAN_PATH)
    return df


def create_labels(df):
    """Assign 5-class climate_bin labels and save labeled data.

    Bin order (Kasting et al., 1993; Kopparapu et al., 2013; Fulton et al.,
    2017):
        4 -- Gas Giant (pl_rade >= 1.6, Fulton gap)
        0 -- Runaway Greenhouse (pl_insol > 1.5)
        1 -- Moist Greenhouse (pl_insol > 1.1)
        2 -- Stable Temperate (pl_insol >= 0.35)
        3 -- Snowball (pl_insol < 0.35)

    Returns
    -------
    pandas.DataFrame
        Labeled dataframe saved to data/processed/labeled_data.csv.
    """

    def climate_bin(row):
        if row["pl_rade"] >= 1.6:
            return 4
        if row["pl_insol"] > 1.5:
            return 0
        if row["pl_insol"] > 1.1:
            return 1
        if row["pl_insol"] >= 0.35:
            return 2
        return 3

    df = df.copy()
    df["climate_bin"] = df.apply(climate_bin, axis=1)
    print("climate_bin value_counts:")
    print(df["climate_bin"].value_counts().sort_index())

    df.to_csv(LABELED_PATH, index=False)
    print("Saved labeled data to", LABELED_PATH)
    return df


def main():
    """Run the full preprocessing pipeline."""
    load_and_clean()
    print("Loaded clean_data.csv for labeling")
    clean_df = pd.read_csv(CLEAN_PATH)
    labeled_df = create_labels(clean_df)
    print("PREPROCESSING COMPLETE")
    print("Final row count:", len(labeled_df))
    print("Class distribution:")
    print(labeled_df["climate_bin"].value_counts().sort_index())


if __name__ == "__main__":
    main()
