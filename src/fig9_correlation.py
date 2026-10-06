import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

FEATURES = [
    "pl_insol",
    "pl_rade",
    "pl_eqt",
    "st_teff",
    "st_mass",
    "co2_frac",
    "h2o_frac",
    "ch4_frac",
]


def synthetic_data(n=500):
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "pl_insol": rng.lognormal(0.0, 1.0, n),
            "pl_rade": rng.lognormal(0.0, 0.8, n),
            "pl_eqt": rng.uniform(200, 3000, n),
            "st_teff": rng.normal(5780, 800, n),
            "st_mass": rng.lognormal(0.0, 0.3, n),
            "co2_frac": rng.beta(2.0, 5.0, n),
            "h2o_frac": rng.beta(2.0, 5.0, n),
            "ch4_frac": rng.beta(1.5, 20.0, n),
        }
    )


def load_data():
    for path in ("data/processed/labeled_data.csv", "data/processed/clean_data.csv"):
        if os.path.exists(path):
            try:
                return pd.read_csv(path)
            except (OSError, pd.errors.ParserError) as exc:
                print(f"Could not read {path}: {exc}")

    print("No processed data found; generating synthetic data")
    return synthetic_data()


df = load_data()

cols = [c for c in FEATURES if c in df.columns]
if len(cols) < 3:
    print(f"Only {len(cols)} of {len(FEATURES)} feature columns found; using synthetic data")
    df = synthetic_data()
    cols = [c for c in FEATURES if c in df.columns]

data = df[cols].dropna()
print(f"Correlating {len(cols)} features over {len(data)} rows: {', '.join(cols)}")

corr = data[cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    square=True,
    linewidths=0.5,
    vmin=-1,
    vmax=1,
    ax=ax,
)

os.makedirs("figures", exist_ok=True)
fig.savefig("figures/fig9_correlation.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("Figure 9 saved")
