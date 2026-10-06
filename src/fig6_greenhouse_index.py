import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

CANDIDATE_FILES = [
    "data/processed/labeled_data.csv",
    "data/processed/clean_data.csv",
]


def load_data():
    for path in CANDIDATE_FILES:
        if os.path.exists(path):
            try:
                return pd.read_csv(path), path
            except (OSError, pd.errors.ParserError) as exc:
                print(f"Could not read {path}: {exc}")
    print("No processed data found; generating synthetic data")
    rng = np.random.default_rng(42)
    n = 500
    n_rocky = 200
    pl_rade = np.concatenate(
        [rng.uniform(0.3, 1.6, n_rocky), rng.uniform(1.6, 12.0, n - n_rocky)]
    )
    co2_frac = rng.beta(2.0, 5.0, n)
    h2o_frac = rng.beta(2.0, 5.0, n)
    ch4_frac = rng.beta(1.5, 20.0, n)
    return (
        pd.DataFrame(
            {
                "pl_rade": pl_rade,
                "co2_frac": co2_frac,
                "h2o_frac": h2o_frac,
                "ch4_frac": ch4_frac,
            }
        ),
        None,
    )


df, source = load_data()
if source is not None:
    print(f"Loaded {source}: {len(df)} rows")

if "greenhouse_index" not in df.columns:
    df["greenhouse_index"] = df["co2_frac"] + df["h2o_frac"] + df["ch4_frac"]

df["category"] = np.where(
    df["pl_rade"] < 1.6, "Rocky (R < 1.6)", "Gas Giant (R >= 1.6)"
)
df = df.dropna(subset=["greenhouse_index", "pl_rade"])

fig, ax = plt.subplots(figsize=(8, 6))

sns.violinplot(
    data=df,
    x="category",
    y="greenhouse_index",
    hue="category",
    hue_order=["Rocky (R < 1.6)", "Gas Giant (R >= 1.6)"],
    palette=["steelblue", "orange"],
    legend=False,
    ax=ax,
)

ax.set_xlabel("Planetary Radius Category")
ax.set_ylabel("Greenhouse Index (f_CO2 + f_H2O + f_CH4)")
ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)

os.makedirs("figures", exist_ok=True)
fig.savefig("figures/fig6_greenhouse_index.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("Figure 6 saved")
