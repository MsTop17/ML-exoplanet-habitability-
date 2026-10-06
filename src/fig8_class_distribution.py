import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABELS = ["Runaway GHG", "Moist GHG", "Stable Temp", "Snowball", "Gas Giant"]
COLORS = ["red", "orange", "green", "blue", "gray"]


def assign_bin(pl_rade, pl_insol):
    if pl_rade >= 1.6:
        return 4
    if pl_insol > 1.5:
        return 0
    if pl_insol > 1.1:
        return 1
    if pl_insol >= 0.35:
        return 2
    return 3


def load_data():
    if os.path.exists("data/processed/labeled_data.csv"):
        return pd.read_csv("data/processed/labeled_data.csv")

    if os.path.exists("data/processed/clean_data.csv"):
        df = pd.read_csv("data/processed/clean_data.csv")
        df["climate_bin"] = [
            assign_bin(r, s) for r, s in zip(df["pl_rade"], df["pl_insol"])
        ]
        return df

    print("No processed data found; generating synthetic data")
    rng = np.random.default_rng(42)
    n = 1000
    n_giant = 800
    rocky_insol = np.concatenate(
        [
            rng.uniform(1.5, 4.0, 120),
            rng.uniform(1.1, 1.5, 30),
            rng.uniform(0.35, 1.1, 40),
            rng.uniform(0.01, 0.35, n - n_giant - 120 - 30 - 40),
        ]
    )
    pl_insol = np.concatenate(
        [rocky_insol, rng.uniform(0.01, 20.0, n_giant)]
    )
    pl_rade = np.concatenate(
        [rng.uniform(0.3, 1.6, len(rocky_insol)), rng.uniform(1.6, 15.0, n_giant)]
    )
    order = rng.permutation(n)
    return pd.DataFrame(
        {"pl_rade": pl_rade[order], "pl_insol": pl_insol[order]}
    )


df = load_data()

if "climate_bin" not in df.columns:
    df["climate_bin"] = [
        assign_bin(r, s) for r, s in zip(df["pl_rade"], df["pl_insol"])
    ]

counts = df["climate_bin"].value_counts().reindex(range(5), fill_value=0)

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    np.arange(5),
    counts.values,
    color=COLORS,
    edgecolor="black",
    linewidth=0.8,
)

offset = max(counts.max() * 0.01, 1)
for bar, value in zip(bars, counts.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + offset,
        f"{value:,}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

ax.set_xticks(np.arange(5))
ax.set_xticklabels(LABELS)
ax.set_xlabel("Climate Stability Bin")
ax.set_ylabel("Number of Planets")
ax.set_ylim(0, counts.max() * 1.12)
ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
ax.set_axisbelow(True)

os.makedirs("figures", exist_ok=True)
fig.savefig("figures/fig8_class_distribution.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("Figure 8 saved")
